from abc import abstractmethod
from collections.abc import Generator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets.st_dataset_base import STDatasetBase, STTrainDataItem, STValDataItem
from nicheflow.preprocessing import H5ADDatasetDataclass, load_h5ad_dataset_dataclass
from nicheflow.utils.datasets import create_kmeans_regions, init_worker_rng
from nicheflow.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class MicroEnvDatasetBase(STDatasetBase):
    def __init__(
        self,
        ds: H5ADDatasetDataclass,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
    ) -> None:
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
        )

        # 此属性必须由子类设置
        self.n_microenvs_per_slice: int
        self.resample_n_microenvs: int | None = None
        self.per_microenv_transforms = per_microenv_transforms
        # 保存来自数据集类的重要属性（记录时间点的邻居索引以构成微环境）
        self.timepoint_neighboring_indices = ds.timepoint_neighboring_indices

    @abstractmethod
    def _sample_microenvs_idxs(self, timepoint: str) -> list[list[int]]:
        raise NotImplementedError("The method `_sample_microenvs_idxs` must be implemented in child classes!")

    @abstractmethod
    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        raise NotImplementedError("The method `_get_timepoints` must be implemented in child classes!")

    def _mini_batch_ot(
        self, microenvs_t1: list[Data], microenvs_t2: list[Data]
    ) -> tuple[list[Data], list[Data]]:
        '''
        从采样集合中，获得重采样的细胞微环境集合
        resample ---> resampled_microenvs_t1, resampled_microenvs_t2
        '''
        
        # 提取各个微环境中的基因表达矩阵(X)和空间坐标(pos)
        X_t1 = torch.stack([el.x for el in microenvs_t1])
        X_t2 = torch.stack([el.x for el in microenvs_t2])
        pos_t1 = torch.stack([el.pos for el in microenvs_t1])
        pos_t2 = torch.stack([el.pos for el in microenvs_t2])

        features_size = X_t1.size(-1)       # 基因表达特征的维度
        coordinates_size = pos_t1.size(-1)  # 空间坐标的维度


        # 把基因表达特征和空间位置特征拼接在一起
        # 然后通过计算均值(mean)，池化得到每个微环境的中心表示(centroid)
        # 维度变为: (微环境数量_n_microenvs_per_slice, 基因数 + 坐标轴数)
        source = torch.cat([X_t1, pos_t1], dim=-1).mean(dim=1)
        target = torch.cat([X_t2, pos_t2], dim=-1).mean(dim=1)

        # 构建加权张量，对特征(ot_lambda)和空间坐标(1-ot_lambda)分别加权
        lambda_tensor = torch.cat(
            [
                torch.repeat_interleave(self.ot_lambda, features_size),
                torch.repeat_interleave(1 - self.ot_lambda, coordinates_size),
            ]
        )
        source *= lambda_tensor
        target *= lambda_tensor


        # 执行最优传输OT, 计算源(t1)到目标(t2)的匹配方案 pi
        pi = self.ot_plan_sampler.get_map(x0=source, x1=target)

        # 根据pi，从中采样 源-目标微环境索引对 resample
        source_idxs, target_idxs = self.ot_plan_sampler.sample_map(
            pi,
            batch_size=self.n_microenvs_per_slice
            if self.resample_n_microenvs is None
            else self.resample_n_microenvs,
            replace=False,
        )

        resampled_microenvs_t1 = [microenvs_t1[i] for i in source_idxs]
        resampled_microenvs_t2 = [microenvs_t2[i] for i in target_idxs]

        return resampled_microenvs_t1, resampled_microenvs_t2

    def _get_microenvs_t1_t2(self, index: int | None) -> tuple[list[Data], list[Data], str, str]:
        '''
        获取并配对两个连续时间节点的细胞微环境集合

        '''

        t1, t2 = self._get_timepoints(index=index)
        # 在定义的空间区域内均匀采样 提取微环境的中心索引
        microenv_idxs_t1 = self._sample_microenvs_idxs(t1)
        microenv_idxs_t2 = self._sample_microenvs_idxs(t2)

        # 切割截取点云子图(subgraph)，
        # 并分别数据增强，封装成 torch_geometric输入数据
        microenvs_t1: list[Data] = [
            self.per_microenv_transforms(
                self.timepoint_pc[t1].subgraph(torch.Tensor(idx).to(torch.int32))
            )
            for idx in microenv_idxs_t1
        ]
        microenvs_t2: list[Data] = [
            self.per_microenv_transforms(
                self.timepoint_pc[t2].subgraph(torch.Tensor(idx).to(torch.int32))
            )
            for idx in microenv_idxs_t2
        ]

        # 执行批内最优传输匹配组合
        microenvs_t1, microenvs_t2 = self._mini_batch_ot(
            microenvs_t1=microenvs_t1, microenvs_t2=microenvs_t2
        )

        return microenvs_t1, microenvs_t2, t1, t2


class InfiniteMicroEnvDataset(IterableDataset, MicroEnvDatasetBase):
    '''
    持续生成 匹配后的微环境对，用于 training 阶段的微环境数据加载器
    '''
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        n_microenvs_per_slice: int = 256,
        resample_n_microenvs: int = 64,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
    ) -> None:
        ds = load_h5ad_dataset_dataclass(data_fp)
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
            per_microenv_transforms=per_microenv_transforms,
        )
        self.seed = seed
        # 使用基于种子的生成器来对数据对进行采样
        # 并在 K区域 内对微环境进行采样
        self.rng = None

        self.k_regions = k_regions
        self.n_microenvs_per_slice = n_microenvs_per_slice
        self.resample_n_microenvs = resample_n_microenvs

        # 创建 KMeans 分区(将空间划分为 K 个区域)
        self.timepoint_regions_to_idx: dict[str, dict[int, list[int]]] = create_kmeans_regions(
            ds=ds, timepoint_pc=self.timepoint_pc, k_regions=self.k_regions, seed=self.seed
        )

        # 预先计算出我们要从每个区域里采集多少个微环境
        self.n_microenvs_per_region = self.n_microenvs_per_slice // self.k_regions
        if self.n_microenvs_per_region == 0:
            _logger.warning(
                "The number of microenvironments per slice must be larger than the number of regions!"
                + f"Got {self.n_microenvs_per_slice} microenvironments but only {self.k_regions} regions."
            )
            _logger.warning("Setting the microenvironments per slice to 1")
            self.n_microenvs_per_slice = k_regions
            self.n_microenvs_per_region = 1

    def _sample_microenvs_idxs(self, timepoint: str) -> list[list[int]]:
        '''
        微环境索引采样：从每个区域的候选点中随机抽取指定数量的中心点索引。
        '''
        region_to_idxs = self.timepoint_regions_to_idx[timepoint]

        selected_idxs: list[int] = []
        for region_id, region_idxs_list in region_to_idxs.items():
            region_idxs = np.array(region_idxs_list)

            if len(region_idxs) < self.n_microenvs_per_region:
                _logger.warning(
                    f"Region {region_id} at time {timepoint} has less microenvironemnts "
                    + f"than the microenviornments per region. It has {len(region_idxs)} "
                    + f"but we sample {self.n_microenvs_per_region}. Using `replace=True`"
                    + " during sampling."
                )
                sampled = self.rng.choice(
                    region_idxs, size=self.n_microenvs_per_region, replace=True
                )# 如果某个区域细胞太少，会自动切换到 replace=True（有放回抽样）。
            else:
                sampled = self.rng.choice(
                    region_idxs, size=self.n_microenvs_per_region, replace=False
                )
            selected_idxs.extend(sampled)

        return [self.timepoint_neighboring_indices[timepoint][i] for i in selected_idxs]

    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        '''
        时间点对抽取：随机挑出一对连续的时间点
        '''
        pair_idx = self.rng.integers(self.num_pairs)
        t1, t2 = self.consecutive_pairs[pair_idx]
        return t1, t2

    def __iter__(self) -> Generator[STTrainDataItem]:
        self.rng = init_worker_rng(seed=self.seed)
        while True:
            # 在无穷循环的训练数据集中，因为使用纯随机采样，因此传入 index = None
            microenvs_t1, microenvs_t2, t1, t2 = self._get_microenvs_t1_t2(index=None)
            yield {
                # 第一条连续切片（源分布）
                "X_t1": torch.stack([pc.x for pc in microenvs_t1]),
                "pos_t1": torch.stack([pc.pos for pc in microenvs_t1]),
                "t1_ohe": self.timepoint_pc[t1].t_ohe,  # One-Hot Encoding
                # 第二条连续切片（目标分布）
                "X_t2": torch.stack([pc.x for pc in microenvs_t2]),
                "pos_t2": torch.stack([pc.pos for pc in microenvs_t2]),
                "t2_ohe": self.timepoint_pc[t2].t_ohe,
            }


class TestMicroEnvDataset(Dataset, MicroEnvDatasetBase):
    '''验证和测试阶段，'''
    def __init__(
        self,
        data_fp: str,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
        upsample_factor: int = 1,
    ) -> None:
        ds = load_h5ad_dataset_dataclass(data_fp)
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
            per_microenv_transforms=per_microenv_transforms,
        )

        # 在测试集中，我们将每对连续切片经过 `upsample_factor` (上采样因子) 倍的多次采样与评估
        self.upsample_factor = upsample_factor
        self.length = upsample_factor * self.num_pairs

        self.n_microenvs_per_slice = ds.test_microenvs
        self.subsampled_timepoint_idx = ds.subsampled_timepoint_idx

    def _sample_microenvs_idxs(self, timepoint: str) -> list[list[int]]:
        '''固定采样实现：直接使用均匀网格采样的中心点索引，便于复现'''
        return [
            self.timepoint_neighboring_indices[timepoint][i]
            for i in self.subsampled_timepoint_idx[timepoint]
        ]

    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        if index is None:
            raise ValueError("Index cannot be None in TestMicroEnvDataset.")

        pair_idx = index // self.upsample_factor

        if pair_idx >= len(self.consecutive_pairs):
            raise ValueError(f"Index `{index}` is out of bounds for the number of pairs.")

        t1, t2 = self.consecutive_pairs[pair_idx]
        return t1, t2

    def __getitem__(self, index: int) -> STValDataItem:
        '''单样本获取'''
        microenvs_t1, microenvs_t2, t1, t2 = self._get_microenvs_t1_t2(index=index)
        return {
            "X_t1": torch.stack([pc.x for pc in microenvs_t1]),
            "pos_t1": torch.stack([pc.pos for pc in microenvs_t1]),
            "t1_ohe": self.timepoint_pc[t1].t_ohe,
            "X_t2": torch.stack([pc.x for pc in microenvs_t2]),
            "pos_t2": torch.stack([pc.pos for pc in microenvs_t2]),
            "t2_ohe": self.timepoint_pc[t2].t_ohe,
            
            # 我们还需要在整个时间切片范围内执行全局层面的分布评估
            # (例如生成所有细胞点云的Wasserstein距离)
            # 因此这里连同全部的时间全量空间坐标和细胞类型注释(Cell Types)一并传递
            "global_pos_t2": self.timepoint_pc[t2].pos, # 整体坐标参考
            "global_ct_t2": self.timepoint_pc[t2].ct,   # 整体分类参考
        }

    def __len__(self) -> int:
        return self.length
