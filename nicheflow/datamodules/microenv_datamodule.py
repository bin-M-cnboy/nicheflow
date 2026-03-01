from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets import InfiniteMicroEnvDataset, TestMicroEnvDataset
from nicheflow.utils import microenv_train_collate, microenv_val_collate


class MicroEnvDataModule(LightningDataModule):
    """
    点云微环境数据模块：负责管理训练和测试数据集的生命周期、Dataloader 的配置。
    """
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        n_microenvs_per_slice: int = 256,
        resample_n_microenvs: int = 64,
        ot_lambda: float = 0.1,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
        val_upsample_factor: int = 1,
        train_batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        # 初始化采样参数
        self.seed = int(seed)
        self.k_regions = k_regions
        self.n_microenvs_per_slice = n_microenvs_per_slice
        self.resample_n_microenvs = resample_n_microenvs
        self.val_upsample_factor = val_upsample_factor

        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        # 封装训练和测试共用的数据集基础参数
        self.common_dataset_args = {
            "data_fp": data_fp,
            "ot_lambda": ot_lambda,
            "ot_plan_sampler": ot_plan_sampler,
            "per_pc_transforms": per_pc_transforms,
            "per_microenv_transforms": per_microenv_transforms,
        }

    def prepare_data(self) -> None:
        """
        准备数据：实例化具体的 Dataset 类
        """
        # 训练集：使用无限迭代模型，支持空间分区采样
        self.train_dataset = InfiniteMicroEnvDataset(
            **self.common_dataset_args,
            seed=self.seed,
            k_regions=self.k_regions,
            n_microenvs_per_slice=self.n_microenvs_per_slice,
            resample_n_microenvs=self.resample_n_microenvs,
        )

        # 测试集：使用固定采样点，支持评估上采样
        self.test_dataset = TestMicroEnvDataset(
            **self.common_dataset_args, upsample_factor=self.val_upsample_factor
        )

    def train_dataloader(self) -> DataLoader:
        """
        配置训练阶段的 DataLoader
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # 开启内存锁定，加速 GPU 数据拷贝
            collate_fn=microenv_train_collate,  # 专门用于处理点云微环境 Batch 的整理函数
        )

    def eval_dl(self) -> DataLoader:
        """
        内部评估加载器配置（用于 Val 和 Test）
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,  # 测试阶段通常按样本逐个评估
            num_workers=self.num_workers,
            shuffle=False,  # 测试不打乱顺序
            pin_memory=True,
            collate_fn=microenv_val_collate,  # 测试集专用的整理函数（包含全局参考数据）
        )

    def val_dataloader(self) -> DataLoader:
        """ 验证阶段加载器 """
        return self.eval_dl()

    def test_dataloader(self) -> DataLoader:
        """ 测试阶段加载器 """
        return self.eval_dl()
