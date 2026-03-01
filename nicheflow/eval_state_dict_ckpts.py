from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from omegaconf import OmegaConf

from nicheflow.tasks import FlowMatching
from nicheflow.utils import RankedLogger, manual_seed

_logger = RankedLogger(__name__, rank_zero_only=True)

# 初始化 hydra
initialize(config_path="../configs", version_base=None)

# 为 OmegaConf 注册解析器
OmegaConf.register_new_resolver("add", lambda x, y: x + y)

# 定义一些常数常量
base_experiment_path = Path("configs/experiment")
base_ckpt_path = Path("ckpts")
base_output_path = Path("outputs/eval")
models = ["nicheflow", "rpcflow", "spflow"]
variants = ["cfm", "gvfm", "glvfm"]
datasets = ["med", "abd", "mba"]

model_to_ckpt = {"nicheflow": "NicheFlow", "rpcflow": "RPCFlow", "spflow": "SPFlow"}

# 验证/测试的循环次数
eval_runs = 5


def evaluate(experiment_override: str, ckpt_path: Path):
    config = compose(
        config_name="train",
        overrides=[f"experiment={experiment_override}"],
    )
    # 覆盖这些变量以解决 hydra 实例配置问题
    config.paths.output_dir = Path(config.paths.root_dir).joinpath("outputs")
    config.paths.work_dir = config.paths.root_dir

    OmegaConf.resolve(config)

    _logger.info(f"Instantiating datamodule <{config.data.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.data.datamodule)

    _logger.info(f"Instantiating model <{config.model._target_}>")
    model: FlowMatching = instantiate(config.model)

    # 加载状态字典（即模型的权重参数 State Dict）
    ckpt = torch.load(
        ckpt_path,
        weights_only=False,
        map_location="cpu",
    )
    # 将权重仅加载到模型的骨干网络上
    model.flow.backbone.load_state_dict(ckpt)

    _logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=None, logger=False, accelerator="auto"
    )

    result_list = []

    # 进行测试和评估
    for run_id in range(eval_runs):
        _logger.info(f"Evaluation run {run_id + 1}/{eval_runs}")
        manual_seed(int(config.seed) + run_id)

        # 启动测试，由手动构造并加载好权重的 model 执行
        results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=None,  # 由于上述已经手动加载 state_dict, 这里不需要利用 lightning 加载 ckpt
        )

        result_list.append({**results[0], "run_id": run_id})

    # 将结果输出到 CSV 文件
    df = pd.DataFrame(result_list)
    base_output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_output_path.joinpath(f"{config.logger.wandb.name}.csv"), index=False)


def main() -> None:
    for model in models:
        for variant in variants:
            for dataset in datasets:
                config_path = base_experiment_path.joinpath(model, variant, f"{dataset}.yaml")
                ckpt_path = base_ckpt_path.joinpath(
                    f"{model_to_ckpt[model]}_{variant.upper()}_{dataset.upper()}.ckpt"
                )
                if not config_path.exists():
                    raise FileNotFoundError(f"Cannot find configuration file {config_path}")

                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Cannot find checkpoint path {ckpt_path}")

                evaluate(
                    experiment_override=f"{model}/{variant}/{dataset}",
                    ckpt_path=ckpt_path,
                )


if __name__ == "__main__":
    main()
