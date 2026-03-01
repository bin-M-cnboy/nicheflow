from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from nicheflow.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    manual_seed,
    print_config,
    print_exceptions,
)

eval_runs = 5
base_output_path = Path("outputs/eval_lightning")
_logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
@print_exceptions
def main(config: DictConfig) -> float | None:
    # 注册解析规则
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.resolve(config)
    print_config(config)

    # 确保检查点 (ckpt) 路径已提供
    ckpt_path = config.ckpt_path
    if ckpt_path is None:
        raise ValueError("You need to provide a checkpoint path for evaluation!")

    _logger.info(f"正在实例化 datamodule <{config.data.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.data.datamodule)

    _logger.info(f"正在实例化 model <{config.model._target_}>")
    model: LightningModule = instantiate(config.model)

    _logger.info("实例化各类回调函数 callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(config.get("callbacks"))

    _logger.info("实例化日志记录器 loggers...")
    logger: list[Logger] = instantiate_loggers(config.get("logger"))

    _logger.info(f"正在实例化 trainer (训练/推理器) <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        _logger.info("正在记录实验超参数！")
        log_hyperparameters(object_dict)

    result_list = []
    # 循环执行测试/验证
    for run_id in range(eval_runs):
        _logger.info(f"评估轮次 {run_id + 1}/{eval_runs}")
        manual_seed(int(config.seed) + run_id)

        # 挂载预先定好的 ckpt 给 Pytorch Lightning，启动自动加载和测试流程
        results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

        result_list.append({**results[0], "run_id": run_id})

    # 将结果保存为 CSV 便于后续统计
    df = pd.DataFrame(result_list)
    base_output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_output_path.joinpath(f"{config.logger.wandb.name}.csv"), index=False)


if __name__ == "__main__":
    main()
