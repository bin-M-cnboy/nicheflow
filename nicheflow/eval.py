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
    # Resolve
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.resolve(config)
    print_config(config)

    # Ensure the checkpoint path is provided.
    ckpt_path = config.ckpt_path
    if ckpt_path is None:
        raise ValueError("You need to provide a checkpoint path for evaluation!")

    _logger.info(f"Instantiating datamodule <{config.data.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.data.datamodule)

    _logger.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = instantiate(config.model)

    _logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(config.get("callbacks"))

    _logger.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(config.get("logger"))

    _logger.info(f"Instantiating trainer <{config.trainer._target_}>")
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
        _logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    result_list = []
    # Evaluate
    for run_id in range(eval_runs):
        _logger.info(f"Evaluation run {run_id + 1}/{eval_runs}")
        manual_seed(int(config.seed) + run_id)

        results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

        result_list.append({**results[0], "run_id": run_id})

    df = pd.DataFrame(result_list)
    base_output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_output_path.joinpath(f"{config.logger.wandb.name}.csv"), index=False)


if __name__ == "__main__":
    main()
