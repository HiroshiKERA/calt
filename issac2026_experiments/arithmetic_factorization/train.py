import os

import click
from omegaconf import OmegaConf

from calt.io import IOPipeline, ReversedOrderLoadPreprocessor
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


@click.command()
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode with reduced epochs and data for quick testing",
)
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/train.yaml",
    help="Path to train config YAML (model, train, data).",
)
@click.option(
    "--wandb_runname_postfix",
    type=str,
    default=None,
    help="Postfix appended to wandb run name for distinguishing runs.",
)
@click.option(
    "--target_reversed",
    is_flag=True,
    help="Target = factors in reversed order (ReversedOrderLoadPreprocessor). Input is still n; only the order of factors in target is reversed (e.g. 2,3,5 -> 5,3,2).",
)
def main(
    dryrun: bool,
    config_path: str,
    wandb_runname_postfix: str | None,
    target_reversed: bool,
):
    """Train a model for integer_factorization task."""
    cfg = OmegaConf.load(config_path)

    if dryrun:
        apply_dryrun_settings(cfg)

    if (
        wandb_runname_postfix
        and hasattr(cfg.train, "wandb")
        and hasattr(cfg.train.wandb, "name")
    ):
        base_name = cfg.train.wandb.name or "run"
        cfg.train.wandb.name = f"{base_name}_{wandb_runname_postfix}"

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    if wandb_runname_postfix:
        save_dir = save_dir.rstrip("/") + "_" + wandb_runname_postfix
        cfg.train.save_dir = save_dir
    if target_reversed:
        save_dir = save_dir.rstrip("/") + "/reversed"
        cfg.train.save_dir = save_dir
        if hasattr(cfg.train, "wandb") and hasattr(cfg.train.wandb, "name"):
            base_name = cfg.train.wandb.name or "run"
            cfg.train.wandb.name = f"{base_name}_reversed"
    os.makedirs(save_dir, exist_ok=True)
    # Save target_mode so load_from_checkpoint uses the same dataset preprocessor at eval time
    cfg.data.target_mode = "reversed" if target_reversed else "full"
    if cfg.data.target_mode == "reversed":
        cfg.data.target_delimiter = ","
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)
    if target_reversed:
        io_pipeline.dataset_load_preprocessor = ReversedOrderLoadPreprocessor(
            delimiter=","
        )
    io_dict = io_pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    success_rate = trainer_pipeline.evaluate_and_save_generation()
    print(f"Success rate: {100 * success_rate:.1f}%")


if __name__ == "__main__":
    main()
