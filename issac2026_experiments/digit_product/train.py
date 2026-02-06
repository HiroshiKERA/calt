"""Train model for Prod task: digit-level product (L=10)."""

import os

import click
from omegaconf import OmegaConf

from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/train.yaml",
    help="Path to train config YAML.",
)
@click.option("--dryrun", is_flag=True, help="Run in dryrun mode.")
@click.option(
    "--target_reversed",
    is_flag=True,
    help="Target sequence reversed (for evaluating reverse vs non-reverse performance).",
)
def main(config_path: str, dryrun: bool, target_reversed: bool) -> None:
    """Train a model for Prod (digit-level product, L=10)."""
    cfg = OmegaConf.load(config_path)
    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    if target_reversed:
        save_dir = save_dir.rstrip("/") + "/reversed"
        cfg.train.save_dir = save_dir
        if hasattr(cfg.train, "wandb") and hasattr(cfg.train.wandb, "name"):
            base = cfg.train.wandb.name or "run"
            cfg.train.wandb.name = f"{base}_reversed"
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_dict = IOPipeline.from_config(cfg.data).build()
    if target_reversed:
        for key in ("train_dataset", "val_dataset", "test_dataset"):
            if key in io_dict and io_dict[key] is not None:
                ds = io_dict[key]
                ds.target_texts = [t[::-1] for t in ds.target_texts]
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    success_rate = trainer_pipeline.evaluate_and_save_generation()
    print(f"Success rate: {100 * success_rate:.1f}%")


if __name__ == "__main__":
    main()
