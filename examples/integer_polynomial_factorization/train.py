import os

import click
from omegaconf import OmegaConf

from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


@click.command()
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode with reduced epochs and data for quick testing",
)
def main(dryrun: bool):
    """Train a model for integer_polynomial_factorization task."""
    cfg = OmegaConf.load("configs/train.yaml")

    if dryrun:
        print("=" * 80)
        print("Running in DRYRUN mode")
        print("=" * 80)
        apply_dryrun_settings(cfg)
        print(f"Training epochs: {cfg.train.num_train_epochs}")
        print(f"Batch size: {cfg.train.batch_size}")
        print(f"Test batch size: {cfg.train.test_batch_size}")
        print(f"Training samples: {cfg.data.num_train_samples}")
        print(f"Test samples: {cfg.data.num_test_samples}")
        print("=" * 80)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_dict = IOPipeline.from_config(cfg.data).build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    success_rate = trainer_pipeline.evaluate_and_save_generation()
    print(f"Success rate: {100 * success_rate:.1f}%")


if __name__ == "__main__":
    main()
