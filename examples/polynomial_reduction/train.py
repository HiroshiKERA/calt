import os

import click
from load_preprocessor import PolynomialReductionLoadPreprocessor
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
@click.option("--order", type=click.Choice(["grevlex", "lex"]), default="grevlex")
@click.option(
    "--pattern", type=int, default=1, help="1=remainder only, 2=quotients|remainder"
)
def main(dryrun: bool, order: str, pattern: int):
    """Train a model for polynomial_reduction task."""
    cfg = OmegaConf.load("configs/train.yaml")

    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)
    io_pipeline.dataset_load_preprocessor = PolynomialReductionLoadPreprocessor(
        order=order, pattern=pattern
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
