import os

import click
from omegaconf import OmegaConf
from sage.all import ZZ, PolynomialRing

from calt.io import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    TextToSageLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings

# C/E expanded form: text -> SageMath (TextToSage) -> C/E string (ExpandedForm)
R = PolynomialRing(ZZ, "x0,x1,x2")
TEXT_DELIMITER = " | "
text_to_sage = TextToSageLoadPreprocessor(delimiter=TEXT_DELIMITER, ring=R)
expanded_form = ExpandedFormLoadPreprocessor(delimiter=TEXT_DELIMITER)
dataset_load_preprocessor = ChainLoadPreprocessor(text_to_sage, expanded_form)


@click.command()
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode with reduced epochs and data for quick testing",
)
def main(dryrun: bool):
    """Train a model for polynomial addition (C/E expanded form)."""
    cfg = OmegaConf.load("configs/train.yaml")

    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)
    io_pipeline.dataset_load_preprocessor = dataset_load_preprocessor
    io_dict = io_pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    success_rate = trainer_pipeline.evaluate_and_save_generation()
    print(f"Success rate: {100 * success_rate:.1f}%")


if __name__ == "__main__":
    main()
