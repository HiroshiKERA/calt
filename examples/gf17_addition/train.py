import os

import click
from omegaconf import OmegaConf

from calt.io import (
    IOPipeline,
    LastElementLoadPreprocessor,
    ReversedOrderLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


@click.command()
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode with reduced epochs and data for quick testing",
)
def main(dryrun: bool):
    """Train a model for gf17_addition task.

    You can choose from 3 patterns (enable exactly one):
    - Standard: target is the full cumulative sum (11,4,11,4 # 11,15,9,13)
    - Last element only: target is only the last value (11,4,11,4 # 13)
    - Reversed: target sequence is reversed (11,4,11,4 # 13,9,15,11)
    """
    cfg = OmegaConf.load("configs/train.yaml")

    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)

    # --- Enable exactly one of the following (comment out the others) ---
    # Standard: Do nothing (target is the full cumulative sum)
    # io_pipeline.dataset_load_preprocessor = None  # default

    # Last element only: target is just the last value (11,15,9,13 → 13)
    # io_pipeline.dataset_load_preprocessor = LastElementLoadPreprocessor(delimiter=",")

    # Reversed: reverse the target sequence (11,15,9,13 → 13,9,15,11)
    # io_pipeline.dataset_load_preprocessor = ReversedOrderLoadPreprocessor(delimiter=",")

    io_dict = io_pipeline.build()

    # プリプロセッサ適用後のサンプルを数件表示（適用確認用）
    train_ds = io_dict["train_dataset"]
    n_show = min(5, len(train_ds.input_texts))
    print(f"[Preprocessor check] train_dataset: {len(train_ds.input_texts)} samples, showing first {n_show}:")
    for i in range(n_show):
        inp = train_ds.input_texts[i]
        tgt = train_ds.target_texts[i]
        inp_short = inp if len(inp) <= 50 else inp[:47] + "..."
        tgt_short = tgt if len(tgt) <= 50 else tgt[:47] + "..."
        print(f"  [{i}] input:  {inp_short!r}")
        print(f"      target: {tgt_short!r}")
    print()

    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    success_rate = trainer_pipeline.evaluate_and_save_generation()
    print(f"Success rate: {100 * success_rate:.1f}%")


if __name__ == "__main__":
    main()
