import os

import click
from omegaconf import OmegaConf

from calt.io import IOPipeline, LastElementLoadPreprocessor
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/train.yaml",
    help="Path to train config YAML (model, train, data).",
)
@click.option(
    "--target_mode",
    type=click.Choice(["full", "last_element"]),
    default="last_element",
    help="Target format: 'full' = full cumulative (3,5,5 → 3,8,13), 'last_element' = last value only (3,5,5 → 13).",
)
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode with reduced epochs and data for quick testing",
)
@click.option(
    "--wandb_runname_postfix",
    type=str,
    default=None,
    help="Postfix appended to wandb run name (e.g. 'full', 'last_element') for distinguishing runs.",
)
def main(
    config_path: str, target_mode: str, dryrun: bool, wandb_runname_postfix: str | None
):
    """Train a model for arithmetic addition task."""
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
    os.makedirs(save_dir, exist_ok=True)
    # Save target_mode so load_from_checkpoint uses the same dataset preprocessor at eval time
    cfg.data.target_mode = target_mode
    if target_mode == "last_element":
        cfg.data.target_delimiter = ","
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)

    if target_mode == "last_element":
        io_pipeline.dataset_load_preprocessor = LastElementLoadPreprocessor(
            delimiter=","
        )
    else:
        io_pipeline.dataset_load_preprocessor = None  # full cumulative

    io_dict = io_pipeline.build()

    # プリプロセッサ適用後のサンプルを数件表示（適用確認用）
    train_ds = io_dict["train_dataset"]
    n_show = min(5, len(train_ds.input_texts))
    print(
        f"[Preprocessor check] train_dataset: {len(train_ds.input_texts)} samples, showing first {n_show}:"
    )
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
