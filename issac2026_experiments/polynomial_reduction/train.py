"""Train a model for polynomial reduction (univariate quo/rem; C/E expanded form)."""

import os

import click
from omegaconf import OmegaConf
from sage.all import GF, PolynomialRing, QQ, RR, ZZ

from calt.io import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    TextToSageLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings


def get_ring(data_cfg):
    """Build a Sage polynomial ring from data_cfg.sampler (field_str, symbols, order)."""
    field_str = str(data_cfg.sampler.field_str)
    symbols = str(data_cfg.sampler.symbols)
    order = str(data_cfg.sampler.get("order", "lex"))
    if order == "grevlex":
        order = "degrevlex"
    if field_str in ("ZZ", "QQ", "RR"):
        field = {"ZZ": ZZ, "QQ": QQ, "RR": RR}[field_str]
    elif field_str.startswith("GF"):
        p = int(
            field_str[3:-1] if field_str.startswith("GF(") else field_str[2:]
        )
        if p <= 1:
            raise ValueError(f"Field size must be > 1: {field_str!r}")
        field = GF(p)
    else:
        raise ValueError(f"Unsupported field_str: {field_str!r}")
    return PolynomialRing(field, symbols, order=order)


@click.command()
@click.option(
    "--train_config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to train config YAML (model, train, data paths, lexer).",
)
@click.option(
    "--data_config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to data config YAML (sampler, problem_generator, dataset).",
)
@click.option(
    "--target_mode",
    type=click.Choice(["full", "last_element"]),
    default="full",
    help="Target format: 'full' = quotient | remainder; 'last_element' = remainder only.",
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
    train_config_path: str,
    data_config_path: str,
    target_mode: str,
    dryrun: bool,
    wandb_runname_postfix: str | None,
):
    """Train a model for polynomial reduction (C/E expanded form)."""
    cfg = OmegaConf.load(train_config_path)
    data_cfg = OmegaConf.load(data_config_path)

    if dryrun:
        apply_dryrun_settings(cfg)

    if wandb_runname_postfix and hasattr(cfg.train, "wandb") and hasattr(cfg.train.wandb, "name"):
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
        cfg.data.target_delimiter = " | "
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))
    R = get_ring(data_cfg)

    # C/E expanded form: text -> SageMath (TextToSage) -> C/E string (ExpandedForm)
    text_delimiter = " | "
    text_to_sage = TextToSageLoadPreprocessor(delimiter=text_delimiter, ring=R)
    expanded_form = ExpandedFormLoadPreprocessor(delimiter=text_delimiter)
    dataset_load_preprocessor = ChainLoadPreprocessor(text_to_sage, expanded_form)

    io_pipeline = IOPipeline.from_config(cfg.data)
    io_pipeline.dataset_load_preprocessor = dataset_load_preprocessor
    io_dict = io_pipeline.build()

    if target_mode == "last_element":
        for key in ("train_dataset", "val_dataset", "test_dataset"):
            if key in io_dict and io_dict[key] is not None:
                ds = io_dict[key]
                ds.target_texts = [
                    t.rsplit(text_delimiter, 1)[-1].strip() if text_delimiter in t else t
                    for t in ds.target_texts
                ]

    # Show a few samples after preprocessor for verification
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
