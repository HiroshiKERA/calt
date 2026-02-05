import os

import click
from omegaconf import OmegaConf
from sage.all import GF, PolynomialRing, QQ, RR, ZZ

from calt.io import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    LastElementLoadPreprocessor,
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
    help="Target format: 'full' = full C/E expanded form; 'last_element' = last polynomial only (not implemented yet).",
)
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode with reduced epochs and data for quick testing",
)
def main(train_config_path: str, data_config_path: str, target_mode: str, dryrun: bool):
    """Train a model for polynomial multiplication (C/E expanded form)."""
    if target_mode != "full":
        raise NotImplementedError(f"Only target_mode 'full' is supported for now, got {target_mode!r}")
    cfg = OmegaConf.load(train_config_path)
    data_cfg = OmegaConf.load(data_config_path)

    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
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
