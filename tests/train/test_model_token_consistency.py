"""Tests that generic and BART models use the same special token IDs for a given setup.

When both models are built from the same config and tokenizer, pad_token_id,
bos_token_id, and eos_token_id must match each other and the tokenizer so that
generation and decoding behave consistently across model types.
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.models.bart.config_mapping import create_bart_config
from calt.models.generic.config_mapping import create_transformer_config

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

# Examples that have both lexer config and sample data
CONSISTENCY_TEST_EXAMPLES = ["gf17_addition", "eigvec_3x3"]


def get_sample_data_path(example_dir: str) -> Path:
    return SAMPLE_DATA_DIR / example_dir


def _build_tokenizer_and_model_config(example_dir: str):
    """Build tokenizer and unified model_config from an example. Returns (tokenizer, model_config)."""
    example_path = EXAMPLES_DIR / example_dir
    config_path = example_path / "configs" / "train.yaml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    sample_data_path = get_sample_data_path(example_dir)
    train_sample = sample_data_path / "train_raw.txt"
    test_sample = sample_data_path / "test_raw.txt"
    if not train_sample.exists() or not test_sample.exists():
        pytest.skip(
            f"Sample data not found for {example_dir}. "
            f"Sample data should be in {sample_data_path}"
        )

    original_cwd = os.getcwd()
    try:
        os.chdir(example_path)
        cfg = OmegaConf.load("configs/train.yaml")
        cfg.data.train_dataset_path = str(train_sample.resolve())
        cfg.data.test_dataset_path = str(test_sample.resolve())
        io_dict = IOPipeline.from_config(cfg.data).build()
        tokenizer = io_dict["tokenizer"]
        model_config = cfg.model
        return tokenizer, model_config
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("example_dir", CONSISTENCY_TEST_EXAMPLES)
def test_generic_and_bart_config_same_special_token_ids(example_dir: str):
    """Generic and BART configs must have identical pad/bos/eos token IDs for the same setup."""
    tokenizer, model_config = _build_tokenizer_and_model_config(example_dir)

    transformer_config = create_transformer_config(model_config, tokenizer)
    bart_config = create_bart_config(model_config, tokenizer)

    assert transformer_config.pad_token_id == bart_config.pad_token_id, (
        f"pad_token_id mismatch: generic={transformer_config.pad_token_id}, "
        f"bart={bart_config.pad_token_id}"
    )
    assert transformer_config.bos_token_id == bart_config.bos_token_id, (
        f"bos_token_id mismatch: generic={transformer_config.bos_token_id}, "
        f"bart={bart_config.bos_token_id}"
    )
    assert transformer_config.eos_token_id == bart_config.eos_token_id, (
        f"eos_token_id mismatch: generic={transformer_config.eos_token_id}, "
        f"bart={bart_config.eos_token_id}"
    )

    if tokenizer.pad_token_id is not None:
        assert transformer_config.pad_token_id == tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        assert transformer_config.bos_token_id == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        assert transformer_config.eos_token_id == tokenizer.eos_token_id


@pytest.mark.parametrize("example_dir", CONSISTENCY_TEST_EXAMPLES)
def test_model_pipeline_same_special_token_ids(example_dir: str):
    """Models built via ModelPipeline (generic vs bart) must have same special token IDs."""
    example_path = EXAMPLES_DIR / example_dir
    config_path = example_path / "configs" / "train.yaml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    sample_data_path = get_sample_data_path(example_dir)
    train_sample = sample_data_path / "train_raw.txt"
    test_sample = sample_data_path / "test_raw.txt"
    if not train_sample.exists() or not test_sample.exists():
        pytest.skip(
            f"Sample data not found for {example_dir}. "
            f"Sample data should be in {sample_data_path}"
        )

    original_cwd = os.getcwd()
    try:
        os.chdir(example_path)
        cfg = OmegaConf.load("configs/train.yaml")
        cfg.data.train_dataset_path = str(train_sample.resolve())
        cfg.data.test_dataset_path = str(test_sample.resolve())
        io_dict = IOPipeline.from_config(cfg.data).build()

        cfg.model.model_type = "generic"
        model_generic = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
        cfg.model.model_type = "bart"
        model_bart = ModelPipeline.from_io_dict(cfg.model, io_dict).build()

        gen_cfg = model_generic.config
        bart_cfg = model_bart.config

        assert gen_cfg.pad_token_id == bart_cfg.pad_token_id, (
            f"pad_token_id mismatch: generic={gen_cfg.pad_token_id}, bart={bart_cfg.pad_token_id}"
        )
        assert gen_cfg.bos_token_id == bart_cfg.bos_token_id, (
            f"bos_token_id mismatch: generic={gen_cfg.bos_token_id}, bart={bart_cfg.bos_token_id}"
        )
        assert gen_cfg.eos_token_id == bart_cfg.eos_token_id, (
            f"eos_token_id mismatch: generic={gen_cfg.eos_token_id}, bart={bart_cfg.eos_token_id}"
        )
    finally:
        os.chdir(original_cwd)
