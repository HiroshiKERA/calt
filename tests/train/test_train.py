"""Simplified tests for training pipeline.

This module tests that training pipeline can be set up and run with minimal configuration.
Uses a single example to keep tests fast and lightweight.
"""

import os
import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

# Get the examples directory path
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

# Use a single simple example for testing
TEST_EXAMPLE = "gf17_addition"

# Get sample data directory
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"


def get_sample_data_path(example_dir: str) -> Path:
    """Get the path to sample data for an example."""
    return SAMPLE_DATA_DIR / example_dir


def test_training_pipeline_setup():
    """Test that training pipeline can be set up with minimal config."""
    example_path = EXAMPLES_DIR / TEST_EXAMPLE
    config_path = example_path / "configs" / "train.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    # Check if sample data exists
    sample_data_path = get_sample_data_path(TEST_EXAMPLE)
    train_sample = sample_data_path / "train_raw.txt"
    test_sample = sample_data_path / "test_raw.txt"

    if not train_sample.exists() or not test_sample.exists():
        pytest.skip(
            f"Sample data not found for {TEST_EXAMPLE}. "
            f"Sample data should be in {sample_data_path}"
        )

    original_cwd = os.getcwd()
    try:
        os.chdir(example_path)

        # Load config
        cfg = OmegaConf.load("configs/train.yaml")

        # Override dataset paths to use sample data
        cfg.data.train_dataset_path = str(train_sample.resolve())
        cfg.data.test_dataset_path = str(test_sample.resolve())

        # Apply minimal settings for fast test
        cfg.train.num_train_epochs = 1
        # Use larger batch size to avoid division by device count issues
        cfg.train.batch_size = 8
        cfg.train.test_batch_size = 8
        cfg.train.num_workers = 0
        if hasattr(cfg, "wandb"):
            cfg.wandb.no_wandb = True
        else:
            cfg.wandb = OmegaConf.create({"no_wandb": True})

        # Test pipeline setup
        io_dict = IOPipeline.from_config(cfg.data).build()
        model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
        trainer_pipeline = TrainerPipeline.from_io_dict(
            cfg.train, model, io_dict, cfg.get("wandb")
        ).build()

        # Verify components are set up correctly
        assert trainer_pipeline.trainer is not None
        assert trainer_pipeline.model is not None
        assert trainer_pipeline.tokenizer is not None
        assert trainer_pipeline.train_dataset is not None

    finally:
        os.chdir(original_cwd)


def test_training_pipeline_minimal_run():
    """Test that training pipeline can run with minimal training steps."""
    example_path = EXAMPLES_DIR / TEST_EXAMPLE
    config_path = example_path / "configs" / "train.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    # Check if sample data exists
    sample_data_path = get_sample_data_path(TEST_EXAMPLE)
    train_sample = sample_data_path / "train_raw.txt"
    test_sample = sample_data_path / "test_raw.txt"

    if not train_sample.exists() or not test_sample.exists():
        pytest.skip(
            f"Sample data not found for {TEST_EXAMPLE}. "
            f"Sample data should be in {sample_data_path}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(example_path)

            # Load config
            cfg = OmegaConf.load("configs/train.yaml")

            # Override dataset paths to use sample data
            cfg.data.train_dataset_path = str(train_sample.resolve())
            cfg.data.test_dataset_path = str(test_sample.resolve())

            # Apply minimal settings for fast test
            cfg.train.save_dir = str(Path(tmpdir) / "test_output")
            cfg.train.num_train_epochs = 1
            # Use larger batch size to avoid division by device count issues
            cfg.train.batch_size = 8
            cfg.train.test_batch_size = 8
            cfg.train.num_workers = 0
            if hasattr(cfg, "wandb"):
                cfg.wandb.no_wandb = True
            else:
                cfg.wandb = OmegaConf.create({"no_wandb": True})

            # Run minimal training
            io_dict = IOPipeline.from_config(cfg.data).build()
            model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
            trainer_pipeline = TrainerPipeline.from_io_dict(
                cfg.train, model, io_dict, cfg.get("wandb")
            ).build()

            # Run training (should complete quickly with minimal settings)
            trainer_pipeline.train()

            # Verify metrics are computed
            metrics = trainer_pipeline.trainer.evaluate()
            assert "eval_token_accuracy" in metrics or "token_accuracy" in metrics
            assert "eval_success_rate" in metrics or "success_rate" in metrics

        finally:
            os.chdir(original_cwd)
