"""Tests for checkpoint resume functionality.

This module tests that training can be resumed from a saved checkpoint.
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

# Get sample data directory
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"


def test_resume_from_checkpoint():
    """Test that training can be resumed from a checkpoint."""
    # Use gf17_addition as test example (simpler and faster)
    example_dir = "gf17_addition"
    example_path = EXAMPLES_DIR / example_dir
    config_path = example_path / "configs" / "train.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    # Check if sample data exists
    sample_data_path = SAMPLE_DATA_DIR / example_dir
    train_sample = sample_data_path / "train_raw.txt"
    test_sample = sample_data_path / "test_raw.txt"

    if not train_sample.exists() or not test_sample.exists():
        pytest.skip(
            f"Sample data not found for {example_dir}. "
            f"Sample data should be in {sample_data_path}"
        )

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "checkpoint_test"
        save_dir.mkdir(parents=True)

        original_cwd = os.getcwd()
        try:
            os.chdir(example_path)

            # Load config
            cfg = OmegaConf.load("configs/train.yaml")

            # Override dataset paths to use sample data
            cfg.data.train_dataset_path = str(train_sample.resolve())
            cfg.data.test_dataset_path = str(test_sample.resolve())

            # Apply minimal training settings for quick test
            cfg.train.save_dir = str(save_dir)
            cfg.train.num_train_epochs = 1
            # Use larger batch size to avoid division by device count issues
            cfg.train.batch_size = 8
            cfg.train.test_batch_size = 8
            cfg.train.num_workers = 0
            cfg.train.seed = 42

            # Disable wandb for test
            if hasattr(cfg, "wandb"):
                cfg.wandb.no_wandb = True
            else:
                cfg.wandb = OmegaConf.create({"no_wandb": True})

            # Save config to save_dir
            os.makedirs(save_dir, exist_ok=True)
            OmegaConf.save(cfg, save_dir / "train.yaml")

            # Step 1: Initial training and save
            print("\n=== Step 1: Initial training ===")
            io_dict = IOPipeline.from_config(cfg.data).build()
            model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
            trainer_pipeline = TrainerPipeline.from_io_dict(
                cfg.train, model, io_dict, cfg.get("wandb")
            ).build()

            # Train for a short time
            trainer_pipeline.train()
            # Save model and tokenizer to save_dir
            trainer_pipeline.save_model(output_dir=str(save_dir))

            # Verify checkpoint was created
            # Note: HuggingFace Trainer saves to output_dir/model and output_dir/tokenizer
            # But we're saving directly to save_dir, so check for model files
            assert (
                (save_dir / "config.json").exists()
                or (save_dir / "pytorch_model.bin").exists()
                or (save_dir / "model.safetensors").exists()
            ), (
                f"Model files should exist in {save_dir}. "
                f"Contents: {list(save_dir.iterdir())}"
            )
            assert (save_dir / "tokenizer_config.json").exists() or (
                save_dir / "vocab.json"
            ).exists(), (
                f"Tokenizer files should exist in {save_dir}. "
                f"Contents: {list(save_dir.iterdir())}"
            )
            assert (save_dir / "train.yaml").exists(), "train.yaml should exist"

            # Step 2: Resume from checkpoint using TrainerPipeline.resume_from_checkpoint
            print(
                "\n=== Step 2: Resume from checkpoint (TrainerPipeline.resume_from_checkpoint) ==="
            )
            resumed_trainer = TrainerPipeline.resume_from_checkpoint(
                str(save_dir), resume_from_checkpoint=True
            )

            # Verify resumed trainer has correct components
            assert resumed_trainer.model is not None, "Model should be loaded"
            assert resumed_trainer.tokenizer is not None, "Tokenizer should be loaded"
            assert resumed_trainer.train_dataset is not None, (
                "Train dataset should be loaded"
            )
            assert resumed_trainer.eval_dataset is not None, (
                "Eval dataset should be loaded"
            )
            assert resumed_trainer.data_collator is not None, (
                "Data collator should be loaded"
            )

            # Build the resumed trainer
            resumed_trainer.build()

            # Verify trainer can continue training
            resumed_trainer.train()

            print("\n✅ Checkpoint resume test passed!")

        finally:
            os.chdir(original_cwd)


def test_resume_from_checkpoint_invalid_directory():
    """Test that resume_from_checkpoint raises appropriate errors for invalid directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_dir = Path(tmpdir) / "nonexistent"

        # Test with non-existent directory
        with pytest.raises(ValueError, match=r"Save directory does not exist"):
            TrainerPipeline.resume_from_checkpoint(str(invalid_dir))

        # Test with directory without train.yaml
        invalid_dir.mkdir()
        with pytest.raises(ValueError, match=r"train.yaml not found"):
            TrainerPipeline.resume_from_checkpoint(str(invalid_dir))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
