"""Tests for train pipeline execution.

This module tests that all examples can be run in dryrun mode.
It can also be executed directly as a script.
"""

import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from calt.io.pipeline import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

# Get the examples directory path
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

# List of example directories to run
EXAMPLE_DIRS = [
    "eigvec_3x3",
    "gf17_addition",
    "integer_factorization",
    "integer_polynomial_factorization",
    "rational_polynomial_factorization",
]


def apply_dryrun_settings(cfg):
    """Apply dryrun settings to config."""
    cfg.train.dryrun = True
    cfg.train.num_train_epochs = 1
    # Set num_train_samples and num_test_samples for dryrun
    cfg.data.num_train_samples = 1000
    cfg.data.num_test_samples = 100
    cfg.wandb.group = "dryrun"
    cfg.wandb.no_wandb = True
    cfg.train.output_dir = f"results/dryrun/{cfg.wandb.name}"

    print("-" * 100)
    print(f"Dryrun mode enabled for {cfg.wandb.name}")
    print("-" * 100)
    print(f"output_dir: {cfg.train.output_dir}")
    print(f"num_train_epochs: {cfg.train.num_train_epochs}")
    print(f"num_train_samples: {cfg.data.num_train_samples}")
    print(f"num_test_samples: {cfg.data.num_test_samples}")
    print("-" * 100)


def run_example(example_dir: str):
    """Run a single example."""
    example_path = EXAMPLES_DIR / example_dir
    config_path = example_path / "configs" / "train.yaml"

    if not config_path.exists():
        print(f"⚠️  Skipping {example_dir}: config file not found at {config_path}")
        return False

    print(f"\n{'=' * 100}")
    print(f"Running example: {example_dir}")
    print(f"{'=' * 100}\n")

    try:
        # Change to example directory
        original_cwd = os.getcwd()
        os.chdir(example_path)

        # Load config
        cfg = OmegaConf.load("configs/train.yaml")

        # Apply dryrun settings
        apply_dryrun_settings(cfg)

        # Create output directory
        os.makedirs(cfg.train.output_dir, exist_ok=True)

        # Run training pipeline
        io_dict = IOPipeline.from_config(cfg.data).build()
        model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
        trainer = TrainerPipeline.from_io_dict(
            cfg.train, model, io_dict, cfg.wandb
        ).build()

        trainer.train()
        success_rate = trainer.evaluate_and_save_generation()
        print(f"Success rate: {100 * success_rate:.1f}%")

        print(f"\n✅ Completed: {example_dir}\n")
        return True

    except Exception as e:
        print(f"\n❌ Error in {example_dir}: {e}\n")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """Run all examples sequentially."""
    print("=" * 100)
    print("Running all examples in dryrun mode")
    print("=" * 100)

    results = {}
    for example_dir in EXAMPLE_DIRS:
        results[example_dir] = run_example(example_dir)

    # Print summary
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    for example_dir, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {example_dir}")

    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)


def test_examples_directory_exists():
    """Test that examples directory exists."""
    assert EXAMPLES_DIR.exists(), f"Examples directory not found: {EXAMPLES_DIR}"
    assert EXAMPLES_DIR.is_dir(), f"Examples path is not a directory: {EXAMPLES_DIR}"


def test_example_dirs_exist():
    """Test that all example directories exist."""
    for example_dir in EXAMPLE_DIRS:
        example_path = EXAMPLES_DIR / example_dir
        assert example_path.exists(), f"Example directory not found: {example_path}"
        assert example_path.is_dir(), f"Example path is not a directory: {example_path}"


def test_example_configs_exist():
    """Test that all example config files exist."""
    for example_dir in EXAMPLE_DIRS:
        config_path = EXAMPLES_DIR / example_dir / "configs" / "train.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"


def test_apply_dryrun_settings():
    """Test that dryrun settings are applied correctly."""
    # Load a sample config
    config_path = EXAMPLES_DIR / EXAMPLE_DIRS[0] / "configs" / "train.yaml"
    cfg = OmegaConf.load(config_path)

    # Apply dryrun settings
    apply_dryrun_settings(cfg)

    # Verify settings
    assert cfg.train.dryrun is True
    assert cfg.train.num_train_epochs == 1
    assert cfg.data.num_train_samples == 1000
    assert cfg.data.num_test_samples == 100
    assert cfg.wandb.group == "dryrun"
    assert cfg.wandb.no_wandb is True
    assert "dryrun" in cfg.train.output_dir


@pytest.mark.parametrize("example_dir", EXAMPLE_DIRS)
def test_run_example_dryrun(example_dir):
    """Test that each example can be run in dryrun mode.

    This test actually runs the training pipeline, so it may take some time.
    """
    result = run_example(example_dir)
    assert result is True, f"Example {example_dir} failed to run"


def test_all_examples_run():
    """Test that all examples can be run sequentially.

    This is an integration test that runs all examples.
    Note: This test may take a long time to complete.
    """
    results = {}
    for example_dir in EXAMPLE_DIRS:
        results[example_dir] = run_example(example_dir)

    # Check that all examples passed
    failed = [name for name, success in results.items() if not success]
    assert len(failed) == 0, f"The following examples failed: {', '.join(failed)}"


if __name__ == "__main__":
    main()
