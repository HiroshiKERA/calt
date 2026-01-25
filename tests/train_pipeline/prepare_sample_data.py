"""Prepare sample data for tests.

This script extracts a small number of samples from each example's dataset
and saves them to tests/train_pipeline/sample_data/ for use in CI tests.
"""

from pathlib import Path

# Get paths
SCRIPT_DIR = Path(__file__).parent
SAMPLE_DATA_DIR = SCRIPT_DIR / "sample_data"
EXAMPLES_DIR = SCRIPT_DIR.parent.parent / "examples"

# List of example directories
EXAMPLE_DIRS = [
    "eigvec_3x3",
    "gf17_addition",
    "integer_factorization",
    "integer_polynomial_factorization",
    "rational_polynomial_factorization",
]

# Number of samples to extract
NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES = 10


def extract_samples(source_file: Path, dest_file: Path, num_samples: int):
    """Extract first N samples from source file to dest file."""
    if not source_file.exists():
        print(f"⚠️  Source file not found: {source_file}")
        return False

    dest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(source_file, "r") as src, open(dest_file, "w") as dst:
        for i, line in enumerate(src):
            if i >= num_samples:
                break
            dst.write(line)

    print(f"✅ Extracted {min(num_samples, i + 1)} samples from {source_file.name}")
    return True


def prepare_sample_data():
    """Prepare sample data for all examples."""
    print("=" * 100)
    print("Preparing sample data for tests")
    print("=" * 100)

    # Create sample_data directory
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for example_dir in EXAMPLE_DIRS:
        print(f"\nProcessing {example_dir}...")
        example_path = EXAMPLES_DIR / example_dir
        sample_example_path = SAMPLE_DATA_DIR / example_dir

        # Check if example directory exists
        if not example_path.exists():
            print(f"⚠️  Example directory not found: {example_path}")
            continue

        # Load config to get dataset paths
        config_path = example_path / "configs" / "train.yaml"
        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}")
            continue

        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        train_source = example_path / cfg.data.train_dataset_path
        test_source = example_path / cfg.data.test_dataset_path

        # Extract samples
        train_dest = sample_example_path / "train_raw.txt"
        test_dest = sample_example_path / "test_raw.txt"

        success = True
        if train_source.exists():
            success &= extract_samples(train_source, train_dest, NUM_TRAIN_SAMPLES)
        else:
            print(f"⚠️  Train dataset not found: {train_source}")

        if test_source.exists():
            success &= extract_samples(test_source, test_dest, NUM_TEST_SAMPLES)
        else:
            print(f"⚠️  Test dataset not found: {test_source}")

        if success:
            print(f"✅ Sample data prepared for {example_dir}")
        else:
            print(f"❌ Failed to prepare sample data for {example_dir}")

    print("\n" + "=" * 100)
    print("Sample data preparation completed")
    print("=" * 100)


if __name__ == "__main__":
    prepare_sample_data()
