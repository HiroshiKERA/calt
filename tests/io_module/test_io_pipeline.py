import os
import tempfile

import pytest
from transformers import PreTrainedTokenizerFast

from calt.io.base import StandardDataCollator, StandardDataset
from calt.io.pipeline import IOPipeline
from calt.io.vocabulary import VocabConfig


# Fixtures
@pytest.fixture
def sample_data_file():
    """Create a temporary data file with sample data."""
    data = [
        "2*x0^2 + 3*x1#5*x0^2 + 3*x1",
        "x0 + x1#x0 + x1",
        "1#1",
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(data))
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def preprocessor():
    """Create a preprocessor for testing."""
    # PolynomialToInternalProcessor has been removed, use None for now
    # Tests that require preprocessor will be skipped or updated to use UnifiedLexer
    return None


@pytest.fixture
def vocab_config():
    """Create a vocab config for testing."""
    return VocabConfig([], {}).from_config(
        {
            "range": {
                "coefficients": ["C", -10, 10],
                "exponents": ["E", 0, 3],
                "variables": ["x", 0, 2],
            },
            "misc": ["+"],
            "special_tokens": {},
            "flags": {},
        }
    )


# -- IOPipeline Tests --


def test_io_pipeline_init():
    """Test IOPipeline initialization."""
    vocab_config = VocabConfig([], {}).from_config(
        {
            "range": {
                "coefficients": ["C", -5, 5],
                "exponents": ["E", 0, 3],
                "variables": ["x", 0, 2],
            },
            "misc": ["+"],
            "special_tokens": {},
            "flags": {},
        }
    )

    pipeline = IOPipeline(
        train_dataset_path="train.txt",
        test_dataset_path="test.txt",
        num_train_samples=100,
        num_test_samples=10,
        vocab_config=vocab_config,
        preprocessor=None,
    )

    assert pipeline.train_dataset_path == "train.txt"
    assert pipeline.test_dataset_path == "test.txt"
    assert pipeline.num_train_samples == 100
    assert pipeline.num_test_samples == 10
    assert pipeline.vocab_config is not None


def test_io_pipeline_get_vocab_config_from_vocab_config(vocab_config):
    """Test get_vocab_config when VocabConfig is provided."""
    pipeline = IOPipeline(vocab_config=vocab_config)
    assert pipeline.vocab_config == vocab_config


def test_io_pipeline_get_vocab_config_from_dict():
    """Test get_vocab_config when dict is provided."""
    from calt.io.vocabulary import VocabConfig

    config_dict = {
        "range": {
            "coeff": ["C", -5, 5],
            "exp": ["E", 0, 3],
        },
        "misc": ["+"],
    }

    pipeline = IOPipeline(vocab_config=config_dict)
    assert pipeline.vocab_config is not None
    assert isinstance(pipeline.vocab_config, VocabConfig)


def test_io_pipeline_get_vocab_config_from_yaml_file():
    """Test get_vocab_config when YAML file path is provided."""
    import yaml

    config = {
        "range": {
            "coeff": ["C", -5, 5],
            "exp": ["E", 0, 3],
        },
        "misc": ["+"],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name

    try:
        pipeline = IOPipeline(vocab_config=temp_path)
        assert pipeline.vocab_config is not None
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_io_pipeline_get_vocab_config_none():
    """Test get_vocab_config when None is provided."""
    pipeline = IOPipeline(vocab_config=None)
    assert pipeline.vocab_config is None


def test_io_pipeline_build(sample_data_file, preprocessor, vocab_config):
    """Test IOPipeline.build() method."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        num_train_samples=2,
        num_test_samples=1,
        vocab_config=vocab_config,
        preprocessor=preprocessor,
    )

    result = pipeline.build()

    # Check return structure
    assert "train_dataset" in result
    assert "test_dataset" in result
    assert "tokenizer" in result
    assert "data_collator" in result

    # Check types
    assert isinstance(result["train_dataset"], StandardDataset)
    assert isinstance(result["test_dataset"], StandardDataset)
    assert isinstance(result["tokenizer"], PreTrainedTokenizerFast)
    assert isinstance(result["data_collator"], StandardDataCollator)

    # Check dataset lengths
    assert len(result["train_dataset"]) == 2
    assert len(result["test_dataset"]) == 1

    # Check that attributes are set
    assert pipeline.train_dataset is not None
    assert pipeline.test_dataset is not None
    assert pipeline.tokenizer is not None
    assert pipeline.data_collator is not None


def test_io_pipeline_build_without_vocab_config_raises_error(
    sample_data_file, preprocessor
):
    """Test that build() raises error when vocab_config is None."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        vocab_config=None,
        preprocessor=preprocessor,
    )

    with pytest.raises(ValueError, match="vocab_config must be provided"):
        pipeline.build()


def test_io_pipeline_build_with_none_paths_raises_error(vocab_config, preprocessor):
    """Test that build() handles None paths correctly."""
    pipeline = IOPipeline(
        train_dataset_path=None,
        test_dataset_path=None,
        vocab_config=vocab_config,
        preprocessor=preprocessor,
    )

    # Should raise FileNotFoundError when trying to load None path
    with pytest.raises((FileNotFoundError, TypeError)):
        pipeline.build()


def test_io_pipeline_build_without_preprocessor(sample_data_file, vocab_config):
    """Test IOPipeline.build() without preprocessor."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        num_train_samples=2,
        vocab_config=vocab_config,
        preprocessor=None,
    )

    # build() succeeds, and accessing dataset items returns raw text
    result = pipeline.build()
    train_dataset = result["train_dataset"]

    # Accessing dataset items should work and return raw text
    item = train_dataset[0]
    assert "input" in item
    assert "target" in item
    assert isinstance(item["input"], str)
    assert isinstance(item["target"], str)


def test_io_pipeline_build_dataset_processing(sample_data_file, vocab_config):
    """Test that datasets return raw text when preprocessor is None."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        num_train_samples=2,
        vocab_config=vocab_config,
        preprocessor=None,
    )

    result = pipeline.build()
    train_dataset = result["train_dataset"]

    # Get first item and check it's raw text
    item = train_dataset[0]
    assert "input" in item
    assert "target" in item

    # Check that input and target are strings (raw text when preprocessor is None)
    assert isinstance(item["input"], str)
    assert isinstance(item["target"], str)


def test_io_pipeline_build_tokenizer_works(
    sample_data_file, preprocessor, vocab_config
):
    """Test that tokenizer is correctly created and works."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        num_train_samples=2,
        vocab_config=vocab_config,
        preprocessor=preprocessor,
    )

    result = pipeline.build()
    tokenizer = result["tokenizer"]

    # Test encoding
    text = "C2 E2 E0 C3 E1 E0"
    encoded = tokenizer.encode(text)
    assert len(encoded) > 0

    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)


def test_io_pipeline_build_data_collator_works(sample_data_file, vocab_config):
    """Test that data collator works correctly."""
    # Skip this test when preprocessor is None because raw text may contain unknown tokens
    # This test requires a preprocessor to tokenize the text properly
    pytest.skip("This test requires a preprocessor to tokenize text properly")
