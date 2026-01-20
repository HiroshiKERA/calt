import pytest
import tempfile
import os

from calt.io.pipeline import IOPipeline
from calt.io.preprocessors import PolynomialToInternalProcessor
from calt.io.vocabs import get_monomial_vocab
from calt.io.base import StandardDataset, StandardDataCollator
from transformers import PreTrainedTokenizerFast


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
    return PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=10)


@pytest.fixture
def vocab_config():
    """Create a vocab config for testing."""
    return get_monomial_vocab(num_variables=2, min_coeff=-10, max_coeff=10, min_degree=0, max_degree=3)


# -- IOPipeline Tests --

def test_io_pipeline_init():
    """Test IOPipeline initialization."""
    vocab_config = get_monomial_vocab(num_variables=2, min_coeff=-5, max_coeff=5, min_degree=0, max_degree=3)
    preprocessor = PolynomialToInternalProcessor(num_variables=2, max_degree=3, max_coeff=5)
    
    pipeline = IOPipeline(
        train_dataset_path="train.txt",
        test_dataset_path="test.txt",
        num_train_samples=100,
        num_test_samples=10,
        vocab_config=vocab_config,
        preprocessor=preprocessor,
    )
    
    assert pipeline.train_dataset_path == "train.txt"
    assert pipeline.test_dataset_path == "test.txt"
    assert pipeline.num_train_samples == 100
    assert pipeline.num_test_samples == 10
    assert pipeline.vocab_config is not None
    assert pipeline.preprocessor is not None


def test_io_pipeline_get_vocab_config_from_vocab_config(vocab_config):
    """Test get_vocab_config when VocabConfig is provided."""
    pipeline = IOPipeline(vocab_config=vocab_config)
    assert pipeline.vocab_config == vocab_config


def test_io_pipeline_get_vocab_config_from_dict():
    """Test get_vocab_config when dict is provided."""
    from calt.io.vocabs.base import VocabConfig
    
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


def test_io_pipeline_build_without_vocab_config_raises_error(sample_data_file, preprocessor):
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
    
    # build() succeeds, but accessing dataset items will fail because preprocessor is None
    result = pipeline.build()
    train_dataset = result["train_dataset"]
    
    # Accessing dataset items should fail because preprocessor is None
    with pytest.raises((TypeError, AttributeError)):
        _ = train_dataset[0]


def test_io_pipeline_build_dataset_processing(sample_data_file, preprocessor, vocab_config):
    """Test that datasets are correctly processed by preprocessor."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        num_train_samples=2,
        vocab_config=vocab_config,
        preprocessor=preprocessor,
    )
    
    result = pipeline.build()
    train_dataset = result["train_dataset"]
    
    # Get first item and check it's processed
    item = train_dataset[0]
    assert "input" in item
    assert "target" in item
    
    # Check that input and target are strings (processed by preprocessor)
    assert isinstance(item["input"], str)
    assert isinstance(item["target"], str)
    
    # Check that they contain token-like strings (C, E tokens)
    assert "C" in item["input"] or "E" in item["input"]


def test_io_pipeline_build_tokenizer_works(sample_data_file, preprocessor, vocab_config):
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


def test_io_pipeline_build_data_collator_works(sample_data_file, preprocessor, vocab_config):
    """Test that data collator works correctly."""
    pipeline = IOPipeline(
        train_dataset_path=sample_data_file,
        test_dataset_path=sample_data_file,
        num_train_samples=2,
        vocab_config=vocab_config,
        preprocessor=preprocessor,
    )
    
    result = pipeline.build()
    dataset = result["train_dataset"]
    data_collator = result["data_collator"]
    
    # Create a batch
    batch = [dataset[i] for i in range(len(dataset))]
    
    # Collate the batch
    collated = data_collator(batch)
    
    # Check that collated batch has expected keys
    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "decoder_input_ids" in collated
    assert "decoder_attention_mask" in collated
    assert "labels" in collated
