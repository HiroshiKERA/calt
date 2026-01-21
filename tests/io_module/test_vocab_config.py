import os
import tempfile

import pytest
import yaml

from calt.io.vocabulary import VocabConfig


# Fixtures for VocabConfig
@pytest.fixture
def simple_vocab_config():
    """Create a simple VocabConfig with minimal vocab."""
    vocab = ["C1", "C2", "E0", "E1"]
    special_tokens = {}
    return VocabConfig(
        vocab=vocab,
        special_tokens=special_tokens,
        include_base_vocab=False,
        include_base_special_tokens=False,
    )


@pytest.fixture
def vocab_config_with_base():
    """Create a VocabConfig that includes base vocab and special tokens."""
    vocab = ["C1", "C2", "E0", "E1"]
    special_tokens = {}
    return VocabConfig(
        vocab=vocab,
        special_tokens=special_tokens,
        include_base_vocab=True,
        include_base_special_tokens=True,
    )


@pytest.fixture
def vocab_config_without_base():
    """Create a VocabConfig without base vocab and special tokens."""
    vocab = ["C1", "C2", "E0", "E1"]
    special_tokens = {"pad_token": "[PAD]"}
    return VocabConfig(
        vocab=vocab,
        special_tokens=special_tokens,
        include_base_vocab=False,
        include_base_special_tokens=False,
    )


@pytest.fixture
def yaml_config_file():
    """Create a temporary YAML config file for testing."""
    config = {
        "range": {
            "coeff": ["C", -2, 2],
            "exp": ["E", 0, 3],
        },
        "misc": ["+", "*"],
        "special_tokens": {"pad_token": "[PAD]"},
        "include_base_vocab": True,
        "include_base_special_tokens": True,
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# -- VocabConfig Tests --

def test_vocab_config_init(simple_vocab_config):
    """Test basic VocabConfig initialization."""
    assert simple_vocab_config.vocab == ["C1", "C2", "E0", "E1"]
    assert simple_vocab_config.special_tokens == {}


def test_vocab_config_with_base_includes_defaults(vocab_config_with_base):
    """Test that VocabConfig includes base vocab and special tokens by default."""
    vocab = vocab_config_with_base.get_vocab()
    special_tokens = vocab_config_with_base.get_special_tokens()
    
    # Should include base vocab [SEP]
    assert "[SEP]" in vocab
    assert "C1" in vocab
    assert "C2" in vocab
    
    # Should include base special tokens
    assert "pad_token" in special_tokens
    assert special_tokens["pad_token"] == "[PAD]"
    assert special_tokens["bos_token"] == "<s>"
    assert special_tokens["eos_token"] == "</s>"
    assert special_tokens["cls_token"] == "[CLS]"


def test_vocab_config_without_base_excludes_defaults(vocab_config_without_base):
    """Test that VocabConfig excludes base vocab and special tokens when specified."""
    vocab = vocab_config_without_base.get_vocab()
    special_tokens = vocab_config_without_base.get_special_tokens()
    
    # Should not include base vocab [SEP]
    assert "[SEP]" not in vocab
    assert "C1" in vocab
    assert "C2" in vocab
    
    # Should only include custom special tokens
    assert "pad_token" in special_tokens
    assert special_tokens["pad_token"] == "[PAD]"
    assert "bos_token" not in special_tokens


def test_vocab_config_from_config_dict():
    """Test creating VocabConfig from a dictionary."""
    # New format: range, misc, special_tokens, flags
    config = {
        "range": {
            "coeff": ["C", -2, 2],
            "exp": ["E", 0, 3],
        },
        "misc": ["+", "*"],
        "special_tokens": {"pad_token": "[PAD]"},
        "flags": {
            "include_base_vocab": True,
            "include_base_special_tokens": True,
        },
    }
    
    vocab_config = VocabConfig([], {})
    vocab_config = vocab_config.from_config(config)
    
    vocab = vocab_config.get_vocab()
    special_tokens = vocab_config.get_special_tokens()
    
    # Check that range vocab was generated
    assert "C-2" in vocab or "C2" in vocab
    assert "E0" in vocab
    assert "E3" in vocab
    assert "+" in vocab
    assert "*" in vocab
    
    # Check special tokens
    assert "pad_token" in special_tokens


def test_vocab_config_from_config_file(yaml_config_file):
    """Test creating VocabConfig from a YAML file."""
    vocab_config = VocabConfig([], {})
    vocab_config = vocab_config.from_config(yaml_config_file)
    
    vocab = vocab_config.get_vocab()
    special_tokens = vocab_config.get_special_tokens()
    
    # Should include base vocab
    assert "[SEP]" in vocab
    
    # Should include range vocab
    assert "C-2" in vocab or "C2" in vocab
    assert "E0" in vocab
    assert "E3" in vocab
    
    # Should include misc vocab
    assert "+" in vocab
    assert "*" in vocab
    
    # Should include special tokens
    assert "pad_token" in special_tokens


def test_vocab_config_save_and_load():
    """Test saving and loading VocabConfig."""
    vocab = ["C1", "C2", "E0", "E1"]
    special_tokens = {"pad_token": "[PAD]"}
    vocab_config = VocabConfig(
        vocab=vocab,
        special_tokens=special_tokens,
        include_base_vocab=False,
        include_base_special_tokens=False,
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name
    
    try:
        vocab_config.save(temp_path)
        
        # Load and verify
        with open(temp_path, "r") as f:
            loaded_config = yaml.safe_load(f)
        
        assert "vocab" in loaded_config
        assert "special_tokens" in loaded_config
        assert loaded_config["vocab"] == vocab
        assert loaded_config["special_tokens"] == special_tokens
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

