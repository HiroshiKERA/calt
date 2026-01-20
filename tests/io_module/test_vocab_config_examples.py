import pytest
import os

from calt.io.vocabulary import VocabConfig
from calt.io.tokenizer import get_tokenizer


def test_vocab_config_from_examples_config():
    """Test loading VocabConfig from examples/demos/configs/lexer.yaml."""
    import yaml
    from pathlib import Path
    
    # Load lexer.yaml and extract vocab section
    lexer_config_path = Path("examples/demos/configs/lexer.yaml")
    if not lexer_config_path.exists():
        pytest.skip(f"Config file not found: {lexer_config_path}")
    
    with open(lexer_config_path, 'r') as f:
        lexer_config = yaml.safe_load(f)
    
    vocab_config_dict = lexer_config.get("vocab", {})
    vocab_config = VocabConfig([], {})
    vocab_config = vocab_config.from_config(vocab_config_dict)
    
    vocab = vocab_config.get_vocab()
    special_tokens = vocab_config.get_special_tokens()
    
    # Should include base vocab [SEP]
    assert "[SEP]" in vocab
    
    # Should include coefficient tokens from range
    assert "C-50" in vocab
    assert "C0" in vocab
    assert "C50" in vocab
    
    # Should include exponent tokens from range
    assert "E0" in vocab
    assert "E20" in vocab
    
    # Should include variable tokens from range
    assert "x0" in vocab
    assert "x1" in vocab
    assert "x2" in vocab
    
    # Should include misc vocab
    assert "+" in vocab
    assert "*" in vocab
    assert "^" in vocab
    assert "(" in vocab
    assert ")" in vocab
    
    # Should include base special tokens
    assert "pad_token" in special_tokens
    assert special_tokens["pad_token"] == "[PAD]"
    assert special_tokens["bos_token"] == "<s>"
    assert special_tokens["eos_token"] == "</s>"
    assert special_tokens["cls_token"] == "[CLS]"


def test_vocab_config_from_examples_config_creates_tokenizer():
    """Test that VocabConfig from examples/demos/configs/lexer.yaml can create a tokenizer."""
    import yaml
    from pathlib import Path
    
    # Load lexer.yaml and extract vocab section
    lexer_config_path = Path("examples/demos/configs/lexer.yaml")
    if not lexer_config_path.exists():
        pytest.skip(f"Config file not found: {lexer_config_path}")
    
    with open(lexer_config_path, 'r') as f:
        lexer_config = yaml.safe_load(f)
    
    vocab_config_dict = lexer_config.get("vocab", {})
    vocab_config = VocabConfig([], {})
    vocab_config = vocab_config.from_config(vocab_config_dict)
    
    tokenizer = get_tokenizer(vocab_config=vocab_config)
    
    assert tokenizer is not None
    assert len(tokenizer.vocab) > 0
    
    # Test encoding
    text = "C2 E1 C3 E0"
    encoded = tokenizer.encode(text)
    assert len(encoded) > 0
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)


