import os
import tempfile
import pytest
from transformers import PreTrainedTokenizerFast
from calt.data_loader.utils.tokenizer import set_tokenizer


def test_set_tokenizer_dynamic_creation_gf():
    """Test dynamic creation of a tokenizer for a GF field."""
    tokenizer = set_tokenizer(field="GF31", max_degree=10, max_length=512)

    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.model_max_length == 512

    # Check special tokens
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.cls_token == "[CLS]"

    # Check vocabulary
    # For GF(31), coeffs are -30 to 30. That's 61 tokens.
    # CONSTS = [C-30, ..., C30] + [C] = 61 + 1 = 62
    # ECONSTS = [E0, ..., E10] = 11
    # Others = [SEP] = 1
    # Special = [PAD], <s>, </s>, [CLS] = 4
    # Total = 62 + 11 + 1 + 4 = 78

    assert len(tokenizer.vocab) == 78
    assert "C30" in tokenizer.get_vocab()
    assert "C-30" in tokenizer.get_vocab()
    assert "E10" in tokenizer.get_vocab()


def test_set_tokenizer_dynamic_creation_zz():
    """Test dynamic creation of a tokenizer for the ZZ field."""
    tokenizer = set_tokenizer(field="ZZ", max_coeff=50, max_degree=5)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # Check vocabulary
    # For ZZ, max_coeff=50, coeffs are -50 to 50. That's 101 tokens.
    # CONSTS = [C-50, ..., C50] + [C] = 101 + 1 = 102
    # ECONSTS = [E0, ..., E5] = 6
    # Others = [SEP] = 1
    # Special = 4
    # Total = 102 + 6 + 1 + 4 = 113
    assert len(tokenizer.vocab) == 113
    assert "C50" in tokenizer.get_vocab()
    assert "C-50" in tokenizer.get_vocab()
    assert "E5" in tokenizer.get_vocab()


def test_set_tokenizer_from_file():
    """Test loading a tokenizer from a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a tokenizer and save it to a file
        original_tokenizer = set_tokenizer(
            field="GF17",
            max_degree=8,
        )
        tokenizer_path = os.path.join(tmpdir, "tokenizer.json")
        original_tokenizer.save_pretrained(tokenizer_path)

        # 2. Load the tokenizer from the file
        breakpoint()
        loaded_tokenizer = set_tokenizer(tokenizer_path=tokenizer_path)

        # 3. Verify the loaded tokenizer
        assert isinstance(loaded_tokenizer, PreTrainedTokenizerFast)
        assert loaded_tokenizer.vocab_size == original_tokenizer.vocab_size
        assert loaded_tokenizer.get_vocab() == original_tokenizer.get_vocab()

        # Test encoding
        text = "C1 E2 C-3 E1"
        original_encoding = original_tokenizer.encode(text)
        loaded_encoding = loaded_tokenizer.encode(text)
        assert original_encoding == loaded_encoding


def test_invalid_field():
    """Test that an invalid field raises a ValueError."""
    with pytest.raises(ValueError, match="unknown field: BLAH"):
        set_tokenizer(field="BLAH")

    with pytest.raises(ValueError, match="Invalid field specification for GF"):
        set_tokenizer(field="GF-invalid")
