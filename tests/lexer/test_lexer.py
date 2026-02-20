"""Tests for UnifiedLexer and NumberPolicy."""

from pathlib import Path

import pytest

from calt.io.preprocessor import NumberPolicy, UnifiedLexer
from calt.io.vocabulary import VocabConfig

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture
def vocab_config():
    """Create a test vocab config."""
    vocab_dict = {
        "range": {
            "coefficients": ["C", -50, 50],
            "exponents": ["E", 0, 20],
            "variables": ["x", 0, 2],
        },
        "misc": ["+", "*", "^", "(", ")"],
        "special_tokens": {"unk_token": "<unk>"},
        "flags": {
            "include_base_vocab": True,
            "include_base_special_tokens": True,
        },
    }
    return VocabConfig([], {}).from_config(vocab_dict)


class TestNumberPolicy:
    """Tests for NumberPolicy."""

    def test_separate_sign(self):
        """Test number policy with separate sign."""
        policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        assert policy.process_number("-50") == ["-", "50"]
        assert policy.process_number("+123") == ["+", "123"]
        assert policy.process_number("42") == ["42"]

    def test_attach_sign(self):
        """Test number policy with attach sign."""
        policy = NumberPolicy(sign=True, digit_group=0, allow_float=False)
        assert policy.process_number("-50") == ["-50"]
        assert policy.process_number("123") == ["123"]

    def test_digit_grouping(self):
        """Test digit grouping."""
        policy = NumberPolicy(sign=False, digit_group=2, allow_float=False)
        assert policy.process_number("12345") == ["12", "34", "5"]
        assert policy.process_number("12") == ["12"]
        assert policy.process_number("1") == ["1"]

    def test_float_handling(self):
        """Test float number handling."""
        policy = NumberPolicy(sign=False, digit_group=1, allow_float=True)
        assert policy.process_number("3.14") == ["3", ".", "1", "4"]
        assert policy.process_number("-2.5") == ["-", "2", ".", "5"]
        assert policy.process_number(".5") == ["0", ".", "5"]

    def test_invalid_sign(self):
        """Test that invalid sign raises error."""
        with pytest.raises(ValueError, match="sign must be bool"):
            NumberPolicy(sign="invalid")

    def test_invalid_digit_group(self):
        """Test that negative digit_group raises error."""
        with pytest.raises(ValueError, match="digit_group must be"):
            NumberPolicy(digit_group=-1)


class TestUnifiedLexer:
    """Tests for UnifiedLexer."""

    def _save_test_result(
        self,
        test_name: str,
        input_text: str,
        tokens: list[str],
        token_text: str,
        config: dict,
    ):
        """Save test result to human-readable text file."""
        output_file = OUTPUT_DIR / f"{test_name}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Test: {test_name}\n")
            f.write("=" * 80 + "\n\n")

            # Configuration
            f.write("Configuration:\n")
            f.write("-" * 80 + "\n")
            if "number_policy" in config:
                np = config["number_policy"]
                f.write("  Number Policy:\n")
                f.write(f"    sign: {np.get('sign', 'N/A')}\n")
                f.write(f"    digit_group: {np.get('digit_group', 'N/A')}\n")
                f.write(f"    allow_float: {np.get('allow_float', 'N/A')}\n")
            f.write(f"  strict: {config.get('strict', 'N/A')}\n")
            f.write(
                f"  include_base_vocab: {config.get('include_base_vocab', 'N/A')}\n"
            )
            f.write("\n")

            # Input
            f.write("Input:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {input_text}\n")
            f.write("\n")

            # Output - Tokens
            f.write("Output (Tokens):\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {tokens}\n")
            f.write(f"  ({len(tokens)} tokens)\n")
            f.write("\n")

            # Output - Token Text
            f.write("Output (Token Text):\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {token_text}\n")
            f.write("\n")

            # Comparison
            f.write("Comparison:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Input length:  {len(input_text)} characters\n")
            f.write(f"  Output length: {len(token_text)} characters\n")
            f.write(f"  Token count:   {len(tokens)} tokens\n")

    def test_example_1_polynomial(self, vocab_config):
        """Test Example 1 from spec: Polynomial."""
        number_policy = NumberPolicy(sign=False, digit_group=1, allow_float=True)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "C-50*x1^2 + 3.14"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)
        expected = ["C-50", "*", "x1", "^", "2", "+", "3", ".", "1", "4"]

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 1,
                "allow_float": True,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result("example_1_polynomial", text, tokens, token_text, config)

        assert tokens == expected

    def test_example_2_separator_priority(self, vocab_config):
        """Test Example 2 from spec: Separator Priority."""
        number_policy = NumberPolicy(sign=False, digit_group=1, allow_float=True)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "x0||x1|x2"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)
        expected = ["x0", "||", "x1", "|", "x2"]

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 1,
                "allow_float": True,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "example_2_separator_priority", text, tokens, token_text, config
        )

        assert tokens == expected

    def test_longest_match(self, vocab_config):
        """Test that longest-match is enforced."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        # || should be matched before |
        text = "x0||x1"
        tokens = lexer.tokenize(text)
        assert "||" in tokens
        assert tokens.count("|") == 0  # Single | should not appear

    def test_number_tokenization_separate(self, vocab_config):
        """Test number tokenization with separate sign."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "-50 + 123"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "number_tokenization_separate", text, tokens, token_text, config
        )

        assert tokens == ["-", "50", "+", "123"]

    def test_number_tokenization_attach(self, vocab_config):
        """Test number tokenization with attach sign."""
        number_policy = NumberPolicy(sign=True, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "-50 + 123"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        # Save test result
        config = {
            "number_policy": {
                "sign": True,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "number_tokenization_attach", text, tokens, token_text, config
        )

        assert tokens == ["-50", "+", "123"]

    def test_digit_grouping(self, vocab_config):
        """Test digit grouping in numbers."""
        number_policy = NumberPolicy(sign=False, digit_group=2, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "12345"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 2,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result("digit_grouping", text, tokens, token_text, config)

        assert tokens == ["12", "34", "5"]

    def test_float_tokenization(self, vocab_config):
        """Test float number tokenization."""
        number_policy = NumberPolicy(sign=False, digit_group=1, allow_float=True)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "3.14"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 1,
                "allow_float": True,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result("float_tokenization", text, tokens, token_text, config)

        assert tokens == ["3", ".", "1", "4"]

    def test_whitespace_handling(self, vocab_config):
        """Test that whitespace is properly skipped."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "x0  +  x1"
        tokens = lexer.tokenize(text)
        assert tokens == ["x0", "+", "x1"]

    def test_to_token_text(self, vocab_config):
        """Test to_token_text method."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "x0 + x1"
        token_text = lexer.to_token_text(text)
        assert token_text == "x0 + x1"

    def test_abstract_preprocessor_interface(self, vocab_config):
        """Test AbstractPreprocessor interface."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "x0 + x1"
        encoded = lexer.encode(text)
        decoded = lexer.decode(encoded)

        assert isinstance(encoded, str)
        assert isinstance(decoded, str)
        # Decode is lossy (whitespace removed)
        assert decoded.replace(" ", "") == text.replace(" ", "")

    def test_strict_mode_raises_error(self, vocab_config):
        """Test that strict mode raises error on unknown characters."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy, strict=True)

        text = "x0 + x1 @ invalid"
        error_occurred = False
        try:
            lexer.tokenize(text)
        except ValueError as e:
            error_occurred = True
            error_msg = str(e)

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }

        output_file = OUTPUT_DIR / "strict_mode_raises_error.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Test: strict_mode_raises_error\n")
            f.write("=" * 80 + "\n\n")

            # Configuration
            f.write("Configuration:\n")
            f.write("-" * 80 + "\n")
            np = config["number_policy"]
            f.write("  Number Policy:\n")
            f.write(f"    sign: {np.get('sign', 'N/A')}\n")
            f.write(f"    digit_group: {np.get('digit_group', 'N/A')}\n")
            f.write(f"    allow_float: {np.get('allow_float', 'N/A')}\n")
            f.write(f"  strict: {config.get('strict', 'N/A')}\n")
            f.write(
                f"  include_base_vocab: {config.get('include_base_vocab', 'N/A')}\n"
            )
            f.write("\n")

            # Input
            f.write("Input:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  {text}\n")
            f.write("\n")

            # Error
            f.write("Result:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Error occurred: {error_occurred}\n")
            if error_occurred:
                f.write(f"  Error message: {error_msg}\n")

        assert error_occurred

    def test_non_strict_mode_emits_unk(self, vocab_config):
        """Test that non-strict mode emits unk token."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy, strict=False)

        text = "x0 + x1 @ invalid"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        # Save test result
        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": False,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "non_strict_mode_emits_unk", text, tokens, token_text, config
        )

        unk_token = vocab_config.get_special_tokens().get("unk_token", "<unk>")
        assert unk_token in tokens or "<unk>" in tokens

    def test_identifier_tokenization(self, vocab_config):
        """Test identifier tokenization."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "abc_123 + xyz"
        tokens = lexer.tokenize(text)
        assert "abc_123" in tokens
        assert "xyz" in tokens

    def test_range_tokens_priority(self, vocab_config):
        """Test that range tokens have highest priority."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        # C-50 should be matched as a single token, not C, -, 50
        text = "C-50"
        tokens = lexer.tokenize(text)
        assert tokens == ["C-50"]

    def test_base_vocab_inclusion(self, vocab_config):
        """Test that base vocab tokens are included."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(
            vocab_config, number_policy=number_policy, include_base_vocab=True
        )

        text = "x0 + x1"
        tokens = lexer.tokenize(text)
        assert "+" in tokens

    def test_without_base_vocab(self, vocab_config):
        """Test lexer without base vocab."""
        # Create vocab config without base vocab
        vocab_dict = {
            "range": {
                "variables": ["x", 0, 2],
            },
            "misc": ["+"],
            "special_tokens": {},
            "flags": {
                "include_base_vocab": False,
                "include_base_special_tokens": False,
            },
        }
        vocab_config_no_base = VocabConfig([], {}).from_config(vocab_dict)

        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(
            vocab_config_no_base, number_policy=number_policy, include_base_vocab=False
        )

        text = "x0 + x1"
        tokens = lexer.tokenize(text)
        assert "+" in tokens  # Should still work if + is in misc

    def test_very_complex_polynomial_matrix_mix(self, vocab_config):
        """Test very complex expression mixing polynomials, matrices, and sequences."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "[C-25*x0^3 + C10*x1|x0 + x1^2][C5*x0||x1||x2]"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "very_complex_polynomial_matrix_mix", text, tokens, token_text, config
        )

        # Check various components
        assert "[" in tokens
        assert "]" in tokens
        assert "||" in tokens
        assert "|" in tokens
        assert "C-25" in tokens
        assert "^" in tokens

    def test_sequence_of_polynomials(self, vocab_config):
        """Test sequence of polynomial expressions."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "C-30*x0^2||C20*x1^3||C10*x0*x1||C5"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "sequence_of_polynomials", text, tokens, token_text, config
        )

        # Check separators and coefficients
        assert "||" in tokens
        assert "C-30" in tokens
        assert "C20" in tokens
        assert "C10" in tokens
        assert "C5" in tokens

    def test_matrix_with_nested_expressions(self, vocab_config):
        """Test matrix with nested polynomial expressions."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "[(x0 + x1)^2|x0*x1][(x0 - x1)^2|x0/x1]"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "matrix_with_nested_expressions", text, tokens, token_text, config
        )

        # Check brackets and operators
        assert "[" in tokens
        assert "]" in tokens
        assert "(" in tokens
        assert ")" in tokens
        assert "|" in tokens
        assert "^" in tokens

    def test_mixed_arithmetic_sequence(self, vocab_config):
        """Test mixed arithmetic operations in a sequence."""
        number_policy = NumberPolicy(sign=False, digit_group=1, allow_float=True)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "x0 + 3.14||x1 * 2.5||x2 - 1.0||x0 / 0.5"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 1,
                "allow_float": True,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "mixed_arithmetic_sequence", text, tokens, token_text, config
        )

        # Check operators and floats
        assert "||" in tokens
        assert "+" in tokens
        assert "*" in tokens
        assert "-" in tokens
        assert "/" in tokens
        assert "." in tokens

    def test_complex_exponent_chain(self, vocab_config):
        """Test complex chain of exponents."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "C-50*x0^10*x1^5*x2^3 + C30*x0^2*x1^1 - C5"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "complex_exponent_chain", text, tokens, token_text, config
        )

        # Check exponents and coefficients
        assert "^" in tokens
        assert "C-50" in tokens
        assert "C30" in tokens
        assert "E10" in tokens or "10" in tokens
        assert "E5" in tokens or "5" in tokens

    def test_matrix_sequence_combination(self, vocab_config):
        """Test combination of matrix and sequence expressions."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "[x0|x1]||[x2|x0]||[x1|x2]"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "matrix_sequence_combination", text, tokens, token_text, config
        )

        # Check matrix and sequence separators
        assert "[" in tokens
        assert "]" in tokens
        assert "||" in tokens
        assert "|" in tokens

    def test_deeply_nested_polynomial(self, vocab_config):
        """Test deeply nested polynomial expression."""
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_config, number_policy=number_policy)

        text = "((C-25*x0^2 + C10*x1) * (x0 + x1)) + ((C5*x0 - C3) * x1)"
        tokens = lexer.tokenize(text)
        token_text = lexer.to_token_text(text)

        config = {
            "number_policy": {
                "sign": False,
                "digit_group": 0,
                "allow_float": False,
            },
            "strict": True,
            "include_base_vocab": True,
        }
        self._save_test_result(
            "deeply_nested_polynomial", text, tokens, token_text, config
        )

        # Check nesting
        assert "(" in tokens
        assert ")" in tokens
        assert tokens.count("(") == tokens.count(")")
        assert "C-25" in tokens
        assert "*" in tokens

    def test_digit_group_zero_prioritizes_number_pattern(self, vocab_config):
        """Test that digit_group=0 prioritizes number pattern over reserved tokens."""
        # Create vocab with number tokens that could match parts of longer numbers
        vocab_dict = {
            "range": {
                "numbers": ["", 0, 111],  # Includes 111
            },
            "misc": ["+", "*"],
            "special_tokens": {},
            "flags": {
                "include_base_vocab": True,
                "include_base_special_tokens": True,
            },
        }
        vocab_with_numbers = VocabConfig([], {}).from_config(vocab_dict)

        # digit_group=0: numbers should not be split
        number_policy = NumberPolicy(sign=False, digit_group=0, allow_float=False)
        lexer = UnifiedLexer(vocab_with_numbers, number_policy=number_policy)

        # 1111111 should be a single token, not split into 111, 111, 1
        text = "1111111"
        tokens = lexer.tokenize(text)
        assert tokens == ["1111111"], f"Expected ['1111111'], got {tokens}"

        # 111 should also work (it's in vocab)
        text2 = "111"
        tokens2 = lexer.tokenize(text2)
        assert tokens2 == ["111"], f"Expected ['111'], got {tokens2}"

        # Mixed case
        text3 = "111+1111111"
        tokens3 = lexer.tokenize(text3)
        assert tokens3 == ["111", "+", "1111111"], (
            f"Expected ['111', '+', '1111111'], got {tokens3}"
        )

    def test_zero_padded_digit_tokens_auto_added(self, vocab_config):
        """Test that zero-padded digit tokens are automatically added when digit_group > 0."""
        vocab_dict = {
            "range": {
                "numbers": ["", 0, 1000],
            },
            "misc": ["|"],
            "special_tokens": {},
            "flags": {
                "include_base_vocab": True,
                "include_base_special_tokens": True,
            },
        }
        vocab_config_test = VocabConfig([], {}).from_config(vocab_dict)

        # digit_group=3: should add 000-999, 00-99, 0-9
        number_policy = NumberPolicy(sign=False, digit_group=3, allow_float=False)
        lexer = UnifiedLexer(vocab_config_test, number_policy=number_policy)

        vocab_list = lexer.vocab_config.get_vocab()

        # Check that zero-padded tokens are present
        assert "006" in vocab_list, "006 should be in vocab"
        assert "009" in vocab_list, "009 should be in vocab"
        assert "02" in vocab_list, "02 should be in vocab"
        assert "05" in vocab_list, "05 should be in vocab"
        assert "06" in vocab_list, "06 should be in vocab"
        assert "000" in vocab_list, "000 should be in vocab"
        assert "999" in vocab_list, "999 should be in vocab"

        # Test tokenization with zero-padded numbers
        text = "1006"  # Should be split into ['100', '6']
        tokens = lexer.tokenize(text)
        assert "6" in tokens, f"Expected '6' in tokens, got {tokens}"

        # Test with a number that produces zero-padded token
        text2 = "2009"  # Should be split into ['200', '9']
        tokens2 = lexer.tokenize(text2)
        assert "9" in tokens2, f"Expected '9' in tokens2, got {tokens2}"

    def test_attach_sign_with_digit_group_zero(self, vocab_config):
        """Test attach_sign with digit_group=0 and vocab containing negative numbers."""
        vocab_dict = {
            "range": {
                "numbers": ["", -99, 99],
            },
            "misc": [",", ";", "[", "]", "-"],
            "special_tokens": {},
            "flags": {
                "include_base_vocab": True,
                "include_base_special_tokens": True,
            },
        }
        vocab_config_test = VocabConfig([], {}).from_config(vocab_dict)

        # attach_sign: true, digit_group=0
        number_policy = NumberPolicy(sign=True, digit_group=0, allow_float=True)
        lexer = UnifiedLexer(vocab_config_test, number_policy=number_policy)

        # Test negative numbers
        text = "-10 + -11 - -13"
        tokens = lexer.tokenize(text)
        # Should be: ['-10', '+', '-11', '-', '-13']
        assert "-10" in tokens, f"Expected '-10' in tokens, got {tokens}"
        assert "-11" in tokens, f"Expected '-11' in tokens, got {tokens}"
        assert "-13" in tokens, f"Expected '-13' in tokens, got {tokens}"
