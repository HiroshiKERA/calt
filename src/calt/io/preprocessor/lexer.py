"""
Unified regex-based lexer/preprocessor for CALT.

This module implements a unified lexer that converts raw input strings into
token sequences based on vocabulary configuration. It follows the specification
in _blueprints/lexer.txt.
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..vocabulary.config import VocabConfig
from .base import AbstractPreProcessor

# Base vocabulary as per specification
BASE_VOCAB_TOKENS = {
    "separators": ("||", "|"),
    "operators": ("+", "-", "*", "^", "/"),
    "brackets": ("(", ")", "[", "]"),
}


@dataclass
class NumberPolicy:
    """Policy for tokenizing numbers.

    Attributes:
        sign: How to handle sign. True (attach) means sign is part of the number,
            False (separate) means sign is a separate token.
        digit_group: Group digits. 0 = no split, d>=1 = split every d digits.
        allow_float: Whether to allow floating point numbers.
    """

    sign: bool = False  # True = attach, False = separate
    digit_group: int = 0  # 0 = no split, d>=1 = split every d digits
    allow_float: bool = True

    def __post_init__(self):
        if not isinstance(self.sign, bool):
            raise ValueError(
                f"sign must be bool (True=attach, False=separate), got {type(self.sign).__name__}: {self.sign}"
            )
        if self.digit_group < 0:
            raise ValueError(f"digit_group must be >= 0, got {self.digit_group}")

    def process_number(self, number_str: str) -> list[str]:
        """Process a number string according to the policy.

        Args:
            number_str: Number string (may include sign and/or decimal point).

        Returns:
            List of tokens representing the number.
        """
        tokens = []
        has_negative_sign = False

        # Handle sign
        if number_str.startswith("-"):
            if not self.sign:  # sign == False (separate)
                tokens.append("-")
                number_str = number_str[1:]
            else:
                # sign == True (attach): keep minus sign with the number
                # If digit_group > 0, attach - to the first digit group
                has_negative_sign = True
                number_str = number_str[1:]
        elif number_str.startswith("+"):
            if not self.sign:  # sign == False (separate)
                tokens.append("+")
                number_str = number_str[1:]
            # else: ignore leading +

        # Handle float
        if "." in number_str and self.allow_float:
            parts = number_str.split(".", 1)
            integer_part = parts[0] or "0"
            fractional_part = parts[1]

            # Process integer part
            if self.digit_group > 0:
                # For sign=True (attach) with digit_group, attach - to the first group
                if has_negative_sign and self.sign:
                    tokens.extend(self._group_digits(integer_part, prefix="-"))
                else:
                    tokens.extend(self._group_digits(integer_part))
            else:
                if has_negative_sign and self.sign:
                    tokens.append("-" + integer_part)
                else:
                    tokens.append(integer_part)

            # Add dot token (always use ".")
            tokens.append(".")

            # Process fractional part
            if self.digit_group > 0:
                tokens.extend(self._group_digits(fractional_part))
            else:
                tokens.append(fractional_part)
        else:
            # Integer
            if self.digit_group > 0:
                # For sign=True (attach) with digit_group, attach - to the first group
                if has_negative_sign and self.sign:
                    tokens.extend(self._group_digits(number_str, prefix="-"))
                else:
                    tokens.extend(self._group_digits(number_str))
            else:
                if has_negative_sign and self.sign:
                    tokens.append("-" + number_str)
                else:
                    tokens.append(number_str)

        return tokens

    def _group_digits(self, digits: str, prefix: str = "") -> list[str]:
        """Group digits according to digit_group.

        Args:
            digits: String of digits.
            prefix: Optional prefix to attach to the first group (e.g., "-").

        Returns:
            List of digit groups.
        """
        if self.digit_group == 0:
            return [prefix + digits] if prefix else [digits]

        groups = []
        for i in range(0, len(digits), self.digit_group):
            group = digits[i : i + self.digit_group]
            # Attach prefix only to the first group
            if i == 0 and prefix:
                groups.append(prefix + group)
            else:
                groups.append(group)
        return groups


class UnifiedLexer(AbstractPreProcessor):
    """Unified regex-based lexer for tokenizing input strings.

    This lexer converts raw input strings into token sequences based on
    vocabulary configuration. It follows longest-match principle for
    reserved tokens and supports configurable number tokenization.

    Example:
        >>> from calt.io.vocabulary import VocabConfig
        >>> from calt.io.preprocessor import UnifiedLexer, NumberPolicy
        >>>
        >>> vocab_config = VocabConfig([], {}).from_config("config/vocab.yaml")
        >>> number_policy = NumberPolicy(sign=False, digit_group=1)
        >>> lexer = UnifiedLexer(vocab_config, number_policy=number_policy)
        >>>
        >>> tokens = lexer.tokenize("C-50*x1^2 + 3.14")
        >>> # Returns: ["C-50", "*", "x1", "^", "2", "+", "3", ".", "1", "4"]
    """

    def __init__(
        self,
        vocab_config: VocabConfig,
        number_policy: Optional[NumberPolicy] = None,
        strict: bool = True,
        include_base_vocab: bool = True,
    ):
        """Initialize the unified lexer.

        Args:
            vocab_config: Vocabulary configuration.
            number_policy: Policy for tokenizing numbers. If None, uses default.
            strict: If True, raise error on unknown characters. If False, emit <unk>.
            include_base_vocab: Whether to include base vocabulary tokens.
        """
        # Initialize post-processor base class
        super().__init__()

        self.number_policy = number_policy or NumberPolicy()
        self.strict = strict
        self.include_base_vocab = include_base_vocab

        # Extend vocab_config with required tokens based on number_policy
        self.vocab_config = self._extend_vocab_for_number_policy(vocab_config)

        # Build reserved tokens
        self._build_reserved_tokens()

        # Build regex patterns
        self._build_patterns()

    def _extend_vocab_for_number_policy(self, vocab_config: VocabConfig) -> VocabConfig:
        """Extend vocab_config with tokens required by number_policy settings.

        If allow_float is true, automatically adds:
        - "." (dot token)
        - "-0" (if attach_sign is true, i.e., attach sign)
        - Fractional part tokens (if digit_group > 0, e.g., "00", "01", ..., "99" for digit_group=2)

        If digit_group > 0, automatically adds zero-padded digit group tokens
        (e.g., "00", "01", ..., "99" for digit_group=2) for both integer and fractional parts.

        Args:
            vocab_config: Original vocabulary configuration.

        Returns:
            New VocabConfig with required tokens added (or original if no additions needed).
        """
        vocab_list = vocab_config.get_vocab()
        existing_vocab = set(vocab_list)
        tokens_to_add = []

        if self.number_policy.allow_float:
            # Add "." if not present
            if "." not in existing_vocab:
                tokens_to_add.append(".")

            # Add "-0" if sign is True (attach sign, i.e., attach_sign=true) and not present
            # For floats like -0.5, this generates "-0", ".", "5" tokens
            if self.number_policy.sign and "-0" not in existing_vocab:
                tokens_to_add.append("-0")

        # If digit_group > 0, add zero-padded digit group tokens
        # These are needed for both integer and fractional parts when numbers are split
        # e.g., digit_group=2: "00", "01", ..., "99"
        # e.g., digit_group=3: "000", "001", ..., "999"
        # Also add shorter zero-padded tokens (1-digit, 2-digit, etc.) up to digit_group
        # because numbers with fewer digits than digit_group can still produce these tokens
        if self.number_policy.digit_group > 0:
            # Generate all possible zero-padded digit group tokens for all lengths up to digit_group
            # e.g., digit_group=3: "0", "1", ..., "9" (1-digit), "00", "01", ..., "99" (2-digit), "000", "001", ..., "999" (3-digit)
            for length in range(1, self.number_policy.digit_group + 1):
                max_value = 10**length - 1
                for i in range(max_value + 1):
                    token = str(i).zfill(length)
                    if token not in existing_vocab and token not in tokens_to_add:
                        tokens_to_add.append(token)

        if not tokens_to_add:
            return vocab_config

        # Create new VocabConfig with added tokens
        new_vocab = vocab_list + tokens_to_add
        return VocabConfig(
            vocab=new_vocab,
            special_tokens=vocab_config.get_special_tokens(),
            include_base_vocab=False,  # Already included in vocab_list
            include_base_special_tokens=False,  # Already included in special_tokens
        )

    def _build_reserved_tokens(self):
        """Build the list of reserved tokens in priority order."""
        reserved = []

        # 1. Range-expanded tokens (highest priority)
        vocab = self.vocab_config.get_vocab()
        for token in vocab:
            if token not in reserved:
                reserved.append(token)

        # 2. Misc tokens (already in vocab from VocabConfig)
        # (handled above)

        # 3. Base vocab tokens
        if self.include_base_vocab:
            for category in BASE_VOCAB_TOKENS.values():
                for token in category:
                    if token not in reserved:
                        reserved.append(token)

        # Sort by descending length for longest-match
        self.reserved_tokens = sorted(reserved, key=len, reverse=True)

    def _build_patterns(self):
        """Build regex patterns for tokenization."""
        # Escape reserved tokens for regex
        escaped_tokens = [re.escape(token) for token in self.reserved_tokens]

        # Build reserved token pattern (longest-match)
        if escaped_tokens:
            reserved_pattern = "|".join(escaped_tokens)
        else:
            reserved_pattern = "(?!)"  # Never matches

        # Number pattern (may include sign and/or decimal point)
        # If sign is True (attach), we need to match signed numbers before operators
        if self.number_policy.sign:
            # Match signed numbers (negative lookbehind to ensure - is not preceded by word char)
            # This allows -50 but not a-50 (where - is an operator)
            if self.number_policy.allow_float:
                number_pattern = r"(?<![A-Za-z0-9_])-?\d+\.?\d*|\.\d+"
            else:
                number_pattern = r"(?<![A-Za-z0-9_])-?\d+"
        else:
            # sign == False (separate): match unsigned numbers, sign will be matched as operator
            if self.number_policy.allow_float:
                number_pattern = r"\d+\.?\d*|\.\d+"
            else:
                number_pattern = r"\d+"

        # Identifier pattern
        identifier_pattern = r"[A-Za-z_][A-Za-z_0-9]*"

        # Combined pattern: number first if attach mode, then reserved tokens, then identifier
        # This ensures signed numbers are matched before operators in attach mode
        # If digit_group=0, prioritize number pattern over reserved tokens to avoid splitting
        # numbers that are longer than vocab tokens (e.g., 1111111 should not be split into 111, 111, 1)
        if self.number_policy.sign:
            self.pattern = re.compile(
                f"({number_pattern})|({reserved_pattern})|({identifier_pattern})"
            )
        else:
            # sign == False (separate)
            if self.number_policy.digit_group == 0:
                # digit_group=0: numbers should not be split, so prioritize number pattern
                # This prevents numbers like 1111111 from being split when vocab contains 111
                self.pattern = re.compile(
                    f"({number_pattern})|({reserved_pattern})|({identifier_pattern})"
                )
            else:
                # digit_group > 0: numbers will be split, so reserved tokens can take priority
                self.pattern = re.compile(
                    f"({reserved_pattern})|({number_pattern})|({identifier_pattern})"
                )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize input text into a list of tokens.

        Args:
            text: Input text string.

        Returns:
            List of tokens.

        Raises:
            ValueError: If strict=True and unknown character is encountered.
        """
        tokens = []
        pos = 0
        text_len = len(text)

        while pos < text_len:
            # Skip whitespace
            if text[pos].isspace():
                pos += 1
                continue

            # Try to match at current position
            match = self.pattern.match(text, pos)

            if match:
                # Check which group matched
                # Group order depends on sign mode and digit_group
                if self.number_policy.sign:
                    # Pattern order: number, reserved, identifier
                    if match.group(1):  # Number
                        number_str = match.group(1)
                        number_tokens = self.number_policy.process_number(number_str)
                        tokens.extend(number_tokens)
                        pos = match.end()
                    elif match.group(2):  # Reserved token
                        tokens.append(match.group(2))
                        pos = match.end()
                    elif match.group(3):  # Identifier
                        tokens.append(match.group(3))
                        pos = match.end()
                    else:
                        raise RuntimeError("Pattern matched but no group matched")
                else:
                    # sign == False (separate)
                    if self.number_policy.digit_group == 0:
                        # Pattern order: number, reserved, identifier (number prioritized)
                        if match.group(1):  # Number
                            number_str = match.group(1)
                            number_tokens = self.number_policy.process_number(
                                number_str
                            )
                            tokens.extend(number_tokens)
                            pos = match.end()
                        elif match.group(2):  # Reserved token
                            tokens.append(match.group(2))
                            pos = match.end()
                        elif match.group(3):  # Identifier
                            tokens.append(match.group(3))
                            pos = match.end()
                        else:
                            raise RuntimeError("Pattern matched but no group matched")
                    else:
                        # Pattern order: reserved, number, identifier
                        if match.group(1):  # Reserved token
                            tokens.append(match.group(1))
                            pos = match.end()
                        elif match.group(2):  # Number
                            number_str = match.group(2)
                            number_tokens = self.number_policy.process_number(
                                number_str
                            )
                            tokens.extend(number_tokens)
                            pos = match.end()
                        elif match.group(3):  # Identifier
                            tokens.append(match.group(3))
                            pos = match.end()
                        else:
                            raise RuntimeError("Pattern matched but no group matched")
            else:
                # No match - unknown character
                if self.strict:
                    raise ValueError(
                        f"Unknown character '{text[pos]}' at position {pos} in: {text[: pos + 20]}"
                    )
                else:
                    # Emit <unk>
                    unk_token = self.vocab_config.get_special_tokens().get(
                        "unk_token", "<unk>"
                    )
                    tokens.append(unk_token)
                    pos += 1

        return tokens

    def to_token_text(self, text: str) -> str:
        """Convert input text to space-joined token text.

        Args:
            text: Input text string.

        Returns:
            Space-joined token string.
        """
        tokens = self.tokenize(text)
        return " ".join(tokens)

    # AbstractPreProcessor interface
    def encode(self, text: str) -> str:
        """Encode text to token text (implements AbstractPreProcessor).

        Args:
            text: Input text string.

        Returns:
            Space-joined token string.
        """
        return self.to_token_text(text)

    def decode(self, tokens: str) -> str:
        """Decode token text back to original text.

        Note: This is a lossy operation. The lexer does not maintain
        full reversibility (e.g., whitespace is lost).

        Args:
            tokens: Space-joined token string.

        Returns:
            Original text (approximate, whitespace may differ).
        """
        # Simple reversal: just remove spaces
        # This is lossy but matches the specification
        return tokens.replace(" ", "")
