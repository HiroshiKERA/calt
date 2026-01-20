"""
Unified regex-based lexer/preprocessor for CALT.

This module implements a unified lexer that converts raw input strings into
token sequences based on vocabulary configuration. It follows the specification
in _blueprints/lexer.txt.
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..vocabs.base import VocabConfig, BASE_VOCAB, BASE_SPECIAL_TOKENS
from .base import AbstractPreprocessor


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
        sign: How to handle sign. "separate" means sign is a separate token,
            "attach" means sign is part of the number.
        digit_group: Group digits. 0 = no split, d>=1 = split every d digits.
        allow_float: Whether to allow floating point numbers.
        dot_token: Token used for decimal point (default: ".").
    """
    sign: str = "separate"  # "separate" | "attach"
    digit_group: int = 0  # 0 = no split, d>=1 = split every d digits
    allow_float: bool = True
    dot_token: str = "."
    
    def __post_init__(self):
        if self.sign not in ("separate", "attach"):
            raise ValueError(f"sign must be 'separate' or 'attach', got {self.sign}")
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
        
        # Handle sign
        if number_str.startswith("-"):
            if self.sign == "separate":
                tokens.append("-")
                number_str = number_str[1:]
            else:
                # sign == "attach", keep the minus sign
                pass
        elif number_str.startswith("+"):
            if self.sign == "separate":
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
                tokens.extend(self._group_digits(integer_part))
            else:
                tokens.append(integer_part)
            
            # Add dot token
            tokens.append(self.dot_token)
            
            # Process fractional part
            if self.digit_group > 0:
                tokens.extend(self._group_digits(fractional_part))
            else:
                tokens.append(fractional_part)
        else:
            # Integer
            if self.digit_group > 0:
                tokens.extend(self._group_digits(number_str))
            else:
                tokens.append(number_str)
        
        return tokens
    
    def _group_digits(self, digits: str) -> list[str]:
        """Group digits according to digit_group.
        
        Args:
            digits: String of digits.
        
        Returns:
            List of digit groups.
        """
        if self.digit_group == 0:
            return [digits]
        
        groups = []
        for i in range(0, len(digits), self.digit_group):
            groups.append(digits[i:i + self.digit_group])
        return groups


class UnifiedLexer(AbstractPreprocessor):
    """Unified regex-based lexer for tokenizing input strings.
    
    This lexer converts raw input strings into token sequences based on
    vocabulary configuration. It follows longest-match principle for
    reserved tokens and supports configurable number tokenization.
    
    Example:
        >>> from calt.io.vocabs.base import VocabConfig
        >>> from calt.io.preprocessors.lexer import UnifiedLexer, NumberPolicy
        >>> 
        >>> vocab_config = VocabConfig([], {}).from_config("config/vocab.yaml")
        >>> number_policy = NumberPolicy(sign="separate", digit_group=1)
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
        # For AbstractPreprocessor compatibility, we need num_variables, max_degree, max_coeff
        # These are not used by the lexer, so we use dummy values
        super().__init__(num_variables=0, max_degree=0, max_coeff=1)
        
        self.vocab_config = vocab_config
        self.number_policy = number_policy or NumberPolicy()
        self.strict = strict
        self.include_base_vocab = include_base_vocab
        
        # Build reserved tokens
        self._build_reserved_tokens()
        
        # Build regex patterns
        self._build_patterns()
    
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
        # If sign is "attach", we need to match signed numbers before operators
        if self.number_policy.sign == "attach":
            # Match signed numbers (negative lookbehind to ensure - is not preceded by word char)
            # This allows -50 but not a-50 (where - is an operator)
            if self.number_policy.allow_float:
                number_pattern = r"(?<![A-Za-z0-9_])-?\d+\.?\d*|\.\d+"
            else:
                number_pattern = r"(?<![A-Za-z0-9_])-?\d+"
        else:
            # sign == "separate": match unsigned numbers, sign will be matched as operator
            if self.number_policy.allow_float:
                number_pattern = r"\d+\.?\d*|\.\d+"
            else:
                number_pattern = r"\d+"
        
        # Identifier pattern
        identifier_pattern = r"[A-Za-z_][A-Za-z_0-9]*"
        
        # Combined pattern: number first if attach mode, then reserved tokens, then identifier
        # This ensures signed numbers are matched before operators in attach mode
        if self.number_policy.sign == "attach":
            self.pattern = re.compile(
                f"({number_pattern})|({reserved_pattern})|({identifier_pattern})"
            )
        else:
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
                # Group order depends on sign mode
                if self.number_policy.sign == "attach":
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
                    # Pattern order: reserved, number, identifier
                    if match.group(1):  # Reserved token
                        tokens.append(match.group(1))
                        pos = match.end()
                    elif match.group(2):  # Number
                        number_str = match.group(2)
                        number_tokens = self.number_policy.process_number(number_str)
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
                        f"Unknown character '{text[pos]}' at position {pos} in: {text[:pos+20]}"
                    )
                else:
                    # Emit <unk>
                    unk_token = self.vocab_config.get_special_tokens().get("unk_token", "<unk>")
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
    
    # AbstractPreprocessor interface
    def encode(self, text: str) -> str:
        """Encode text to token text (implements AbstractPreprocessor).
        
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
