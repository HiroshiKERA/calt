"""Preprocessor modules for converting symbolic expressions to internal token representation."""

from ..vocabulary.config import (
    BASE_SPECIAL_TOKENS,
    BASE_VOCAB,
    VocabConfig,
    get_base_special_tokens,
    get_base_vocab,
)
from .base import AbstractPreProcessor, PreProcessorChain, TermParseException
from .lexer import NumberPolicy, UnifiedLexer

# Legacy preprocessors (_*.py) are not imported here. Use UnifiedLexer instead.

__all__ = [
    "AbstractPreProcessor",
    "PreProcessorChain",
    "TermParseException",
    "UnifiedLexer",
    "NumberPolicy",
    "VocabConfig",
    "BASE_VOCAB",
    "BASE_SPECIAL_TOKENS",
    "get_base_vocab",
    "get_base_special_tokens",
]
