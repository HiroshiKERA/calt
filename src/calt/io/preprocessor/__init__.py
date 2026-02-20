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
from .load_preprocessor import (
    DatasetLoadPreprocessor,
    JsonlDefaultLoadPreprocessor,
    PickleDefaultLoadPreprocessor,
    TextDefaultLoadPreprocessor,
    UserCallableLoadPreprocessor,
)
from .load_preprocessors import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    LastElementLoadPreprocessor,
    ReversedOrderLoadPreprocessor,
    TextToSageLoadPreprocessor,
)

# Legacy preprocessors (_*.py) are not imported here. Use UnifiedLexer instead.

__all__ = [
    "AbstractPreProcessor",
    "BASE_SPECIAL_TOKENS",
    "BASE_VOCAB",
    "ChainLoadPreprocessor",
    "DatasetLoadPreprocessor",
    "ExpandedFormLoadPreprocessor",
    "JsonlDefaultLoadPreprocessor",
    "LastElementLoadPreprocessor",
    "ReversedOrderLoadPreprocessor",
    "TextDefaultLoadPreprocessor",
    "TextToSageLoadPreprocessor",
    "NumberPolicy",
    "PickleDefaultLoadPreprocessor",
    "PreProcessorChain",
    "TermParseException",
    "UnifiedLexer",
    "UserCallableLoadPreprocessor",
    "VocabConfig",
    "get_base_special_tokens",
    "get_base_vocab",
]
