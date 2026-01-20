"""
Config mappings for converting OmegaConf configs to model-specific configs.

This module provides functions to convert from the unified config format
(cfg.model) to model-specific config classes (TransformerConfig, BartConfig, etc.).
"""

from .bart import create_bart_config
from .transformer import create_transformer_config

__all__ = [
    "create_bart_config",
    "create_transformer_config",
]
