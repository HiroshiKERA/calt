"""Decoder-only (causal LM) Transformer."""

from .config_mapping import create_decoder_only_config
from .model import DecoderOnlyConfig, DecoderOnlyTransformer

__all__ = [
    "DecoderOnlyTransformer",
    "DecoderOnlyConfig",
    "create_decoder_only_config",
]
