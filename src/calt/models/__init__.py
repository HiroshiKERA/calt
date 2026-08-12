from .base import ModelRegistry, get_model, get_model_from_config
from .decoder_only.model import DecoderOnlyConfig, DecoderOnlyTransformer
from .generic.model import Transformer, TransformerConfig
from .input_embeddings import get_input_embedding, register_input_embedding
from .loader import ModelLoader
from .monomial.model import MonomialTransformer, MonomialTransformerConfig
from .monomial_decoder_only.model import (
    MonomialDecoderOnlyConfig,
    MonomialDecoderOnlyTransformer,
)
from .pipeline import ModelPipeline
from .positional_embeddings import (
    get_positional_embedding,
    register_positional_embedding,
)


# Import loaders lazily to avoid circular imports
def _get_loaders():
    from .bart.loader import BartLoader
    from .generic.loader import TransformerLoader

    return TransformerLoader, BartLoader


__all__ = [
    "ModelRegistry",
    "get_model",
    "get_model_from_config",
    "ModelLoader",
    "ModelPipeline",
    "Transformer",
    "TransformerConfig",
    "MonomialTransformer",
    "MonomialTransformerConfig",
    "MonomialDecoderOnlyTransformer",
    "MonomialDecoderOnlyConfig",
    "DecoderOnlyTransformer",
    "DecoderOnlyConfig",
    # Pluggable embeddings (custom input + positional)
    "get_input_embedding",
    "register_input_embedding",
    "get_positional_embedding",
    "register_positional_embedding",
]
