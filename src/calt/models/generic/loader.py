"""
Loader for Transformer model.

Handles conversion from unified config format (cfg.model) to TransformerConfig
and creates Transformer model instances.
"""

from ..loader import ModelLoader
from .model import Transformer, TransformerConfig

# Default TransformerConfig for reference
_DEFAULT_TRANSFORMER_CONFIG = TransformerConfig()


class TransformerLoader(ModelLoader):
    """Loader for Transformer model.

    Converts cfg.model to TransformerConfig and creates Transformer instances.
    """

    def translate_config(self) -> TransformerConfig:
        """Convert unified config format to TransformerConfig.

        Returns:
            TransformerConfig: Transformer model configuration.
        """
        # Get vocab_size from tokenizer if available, otherwise use default
        vocab_size = (
            len(self.tokenizer.vocab)
            if self.tokenizer is not None
            else getattr(
                self.calt_config, "vocab_size", _DEFAULT_TRANSFORMER_CONFIG.vocab_size
            )
        )

        self.config = TransformerConfig(
            d_model=self.calt_config.d_model,
            attention_heads=self.calt_config.num_encoder_heads,  # Use encoder_heads as default
            num_encoder_layers=self.calt_config.num_encoder_layers,
            num_decoder_layers=self.calt_config.num_decoder_layers,
            dim_feedforward=self.calt_config.encoder_ffn_dim,  # Use encoder_ffn_dim as default
            max_input_len=self.calt_config.max_sequence_length,
            vocab_size=vocab_size,
            use_positional_embedding=getattr(
                self.calt_config, "use_positional_embedding", "generic"
            ),
            # Optional parameters with defaults from _DEFAULT_TRANSFORMER_CONFIG
            dropout=getattr(
                self.calt_config, "dropout", _DEFAULT_TRANSFORMER_CONFIG.dropout
            ),
            activation=getattr(
                self.calt_config, "activation", _DEFAULT_TRANSFORMER_CONFIG.activation
            ),
            init_std=getattr(
                self.calt_config, "init_std", _DEFAULT_TRANSFORMER_CONFIG.init_std
            ),
            seed=getattr(self.calt_config, "seed", _DEFAULT_TRANSFORMER_CONFIG.seed),
        )
        return self.config

    def build_model(self) -> Transformer:
        """Create Transformer model instance.

        Returns:
            Transformer: Transformer model instance.
        """
        if self.config is None:
            raise ValueError(
                "config must be set before building model. Call translate_config() first."
            )

        return Transformer(config=self.config)
