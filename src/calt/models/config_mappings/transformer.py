"""
Config mapping for Transformer model.

Converts from unified config format (cfg.model) to TransformerConfig.
"""

from omegaconf import DictConfig

from ..transformer import TransformerConfig


def create_transformer_config(
    model_config: DictConfig,
    tokenizer=None,  # Not used for transformer, but kept for consistency
) -> TransformerConfig:
    """Create TransformerConfig from unified config format.
    
    Args:
        model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
        tokenizer: Not used for transformer model, but kept for API consistency.
    
    Returns:
        TransformerConfig: Transformer model configuration.
    """
    # Get vocab_size from tokenizer if available, otherwise use default
    vocab_size = len(tokenizer.vocab) if tokenizer is not None else getattr(model_config, 'vocab_size', 1000)
    
    return TransformerConfig(
        d_model=model_config.d_model,
        attention_heads=model_config.num_encoder_heads,  # Use encoder_heads as default
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        dim_feedforward=model_config.encoder_ffn_dim,  # Use encoder_ffn_dim as default
        max_input_len=model_config.max_sequence_length,
        vocab_size=vocab_size,
        use_positional_embedding=getattr(model_config, 'use_positional_embedding', 'learned'),
        # Optional parameters with defaults
        dropout=getattr(model_config, 'dropout', 0.1),
        activation=getattr(model_config, 'activation', 'relu'),
        init_std=getattr(model_config, 'init_std', 0.02),
        seed=getattr(model_config, 'seed', 42),
    )
