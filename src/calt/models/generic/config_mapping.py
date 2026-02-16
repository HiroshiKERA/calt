"""
Config mapping for Transformer model.

Converts from unified config format (cfg.model) to TransformerConfig.
"""

from omegaconf import DictConfig

from .model import TransformerConfig


def create_transformer_config(
    model_config: DictConfig,
    tokenizer=None,
) -> TransformerConfig:
    """Create TransformerConfig from unified config format.

    Args:
        model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
        tokenizer: Tokenizer instance. When provided, pad_token_id, eos_token_id,
            and bos_token_id are taken from it so generation aligns with decoding.

    Returns:
        TransformerConfig: Transformer model configuration.
    """
    # Get vocab_size from tokenizer if available, otherwise use default
    vocab_size = (
        len(tokenizer.vocab)
        if tokenizer is not None
        else getattr(model_config, "vocab_size", 1000)
    )

    # Align special token IDs with tokenizer so generation (BOS start, EOS stop)
    # and batch_decode(skip_special_tokens=True) behave correctly. BART does this;
    # generic was previously using TransformerConfig defaults (0, 1, 2).
    pad_token_id = getattr(model_config, "pad_token_id", None)
    eos_token_id = getattr(model_config, "eos_token_id", None)
    bos_token_id = getattr(model_config, "bos_token_id", None)
    if tokenizer is not None:
        if pad_token_id is None and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        if eos_token_id is None and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        if bos_token_id is None and tokenizer.bos_token_id is not None:
            bos_token_id = tokenizer.bos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    if eos_token_id is None:
        eos_token_id = 1
    if bos_token_id is None:
        bos_token_id = 2

    return TransformerConfig(
        d_model=model_config.d_model,
        attention_heads=model_config.num_encoder_heads,  # Use encoder_heads as default
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        dim_feedforward=model_config.encoder_ffn_dim,  # Use encoder_ffn_dim as default
        max_input_len=model_config.max_sequence_length,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_positional_embedding=getattr(
            model_config, "use_positional_embedding", "generic"
        ),
        # Optional parameters with defaults
        dropout=getattr(model_config, "dropout", 0.1),
        activation=getattr(model_config, "activation", "relu"),
        init_std=getattr(model_config, "init_std", 0.02),
        seed=getattr(model_config, "seed", 42),
    )
