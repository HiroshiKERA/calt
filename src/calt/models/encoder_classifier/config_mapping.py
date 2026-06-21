"""
Config mapping for the encoder-only classification model.

Converts the unified config format (cfg.model) into an EncoderClassifierConfig,
mirroring create_transformer_config but ignoring the decoder-only fields.
"""

from omegaconf import DictConfig

from .model import EncoderClassifierConfig


def create_encoder_classifier_config(
    model_config: DictConfig,
    tokenizer=None,
) -> EncoderClassifierConfig:
    """Create an EncoderClassifierConfig from cfg.model (+ tokenizer for vocab/ids)."""
    vocab_size = (
        len(tokenizer.vocab)
        if tokenizer is not None
        else getattr(model_config, "vocab_size", 1000)
    )

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

    return EncoderClassifierConfig(
        d_model=model_config.d_model,
        attention_heads=model_config.num_encoder_heads,
        num_encoder_layers=model_config.num_encoder_layers,
        dim_feedforward=model_config.encoder_ffn_dim,
        max_input_len=model_config.max_sequence_length,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_positional_embedding=getattr(
            model_config, "use_positional_embedding", "generic"
        ),
        input_embedding_type=getattr(model_config, "input_embedding_type", "token"),
        dropout=getattr(model_config, "dropout", 0.1),
        activation=getattr(model_config, "activation", "relu"),
        init_std=getattr(model_config, "init_std", 0.02),
        seed=getattr(model_config, "seed", 42),
    )
