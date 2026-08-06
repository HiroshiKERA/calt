"""
Config mapping for the decoder-only Transformer.

Converts from unified config format (cfg.model) to DecoderOnlyConfig.
"""

from omegaconf import DictConfig

from .model import DecoderOnlyConfig


def create_decoder_only_config(
    model_config: DictConfig,
    tokenizer=None,
) -> DecoderOnlyConfig:
    """Create DecoderOnlyConfig from unified config format.

    A decoder-only model has one stack, so the encoder/decoder pairs of the
    seq2seq configs collapse: ``num_layers``, ``num_heads`` and ``ffn_dim`` are
    read first, and the decoder-side keys are the fallback. An existing seq2seq
    config therefore runs unchanged, at half the depth of its 6+6 counterpart.

    Args:
        model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
        tokenizer: When provided, vocab_size and the special token ids
            (pad/eos/bos) are taken from it so generation aligns with decoding.

    Returns:
        DecoderOnlyConfig: Decoder-only model configuration.
    """
    vocab_size = (
        len(tokenizer.vocab)
        if tokenizer is not None
        else getattr(model_config, "vocab_size", 1000)
    )

    # Align special token IDs with tokenizer so generation (BOS start, EOS stop)
    # and batch_decode(skip_special_tokens=True) behave correctly.
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

    num_layers = getattr(model_config, "num_layers", None)
    if num_layers is None:
        num_layers = getattr(model_config, "num_decoder_layers", 6)

    attention_heads = getattr(model_config, "num_heads", None)
    if attention_heads is None:
        attention_heads = getattr(
            model_config,
            "num_decoder_heads",
            getattr(model_config, "num_encoder_heads", 8),
        )

    dim_feedforward = getattr(model_config, "ffn_dim", None)
    if dim_feedforward is None:
        dim_feedforward = getattr(
            model_config,
            "decoder_ffn_dim",
            getattr(model_config, "encoder_ffn_dim", 2048),
        )

    return DecoderOnlyConfig(
        d_model=model_config.d_model,
        attention_heads=attention_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        # Problem and solution share one sequence, so the positional table has
        # to span both; the embedding allocates max_input_len * 2 positions.
        max_input_len=model_config.max_sequence_length,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_positional_embedding=getattr(
            model_config, "use_positional_embedding", "generic"
        ),
        embedding_layer_norm=getattr(model_config, "embedding_layer_norm", True),
        dropout=getattr(model_config, "dropout", 0.1),
        activation=getattr(model_config, "activation", "relu"),
        init_std=getattr(model_config, "init_std", 0.02),
        seed=getattr(model_config, "seed", 42),
    )
