"""
Config mapping for the monomial decoder-only model.

Converts the unified config format (cfg.model) into a
MonomialDecoderOnlyConfig. The vocabulary structure (coefficient / exponent /
separator token groups) is derived from the tokenizer by the same helpers the
encoder-decoder monomial model uses, so a config that trains one architecture
trains the other with a single ``model_type`` change.

Expected extra keys on cfg.model: the same as for ``model_type: monomial``
(``num_variables`` or ``variables``, optionally ``monomial_separators``,
``coeff_scale``, ``coeff_noise_std``, ``coeff_loss_weight``).
"""

from omegaconf import DictConfig

from ..monomial.config_mapping import _coeff_fields, _exp_slots
from .model import MonomialDecoderOnlyConfig


def create_monomial_decoder_only_config(
    model_config: DictConfig,
    tokenizer=None,
) -> MonomialDecoderOnlyConfig:
    """Create a MonomialDecoderOnlyConfig from cfg.model and the tokenizer."""
    if tokenizer is None:
        raise ValueError(
            "The monomial decoder-only model requires the tokenizer to derive "
            "its coefficient/exponent/separator token groups."
        )

    vocab = tokenizer.get_vocab()

    coeff_token_ids = _coeff_fields(vocab)
    if not coeff_token_ids:
        raise ValueError(
            "No coefficient tokens ('C<int>' or 'C<int>_<label>') found in the "
            "vocabulary. The monomial models require data in C/E expanded form; "
            "check the lexer/vocab config (range: coefficients)."
        )
    exp_token_ids = _exp_slots(vocab, model_config)
    if not exp_token_ids or not exp_token_ids[0]:
        raise ValueError(
            "No exponent tokens ('E<int>') found in the vocabulary. The monomial "
            "models require data in C/E expanded form; check the lexer/vocab "
            "config (range: exponents)."
        )

    separators = getattr(model_config, "monomial_separators", None)
    if separators is None:
        separators = [t for t in ("+", "||") if t in vocab]
    else:
        missing = [t for t in separators if t not in vocab]
        if missing:
            raise ValueError(
                f"monomial_separators {missing} are not in the vocabulary."
            )
    separator_token_ids = [vocab[t] for t in separators]

    pad_token_id = getattr(model_config, "pad_token_id", None)
    eos_token_id = getattr(model_config, "eos_token_id", None)
    bos_token_id = getattr(model_config, "bos_token_id", None)
    if pad_token_id is None and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    if eos_token_id is None and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None and tokenizer.bos_token_id is not None:
        bos_token_id = tokenizer.bos_token_id
    if pad_token_id is None or eos_token_id is None or bos_token_id is None:
        raise ValueError(
            "The monomial decoder-only model needs pad/eos/bos token ids; the "
            "tokenizer does not define them and cfg.model does not override them."
        )

    # One stack, so the encoder/decoder pairs of the seq2seq configs collapse:
    # the decoder-side keys are the fallback, matching create_decoder_only_config.
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

    return MonomialDecoderOnlyConfig(
        d_model=model_config.d_model,
        attention_heads=attention_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_input_len=model_config.max_sequence_length,
        vocab_size=len(tokenizer),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_positional_embedding=getattr(
            model_config, "use_positional_embedding", "generic"
        ),
        embedding_layer_norm=getattr(model_config, "embedding_layer_norm", True),
        coeff_token_ids=coeff_token_ids,
        exp_token_ids=exp_token_ids,
        separator_token_ids=separator_token_ids,
        coeff_scale=getattr(model_config, "coeff_scale", 1.0),
        coeff_noise_std=getattr(model_config, "coeff_noise_std", 0.0),
        coeff_loss_weight=getattr(model_config, "coeff_loss_weight", 1.0),
        dropout=getattr(model_config, "dropout", 0.1),
        activation=getattr(model_config, "activation", "relu"),
        init_std=getattr(model_config, "init_std", 0.02),
        seed=getattr(model_config, "seed", 42),
    )
