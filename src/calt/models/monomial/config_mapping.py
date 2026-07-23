"""
Config mapping for the monomial-embedding model.

Converts the unified config format (cfg.model) into a MonomialTransformerConfig,
mirroring create_transformer_config, and derives the monomial vocabulary
structure (coefficient / exponent / separator token groups) from the tokenizer.

The tokenizer is required: the monomial heads classify over the coefficient and
exponent token groups, which only the tokenizer's vocabulary defines.

Expected extra keys on cfg.model:
    num_variables (int): Number of variables in the polynomial ring, i.e. how
        many ``E`` tokens follow each ``C`` token in the expanded form. Required
        (unless ``variables`` is given) because shared ``E<k>`` tokens carry no
        variable identity.
    variables (list[str], optional): Alternative to num_variables for
        per-variable vocabularies (``Ex2 Ey0`` style): exponent tokens are then
        matched as ``E<name><exp>`` per variable name.
    monomial_separators (list[str], optional): Continuation separator tokens in
        follow-slot order. Defaults to those of ``["+", "||"]`` present in the
        vocabulary.
"""

import re

from omegaconf import DictConfig

from .model import MonomialTransformerConfig

_COEFF_RE = re.compile(r"C(-?\d+)(?:_(.+))?$")
_SHARED_EXP_RE = re.compile(r"E(\d+)$")


def _coeff_fields(vocab: dict) -> list:
    """Group coefficient tokens by field label, ids ordered by coefficient value.

    Plain ``C<int>`` tokens form a single unlabeled field; ``C<int>_<label>``
    tokens form one field per label (multi-modulus vocabularies). Field order
    follows first appearance in id order, matching the reference implementation.
    """
    fields: dict = {}
    for token, gid in sorted(vocab.items(), key=lambda kv: kv[1]):
        m = _COEFF_RE.fullmatch(token)
        if m:
            fields.setdefault(m.group(2) or "", []).append((int(m.group(1)), gid))
    return [[gid for _, gid in sorted(pairs)] for pairs in fields.values()]


def _exp_slots(vocab: dict, model_config: DictConfig) -> list:
    """One ordered exponent-token id list per variable slot."""
    variables = getattr(model_config, "variables", None)
    if variables:
        slots = []
        for name in variables:
            pattern = re.compile(rf"E{re.escape(str(name))}(\d+)$")
            pairs = sorted(
                (int(m.group(1)), gid)
                for token, gid in vocab.items()
                if (m := pattern.fullmatch(token))
            )
            slots.append([gid for _, gid in pairs])
        return slots

    num_variables = getattr(model_config, "num_variables", None)
    if not num_variables:
        raise ValueError(
            "The monomial model needs model.num_variables (or model.variables) "
            "in the config: shared 'E<k>' exponent tokens carry no variable "
            "identity, so the number of exponent slots per monomial cannot be "
            "inferred from the vocabulary."
        )
    shared = sorted(
        (int(m.group(1)), gid)
        for token, gid in vocab.items()
        if (m := _SHARED_EXP_RE.fullmatch(token))
    )
    return [[gid for _, gid in shared]] * int(num_variables)


def create_monomial_config(
    model_config: DictConfig,
    tokenizer=None,
) -> MonomialTransformerConfig:
    """Create a MonomialTransformerConfig from cfg.model and the tokenizer."""
    if tokenizer is None:
        raise ValueError(
            "The monomial model requires the tokenizer to derive its "
            "coefficient/exponent/separator token groups."
        )

    vocab = tokenizer.get_vocab()

    coeff_token_ids = _coeff_fields(vocab)
    if not coeff_token_ids:
        raise ValueError(
            "No coefficient tokens ('C<int>' or 'C<int>_<label>') found in the "
            "vocabulary. The monomial model requires data in C/E expanded form; "
            "check the lexer/vocab config (range: coefficients)."
        )
    exp_token_ids = _exp_slots(vocab, model_config)
    if not exp_token_ids or not exp_token_ids[0]:
        raise ValueError(
            "No exponent tokens ('E<int>') found in the vocabulary. The monomial "
            "model requires data in C/E expanded form; check the lexer/vocab "
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
            "The monomial model needs pad/eos/bos token ids; the tokenizer "
            "does not define them and cfg.model does not override them."
        )

    return MonomialTransformerConfig(
        d_model=model_config.d_model,
        attention_heads=model_config.num_encoder_heads,
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        dim_feedforward=model_config.encoder_ffn_dim,
        max_input_len=model_config.max_sequence_length,
        vocab_size=len(tokenizer),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        use_positional_embedding=getattr(
            model_config, "use_positional_embedding", "generic"
        ),
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
