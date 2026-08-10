"""Regression tests for two defects that silently degraded seq2seq accuracy.

1. ``generic``'s decoder cross-attention ignored the encoder padding mask
   (``memory_key_padding_mask`` was never passed to ``nn.Transformer``), so a
   sample's prediction depended on how much padding its batch mates forced onto
   it. With ``padding="longest"`` that is a large fraction of the memory.

2. ``MonomialEmbedding`` embedded every slot of the monomial out of one shared
   table and summed, making the result symmetric in the variables: x^3*y^2 and
   x^2*y^3 produced the same vector.

Both are invisible in the loss curve and only show up as a quality ceiling, so
they are pinned here as invariants.
"""

import pytest
import torch

from calt.models.generic.model import Transformer, TransformerConfig
from calt.models.monomial.model import MonomialEmbedding

PAD_ID = 0


def _generic_model(vocab_size: int = 24, max_len: int = 64) -> Transformer:
    model = Transformer(
        TransformerConfig(
            d_model=32,
            attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            vocab_size=vocab_size,
            max_input_len=max_len,
            pad_token_id=PAD_ID,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def test_generic_loss_invariant_to_encoder_padding():
    """Appending PAD to the encoder input must not change the loss."""
    torch.manual_seed(0)
    model = _generic_model()

    src = torch.tensor([[5, 6, 7, 8, 9]])
    dec_in = torch.tensor([[2, 10, 11, 12]])
    labels = torch.tensor([[10, 11, 12, 1]])

    with torch.no_grad():
        tight = model(
            input_ids=src,
            attention_mask=torch.ones_like(src),
            decoder_input_ids=dec_in,
            decoder_attention_mask=torch.ones_like(dec_in),
            labels=labels,
        ).loss

        pad_width = 11
        src_padded = torch.cat([src, src.new_full((1, pad_width), PAD_ID)], dim=1)
        mask_padded = torch.cat(
            [torch.ones_like(src), src.new_zeros((1, pad_width))], dim=1
        )
        padded = model(
            input_ids=src_padded,
            attention_mask=mask_padded,
            decoder_input_ids=dec_in,
            decoder_attention_mask=torch.ones_like(dec_in),
            labels=labels,
        ).loss

    torch.testing.assert_close(tight, padded, rtol=1e-5, atol=1e-6)


def test_generic_generation_invariant_to_encoder_padding():
    """Generation must not depend on the batch's padding either."""
    torch.manual_seed(0)
    model = _generic_model()

    src = torch.tensor([[5, 6, 7, 8, 9]])
    pad_width = 9
    src_padded = torch.cat([src, src.new_full((1, pad_width), PAD_ID)], dim=1)
    mask_padded = torch.cat(
        [torch.ones_like(src), src.new_zeros((1, pad_width))], dim=1
    )

    with torch.no_grad():
        tight = model.generate(src, attention_mask=torch.ones_like(src), max_length=8)
        padded = model.generate(src_padded, attention_mask=mask_padded, max_length=8)

    assert torch.equal(tight, padded), (
        f"padding changed the generated sequence: {tight.tolist()} vs {padded.tolist()}"
    )


def test_embeddings_are_normalized_before_the_stack():
    """token+position must enter the stack at unit scale, as in BART.

    Left unnormalized the residual stream starts ~35x smaller than what the
    layers add to it. Measured cost on a GF7 cumulative-product memorization
    probe: coefficient accuracy stuck near chance (0.225 at epoch 36 vs 0.688
    with the norm) and exact match flat at 0.000 vs 0.220.
    """
    torch.manual_seed(0)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])

    normed = _generic_model()  # default config
    assert normed.embedding_layer_norm is not None, (
        "embedding_layer_norm must stay on by default"
    )
    with torch.no_grad():
        scale = normed._compute_embeddings(ids).std().item()
    assert 0.5 < scale < 2.0, f"embeddings enter the stack at std={scale:.3f}"

    raw = Transformer(
        TransformerConfig(
            d_model=32,
            attention_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            vocab_size=24,
            max_input_len=64,
            pad_token_id=PAD_ID,
            dropout=0.0,
            embedding_layer_norm=False,
        )
    ).eval()
    with torch.no_grad():
        assert raw._compute_embeddings(ids).std().item() < 0.2


def test_pipeline_built_model_has_the_embedding_norm_by_default():
    """A config that never mentions the key must still get the LayerNorm.

    The fix is only worth anything if it reaches models built the normal way,
    through ModelPipeline from a YAML. A config_mapping default of False would
    silently switch it back off for every existing config.
    """
    from omegaconf import OmegaConf

    from calt.io.tokenizer import get_tokenizer
    from calt.io.vocabulary.config import VocabConfig
    from calt.models import ModelPipeline

    tokenizer = get_tokenizer(
        VocabConfig(vocab=["C1", "C2", "E0", "E1", "+", "|"], special_tokens={})
    )
    cfg = OmegaConf.create(
        {
            "model_type": "generic",
            "d_model": 32,
            "num_encoder_heads": 2,
            "num_decoder_heads": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "encoder_ffn_dim": 32,
            "decoder_ffn_dim": 32,
            "max_sequence_length": 64,
            "num_variables": 2,
        }
    )
    model = ModelPipeline(cfg, tokenizer).build()
    assert model.embedding_layer_norm is not None

    cfg.embedding_layer_norm = False  # ...and it must still be switchable off
    assert ModelPipeline(cfg, tokenizer).build().embedding_layer_norm is None


def test_monomial_embedding_is_not_symmetric_in_the_variables():
    """x^a*y^b and x^b*y^a must not collapse to the same vector."""
    torch.manual_seed(0)
    emb = MonomialEmbedding(
        vocab_size=30, d_model=64, num_coeff_fields=1, num_variables=2
    )

    # grid columns: [coeff, exp_x, exp_y, follow]
    xa_yb = torch.tensor([[[5, 7, 9, 24]]])
    xb_ya = torch.tensor([[[5, 9, 7, 24]]])

    with torch.no_grad():
        assert not torch.equal(emb(xa_yb), emb(xb_ya)), (
            "exponent slots share one embedding block, so the monomial vector "
            "only encodes the exponent multiset"
        )
        # ...while the same monomial must still embed identically.
        torch.testing.assert_close(emb(xa_yb), emb(xa_yb.clone()))


def test_monomial_embedding_rejects_wrong_grid_width():
    emb = MonomialEmbedding(
        vocab_size=30, d_model=16, num_coeff_fields=1, num_variables=2
    )
    with pytest.raises(ValueError, match="grid width"):
        emb(torch.tensor([[[5, 7, 24]]]))
