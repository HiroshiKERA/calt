"""
A decoder-only Transformer that reads and writes whole monomials.

This is the monomial-embedding model of ``calt.models.monomial`` with the
encoder removed: a single causal stack over the concatenation of the problem and
its solution, both folded to one sequence position per monomial.

    monomials   [C1 E1 E0 +] [C2 E0 E1 <] | ans | [C3 E1 E1 +] [C1 E0 E0 <]
                |----------- problem -----------|  |------- solution -------|
    loss                                              ^^^^^^^^^^^^^^^^^^^^^^

Two things are combined here, and both matter for what the model can express:

  - *the monomial embedding* (Kera et al., arXiv 2505.23696): a coefficient, its
    exponents and the separator that follows them occupy one position instead of
    ``n_vars + 2``, and each slot reads from its own block of the table so that
    ``x^3 y^2`` and ``x^2 y^3`` never collapse to the same vector;
  - *the decoder-only layout*: next-token prediction starts in the middle, at
    the first solution monomial, because a model asked to reproduce the problem
    from nothing could not.  The problem part is context and carries no loss.

The learnable answer marker sits between the two halves and plays the role the
``<bos>`` of the flat decoder-only model plays: it is the position whose output
predicts the first solution monomial.

Unlike the encoder-decoder monomial model, which keeps independent source and
target embedding tables, one stack sees one stream here, so problem and solution
share a single table.

Select it with ``model_type: monomial_decoder_only``.  Like every other model in
this package it consumes the seq2seq collator's batch unchanged
(``input_ids`` / ``attention_mask`` / ``decoder_input_ids`` / ``labels``) and
returns predictions and generations aligned with ``labels``.  The data must be
in C/E expanded form and ``model.num_variables`` must match it.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from ..monomial.model import (
    MonomialDecodeHead,
    MonomialEmbedding,
    _build_gid_map,
    _build_local_map,
)
from ..positional_embeddings import get_positional_embedding


def fold_to_grid(
    flat_ids: torch.Tensor, width: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape flat token ids (BOS already stripped) into a monomial grid.

    Args:
        flat_ids (torch.Tensor): ``(batch, seq)`` token ids; every valid sample
            is ``L * width`` tokens, padding may make ``seq`` longer.
        width (int): Tokens per monomial.
        pad_token_id (int): Id used for padding.

    Returns:
        tuple: grid ``(batch, L, width)`` padded up with ``pad_token_id``, and
        mask ``(batch, L)`` with True on valid monomial rows.
    """
    B, S = flat_ids.shape
    L = max(1, -(-S // width))  # ceil division, at least one row
    if L * width > S:
        pad_cols = flat_ids.new_full((B, L * width - S), pad_token_id)
        flat_ids = torch.cat([flat_ids, pad_cols], dim=1)
    grid = flat_ids.reshape(B, L, width)
    mask = grid[..., 0] != pad_token_id
    return grid, mask


class MonomialDecoderOnlyConfig(PretrainedConfig):
    """Configuration for the monomial decoder-only model.

    Same vocabulary structure as ``MonomialTransformerConfig`` — the token
    groups are stored as plain id lists so the config stays JSON-serializable —
    with the encoder/decoder layer counts replaced by a single ``num_layers``.

    Attributes:
        coeff_token_ids (list[list[int]]): One ordered id list per coefficient
            field.
        exp_token_ids (list[list[int]]): One ordered id list per variable slot.
        separator_token_ids (list[int]): Ids of the continuation separators in
            follow-slot order; EOS is appended internally as the last class.
    """

    model_type = "monomial_decoder_only"

    def __init__(
        self,
        d_model: int = 512,
        attention_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        bias: bool = True,
        vocab_size: int = 1000,
        max_input_len: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        use_positional_embedding: str = "generic",
        embedding_layer_norm: bool = True,
        coeff_token_ids: Optional[list] = None,
        exp_token_ids: Optional[list] = None,
        separator_token_ids: Optional[list] = None,
        coeff_scale: float = 1.0,
        coeff_noise_std: float = 0.0,
        coeff_loss_weight: float = 1.0,
        init_std: float = 0.02,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs,
        )
        self.d_model = d_model
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.bias = bias
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.use_positional_embedding = use_positional_embedding
        self.embedding_layer_norm = embedding_layer_norm
        self.coeff_token_ids = coeff_token_ids or [[]]
        self.exp_token_ids = exp_token_ids or [[]]
        self.separator_token_ids = separator_token_ids or []
        self.coeff_scale = coeff_scale
        self.coeff_noise_std = coeff_noise_std
        self.coeff_loss_weight = coeff_loss_weight
        self.init_std = init_std
        self.seed = seed

    @property
    def num_coeff_fields(self) -> int:
        return len(self.coeff_token_ids)

    @property
    def num_variables(self) -> int:
        return len(self.exp_token_ids)

    @property
    def monomial_width(self) -> int:
        """Tokens per monomial: coeff fields + exponent slots + follow slot."""
        return self.num_coeff_fields + self.num_variables + 1

    @property
    def num_follow_classes(self) -> int:
        """Continuation separators + EOS (always the last class)."""
        return len(self.separator_token_ids) + 1


class MonomialDecoderOnlyTransformer(PreTrainedModel):
    """Causal Transformer over ``[problem, answer marker, solution]`` monomials."""

    config_class = MonomialDecoderOnlyConfig

    def __init__(self, config: MonomialDecoderOnlyConfig):
        super().__init__(config)
        self.config = config

        if not config.coeff_token_ids or not config.coeff_token_ids[0]:
            raise ValueError(
                "MonomialDecoderOnlyConfig.coeff_token_ids is empty. Build the "
                "config with create_monomial_decoder_only_config(cfg.model, "
                "tokenizer) so the coefficient/exponent token groups are derived "
                "from the tokenizer."
            )
        if not config.exp_token_ids or not config.exp_token_ids[0]:
            raise ValueError(
                "MonomialDecoderOnlyConfig.exp_token_ids is empty. Set "
                "model.num_variables (or model.variables) in the config so the "
                "exponent slots are known."
            )

        d = config.d_model

        # One stack, one stream, one table: unlike the encoder-decoder model
        # there is no separate source side to keep independent.
        self.emb = MonomialEmbedding(
            config.vocab_size,
            d,
            config.num_coeff_fields,
            config.num_variables,
            coeff_scale=config.coeff_scale,
            coeff_noise_std=config.coeff_noise_std,
        )
        # The position between problem and solution. Its output predicts the
        # first solution monomial, exactly as the flat model's <bos> does.
        self.answer_emb = nn.Parameter(torch.zeros(1, 1, d))

        self.positional_embedding = get_positional_embedding(
            pe_type=config.use_positional_embedding,
            d_model=d,
            max_len=config.max_input_len * 2,
        )
        self.emb_norm = (
            nn.LayerNorm(d, eps=config.layer_norm_eps)
            if getattr(config, "embedding_layer_norm", True)
            else None
        )

        # A decoder without cross-attention is an encoder stack under a causal
        # mask, which is what nn.TransformerEncoder provides.
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
            bias=config.bias,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(d, eps=config.layer_norm_eps)
            if config.norm_first
            else None,
        )

        self.head = MonomialDecodeHead(
            [len(ids) for ids in config.coeff_token_ids],
            [len(ids) for ids in config.exp_token_ids],
            config.num_follow_classes,
            d,
        )

        follow_ids = list(config.separator_token_ids) + [config.eos_token_id]
        self.register_buffer(
            "coeff_gid_map",
            _build_gid_map(config.coeff_token_ids, config.pad_token_id),
        )
        self.register_buffer(
            "exp_gid_map", _build_gid_map(config.exp_token_ids, config.pad_token_id)
        )
        self.register_buffer(
            "follow_gid_map", torch.tensor(follow_ids, dtype=torch.long)
        )
        self.register_buffer(
            "coeff_local_map",
            _build_local_map(config.coeff_token_ids, config.vocab_size),
        )
        self.register_buffer(
            "exp_local_map", _build_local_map(config.exp_token_ids, config.vocab_size)
        )
        self.register_buffer(
            "follow_local_map", _build_local_map([follow_ids], config.vocab_size)[0]
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.answer_emb, std=config.init_std)
        self.seed = config.seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------ #
    # Folding and alignment                                               #
    # ------------------------------------------------------------------ #

    def _fold(self, flat_ids: torch.Tensor) -> tuple:
        return fold_to_grid(
            flat_ids, self.config.monomial_width, self.config.pad_token_id
        )

    def _check_alignment(self, grid: torch.Tensor, mask: torch.Tensor, what: str):
        """Fail fast when sequences are not monomial-aligned.

        Every valid monomial's follow slot must hold a separator or EOS. A stray
        token there means the data is not in C/E expanded form or
        ``num_variables`` doesn't match, which would corrupt training silently.
        """
        f_local = self.follow_local_map[grid[..., -1]]
        if bool(((f_local < 0) & mask).any()):
            raise ValueError(
                f"{what} is not aligned to monomials of width "
                f"{self.config.monomial_width} (= {self.config.num_coeff_fields} "
                f"coefficient field(s) + {self.config.num_variables} exponent "
                "slot(s) + 1 separator). The monomial decoder-only model requires "
                "data in C/E expanded form ('C<c> E<e1> .. E<en>' terms joined by "
                "'+'/'||') and model.num_variables matching the data."
            )

    # ------------------------------------------------------------------ #
    # Packing                                                             #
    # ------------------------------------------------------------------ #

    def _pack(
        self,
        src_grid: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_grid: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple:
        """Concatenate problem and solution per row, dropping problem padding.

        The two halves are folded independently, so a plain ``cat`` would leave
        padded monomial rows in the middle of the sequence and shift every
        solution monomial to a position that depends on the rest of the batch.

        Returns:
            tuple: packed grid ``(batch, total, width)``, boolean key padding
            mask ``(batch, total)``, and the per-row index of the answer marker.
        """
        B, L_src, w = src_grid.shape
        L_tgt = tgt_grid.size(1)
        device = src_grid.device

        src_len = src_mask.sum(dim=1)  # (B,)
        tgt_len = tgt_mask.sum(dim=1)  # (B,)

        total = L_src + 1 + L_tgt
        positions = torch.arange(total, device=device).unsqueeze(0)  # (1, total)
        start = src_len.unsqueeze(1)  # (B, 1), index of the answer marker

        from_src = positions < start
        from_tgt = (positions > start) & (positions <= start + L_tgt)

        src_index = positions.clamp(max=L_src - 1).expand(B, total)
        tgt_index = (positions - start - 1).clamp(0, L_tgt - 1)

        gathered_src = src_grid.gather(
            1, src_index.unsqueeze(-1).expand(-1, -1, w)
        )
        gathered_tgt = tgt_grid.gather(
            1, tgt_index.unsqueeze(-1).expand(-1, -1, w)
        )

        packed = torch.where(from_src.unsqueeze(-1), gathered_src, gathered_tgt)
        packed = torch.where(
            (from_src | from_tgt).unsqueeze(-1),
            packed,
            torch.full_like(packed, self.config.pad_token_id),
        )

        # Everything past the row's own content is padding: the tail of the
        # buffer, and the solution padding folding added.
        real = positions <= (start + tgt_len.unsqueeze(1))
        key_padding_mask = ~real

        return packed, key_padding_mask, src_len

    def _embed_packed(
        self, grid: torch.Tensor, marker_pos: torch.Tensor
    ) -> torch.Tensor:
        """Embed a packed monomial grid, overwriting the marker position.

        The ids sitting at the marker position are meaningless (they are
        whatever padding the packing left there); the learnable answer vector
        replaces them before positions are added.
        """
        embeddings = self.emb(grid)
        positions = torch.arange(grid.size(1), device=grid.device).unsqueeze(0)
        is_marker = (positions == marker_pos.unsqueeze(1)).unsqueeze(-1)
        embeddings = torch.where(
            is_marker, self.answer_emb.to(embeddings.dtype), embeddings
        )

        if self.positional_embedding is not None:
            embeddings = self.positional_embedding(embeddings)
        # Same normalization BART applies before its stack; without it the
        # residual stream enters the transformer far smaller than what the
        # layers add to it.
        if self.emb_norm is not None:
            embeddings = self.emb_norm(embeddings)
        return embeddings

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    def _run_stack(
        self,
        grid: torch.Tensor,
        marker_pos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embeddings = self._embed_packed(grid, marker_pos)
        causal = self._causal_mask(grid.size(1), grid.device)
        return self.transformer(
            embeddings, mask=causal, src_key_padding_mask=key_padding_mask
        )

    # ------------------------------------------------------------------ #
    # Loss                                                                #
    # ------------------------------------------------------------------ #

    def _gid_to_local(self, tgt_grid: torch.Tensor) -> tuple:
        k, n_vars = self.config.num_coeff_fields, self.config.num_variables
        c_local = torch.stack(
            [self.coeff_local_map[j][tgt_grid[..., j]] for j in range(k)], dim=-1
        )
        e_local = torch.stack(
            [self.exp_local_map[v][tgt_grid[..., k + v]] for v in range(n_vars)],
            dim=-1,
        )
        f_local = self.follow_local_map[tgt_grid[..., -1]]
        return c_local, e_local, f_local

    def _compute_loss(
        self,
        logits: dict,
        c_local: torch.Tensor,
        e_local: torch.Tensor,
        f_local: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Factored cross-entropy, averaged over valid solution monomials."""
        B, T = f_local.shape
        ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

        step_loss = torch.zeros(B, T, device=f_local.device)
        for j, logit_j in enumerate(logits["coeffs"]):
            step_loss = step_loss + self.config.coeff_loss_weight * ce(
                logit_j.reshape(B * T, -1), c_local[..., j].reshape(B * T)
            ).view(B, T)
        for v in range(e_local.size(-1)):
            step_loss = step_loss + ce(
                logits["exps"][:, :, v, :].reshape(B * T, -1),
                e_local[..., v].reshape(B * T),
            ).view(B, T)
        step_loss = step_loss + ce(
            logits["follow"].reshape(B * T, -1), f_local.reshape(B * T)
        ).view(B, T)

        step_loss = step_loss * mask.to(step_loss.dtype)
        return step_loss.sum() / mask.sum().clamp(min=1)

    def _predicted_grid(self, logits: dict) -> torch.Tensor:
        """Argmax each head and map local indices back to token ids: (B, T, w)."""
        k = self.config.num_coeff_fields
        parts = [
            self.coeff_gid_map[j][logits["coeffs"][j].argmax(-1)] for j in range(k)
        ]
        exp_local = logits["exps"].argmax(-1)
        parts += [
            self.exp_gid_map[v][exp_local[..., v]]
            for v in range(self.config.num_variables)
        ]
        parts.append(self.follow_gid_map[logits["follow"].argmax(-1)])
        return torch.stack(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # Forward / generate                                                  #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # input_ids = [BOS, content..., EOS, PAD...]; the content+EOS part is
        # monomial-aligned, so strip BOS and fold.
        src_grid, src_mask = self._fold(input_ids[:, 1:])
        self._check_alignment(src_grid, src_mask, "input_ids")

        # labels (= target without BOS, with EOS, -100 on padding) is already
        # the monomial-aligned view of the target; prefer it over
        # decoder_input_ids so teacher forcing and loss share one grid.
        if labels is not None:
            tgt_flat = labels.masked_fill(labels == -100, self.config.pad_token_id)
        elif decoder_input_ids is not None:
            tgt_flat = decoder_input_ids[:, 1:]
        else:
            raise ValueError("Either labels or decoder_input_ids must be provided")
        tgt_grid, tgt_mask = self._fold(tgt_flat)
        self._check_alignment(tgt_grid, tgt_mask, "labels")

        packed, key_padding_mask, src_len = self._pack(
            src_grid, src_mask, tgt_grid, tgt_mask
        )
        hidden = self._run_stack(packed, src_len, key_padding_mask)

        # The marker sits at src_len and predicts solution monomial 0; solution
        # monomial j sits at src_len+1+j and predicts j+1. Gathering src_len+j
        # for j in [0, L_tgt) therefore lines predictions up with tgt_grid.
        L_tgt = tgt_grid.size(1)
        offsets = torch.arange(L_tgt, device=packed.device).unsqueeze(0)
        index = (src_len.unsqueeze(1) + offsets).clamp(max=packed.size(1) - 1)
        target_hidden = hidden.gather(
            1, index.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )
        logits = self.head(target_hidden)

        loss = None
        if labels is not None:
            c_local, e_local, f_local = self._gid_to_local(tgt_grid)
            loss = self._compute_loss(logits, c_local, e_local, f_local, tgt_mask)

        # Flatten predictions back to token ids in the labels' shape so the
        # trainer's token-accuracy / success-rate metrics work unchanged.
        pred_flat = self._predicted_grid(logits).reshape(hidden.size(0), -1)
        if labels is not None:
            pred_flat = pred_flat[:, : labels.size(1)]

        if not return_dict:
            return (loss, pred_flat) if loss is not None else (pred_flat,)

        return CausalLMOutput(loss=loss, logits=pred_flat)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Greedy monomial-level continuation, returned as flat token ids.

        Only the generated part is returned, opening with BOS, so the existing
        exact-match evaluation (``batch_decode(skip_special_tokens=True)``)
        compares it against the labels unchanged. ``max_length`` counts flat
        tokens, like the other models in this package.
        """
        self.eval()
        cfg = self.config
        if pad_token_id is None:
            pad_token_id = cfg.pad_token_id

        src_grid, src_mask = self._fold(input_ids[:, 1:])
        self._check_alignment(src_grid, src_mask, "input_ids")

        B, L_src, w = src_grid.shape
        device = input_ids.device
        eos_class = cfg.num_follow_classes - 1
        max_monomials = max(1, (max_length - 1) // w)

        src_len = src_mask.sum(dim=1)  # (B,), also the marker position

        # The buffer holds the problem, the answer marker, and room for the
        # generated monomials. Each row writes at its own cursor, since problems
        # differ in length.
        total = L_src + 1 + max_monomials
        positions = torch.arange(total, device=device).unsqueeze(0)
        start = src_len.unsqueeze(1)
        src_index = positions.clamp(max=L_src - 1).expand(B, total)
        buffer = torch.full(
            (B, total, w), pad_token_id, dtype=torch.long, device=device
        )
        buffer = torch.where(
            (positions < start).unsqueeze(-1),
            src_grid.gather(1, src_index.unsqueeze(-1).expand(-1, -1, w)),
            buffer,
        )

        cursor = src_len + 1  # first free slot, just past the marker
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = 0

        for _ in range(max_monomials):
            width = int(cursor.max().item())
            real = torch.arange(width, device=device).unsqueeze(0) < cursor.unsqueeze(1)
            hidden = self._run_stack(buffer[:, :width], src_len, ~real)

            last = (cursor - 1).view(B, 1, 1).expand(-1, -1, hidden.size(-1))
            logits = self.head(hidden.gather(1, last))
            row = self._predicted_grid(logits)  # (B, 1, w)

            # Keep finished rows closed by emitting PAD monomials.
            row[finished] = pad_token_id
            follow_class = logits["follow"][:, 0].argmax(-1)
            finished = finished | (follow_class == eos_class)

            buffer.scatter_(
                1, cursor.view(B, 1, 1).expand(-1, -1, w), row
            )
            cursor = cursor + 1
            generated += 1

            if bool(finished.all()):
                break

        # Cut out the answer only: what was written past the marker.
        offsets = torch.arange(generated, device=device).unsqueeze(0)
        index = (start + 1 + offsets).clamp(max=total - 1)
        answer = buffer.gather(1, index.unsqueeze(-1).expand(-1, -1, w))

        bos_col = torch.full((B, 1), cfg.bos_token_id, dtype=torch.long, device=device)
        return torch.cat([bos_col, answer.reshape(B, -1)], dim=1)
