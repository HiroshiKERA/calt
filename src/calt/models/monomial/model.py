"""
Encoder-decoder Transformer with a monomial-structured embedding (HATSolver-style).

Why this model exists
---------------------
Polynomials in the C/E expanded form (see
``calt.io.preprocessor.load_preprocessors.expanded_form``) spend ``1 + n_vars``
tokens per term: one coefficient token (``C<coeff>``) and one exponent token per
variable (``E<e1> E<e2> ...``), with ``+`` between terms and ``||`` between
polynomials. A flat Transformer must learn that these tokens form one semantic
unit. The *monomial embedding* (Kera et al., HATSolver appendix) builds that
structure in: each monomial — coefficient, exponents, and the following
separator — is compressed into a single sequence position, and the decoder
predicts the parts of the next monomial with separate classification heads
(one per coefficient field, one per variable, one for the separator). Sequences
become ``1 + n_vars + 1`` times shorter, and the output space per head is the
coefficient/exponent range instead of the whole vocabulary.

Design that keeps it a drop-in
------------------------------
Like ``encoder_classifier``, this model consumes the *same* batch the seq2seq
collator already produces (``input_ids``, ``attention_mask``, ``labels``), so
neither the IOPipeline nor the data collator change:

  - ``forward`` strips BOS and reshapes the flat token sequence into a
    ``(batch, monomials, width)`` grid — valid because in expanded form every
    monomial occupies exactly ``width = k + n_vars + 1`` tokens (``k``
    coefficient fields, ``n_vars`` exponent slots, and one slot holding ``+``,
    ``||``, or the final EOS);
  - the loss is the factored cross-entropy over the per-part heads, but
    ``logits`` returned to the trainer are the *flat predicted token ids* in the
    same shape as ``labels``, so token accuracy / success-rate metrics work
    unchanged;
  - :meth:`generate` runs the autoregressive loop at monomial level and returns
    flat token ids ``[BOS, C.., E.., sep, ..., EOS]`` so the existing
    exact-match generation evaluation (``tokenizer.batch_decode``) works
    unchanged.

Select it with ``model_type: monomial`` in train.yaml; the config mapping
derives the coefficient/exponent/separator token groups from the tokenizer
(see ``config_mapping.create_monomial_config``). The data must be in expanded
form and the model must know the number of variables (``model.num_variables``).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from ..positional_embeddings import get_positional_embedding


class MonomialTransformerConfig(PretrainedConfig):
    """Configuration for the monomial-embedding encoder-decoder model.

    On top of the usual architecture fields (same names as ``TransformerConfig``),
    it stores the vocabulary structure as plain token-id lists so the config
    stays JSON-serializable and ``save_pretrained``/``from_pretrained`` round-trip:

    Attributes:
        coeff_token_ids (list[list[int]]): One ordered id list per coefficient
            field (single-field ``[[C-50 .. C50]]`` for CALT's expanded form;
            multiple fields support multi-modulus ``C3_f2 C1_f7`` vocabularies).
        exp_token_ids (list[list[int]]): One ordered id list per variable slot.
        separator_token_ids (list[int]): Ids of the continuation separators in
            follow-slot order (e.g. ``[+, ||]``). EOS is appended internally as
            the last follow class.
    """

    model_type = "monomial"

    def __init__(
        self,
        d_model: int = 512,
        attention_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        vocab_size: int = 1000,
        max_input_len: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        use_positional_embedding: str = "generic",
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
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.use_positional_embedding = use_positional_embedding
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


class MonomialEmbedding(nn.Module):
    """Embed a ``(batch, monomials, width)`` id grid into ``(batch, monomials, d)``.

    A single table is shared by all token types; the monomial vector is

        coeff_scale * mean_j(emb[coeff_j]) + mean(emb[exp_1..n_vars], emb[follow])

    ``coeff_noise_std`` optionally adds Gaussian noise to the coefficient part
    during training (ablation knob from the reference implementation).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_coeff_fields: int,
        num_variables: int,
        coeff_scale: float = 1.0,
        coeff_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_coeff_fields = num_coeff_fields
        self.num_variables = num_variables
        self.coeff_scale = coeff_scale
        self.coeff_noise_std = coeff_noise_std
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        k, n_vars = self.num_coeff_fields, self.num_variables
        coeff_part = sum(self.emb(ids[..., j]) for j in range(k)) / k
        if self.coeff_noise_std > 0.0 and self.training:
            coeff_part = (
                coeff_part + torch.randn_like(coeff_part) * self.coeff_noise_std
            )
        exp_part = (
            sum(self.emb(ids[..., k + v]) for v in range(n_vars))
            + self.emb(ids[..., -1])
        ) / (n_vars + 1)
        return self.coeff_scale * coeff_part + exp_part


class MonomialDecodeHead(nn.Module):
    """Parallel classification heads: one per coefficient field, one per
    variable, and one for the follow slot (separators + EOS)."""

    def __init__(
        self,
        coeff_sizes: list,
        exp_sizes: list,
        num_follow_classes: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.coeff_heads = nn.ModuleList(nn.Linear(d_model, s) for s in coeff_sizes)
        self.exp_heads = nn.ModuleList(nn.Linear(d_model, s) for s in exp_sizes)
        self.follow_head = nn.Linear(d_model, num_follow_classes)

    def forward(self, hidden: torch.Tensor) -> dict:
        return {
            "coeffs": [h(hidden) for h in self.coeff_heads],
            "exps": torch.stack([h(hidden) for h in self.exp_heads], dim=2),
            "follow": self.follow_head(hidden),
        }


def _build_gid_map(id_lists: list, pad_id: int) -> torch.Tensor:
    """local index -> token id, shape (n_lists, max_list_len), padded with pad_id."""
    max_len = max(len(ids) for ids in id_lists)
    t = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    for j, ids in enumerate(id_lists):
        t[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return t


def _build_local_map(id_lists: list, vocab_size: int) -> torch.Tensor:
    """token id -> local index, shape (n_lists, vocab_size), -1 = not in list."""
    t = torch.full((len(id_lists), vocab_size), -1, dtype=torch.long)
    for j, ids in enumerate(id_lists):
        for local, gid in enumerate(ids):
            t[j, gid] = local
    return t


class MonomialTransformer(PreTrainedModel):
    """Encoder-decoder Transformer operating on monomial-level positions."""

    config_class = MonomialTransformerConfig

    def __init__(self, config: MonomialTransformerConfig):
        super().__init__(config)
        self.config = config

        if not config.coeff_token_ids or not config.coeff_token_ids[0]:
            raise ValueError(
                "MonomialTransformerConfig.coeff_token_ids is empty. Build the "
                "config with create_monomial_config(cfg.model, tokenizer) so the "
                "coefficient/exponent token groups are derived from the tokenizer."
            )
        if not config.exp_token_ids or not config.exp_token_ids[0]:
            raise ValueError(
                "MonomialTransformerConfig.exp_token_ids is empty. Set "
                "model.num_variables (or model.variables) in the config so the "
                "exponent slots are known."
            )

        d = config.d_model
        # Encoder and decoder embeddings hold independent tables, matching the
        # reference implementation.
        self.src_emb = MonomialEmbedding(
            config.vocab_size,
            d,
            config.num_coeff_fields,
            config.num_variables,
            coeff_scale=config.coeff_scale,
            coeff_noise_std=config.coeff_noise_std,
        )
        self.tgt_emb = MonomialEmbedding(
            config.vocab_size,
            d,
            config.num_coeff_fields,
            config.num_variables,
            coeff_scale=config.coeff_scale,
        )
        # Learnable decoder start position: BOS has no monomial structure, so it
        # gets a dedicated vector instead of going through the embedding.
        self.bos_emb = nn.Parameter(torch.zeros(1, 1, d))

        self.positional_embedding = get_positional_embedding(
            pe_type=config.use_positional_embedding,
            d_model=d,
            max_len=config.max_input_len * 2,
        )

        self.transformer = nn.Transformer(
            d_model=d,
            nhead=config.attention_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=config.batch_first,
            norm_first=config.norm_first,
            bias=config.bias,
        )

        self.head = MonomialDecodeHead(
            [len(ids) for ids in config.coeff_token_ids],
            [len(ids) for ids in config.exp_token_ids],
            config.num_follow_classes,
            d,
        )

        # local index <-> token id maps, registered as buffers so they follow
        # the model across devices and are excluded from optimization.
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
        nn.init.normal_(self.bos_emb, std=config.init_std)
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
    # Flat sequence <-> monomial grid                                     #
    # ------------------------------------------------------------------ #

    def _fold(self, flat_ids: torch.Tensor) -> tuple:
        """Reshape flat token ids (BOS already stripped) into a monomial grid.

        Args:
            flat_ids (torch.Tensor): (batch, seq) token ids; every valid sample
                is ``L * width`` tokens (padding may make seq longer).

        Returns:
            tuple: grid (batch, L, width) padded up with pad_token_id, and
                mask (batch, L) with True on valid monomial rows.
        """
        B, S = flat_ids.shape
        w = self.config.monomial_width
        L = max(1, -(-S // w))  # ceil division, at least one row
        if L * w > S:
            pad_cols = flat_ids.new_full((B, L * w - S), self.config.pad_token_id)
            flat_ids = torch.cat([flat_ids, pad_cols], dim=1)
        grid = flat_ids.reshape(B, L, w)
        mask = grid[..., 0] != self.config.pad_token_id
        return grid, mask

    def _check_alignment(self, grid: torch.Tensor, mask: torch.Tensor, what: str):
        """Fail fast when sequences are not monomial-aligned.

        Every valid monomial's follow slot must hold a separator or EOS. A stray
        token there means the data is not in C/E expanded form or
        ``num_variables`` doesn't match the data, which would otherwise corrupt
        training silently.
        """
        f_local = self.follow_local_map[grid[..., -1]]
        if bool(((f_local < 0) & mask).any()):
            raise ValueError(
                f"{what} is not aligned to monomials of width "
                f"{self.config.monomial_width} (= {self.config.num_coeff_fields} "
                f"coefficient field(s) + {self.config.num_variables} exponent "
                "slot(s) + 1 separator). The monomial model requires data in C/E "
                "expanded form ('C<c> E<e1> .. E<en>' terms joined by '+'/'||') "
                "and model.num_variables matching the data."
            )

    # ------------------------------------------------------------------ #
    # Encoder / decoder                                                    #
    # ------------------------------------------------------------------ #

    def _apply_positions(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.positional_embedding is not None:
            embeddings = self.positional_embedding(embeddings)
        return embeddings

    def _encode(self, src_grid: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self._apply_positions(self.src_emb(src_grid))
        return self.transformer.encoder(x, src_key_padding_mask=~src_mask)

    def _decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_grid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode with the learnable BOS vector prepended; returns (B, T+1, d)
        hidden states where position t predicts monomial t."""
        B = memory.size(0)
        bos = self.bos_emb.expand(B, 1, -1)
        tgt = bos if tgt_grid is None else torch.cat([bos, self.tgt_emb(tgt_grid)], 1)
        tgt = self._apply_positions(tgt)
        causal = self._causal_mask(tgt.size(1), tgt.device)
        return self.transformer.decoder(
            tgt, memory, tgt_mask=causal, memory_key_padding_mask=~src_mask
        )

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
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
        """Factored cross-entropy, averaged over valid monomials.

        Padding rows carry local index -1 everywhere (pad is in no category), so
        ignore_index and the mask are redundant safety nets for each other.
        """
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
        exp_local = logits["exps"].argmax(-1)  # (B, T, n_vars)
        parts += [
            self.exp_gid_map[v][exp_local[..., v]]
            for v in range(self.config.num_variables)
        ]
        parts.append(self.follow_gid_map[logits["follow"].argmax(-1)])
        return torch.stack(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # Forward / generate                                                   #
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
        memory = self._encode(src_grid, src_mask)

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

        dec_grid = tgt_grid[:, :-1] if tgt_grid.size(1) > 1 else None
        hidden = self._decode(memory, src_mask, dec_grid)
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            c_local, e_local, f_local = self._gid_to_local(tgt_grid)
            loss = self._compute_loss(logits, c_local, e_local, f_local, tgt_mask)

        # Flatten predictions back to token ids in the labels' shape so the
        # trainer's token-accuracy/success-rate metrics work unchanged (the
        # generic model also returns argmax'd ids in `logits`).
        pred_flat = self._predicted_grid(logits).reshape(hidden.size(0), -1)
        if labels is not None:
            pred_flat = pred_flat[:, : labels.size(1)]

        if not return_dict:
            return (loss, pred_flat) if loss is not None else (pred_flat,)

        return Seq2SeqLMOutput(loss=loss, logits=pred_flat)

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
        """Greedy monomial-level generation returning flat token ids.

        The output is ``[BOS, C.., E.., sep, ..., EOS, PAD..]`` per row so the
        existing evaluation (``batch_decode(skip_special_tokens=True)`` +
        exact match) works unchanged. ``max_length`` counts flat tokens, like
        the generic model.
        """
        self.eval()
        cfg = self.config
        if pad_token_id is None:
            pad_token_id = cfg.pad_token_id

        src_grid, src_mask = self._fold(input_ids[:, 1:])
        self._check_alignment(src_grid, src_mask, "input_ids")
        memory = self._encode(src_grid, src_mask)

        B = input_ids.size(0)
        device = input_ids.device
        w = cfg.monomial_width
        eos_class = cfg.num_follow_classes - 1
        max_monomials = max(1, (max_length - 1) // w)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        gen_grid: Optional[torch.Tensor] = None

        for _ in range(max_monomials):
            hidden = self._decode(memory, src_mask, gen_grid)
            logits = self.head(hidden[:, -1:, :])
            row = self._predicted_grid(logits)  # (B, 1, w)

            # Keep finished rows closed by emitting PAD monomials.
            row[finished] = pad_token_id
            follow_class = logits["follow"][:, 0].argmax(-1)
            finished = finished | (follow_class == eos_class)

            gen_grid = row if gen_grid is None else torch.cat([gen_grid, row], dim=1)
            if bool(finished.all()):
                break

        bos_col = torch.full((B, 1), cfg.bos_token_id, dtype=torch.long, device=device)
        return torch.cat([bos_col, gen_grid.reshape(B, -1)], dim=1)
