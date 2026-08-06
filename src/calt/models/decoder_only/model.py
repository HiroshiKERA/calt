"""
A decoder-only (causal LM) Transformer usable with the Hugging Face Trainer.

This is the generic encoder-decoder model with the encoder removed: a single
causal self-attention stack over the concatenation of the problem and its
solution.  The problem part is written into the sequence and carries no loss —
next-token prediction starts in the middle, at the first solution token —
because a model asked to reproduce the problem from ``<bos>`` alone could not.

    ids     <bos> 1 3 + 0 8 <eos> <bos> 2 1 <eos>
            |------- problem -------|--- solution ---|
    loss                                 ^^^^^^^^^^^^

The model consumes the same collated batch as the seq2seq models
(``input_ids`` / ``attention_mask`` / ``decoder_input_ids`` / ``labels``) and
returns predictions and generations aligned with ``labels``, so no other part of
the pipeline changes.  The ``<bos>`` that opens ``decoder_input_ids`` doubles as
the separator marking where the answer begins.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import logging

from ..positional_embeddings import get_positional_embedding

logger = logging.get_logger(__name__)


class DecoderOnlyConfig(PretrainedConfig):
    """Configuration for the decoder-only Transformer."""

    model_type = "decoder_only"

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
        init_std: float = 0.02,
        tie_word_embeddings: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        """Configure layer sizes, embeddings, and tokenizer metadata.

        Args:
            d_model (int, optional): Hidden size used across embeddings and blocks.
            attention_heads (int, optional): Number of attention heads per block.
            num_layers (int, optional): Number of causal self-attention blocks.
            dim_feedforward (int, optional): Width of the feed-forward layers.
            dropout (float, optional): Dropout probability applied in the stack.
            activation (str, optional): Activation used inside feed-forward blocks.
            layer_norm_eps (float, optional): Epsilon for layer normalization.
            norm_first (bool, optional): Whether to apply layer norm before attention.
            bias (bool, optional): If linear layers include a bias term.
            vocab_size (int, optional): Vocabulary size for embeddings and head.
            max_input_len (int, optional): Maximum supported sequence length.
            pad_token_id (int, optional): Padding token id.
            eos_token_id (int, optional): End-of-sequence token id.
            bos_token_id (int, optional): Beginning-of-sequence token id, also the
                token that opens the solution part of the sequence.
            use_positional_embedding (str, optional): Positional embedding strategy.
            embedding_layer_norm (bool, optional): Normalize token+position before
                the stack, as BART does.
            init_std (float, optional): Standard deviation for weight init.
            tie_word_embeddings (bool, optional): Whether to share embed/lm_head.
            seed (int, optional): Seed used for deterministic initialization.
            **kwargs: Extra options passed to `PretrainedConfig`.
        """
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
        self.init_std = init_std
        self.tie_word_embeddings = tie_word_embeddings
        self.seed = seed


class DecoderOnlyTransformer(PreTrainedModel):
    """Causal Transformer over ``[problem, solution]``, HF Trainer compatible."""

    config_class = DecoderOnlyConfig

    def __init__(self, config: DecoderOnlyConfig):
        """Build embedding, causal stack, and language modeling head.

        Args:
            config (DecoderOnlyConfig): Model hyper-parameters and tokenizer metadata.
        """
        super().__init__(config)

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.positional_embedding = get_positional_embedding(
            pe_type=config.use_positional_embedding,
            d_model=config.d_model,
            max_len=config.max_input_len * 2,
        )

        # Same reason as the seq2seq model: without normalizing token+position,
        # the residual stream enters the stack far smaller than what the layers
        # add to it and precise value prediction never leaves chance level.
        self.emb_norm = (
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if getattr(config, "embedding_layer_norm", True)
            else None
        )

        # A decoder without cross-attention is exactly an encoder stack driven by
        # a causal mask, which is what nn.TransformerEncoder provides.
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
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
            norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if config.norm_first
            else None,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.tie_weights()
        self.seed = config.seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _init_weights(self, module):
        """Apply CALT-specific initialization to supported modules."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _compute_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Compute token embeddings combined with positional embeddings."""
        embeddings = self.embedding(input_ids)

        if self.positional_embedding is not None:
            embeddings = self.positional_embedding(embeddings)

        if self.emb_norm is not None:
            embeddings = self.emb_norm(embeddings)

        return embeddings

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create an upper-triangular mask preventing attention to the future."""
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    @staticmethod
    def _lengths(mask: Optional[torch.Tensor], width: int, device) -> torch.Tensor:
        """Real (non-padded) length per row; the full width when no mask is given."""
        if mask is None:
            return torch.full((1,), width, dtype=torch.long, device=device)
        return mask.to(torch.bool).sum(dim=1)

    def _run_stack(
        self, ids: torch.LongTensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Embed and run the causal stack over ``ids``."""
        embeddings = self._compute_embeddings(ids)
        causal_mask = self._generate_causal_mask(ids.size(1), ids.device)
        return self.transformer(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

    def _pack(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        target_ids: torch.LongTensor,
        target_mask: Optional[torch.Tensor],
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """Concatenate problem and solution per row, dropping the problem padding.

        The collator pads the two sides independently, so a plain ``cat`` would
        leave padding in the middle of the sequence and shift every solution
        token to a position that depends on the rest of the batch.  Packing keeps
        the solution starting immediately after the problem's last real token.

        Returns:
            tuple: packed ids ``(batch, total)``, boolean key padding mask, and
            the per-row index at which the solution starts.
        """
        device = input_ids.device
        batch, in_width = input_ids.shape
        tgt_width = target_ids.size(1)

        in_len = self._lengths(attention_mask, in_width, device).expand(batch)
        tgt_len = self._lengths(target_mask, tgt_width, device).expand(batch)

        total = in_width + tgt_width
        positions = torch.arange(total, device=device).unsqueeze(0)  # (1, total)
        start = in_len.unsqueeze(1)  # (batch, 1)

        in_index = positions.clamp(max=in_width - 1).expand(batch, total)
        tgt_index = (positions - start).clamp(0, tgt_width - 1)

        from_input = positions < start
        from_target = (positions >= start) & (positions < start + tgt_width)

        packed = torch.where(
            from_input,
            input_ids.gather(1, in_index),
            target_ids.gather(1, tgt_index),
        )
        packed = torch.where(
            from_input | from_target,
            packed,
            torch.full_like(packed, self.config.pad_token_id),
        )

        # Everything past the row's own real content is padding: the tail of the
        # buffer, and the solution padding the collator added.
        real = positions < (start + tgt_len.unsqueeze(1))
        key_padding_mask = ~real

        return packed, key_padding_mask, in_len

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the causal stack over ``[problem, solution]``.

        Args:
            input_ids (torch.LongTensor | None): Problem token ids.
            attention_mask (torch.Tensor | None): Mask aligned with `input_ids`.
            decoder_input_ids (torch.LongTensor | None): Solution inputs, opening
                with the BOS that marks where the answer begins.
            decoder_attention_mask (torch.Tensor | None): Mask for the solution.
            labels (torch.LongTensor | None): Targets aligned with
                `decoder_input_ids`, padding marked as -100.
            return_dict (bool | None): Whether to return a dataclass output.
            **kwargs: Extra arguments kept for HF Trainer compatibility.

        Returns:
            CausalLMOutput: Loss, and predicted ids aligned with `labels`.

        Raises:
            ValueError: If either half of the sequence is missing.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None or decoder_input_ids is None:
            raise ValueError(
                "The decoder-only model needs both input_ids (the problem) and "
                "decoder_input_ids (the solution) to build one causal sequence."
            )

        packed, key_padding_mask, in_len = self._pack(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask
        )
        hidden = self._run_stack(packed, key_padding_mask)

        # Position start+i holds solution token i, so its output predicts
        # labels[:, i]. Gathering that slice back out keeps predictions the same
        # shape as labels, which is what the metric and the seq2seq models assume.
        tgt_width = decoder_input_ids.size(1)
        offsets = torch.arange(tgt_width, device=packed.device).unsqueeze(0)
        index = (in_len.unsqueeze(1) + offsets).clamp(max=packed.size(1) - 1)
        target_hidden = hidden.gather(
            1, index.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )

        logits = self.lm_head(target_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutput(
            loss=loss,
            # Ids rather than logits, matching the seq2seq model in this package:
            # the trainer's metric argmaxes only 3-D predictions.
            logits=torch.argmax(logits, dim=-1),
            hidden_states=None,
            attentions=None,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Continue each problem with its solution, one token at a time.

        Only the generated part is returned, so exact-match evaluation compares
        it against the labels unchanged; the problem stays in the context.

        Args:
            input_ids (torch.LongTensor): Problem token ids.
            attention_mask (torch.Tensor | None): Mask aligned with `input_ids`.
            max_length (int, optional): Maximum number of tokens to generate.
            num_beams (int, optional): Reserved for API parity (unused).
            temperature (float, optional): Sampling temperature.
            do_sample (bool, optional): Whether to sample instead of argmax.
            pad_token_id (int | None): Override for padding id.
            eos_token_id (int | None): Override for end-of-sequence id.
            **kwargs: Extra keyword arguments for API compatibility.

        Returns:
            torch.LongTensor: Generated sequences, opening with BOS, in the same
            layout the seq2seq model returns.
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        device = input_ids.device
        batch, in_width = input_ids.shape
        in_len = self._lengths(attention_mask, in_width, device).expand(batch)

        # The answer opens with BOS, so the buffer holds the problem, that BOS,
        # and room for max_length generated tokens.
        buffer_width = in_width + 1 + max_length
        buffer = torch.full(
            (batch, buffer_width), pad_token_id, dtype=torch.long, device=device
        )
        positions = torch.arange(buffer_width, device=device).unsqueeze(0)
        start = in_len.unsqueeze(1)
        in_index = positions.clamp(max=in_width - 1).expand(batch, buffer_width)
        buffer = torch.where(positions < start, input_ids.gather(1, in_index), buffer)
        buffer.scatter_(1, start, torch.full_like(start, self.config.bos_token_id))

        # Each row writes at its own cursor, since problems differ in length.
        cursor = in_len + 1
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated = 0

        for _ in range(max_length):
            width = int(cursor.max().item())
            real = torch.arange(width, device=device).unsqueeze(0) < cursor.unsqueeze(1)
            hidden = self._run_stack(buffer[:, :width], ~real)

            last = (cursor - 1).view(batch, 1, 1).expand(-1, -1, hidden.size(-1))
            next_token_logits = self.lm_head(hidden.gather(1, last).squeeze(1))

            if do_sample:
                next_token_logits = next_token_logits / temperature
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 1
                )
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Keep finished sequences closed by emitting PAD after EOS.
            next_token[finished.unsqueeze(1)] = pad_token_id
            finished = finished | (next_token.squeeze(1) == eos_token_id)

            buffer.scatter_(1, cursor.unsqueeze(1), next_token)
            cursor = cursor + 1
            generated += 1

            if finished.all():
                break

        # Cut out the answer only: the BOS at `start` plus what was generated.
        width = 1 + generated
        offsets = torch.arange(width, device=device).unsqueeze(0)
        index = (start + offsets).clamp(max=buffer_width - 1)
        return buffer.gather(1, index)
