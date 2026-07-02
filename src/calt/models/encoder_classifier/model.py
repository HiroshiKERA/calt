"""
Encoder-only classification model usable with the Hugging Face Trainer.

Why this model exists
---------------------
For tasks whose answer is a *single token* drawn from a small set — e.g. the
permutation-parity task, whose target is ``"+1"`` or ``"-1"`` — a full
encoder-decoder Transformer is overkill: there is nothing to generate
autoregressively. This model offers an encoder-only alternative that the user
can opt into with ``model_type: encoder_classifier`` instead of the default
encoder-decoder (``generic``/``bart``).

Design that keeps it a drop-in
------------------------------
It consumes the *same* batch the seq2seq collator already produces
(``input_ids``, ``attention_mask``, ``labels``) so neither the IOPipeline nor
the data collator need to change:

  - it encodes ``input_ids``, mean-pools over the non-padded positions, and
    classifies the pooled vector over the vocabulary (so the predicted class IS
    a token id, which decodes back to ``"+1"``/``"-1"`` for free);
  - the classification target is the first non-ignored token of ``labels``
    (for a single-token answer that is exactly the answer token);
  - :meth:`generate` returns ``[BOS, predicted_token, EOS]`` so the existing
    exact-match generation evaluation works unchanged.

``config.is_classification = True`` signals the trainer to score it with
classification accuracy (see ``calt.trainer.trainer.Trainer._compute_metrics``).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from ..input_embeddings import get_input_embedding
from ..positional_embeddings import get_positional_embedding


class EncoderClassifierConfig(PretrainedConfig):
    """Configuration for the encoder-only classification model."""

    model_type = "encoder_classifier"

    def __init__(
        self,
        d_model: int = 512,
        attention_heads: int = 8,
        num_encoder_layers: int = 6,
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
        input_embedding_type: str = "token",
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
        self.input_embedding_type = input_embedding_type
        self.init_std = init_std
        self.seed = seed
        # Read by the trainer to switch to classification-style metrics.
        self.is_classification = True


class EncoderClassifier(PreTrainedModel):
    """Encoder-only Transformer with a single-token classification head."""

    config_class = EncoderClassifierConfig

    def __init__(self, config: EncoderClassifierConfig):
        super().__init__(config)
        self.config = config

        self.embedding = get_input_embedding(
            config.input_embedding_type, config.vocab_size, config.d_model
        )
        self.positional_embedding = get_positional_embedding(
            pe_type=config.use_positional_embedding,
            d_model=config.d_model,
            max_len=config.max_input_len * 2,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=config.batch_first,
            norm_first=config.norm_first,
            bias=config.bias,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        # Classify the pooled representation over the vocabulary so the predicted
        # class is a token id (decodes straight back to the answer string).
        self.classifier = nn.Linear(config.d_model, config.vocab_size, bias=True)

        self.apply(self._init_weights)
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

    def _key_padding_mask(self, attention_mask):
        """Boolean mask where True marks a padded position (PyTorch convention)."""
        if attention_mask is None:
            return None
        if attention_mask.dtype == torch.bool:
            mask = ~attention_mask
        else:
            mask = attention_mask == 0
        return mask.to(dtype=torch.bool, device=attention_mask.device)

    def _encode_and_pool(self, input_ids, attention_mask):
        """Encode ``input_ids`` and mean-pool over the non-padded positions."""
        embeddings = self.embedding(input_ids)
        if self.positional_embedding is not None:
            embeddings = self.positional_embedding(embeddings)

        key_padding_mask = self._key_padding_mask(attention_mask)
        encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)

        if attention_mask is None:
            return encoded.mean(dim=1)
        # Masked mean: average only the valid (non-padded) positions.
        weights = attention_mask.to(encoded.dtype).unsqueeze(-1)
        summed = (encoded * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return summed / counts

    @staticmethod
    def _target_classes(labels, ignore_index=-100):
        """First non-ignored label token per row = the (single) answer token.

        Vectorized (runs every forward): argmax over the boolean "is valid" mask
        returns the first True index per row; rows with no valid token keep
        ``ignore_index`` so CrossEntropyLoss skips them.
        """
        valid = labels != ignore_index
        first_idx = valid.float().argmax(dim=1, keepdim=True)  # first valid position
        target = labels.gather(1, first_idx).squeeze(1)
        return torch.where(
            valid.any(dim=1), target, torch.full_like(target, ignore_index)
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # decoder_input_ids etc. are accepted and ignored
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        pooled = self._encode_and_pool(input_ids, attention_mask)
        logits = self.classifier(pooled)  # (batch, vocab_size)

        loss = None
        if labels is not None:
            target = self._target_classes(labels)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits, target)

        # Return the predicted token id (shape (batch, 1)) so the metric step is
        # light and unambiguous; the trainer scores it as classification.
        predictions = logits.argmax(dim=-1, keepdim=True)

        if not return_dict:
            return (loss, predictions) if loss is not None else (predictions,)

        # Reuse Seq2SeqLMOutput for consistency with the generic encoder-decoder
        # model (which also returns argmax'd token ids in `logits`): the Trainer
        # and evaluate_and_save_generation only read `.loss` and `.logits`, so a
        # classification-specific output type would buy nothing and risk breaking
        # that shared consumption path.
        return Seq2SeqLMOutput(loss=loss, logits=predictions)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Return ``[BOS, predicted_token, EOS]`` so seq2seq decoding works as-is."""
        if bos_token_id is None:
            bos_token_id = self.config.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        pooled = self._encode_and_pool(input_ids, attention_mask)
        pred = self.classifier(pooled).argmax(dim=-1, keepdim=True)  # (batch, 1)

        batch_size = input_ids.size(0)
        device = input_ids.device
        bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        eos = torch.full((batch_size, 1), eos_token_id, dtype=torch.long, device=device)
        return torch.cat([bos, pred, eos], dim=1)
