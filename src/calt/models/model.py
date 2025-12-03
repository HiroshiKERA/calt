"""
A PyTorch Transformer model usable with the Hugging Face Trainer.
Extends the existing Transformer class from transformer.py.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CaltModelConfig(PretrainedConfig):
    """Configuration for the CALT Transformer model.

    Attributes:
        model_type (str): Identifier used by Hugging Face to register the model.
    """

    model_type = "transformer"

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
        use_positional_embedding: str = "learned",
        init_std: float = 0.02,
        tie_word_embeddings: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        """Configure layer sizes, embeddings, and tokenizer metadata.

        Args:
            d_model (int, optional): Hidden size used across embeddings and blocks.
            attention_heads (int, optional): Number of attention heads per block.
            num_encoder_layers (int, optional): Encoder block count.
            num_decoder_layers (int, optional): Decoder block count.
            dim_feedforward (int, optional): Width of the feed-forward layers.
            dropout (float, optional): Dropout probability applied in transformer.
            activation (str, optional): Activation used inside feed-forward blocks.
            layer_norm_eps (float, optional): Epsilon for layer normalization.
            batch_first (bool, optional): Whether tensors follow (batch, seq, dim).
            norm_first (bool, optional): Whether to apply layer norm before attention.
            bias (bool, optional): If linear layers include a bias term.
            vocab_size (int, optional): Vocabulary size for embeddings and head.
            max_input_len (int, optional): Maximum supported sequence length.
            pad_token_id (int, optional): Padding token id.
            eos_token_id (int, optional): End-of-sequence token id.
            bos_token_id (int, optional): Beginning-of-sequence token id.
            use_positional_embedding (str, optional): Positional embedding strategy.
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
        self.init_std = init_std
        self.tie_word_embeddings = tie_word_embeddings
        self.seed = seed


class CaltModel(PreTrainedModel):
    """Transformer model compatible with the Hugging Face Trainer."""

    config_class = CaltModelConfig

    def __init__(self, config: CaltModelConfig):
        """Build embedding, transformer stack, and language modeling head.

        Args:
            config (CaltModelConfig): Model hyper-parameters and tokenizer metadata.
        """
        super().__init__(config)

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        if config.use_positional_embedding == "learned":
            self.positional_embedding = nn.Embedding(
                config.max_input_len, config.d_model
            )
        elif config.use_positional_embedding == "none":
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Unsupported positional embedding type: {config.use_positional_embedding}"
            )

        self.transformer = nn.Transformer(
            d_model=config.d_model,
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

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.tie_weights()
        self.seed = config.seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _init_weights(self, module):
        """Apply CALT-specific initialization to supported modules.

        Args:
            module (nn.Module): Layer receiving the initialization routine.
        """
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
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            embeddings = embeddings + self.positional_embedding(position_ids)

        return embeddings

    def _prepare_key_padding_mask(
        self, mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Convert an attention mask to the boolean form expected by PyTorch.

        Args:
            mask (torch.Tensor | None): Mask originating from the tokenizer.

        Returns:
            torch.Tensor | None: Boolean mask where True means the position is padded.
        """
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            key_padding_mask = ~mask
        else:
            key_padding_mask = mask == 0
        return key_padding_mask.to(dtype=torch.bool, device=mask.device)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create an upper-triangular, causal decoder mask.

        Args:
            size (int): Target sequence length.
            device (torch.device): Device on which to allocate the mask.

        Returns:
            torch.Tensor: Boolean mask preventing attending to future tokens.
        """
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the encoder-decoder forward pass.

        Args:
            input_ids (torch.LongTensor | None): Encoder token ids.
            attention_mask (torch.Tensor | None): Encoder attention mask.
            decoder_input_ids (torch.LongTensor | None): Decoder inputs.
            decoder_attention_mask (torch.Tensor | None): Decoder attention mask.
            labels (torch.LongTensor | None): Target labels for loss computation.
            output_attentions (bool | None): Flag to surface attention tensors.
            output_hidden_states (bool | None): Flag to surface hidden states.
            return_dict (bool | None): Whether to return a dataclass output.
            **kwargs: Extra arguments kept for HF Trainer compatibility.

        Returns:
            Seq2SeqLMOutput | tuple: Loss and logits packaged per HF conventions.

        Raises:
            ValueError: If neither encoder nor decoder inputs are provided.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None and decoder_input_ids is None:
            raise ValueError("Either input_ids or decoder_input_ids must be provided")

        if input_ids is not None:
            encoder_embeddings = self._compute_embeddings(input_ids)
        else:
            encoder_embeddings = None

        if decoder_input_ids is not None:
            decoder_embeddings = self._compute_embeddings(decoder_input_ids)
        else:
            decoder_embeddings = None

        tgt_mask = None
        if decoder_embeddings is not None:
            tgt_seq_len = decoder_embeddings.size(1)
            tgt_mask = self._generate_causal_mask(
                tgt_seq_len, decoder_embeddings.device
            )

        encoder_key_padding_mask = self._prepare_key_padding_mask(attention_mask)
        decoder_key_padding_mask = self._prepare_key_padding_mask(
            decoder_attention_mask
        )
        if encoder_embeddings is None or decoder_embeddings is None:
            raise ValueError(
                "Both encoder_embeddings and decoder_embeddings must be provided"
            )

        transformer_output = self.transformer(
            src=encoder_embeddings,
            tgt=decoder_embeddings,
            src_key_padding_mask=encoder_key_padding_mask,
            tgt_key_padding_mask=decoder_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.lm_head(transformer_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            if loss is not None:
                output = (loss,) + transformer_output
            return output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=torch.argmax(logits, dim=-1),
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
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
        """Autoregressively generate tokens from encoder context.

        Args:
            input_ids (torch.LongTensor): Encoder inputs.
            attention_mask (torch.Tensor | None): Mask aligned with `input_ids`.
            max_length (int, optional): Maximum generated length.
            num_beams (int, optional): Reserved for HF parity (unused).
            temperature (float, optional): Sampling temperature.
            do_sample (bool, optional): Whether to sample instead of argmax.
            pad_token_id (int | None): Override for padding id.
            eos_token_id (int | None): Override for end-of-sequence id.
            **kwargs: Extra keyword arguments for API compatibility.

        Returns:
            torch.LongTensor: Completed decoder sequences including BOS/EOS.
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        encoder_embeddings = self._compute_embeddings(input_ids)

        encoder_key_padding_mask = self._prepare_key_padding_mask(attention_mask)

        if encoder_key_padding_mask is not None:
            encoder_output = self.transformer.encoder(
                encoder_embeddings,
                src_key_padding_mask=encoder_key_padding_mask,
            )
        else:
            encoder_output = self.transformer.encoder(encoder_embeddings)

        decoder_input_ids = torch.full(
            (batch_size, 1), self.config.bos_token_id, device=device
        )

        for _ in range(max_length):
            decoder_embeddings = self._compute_embeddings(decoder_input_ids)

            tgt_mask = self._generate_causal_mask(decoder_embeddings.size(1), device)
            decoder_output = self.transformer.decoder(
                decoder_embeddings,
                memory=encoder_output,
                tgt_mask=tgt_mask,
            )

            next_token_logits = self.lm_head(decoder_output[:, -1, :])

            if do_sample:
                next_token_logits = next_token_logits / temperature
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 1
                )
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return decoder_input_ids
