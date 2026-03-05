"""
GPT-2 wrapper for prompted sequence generation.
"""

from typing import Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2ForPromptedGeneration(PreTrainedModel):
    """GPT-2 model that learns only target tokens after a fixed prompt."""

    config_class = GPT2Config

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.model = GPT2LMHeadModel(config)

    def _concat_prompt_target(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        target_input_ids: torch.LongTensor,
        target_attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if target_attention_mask is None:
            target_attention_mask = torch.ones_like(target_input_ids)
        full_input_ids = torch.cat([input_ids, target_input_ids], dim=1)
        full_attention_mask = torch.cat([attention_mask, target_attention_mask], dim=1)
        return full_input_ids, full_attention_mask

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
        kwargs.pop("num_items_in_batch", None)

        if input_ids is None:
            raise ValueError("input_ids is required for GPT-2 prompted generation")

        target_input_ids = None
        target_attention_mask = None
        if labels is not None:
            # Build target-side teacher-forcing inputs directly from labels.
            # This keeps autoregressive contexts consistent with free generation:
            # prompt -> t1 -> t2 -> ... instead of prompt -> BOS -> t1 -> ...
            target_attention_mask = (labels != -100).long()
            target_input_ids = labels.masked_fill(
                labels == -100, self.config.pad_token_id
            )
        elif decoder_input_ids is not None:
            target_input_ids = decoder_input_ids
            target_attention_mask = decoder_attention_mask

        full_input_ids = input_ids
        full_attention_mask = attention_mask
        if target_input_ids is not None:
            full_input_ids, full_attention_mask = self._concat_prompt_target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_input_ids=target_input_ids,
                target_attention_mask=target_attention_mask,
            )

        full_labels = None
        target_len = (
            labels.size(1)
            if labels is not None
            else (target_input_ids.size(1) if target_input_ids is not None else 0)
        )
        if labels is not None:
            prompt_mask_labels = torch.full_like(input_ids, -100)
            full_labels = torch.cat([prompt_mask_labels, labels], dim=1)

        output = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True if return_dict is None else return_dict,
            **kwargs,
        )

        trimmed_logits = output.logits
        if target_len > 0:
            # GPT2LMHeadModel computes loss with an internal left-shift:
            # shift_logits[..., t] is compared with labels[..., t+1].
            # For target-only labels, metric logits should start from the
            # last prompt position and span `target_len`.
            prompt_len = input_ids.size(1)
            start = max(prompt_len - 1, 0)
            end = min(start + target_len, output.logits.size(1))
            trimmed_logits = output.logits[:, start:end, :]
        if return_dict is False:
            result = (trimmed_logits,) + output[2:]
            if output.loss is not None:
                return (output.loss,) + result
            return result

        return CausalLMOutputWithCrossAttentions(
            loss=output.loss,
            logits=trimmed_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            cross_attentions=output.cross_attentions,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **kwargs,
    ) -> torch.LongTensor:
        kwargs.pop("max_new_tokens", None)
        full_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            **kwargs,
        )
        prompt_len = input_ids.shape[1]
        return full_ids[:, prompt_len:]
