"""
BERT wrapper for single-token classification.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForSingleTokenClassification(PreTrainedModel):
    """BERT classifier that predicts one token ID as class label."""

    config_class = BertConfig

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def _extract_class_labels(self, labels: torch.LongTensor) -> torch.LongTensor:
        if labels.dim() == 1:
            return labels
        valid_mask = labels != -100
        has_valid = valid_mask.any(dim=1)
        first_indices = valid_mask.float().argmax(dim=1)
        batch_indices = torch.arange(labels.size(0), device=labels.device)
        class_labels = labels[batch_indices, first_indices]
        class_labels = torch.where(has_valid, class_labels, torch.zeros_like(class_labels))
        return class_labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # Shared collator provides seq2seq keys; encoder-only BERT ignores them.
        kwargs.pop("decoder_input_ids", None)
        kwargs.pop("decoder_attention_mask", None)
        kwargs.pop("num_items_in_batch", None)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True if return_dict is None else return_dict,
            **kwargs,
        )

        if return_dict is False:
            pooled_output = outputs[1]
            logits = self.classifier(self.dropout(pooled_output))
            loss = None
            if labels is not None:
                class_labels = self._extract_class_labels(labels)
                loss = nn.CrossEntropyLoss()(logits, class_labels)
            result = (logits,) + outputs[2:]
            if loss is not None:
                return (loss,) + result
            return result

        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            class_labels = self._extract_class_labels(labels)
            loss = nn.CrossEntropyLoss()(logits, class_labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
