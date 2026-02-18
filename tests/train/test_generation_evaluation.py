"""Tests for generation evaluation and stop behavior."""

import torch
import torch.nn as nn

from calt.models.generic.model import Transformer, TransformerConfig


class PlannedLMHead(nn.Module):
    """LM head that emits predefined next-token plans per decoding step."""

    def __init__(self, token_plan: list[list[int]], vocab_size: int):
        super().__init__()
        self.token_plan = token_plan
        self.vocab_size = vocab_size
        self.step = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        logits = torch.full(
            (batch_size, self.vocab_size),
            -1e9,
            device=hidden_states.device,
        )
        plan_idx = min(self.step, len(self.token_plan) - 1)
        step_plan = self.token_plan[plan_idx]
        for idx, token_id in enumerate(step_plan):
            logits[idx, token_id] = 0.0
        self.step += 1
        return logits


def test_generate_stops_per_sequence_after_eos():
    """Once a sequence emits EOS, subsequent tokens must be PAD for that row."""
    config = TransformerConfig(
        d_model=8,
        attention_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        vocab_size=20,
        max_input_len=16,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        use_positional_embedding="generic",
    )
    model = Transformer(config)
    model.eval()
    model.lm_head = PlannedLMHead(
        token_plan=[
            [1, 5],  # step 1: row0 finishes, row1 continues
            [7, 1],  # step 2: row0 would emit 7 without masking, row1 finishes
        ],
        vocab_size=config.vocab_size,
    )

    input_ids = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=4,
    )

    assert generated.tolist() == [
        [config.bos_token_id, config.eos_token_id, config.pad_token_id],
        [config.bos_token_id, 5, config.eos_token_id],
    ]

