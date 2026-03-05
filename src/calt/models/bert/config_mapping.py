"""
Config mapping for BERT model.

Converts from unified config format (cfg.model) to BertConfig.
"""

from omegaconf import DictConfig
from transformers import BertConfig, PreTrainedTokenizerFast


def create_bert_config(
    model_config: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> BertConfig:
    """Create BertConfig from unified config format."""
    if tokenizer is None:
        raise ValueError("tokenizer is required for BERT model")

    return BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=model_config.d_model,
        num_hidden_layers=model_config.num_encoder_layers,
        num_attention_heads=model_config.num_encoder_heads,
        intermediate_size=model_config.encoder_ffn_dim,
        hidden_dropout_prob=getattr(model_config, "dropout", 0.1),
        attention_probs_dropout_prob=getattr(model_config, "attention_dropout", 0.1),
        max_position_embeddings=model_config.max_sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_labels=len(tokenizer.vocab),
    )
