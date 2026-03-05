"""
Config mapping for GPT-2 model.

Converts from unified config format (cfg.model) to GPT2Config.
"""

from omegaconf import DictConfig
from transformers import GPT2Config, PreTrainedTokenizerFast


def create_gpt2_config(
    model_config: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> GPT2Config:
    """Create GPT2Config from unified config format.

    Args:
        model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
        tokenizer (PreTrainedTokenizerFast): Tokenizer instance for vocab and token IDs.

    Returns:
        GPT2Config: GPT-2 model configuration.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for GPT-2 model")

    n_positions = model_config.max_sequence_length * 2
    return GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=model_config.d_model,
        n_layer=model_config.num_decoder_layers,
        n_head=model_config.num_decoder_heads,
        n_inner=model_config.decoder_ffn_dim,
        resid_pdrop=getattr(model_config, "dropout", 0.1),
        embd_pdrop=getattr(model_config, "dropout", 0.1),
        attn_pdrop=getattr(model_config, "attention_dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
