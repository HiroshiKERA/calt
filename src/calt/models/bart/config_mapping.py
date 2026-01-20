"""
Config mapping for BART model.

Converts from unified config format (cfg.model) to BartConfig.
"""

from omegaconf import DictConfig
from transformers import BartConfig, PreTrainedTokenizerFast


def create_bart_config(
    model_config: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> BartConfig:
    """Create BartConfig from unified config format.
    
    Args:
        model_config (DictConfig): Model configuration from cfg.model (OmegaConf).
        tokenizer (PreTrainedTokenizerFast): Tokenizer instance for vocab and token IDs.
    
    Returns:
        BartConfig: BART model configuration.
    """
    return BartConfig(
        encoder_layers=model_config.num_encoder_layers,
        encoder_attention_heads=model_config.num_encoder_heads,
        decoder_layers=model_config.num_decoder_layers,
        decoder_attention_heads=model_config.num_decoder_heads,
        vocab_size=len(tokenizer.vocab),
        d_model=model_config.d_model,
        encoder_ffn_dim=model_config.encoder_ffn_dim,
        decoder_ffn_dim=model_config.decoder_ffn_dim,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        unk_token_id=tokenizer.unk_token_id,
        max_position_embeddings=model_config.max_sequence_length,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
