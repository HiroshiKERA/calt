"""
Loader for BART model.

Handles conversion from unified config format (cfg.model) to BartConfig
and creates BART model instances.
"""

from typing import Optional

from omegaconf import DictConfig
from transformers import BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast

from ..loader import ModelLoader

# Default BartConfig for reference
_DEFAULT_BART_CONFIG = BartConfig()


class BartLoader(ModelLoader):
    """Loader for BART model.
    
    Converts cfg.model to BartConfig and creates BART instances.
    """
    
    def translate_config(self) -> BartConfig:
        """Convert unified config format to BartConfig.
        
        Returns:
            BartConfig: BART model configuration.
        
        Raises:
            ValueError: If tokenizer is not provided (required for BART).
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for BART model")
        
        self.config = BartConfig(
            encoder_layers=self.calt_config.num_encoder_layers,
            encoder_attention_heads=self.calt_config.num_encoder_heads,
            decoder_layers=self.calt_config.num_decoder_layers,
            decoder_attention_heads=self.calt_config.num_decoder_heads,
            vocab_size=len(self.tokenizer.vocab),
            d_model=self.calt_config.d_model,
            encoder_ffn_dim=self.calt_config.encoder_ffn_dim,
            decoder_ffn_dim=self.calt_config.decoder_ffn_dim,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
            unk_token_id=self.tokenizer.unk_token_id,
            max_position_embeddings=self.calt_config.max_sequence_length,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            # Optional parameters with defaults from _DEFAULT_BART_CONFIG
            dropout=getattr(self.calt_config, 'dropout', _DEFAULT_BART_CONFIG.dropout),
            activation_function=getattr(self.calt_config, 'activation', _DEFAULT_BART_CONFIG.activation_function),
            attention_dropout=getattr(self.calt_config, 'attention_dropout', _DEFAULT_BART_CONFIG.attention_dropout),
            activation_dropout=getattr(self.calt_config, 'activation_dropout', _DEFAULT_BART_CONFIG.activation_dropout),
            init_std=getattr(self.calt_config, 'init_std', _DEFAULT_BART_CONFIG.init_std),
            encoder_layerdrop=getattr(self.calt_config, 'encoder_layerdrop', _DEFAULT_BART_CONFIG.encoder_layerdrop),
            decoder_layerdrop=getattr(self.calt_config, 'decoder_layerdrop', _DEFAULT_BART_CONFIG.decoder_layerdrop),
            classifier_dropout=getattr(self.calt_config, 'classifier_dropout', _DEFAULT_BART_CONFIG.classifier_dropout),
        )
        return self.config
    
    def build_model(self) -> BartForConditionalGeneration:
        """Create BART model instance.
        
        Returns:
            BartForConditionalGeneration: BART model instance.
        """
        if self.config is None:
            raise ValueError("config must be set before building model. Call translate_config() first.")
        
        return BartForConditionalGeneration(config=self.config)
