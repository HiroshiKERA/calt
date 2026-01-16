from omegaconf import OmegaConf
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration as BartModel

class Abstracta
ModelPipeline:
    def __init__(self, model: nn.Module, config: OmegaConf):
        self.model = model
        self.config = config
    
    def set_config(self, config: OmegaConf):
        self.config = config
        return self
    
    def build(self):
        self.model = self.model.to(self.config.device)
        return self
    

class BartModelPipeline(AbstractModelPipeline):
    def __init__(self, model: nn.Module, config: OmegaConf):
        super().__init__(model, config)
        
        self.config = self.set_config(config)
        
    def build(self):
        self.model = BartforModel(self.translate_config(self.config))
        return self
    
    def translate_config(self, config: OmegaConf):
        model_cfg = BartConfig(
                encoder_layers=config.model.num_encoder_layers,
                encoder_attention_heads=config.model.num_encoder_heads,
                decoder_layers=config.model.num_decoder_layers,
                decoder_attention_heads=config.model.num_decoder_heads,
                vocab_size=len(tokenizer.vocab),
                d_model=config.model.d_model,
                encoder_ffn_dim=config.model.encoder_ffn_dim,
                decoder_ffn_dim=config.model.decoder_ffn_dim,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                cls_token_id=tokenizer.cls_token_id,
                sep_token_id=tokenizer.sep_token_id,
                unk_token_id=tokenizer.unk_token_id,
                max_position_embeddings=config.model.max_sequence_length,
                decoder_start_token_id=tokenizer.bos_token_id,
            )