"""
Hugging Face Trainerで使用できるPyTorch Transformerモデル
既存のtransformer.pyのTransformerクラスを継承・拡張
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import argmax
from typing import Optional, Dict, Any, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.utils import logging



import torch.nn as nn

logger = logging.get_logger(__name__)


class CaltModelConfig(PretrainedConfig):
    """Transformerモデルの設定クラス"""
    
    model_type = "transformer"
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
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
        use_positional_embedding: bool = True,
        init_std: float = 0.02,
        tie_word_embeddings: bool = False,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )
        
        self.d_model = d_model
        self.nhead = nhead
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
    """Hugging FaceのTrainerで使用できるTransformerモデル"""
    
    config_class = CaltModelConfig
    
    def __init__(self, config: CaltModelConfig):
        super().__init__(config)
        
        self.config = config
        
        # 埋め込み層
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置エンべディング（設定に応じて有無を切り替え）
        if config.use_positional_embedding == "learned":
            self.positional_embedding = nn.Embedding(config.max_input_len, config.d_model)
        elif config.use_positional_embedding == "none":
            self.positional_embedding = None
        else:
            raise ValueError(f"Unsupported positional embedding type: {config.use_positional_embedding}")
        
        # Transformerモデル（PyTorchの標準的なTransformerを使用）
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=config.batch_first,
            norm_first=config.norm_first,
            bias=config.bias
        )
        
        # 出力層
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 重みの初期化
        self.apply(self._init_weights)
        self.tie_weights()
       self.seed = config.seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
    def _init_weights(self, module):
        """重みの初期化メソッド"""
        if isinstance(module, nn.Linear):
            # Linear層の重みを正規分布で初期化
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Embedding層の重みを正規分布で初期化
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            # LayerNormの重みを1、バイアスを0で初期化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _prepare_key_padding_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            key_padding_mask = ~mask
        else:
            key_padding_mask = mask == 0
        return key_padding_mask.to(dtype=torch.bool, device=mask.device)
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for the transformer model.
        
        Args:
            input_ids: エンコーダーへの入力トークンID
            attention_mask: エンコーダーのアテンションマスク
            decoder_input_ids: デコーダーへの入力トークンID
            decoder_attention_mask: デコーダーのアテンションマスク
            labels: 教師ラベル（損失計算用）
            output_attentions: アテンション重みを出力するか
            output_hidden_states: 隠れ状態を出力するか
            return_dict: 辞書形式で返すか
        
        Returns:
            モデルの出力（損失、ロジット、隠れ状態など）
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 入力の検証
        if input_ids is None and decoder_input_ids is None:
            raise ValueError("input_ids または decoder_input_ids のいずれかが必要です")
        
        # エンコーダー入力の処理
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            encoder_embeddings = self.embedding(input_ids)
            
            # 位置エンべディングの追加（設定に応じて）
            if self.positional_embedding is not None:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                encoder_embeddings = encoder_embeddings + self.positional_embedding(position_ids)
        else:
            encoder_embeddings = None
        
        # デコーダー入力の処理
        if decoder_input_ids is not None:
            batch_size, seq_len = decoder_input_ids.shape
            decoder_embeddings = self.embedding(decoder_input_ids)

            # 位置エンべディングの追加（設定に応じて）
            if self.positional_embedding is not None:
                position_ids = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0).expand(batch_size, -1)
                decoder_embeddings = decoder_embeddings + self.positional_embedding(position_ids)
        else:
            decoder_embeddings = None
        
        tgt_mask = None
        if decoder_embeddings is not None:
            tgt_seq_len = decoder_embeddings.size(1)
            tgt_mask = self._generate_causal_mask(tgt_seq_len, decoder_embeddings.device)
        
        # Transformerのforward pass
        encoder_key_padding_mask = self._prepare_key_padding_mask(attention_mask)
        decoder_key_padding_mask = self._prepare_key_padding_mask(decoder_attention_mask)
        if encoder_embeddings is not None and decoder_embeddings is not None:
            # エンコーダー-デコーダー形式
            transformer_output = self.transformer(
                src=encoder_embeddings,
                tgt=decoder_embeddings,
                src_key_padding_mask=encoder_key_padding_mask,
                tgt_key_padding_mask=decoder_key_padding_mask,
                tgt_mask=tgt_mask,
            )
        else:
            print("encoder_embeddings or decoder_embeddings is None")
        
        # 出力の処理
        logits = self.lm_head(transformer_output)
        # 損失の計算
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()    
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        if not return_dict:
            if loss is not None:
                output = (loss,) + output
            return output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=argmax(logits, dim=-1),
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
        **kwargs
    ) -> torch.LongTensor:
        """
        テキスト生成のためのメソッド
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # エンコーダー出力を計算
        encoder_embeddings = self.embedding(input_ids)
        
        # 位置エンべディングの追加（設定に応じて）
        if self.positional_embedding is not None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            encoder_embeddings = encoder_embeddings + self.positional_embedding(position_ids)
        
        encoder_key_padding_mask = self._prepare_key_padding_mask(attention_mask)

        if encoder_key_padding_mask is not None:
            encoder_output = self.transformer.encoder(
                encoder_embeddings,
                src_key_padding_mask=encoder_key_padding_mask,
            )
        else:
            encoder_output = self.transformer.encoder(encoder_embeddings)
        
        # デコーダー入力の初期化
        decoder_input_ids = torch.full((batch_size, 1), self.config.bos_token_id, device=device)
        
        for _ in range(max_length):
            # デコーダー埋め込み
            decoder_embeddings = self.embedding(decoder_input_ids)
            
            # 位置エンべディングの追加（設定に応じて）
            if self.positional_embedding is not None:
                seq_len = decoder_input_ids.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                decoder_embeddings = decoder_embeddings + self.positional_embedding(position_ids)
            
            # デコーダーのforward pass
            tgt_mask = self._generate_causal_mask(decoder_embeddings.size(1), device)
            decoder_output = self.transformer.decoder(
                decoder_embeddings,
                memory=encoder_output,
                tgt_mask=tgt_mask,
            )
            
            # 次のトークンの予測
            next_token_logits = self.lm_head(decoder_output[:, -1, :])
            
            if do_sample:
                next_token_logits = next_token_logits / temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # デコーダー入力に追加
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # EOSトークンが生成されたら停止
            if (next_token == eos_token_id).all():
                break
        
        return decoder_input_ids


# モデルの登録
from transformers import AutoConfig, AutoModel

# 設定の登録
AutoConfig.register("calt_model", CaltModelConfig)

# モデルの登録
AutoModel.register(CaltModelConfig, CaltModel)
