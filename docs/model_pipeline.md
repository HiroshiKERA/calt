# Model pipeline

`ModelPipeline` builds a sequence-to-sequence model from the `model` block of your config and the tokenizer produced by [IOPipeline](io_pipeline.md). It is used after `IOPipeline.build()` and before [TrainerPipeline](trainer.md).

- [Overview](trainer.md) — how the three pipelines (IO, Model, Trainer) fit together.
- [Configuration](configuration.md) — the `model` block in `train.yaml` and its keys.

## ModelPipeline

Use `ModelPipeline.from_io_dict(cfg.model, io_dict)` to create a pipeline from the result of `IOPipeline.from_config(cfg.data).build()`. The tokenizer is taken from `io_dict["tokenizer"]`. Call `.build()` to obtain the `PreTrainedModel` instance.

::: calt.models.pipeline.ModelPipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

## Supported model types

Models are created via an internal `ModelRegistry`. The following types are registered by default:

| `model_type` | Description |
|--------------|-------------|
| `generic`, `transformer`, `calt` | CALT generic Transformer (encoder–decoder). |
| `bart` | HuggingFace BART for conditional generation. |
| `encoder_classifier`, `encoder_only` | Encoder-only single-token classification model (see below). |

Set `model_type` in the `model` block of `train.yaml` (e.g. `model_type: generic`). Other keys in the `model` block (e.g. `num_encoder_layers`, `d_model`, `max_sequence_length`) are documented under [Configuration — `model`](configuration.md#trainyaml--model-and-training-modelpipeline-trainerpipeline).

## Encoder-only classification model

For tasks whose answer is a **single token** (e.g. permutation parity, whose target is `+1` / `-1`), a full encoder–decoder is unnecessary. `model_type: encoder_classifier` (alias `encoder_only`) is an encoder-only alternative that encodes the input, mean-pools the encoder output, and classifies it over the vocabulary.

```yaml
model:
  model_type: encoder_classifier   # alias: encoder_only
  num_encoder_layers: 3
  num_encoder_heads: 4
  d_model: 256
  encoder_ffn_dim: 1024
  max_sequence_length: 256
  # decoder_* keys are ignored for this model_type
```

It is a drop-in over the existing seq2seq data path (no IOPipeline / collator change): it consumes the standard `input_ids` / `attention_mask` / `labels` batch and derives the classification target from the first non-ignored token of `labels`. `generate()` returns `[BOS, token, EOS]`, so the exact-match generation evaluation works unchanged. Because the model sets `config.is_classification = True`, the trainer reports `token_accuracy` and `success_rate` as plain classification accuracy. Use it only for fixed single-token answers — variable-length outputs (e.g. Gröbner/border bases) need the encoder–decoder.

## Custom embeddings (input and positional)

Both the **input (token) embedding** and the **positional embedding** are chosen by config and extensible at runtime, symmetric to the model registry. Built-ins keep the previous behavior, so existing configs are unaffected.

| config key | default | built-in values |
|------------|---------|-----------------|
| `model.input_embedding_type` | `token` | `token` (aliases `default`, `learned`) — a plain `nn.Embedding` |
| `model.use_positional_embedding` | `generic` | `generic`/`learned`, `sinusoidal`, `rope`, `none` |

To plug in your own, register a factory **before building the model** (e.g. at the top of your train script), then select it by name in the config:

```python
import torch.nn as nn
from calt.models import register_input_embedding, register_positional_embedding

class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
    def forward(self, input_ids):          # (B, S) long -> (B, S, d_model)
        return self.emb(input_ids)

register_input_embedding(
    "my_emb", lambda vocab_size, d_model, **kw: MyEmbedding(vocab_size, d_model))
register_positional_embedding(
    "my_pe", lambda d_model, max_len, **kw: MyPositional(d_model, max_len))
```

```yaml
model:
  input_embedding_type: my_emb
  use_positional_embedding: my_pe
```

**Factory contract.** An input-embedding factory receives at least `vocab_size` and `d_model` (extra config keys are forwarded as kwargs) and returns an `nn.Module` mapping `input_ids` of shape `(batch, seq)` to `(batch, seq, d_model)`. A positional-embedding factory receives at least `d_model` and `max_len` and returns an `nn.Module` mapping `(batch, seq, d_model)` to `(batch, seq, d_model)` (or `None` for "no positional embedding"). An unknown type raises `ValueError` listing the supported names.

These hooks apply to the `generic` and `encoder_classifier` models (not `bart`, which is HuggingFace's own model). The public helpers are exported from `calt.models`: `register_input_embedding`, `register_positional_embedding`, `get_input_embedding`, `get_positional_embedding`.

## ModelRegistry

To create a model without using the pipeline (e.g. with a custom config), you can use the registry or helpers from `calt.models`: `ModelRegistry`, `get_model_from_config`. See the API reference below.

::: calt.models.base.ModelRegistry
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - __init__
        - create_from_config
        - list_models
        - register
        - register_config_mapping
