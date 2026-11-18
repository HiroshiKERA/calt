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
| `monomial`, `monomial_transformer` | Encoder–decoder with monomial-structured embedding for C/E expanded-form data (see below). |
| `decoder_only`, `decoder` | Decoder-only causal Transformer over `[problem, solution]` (see below). |
| `monomial_decoder_only`, `monomial_decoder` | Decoder-only causal Transformer over monomial positions (see below). |

Set `model_type` in the `model` block of `train.yaml` (e.g. `model_type: generic`). Other keys in the `model` block (e.g. `num_encoder_layers`, `d_model`, `max_sequence_length`) are documented under [Configuration — `model`](configuration.md#trainyaml-model-and-training-modelpipeline-trainerpipeline).

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

## Monomial-embedding model

For polynomial data in **C/E expanded form** (`C<coeff> E<e1> .. E<en>` per term, terms joined by `+`, polynomials by `||` — see [Load preprocessors](io_load_preprocessors.md), `ExpandedFormLoadPreprocessor`), each monomial occupies a fixed number of tokens. `model_type: monomial` (alias `monomial_transformer`) exploits that structure, following the monomial embedding of the border-basis Transformer work of Kera et al. (arXiv 2505.23696):

- **Input:** the coefficient token, the `n_vars` exponent tokens, and the following separator are embedded together as **one sequence position** (mean of the slot embeddings, with a configurable `coeff_scale` on the coefficient part). Sequences become `n_vars + 2` times shorter than in the flat model.
- **Output:** instead of one softmax over the whole vocabulary, the decoder predicts each part of the next monomial with **separate heads** — one per coefficient field, one per variable, and one "follow" head choosing between `+`, `||`, and end-of-sequence.

```yaml
model:
  model_type: monomial       # alias: monomial_transformer
  num_variables: 2           # REQUIRED: exponent slots per monomial
  d_model: 256
  num_encoder_layers: 3
  num_decoder_layers: 3
  num_encoder_heads: 4
  encoder_ffn_dim: 1024
  max_sequence_length: 512
  # optional monomial-specific knobs:
  # coeff_scale: 1.0         # weight of the coefficient part of the embedding
  # coeff_noise_std: 0.0     # train-time Gaussian noise on the coefficient part
  # coeff_loss_weight: 1.0   # weight of the coefficient term in the loss
  # monomial_separators: ["+", "||"]   # defaults to those present in the vocab
```

It is a drop-in over the existing seq2seq data path (no IOPipeline / collator change): `forward()` folds the standard flat `input_ids` / `labels` batch into a `(batch, monomials, width)` grid internally, and `generate()` returns flat token ids (`[BOS, C.., E.., +, ..., EOS]`), so decoding, metrics, and the exact-match generation evaluation work unchanged. The coefficient/exponent/separator token groups are derived from the tokenizer's vocabulary (`C<int>` — or `C<int>_<label>` for multi-field vocabularies — and `E<int>` tokens); `num_variables` is required because shared `E<k>` tokens carry no variable identity. If the data is not monomial-aligned (wrong `num_variables`, or data not in expanded form), the model raises a `ValueError` up front instead of training on a corrupted view.

## Decoder-only model

The arithmetic-reasoning literature generally trains decoder-only models rather than encoder–decoders. `model_type: decoder_only` (alias `decoder`) is the generic model with the encoder removed: a single causal self-attention stack running over the concatenation of the problem and its solution.

The one subtlety is where prediction starts. The problem is written into the sequence as context and carries **no loss** — next-token prediction begins at the first solution token, since a model asked to reproduce the problem from `<bos>` alone could not:

```
ids     <bos> 1 3 + 0 8 <eos> <bos> 2 1 <eos>
        |------- problem -------|--- solution ---|
loss                                 ^^^^^^^^^^^^
```

A decoder-only model has one stack, so it takes `num_layers`, `num_heads` and `ffn_dim`. The decoder-side keys of a seq2seq block are read as a fallback, which lets an existing config run by changing `model_type` alone — at half the depth of its 6+6 counterpart.

```yaml
model:
  model_type: decoder_only   # alias: decoder
  num_layers: 6
  num_heads: 8
  d_model: 512
  ffn_dim: 2048
  max_sequence_length: 256
```

Like the monomial model, it is a drop-in over the existing seq2seq data path: it consumes the standard collated batch (`input_ids` / `attention_mask` / `decoder_input_ids` / `labels`), packs the two halves per example so the independent padding of the two sides never lands between them, and returns predictions and generations aligned with `labels`. `generate()` returns the completion only, not the prompt, so exact-match evaluation compares it against the labels unchanged. The `<bos>` that opens `decoder_input_ids` doubles as the marker for where the answer begins.

Note that the problem and the solution now share one sequence: `max_sequence_length` has to cover both, since the positional table is sized from it.

## Monomial decoder-only model

`model_type: monomial_decoder_only` (alias `monomial_decoder`) combines the two models above: one causal stack, and one sequence position per monomial. It is the model to use when a polynomial task should be trained decoder-only, or when the two architectures have to be compared on the same input representation.

```
positions   [C1 E1 E0 +] [C2 E0 E1 <] | ans | [C3 E1 E1 +] [C1 E0 E0 <]
            |----------- problem -----------|  |------- solution -------|
loss                                              ^^^^^^^^^^^^^^^^^^^^^^
```

The learnable **answer marker** between the two halves plays the role the `<bos>` plays in the flat decoder-only model: it is the position whose output predicts the first solution monomial. As there, the problem carries no loss.

```yaml
model:
  model_type: monomial_decoder_only   # alias: monomial_decoder
  num_variables: 2                    # REQUIRED, as for `monomial`
  num_layers: 6
  num_heads: 8
  d_model: 512
  ffn_dim: 2048
  max_sequence_length: 512
  # monomial_separators: ["+", "||"]  # set this if your separator is not "||"
```

The configuration surface is that of `monomial` for the vocabulary keys and that of `decoder_only` for the architecture keys, so the same config trains either architecture by changing `model_type` alone. Unlike the encoder–decoder monomial model, which keeps independent source and target embedding tables, one stack sees one stream, so problem and solution share a single table.

The data requirements are those of `monomial`: C/E expanded form, and `num_variables` matching the data.

## Custom embeddings (input and positional)

Both the **input (token) embedding** and the **positional embedding** are chosen by config and extensible at runtime, symmetric to the model registry. Built-ins keep the previous behavior, so existing configs are unaffected.

| config key | default | built-in values |
|------------|---------|-----------------|
| `model.input_embedding_type` | `token` | `token` (aliases `default`, `learned`) — a plain `nn.Embedding` |
| `model.input_embedding_kwargs` | `{}` | extra keyword arguments for the input-embedding factory |
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

register_input_embedding("my_emb", MyEmbedding)
register_positional_embedding(
    "my_pe", lambda d_model, max_len, **kw: MyPositional(d_model, max_len))
```

```yaml
model:
  input_embedding_type: my_emb
  use_positional_embedding: my_pe
```

**Passing arguments.** The entries of `model.input_embedding_kwargs` are passed to the input-embedding factory as keyword arguments and saved in the model config, so `from_pretrained` rebuilds the same embedding (register the factory before loading). Values must be JSON-serializable, and `vocab_size` / `d_model` cannot be set this way. Register the class itself and spell out the arguments it takes (e.g. `def __init__(self, vocab_size, d_model, digit_ids=())`): a misspelled key then raises a `TypeError`, while a factory that takes `**kwargs` silently ignores it. The built-in `token` embedding takes no extra arguments. A value that depends on the tokenizer, such as the token ids of the digits, can be computed after building the IO pipeline:

```python
io_dict = IOPipeline.from_config(cfg.data).build()
tokenizer = io_dict["tokenizer"]
cfg.model.input_embedding_kwargs = {
    "digit_ids": [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
}
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
```

`convert_tokens_to_ids` returns `None` for a token missing from the vocabulary, so make sure every digit is in it. Set the kwargs before your script saves its config (e.g. `OmegaConf.save(cfg, ".../train.yaml")`), so that `load_from_checkpoint` rebuilds the same embedding.

**Factory contract.** An input-embedding factory receives at least `vocab_size` and `d_model`, plus the entries of `model.input_embedding_kwargs`, and returns an `nn.Module` mapping `input_ids` of shape `(batch, seq)` to `(batch, seq, d_model)`. A positional-embedding factory receives at least `d_model` and `max_len` and returns an `nn.Module` mapping `(batch, seq, d_model)` to `(batch, seq, d_model)` (or `None` for "no positional embedding"). An unknown type raises `ValueError` listing the supported names.

These hooks apply to the `generic`, `decoder_only` and `encoder_classifier` models (not `bart`, which is HuggingFace's own model). The `monomial` model honors `use_positional_embedding` but not `input_embedding_type`: its input embedding *is* the monomial embedding. Setting `input_embedding_kwargs` for `bart` or the monomial models raises an error. The public helpers are exported from `calt.models`: `register_input_embedding`, `register_positional_embedding`, `get_input_embedding`, `get_positional_embedding`.

## Custom losses and auxiliary tasks

The `generic`, `decoder_only` and `encoder_classifier` models compute their training loss through a registry, the same way they get their input embedding. The default `model.loss_type: cross_entropy` is the only loss built into calt and gives the same loss as before. To use your own, register it **before building or loading the model**, then select it by name:

```python
import torch.nn.functional as F
from calt.models import register_loss

def sequence_loss(logits, labels, strength=1.0, power=2.0):
    tokens = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    ce = tokens.sum() / (labels != -100).sum()
    q = (-tokens.sum(dim=1)).exp()        # probability of the whole answer
    return ce + strength * (1 - q**power).mean()

register_loss("sequence", sequence_loss)
```

```yaml
model:
  loss_type: sequence
  loss_kwargs: {strength: 0.5}
```

- A **function** is called on every batch with the data arguments below plus `model.loss_kwargs`, and returns a scalar tensor.
- A **class** (an `nn.Module`, or a `functools.partial` of one with arguments bound by keyword) is built once with `d_model`, `vocab_size` (each only if its `__init__` takes it; a class that neither defines nor inherits an `__init__`, and so uses `nn.Module`'s, gets neither) and `model.loss_kwargs`. It becomes `model.loss_module`, so the auxiliary heads inside it are trained and saved with the model; its `forward` receives the data arguments. It is built after the model's own layers, so it does not change their initial weights, and its Linear, Embedding and LayerNorm layers get the model's initialization. Other parameters (e.g. a raw `nn.Parameter`) keep the values your `__init__` gives them, except when a checkpoint without them is loaded, where they are left uninitialized. `from_pretrained` may run `__init__` on the meta device, so it should only create layers. Register the class, not an instance.

**Data arguments** (keyword arguments):

| Argument | generic / decoder_only | encoder_classifier |
|---|---|---|
| `logits` | `(B, T, V)` | `(B, V)` |
| `labels` | `(B, T)`: the answer tokens, ending with EOS; padding is -100 | `(B,)`: the answer token; -100 for rows without one |
| `hidden` | `(B, T, d_model)`: the hidden state from which the logits for `labels[:, i]` are computed | `(B, d_model)`: the pooled vector the logits are computed from |
| extra labels | one argument per name (see below) | same |

`labels` always has the shape of `logits` without its last dimension, so `F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))` works for all three models. A loss receives only the data arguments its signature names, unless it takes `**kwargs`. `model.loss_kwargs` is always passed, so a misspelled key raises a `TypeError`. calt warns once when the batch has an extra label the loss does not take, and when the loss takes an argument named `*_label` or `*_labels` that the batch does not have.

**Extra labels.** The function given to `UserCallableLoadPreprocessor` may return a third element, a dict of labels, and each name reaches the loss as an argument:

```python
import torch.nn as nn

def format_problem(problem, answer):
    ...
    return input_text, target_text, {"carry_labels": carry, "length_label": n_digits}

class MultiTaskLoss(nn.Module):
    def __init__(self, d_model, weight=0.1):
        super().__init__()
        self.carry_head = nn.Linear(d_model, 1)   # one value per answer token
        self.length_head = nn.Linear(d_model, 7)  # one value per sample
        self.weight = weight

    def forward(self, logits, labels, hidden, carry_labels, length_label):
        main = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        mask = labels != -100                     # answer tokens and EOS
        carry = F.binary_cross_entropy_with_logits(
            self.carry_head(hidden).squeeze(-1)[mask], carry_labels[mask].float())
        length = F.cross_entropy(self.length_head(hidden[:, 0]), length_label)
        return main + self.weight * (carry + length)

register_loss("multitask", MultiTaskLoss)
```

- A label is either one number per sample (batched as `(B,)`) or a list with one number per answer token (batched as `(B, T)`, aligned with `labels`). The answer tokens are the lexer's tokens of the target plus EOS, i.e. the positions of `labels`. A list must have that many values or one fewer; a list one shorter gets one value appended at the EOS position. Any other length is an error, raised by `IOPipeline.build` before training.
- Positions past a row's answer, and the appended value, are filled with 0 (0.0 for float labels), never -100, since -100 can be a real label value. Use `labels != -100` to find the answer positions; which of them the loss ignores is up to the loss.
- In a batch, the labels are one dict under the key `extra_targets`, which the model's `forward(..., extra_targets=...)` passes on to the loss. (The key avoids the word "label": transformers treats every `forward()` argument named so as a label column.) Call `model(**batch)` with such a batch to try a loss by hand.
- Integers and bools become `torch.long`; a label with any float becomes `torch.float`. Every sample must return the same names. Some names (`labels`, `logits`, `hidden`, `input_ids`, and names starting with `decoder_` or `target_`, among others) are reserved.
- Extra labels work with the normal `IOPipeline.build`, not with the preprocessing caches (`calt preprocess`). `load_from_checkpoint` rebuilds the IO pipeline from `train.yaml`, which cannot hold a Python load preprocessor, so set it again before resuming training.

**Saving and loading.** `loss_type` and `loss_kwargs` are saved in the model config, and the weights of `model.loss_module` in the checkpoint. Call `register_loss` before `from_pretrained` or `load_from_checkpoint`. A model loaded without its loss registered warns and can still generate, but computing a loss raises an error. The evaluation loss (`eval_loss`) includes the auxiliary terms; generation is unaffected.

`bart` and the monomial models do not support this: `loss_type` other than `cross_entropy`, or a non-empty `loss_kwargs`, raises an error, and so do extra labels when the trainer is built.

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
