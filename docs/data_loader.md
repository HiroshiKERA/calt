# Data Loader

Utilities to prepare training/evaluation datasets, tokenizers, and data collators. They convert symbolic expressions (polynomials/integers) into internal token sequences and build batches suitable for training.

## IOPipeline

The main entry point for building data pipelines is :class:`calt.io.IOPipeline`.

::: calt.io.IOPipeline

:meth:`IOPipeline.from_config` consumes a :class:`omegaconf.DictConfig` that contains
paths to the lexer/vocabulary configuration and dataset files, then builds:

- tokenised training and test datasets (`train_dataset`, `test_dataset`)
- a `PreTrainedTokenizerFast` instance (`tokenizer`)
- a `StandardDataCollator` instance (`data_collator`)

These are returned as a dictionary (often called `io_dict` in the examples) and passed
to the model and trainer pipelines.

## Lexer and vocabulary configuration (`lexer.yaml`)

The lexer configuration file controls how raw text (e.g. comma-separated integers,
matrix rows separated by `;`, rational coefficients) is mapped to tokens. Although the
exact keys may vary by task, the following groups are common:

- **Number handling**
  - `number.policy`: numeric type (`integer`, `rational`, `float`, …).
  - `number.digit_group`: how many digits to group into one token.
  - `number.sign`: how to treat signs (e.g. `attach` to keep `-` with the token).
  - float-specific options such as the number of decimal places.

- **Separators**
  - item-level separators (e.g. `,` between numbers),
  - row-level separators (e.g. `;` between matrix rows).

- **Vocabulary and special tokens**
  - `vocab.special_tokens`: definitions for `<pad>`, `<bos>`, `<eos>`, `<unk>`, etc.
  - `vocab.unk_token`: the unknown token name used by the underlying tokenizer.
  - `vocab.extra_tokens`: additional domain-specific symbols.
  - `misc.unk_token`: unknown token name used consistently at the IO layer.

`IOPipeline` uses this configuration to instantiate :class:`calt.io.preprocessor.UnifiedLexer`
and :class:`calt.io.vocabulary.config.VocabConfig`, and finally a HuggingFace-compatible
tokenizer that is shared between the model and trainer.

## Core classes

### Dataset and collator
::: calt.io.base.StandardDataset
::: calt.io.base.StandardDataCollator

### Lexer
::: calt.io.preprocessor.UnifiedLexer
::: calt.io.preprocessor.NumberPolicy

### Vocabulary
::: calt.io.vocabulary.config.VocabConfig

## Visualization utilities (optional)

Quickly render visual diffs between predictions and references.
::: calt.io.visualization.comparison_vis.display_with_diff
::: calt.io.visualization.comparison_vis.load_eval_results
::: calt.io.visualization.comparison_vis.parse_poly
