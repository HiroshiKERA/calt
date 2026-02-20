# Overview

Utilities to prepare training/evaluation datasets, tokenizers, and data collators. They convert symbolic expressions (polynomials/integers) into internal token sequences and build batches suitable for training.

- [Lexer and vocabulary](io_lexer.md) — `lexer.yaml` configuration and tokenisation.
- [Load preprocessors](io_load_preprocessors.md) — optional load-time preprocessing.
- [Visualization](io_visualization.md) — visual diff of predictions vs references.

## IOPipeline

The main entry point is `IOPipeline`. `IOPipeline.from_config` consumes a `omegaconf.DictConfig` with paths to the lexer/vocabulary configuration and dataset files, then builds tokenised training and test datasets (`train_dataset`, `test_dataset`), a `PreTrainedTokenizerFast` tokenizer, and a `StandardDataCollator`. These are returned as a dictionary (`io_dict`) and passed to the model and trainer pipelines.

::: calt.io.IOPipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

For visualization of evaluation results (predictions vs references), see [Visualization](io_visualization.md).
