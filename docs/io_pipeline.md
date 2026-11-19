# Overview

Utilities to prepare training/evaluation datasets, tokenizers, and data collators. They convert symbolic expressions (polynomials/integers) into internal token sequences and build batches suitable for training.

- [Lexer and vocabulary](io_lexer.md) — `lexer.yaml` configuration and tokenisation.
- [Load preprocessors](io_load_preprocessors.md) — optional load-time preprocessing.
- [Visualization](io_visualization.md) — visual diff of predictions vs references.

## IOPipeline

The main entry point is `IOPipeline`. `IOPipeline.from_config` consumes a `omegaconf.DictConfig` with paths to the lexer/vocabulary configuration and dataset files, then builds tokenised training and test datasets (`train_dataset`, `test_dataset`), a `PreTrainedTokenizerFast` tokenizer, and a `StandardDataCollator`. These are returned as a dictionary (`io_dict`) and passed to the model and trainer pipelines.

### Custom data collator

To batch the data your own way, pass a collator class (or a function returning a collator) as `data_collator_factory`. `build()` calls it with the tokenizer it built, `data_collator_factory(tokenizer=tokenizer)`, and returns the collator as `io_dict["data_collator"]`, which `TrainerPipeline.from_io_dict` uses for every batch:

```python
from calt.io import IOPipeline, StandardDataCollator
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline


class MyCollator(StandardDataCollator):
    def __call__(self, batch):
        out = super().__call__(batch)  # input_ids, attention_mask, labels, ...
        # change the batch here
        return out


io_dict = IOPipeline.from_config(cfg.data, data_collator_factory=MyCollator).build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()
trainer.train()
```

- The factory comes from Python code, not from `train.yaml`. `IOPipeline(...)` takes the same argument, and `io_pipeline.data_collator_factory = MyCollator` before `build()` also works. `load_from_checkpoint` rebuilds the pipeline from `train.yaml` with `StandardDataCollator`; to resume with your collator, set `trainer_pipeline.data_collator = MyCollator(tokenizer=trainer_pipeline.tokenizer)` before `trainer_pipeline.build()` (the returned `io_pipeline.data_collator` stays a `StandardDataCollator`).
- Without a factory, `build()` uses `StandardDataCollator` as before.
- With [extra labels](model_pipeline.md#custom-losses-and-auxiliary-tasks), the collator must be a `StandardDataCollator` or a subclass, since `build()` can only give the extra-label spec (the dtype of each label, decided over the whole data) to it; `build()` raises otherwise. The spec is set when the collator has none and kept when the factory set one, so let the factory return a new collator on each call.

::: calt.io.IOPipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

For visualization of evaluation results (predictions vs references), see [Visualization](io_visualization.md).
