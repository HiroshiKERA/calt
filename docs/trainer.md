# Overview

A convenient extension of the HuggingFace `Trainer` and utility helpers for training and
evaluation. It streamlines device placement, metrics computation, and generation result
saving.

- [Model pipeline](model_pipeline.md) — builds the model from configuration; `cfg.model` and supported model types.
- [Configuration](configuration.md) — `data.yaml`, `lexer.yaml`, and `train.yaml`.

<!-- ## Core class -->

::: calt.trainer.trainer.Trainer
    options:
      heading: "Trainer"

## TrainerPipeline

The main entry point for building a trainer from config is `TrainerPipeline`. Use `TrainerPipeline.from_io_dict(cfg.train, model, io_dict)` then `.build()` to obtain a pipeline; call `.train()` to run training and `.evaluate_and_save_generation()` for evaluation.

::: calt.trainer.pipeline.TrainerPipeline
    options:
      heading: "TrainerPipeline"
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - __init__
        - build
        - train
        - save_model
        - evaluate_and_save_generation
        - from_io_dict
        - resume_from_checkpoint

## Pipelines and configuration

High-level example scripts (under `calt/examples/*`) use class-based pipelines to keep
configuration and wiring simple:

- :class:`calt.io.IOPipeline` – builds datasets, tokenizer, and collator.
- :class:`calt.models.ModelPipeline` – builds the model from configuration.
- :class:`calt.trainer.TrainerPipeline` – builds the HuggingFace `Trainer`.

A typical training script looks like:

```python
from omegaconf import OmegaConf
from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

cfg = OmegaConf.load("configs/train.yaml")

io_dict = IOPipeline.from_config(cfg.data).build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
# wandb_config is optional - if not provided, TrainerPipeline will try to get it from cfg.train.wandb
trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

trainer_pipeline.train()
success_rate = trainer_pipeline.evaluate_and_save_generation()
print(f"Success rate: {100 * success_rate:.1f}%")
```

## Resuming training from checkpoint

You can resume training from a saved checkpoint using `TrainerPipeline.resume_from_checkpoint`:

```python
from calt.trainer import TrainerPipeline

# Resume training from a saved directory
trainer_pipeline = TrainerPipeline.resume_from_checkpoint(
    save_dir="./results/my_experiment",
    resume_from_checkpoint=True  # Load saved model weights
)
trainer_pipeline.build()
trainer_pipeline.train()  # Continue training
```

For details on the three configuration files used in these examples
(`data.yaml`, `lexer.yaml`, `train.yaml`), see [Configuration](configuration.md).

