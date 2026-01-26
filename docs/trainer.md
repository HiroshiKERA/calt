# Trainer

A convenient extension of the HuggingFace `Trainer` and utility helpers for training and
evaluation. It streamlines device placement, metrics computation, and generation result
saving.

## Core class

:::: calt.trainer.trainer.Trainer

## Pipelines and configuration

High-level example scripts (under `calt/examples/*`) use class-based pipelines to keep
configuration and wiring simple:

- :class:`calt.io.pipeline.IOPipeline` – builds datasets, tokenizer, and collator.
- :class:`calt.models.pipeline.ModelPipeline` – builds the model from configuration.
- :class:`calt.trainer.pipeline.TrainerPipeline` – builds the HuggingFace `Trainer`.

A typical training script looks like:

```python
from omegaconf import OmegaConf
from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

cfg = OmegaConf.load("configs/train.yaml")

io_dict = IOPipeline.from_config(cfg.data).build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer = TrainerPipeline.from_io_dict(cfg.train, model, io_dict, cfg.wandb).build()

trainer.train()
success_rate = trainer.evaluate_and_save_generation()
print(f"Success rate: {100 * success_rate:.1f}%")
```

For details on the three configuration files used in these examples
(`data.yaml`, `lexer.yaml`, `train.yaml`), see :doc:`configuration`.

