# Trainer

A convenient extension of the HuggingFace `Trainer` and utility helpers for training and
evaluation. It streamlines device placement, metrics computation, and generation result
saving.

## Core class

:::: calt.trainer.trainer.Trainer

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
(`data.yaml`, `lexer.yaml`, `train.yaml`), see :doc:`configuration`.

