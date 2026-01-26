## Example configuration files

Example tasks in `calt/examples/*` share a common configuration pattern based on
three YAML files:

- `configs/data.yaml` – controls dataset generation via `DatasetPipeline`.
- `configs/lexer.yaml` – controls tokenisation and vocabulary via `IOPipeline`.
- `configs/train.yaml` – controls the model, training loop, and WandB logging via
  `ModelPipeline` and `TrainerPipeline`.

All three are loaded with `OmegaConf.load` and passed around as `omegaconf.DictConfig`
objects, so they support dot-style access (e.g. `cfg.data`, `cfg.model`, `cfg.train`,
`cfg.wandb`).

### `data.yaml` – dataset generation (`DatasetPipeline`)

The `dataset` block in `configs/data.yaml` is consumed by
`calt.dataset.pipeline.DatasetPipeline.from_config`:

- `save_dir`: base directory where all splits (train/test/…) are written.
- `num_train_samples`: number of training samples in the `"train"` split.
- `num_test_samples`: number of test samples in the `"test"` split.
- `batch_size`: batch size passed to `DatasetGenerator.run` for efficient multiprocessing.
- `n_jobs`: number of worker processes (`backend="multiprocessing"`).
- `root_seed`: global seed used to derive per-sample seeds.
- `verbose`: whether the generator prints progress logs.
- `backend`: backend name (`"sagemath"` or `"sympy"`), used by `get_backend_classes`.
- `save_text`: whether to write human-readable `.txt` files (e.g. `text_raw.txt`).
- `save_json`: whether to write `.jsonl` files preserving the original structure.

### `lexer.yaml` – IO and vocabulary (`IOPipeline`)

`configs/lexer.yaml` is referenced from the `data` block of `configs/train.yaml` and
consumed inside `calt.io.pipeline.IOPipeline.from_config`. It controls:

- **Number handling**
  - `number.policy`: numeric type (`integer`, `rational`, `float`, …).
  - `number.digit_group`: digit grouping size (e.g. `3`).
  - `number.sign`: sign handling strategy (e.g. `attach`).
  - float-related options such as decimal precision.

- **Separators**
  - item separators (e.g. `,` between numbers),
  - row separators (e.g. `;` between matrix rows).

- **Vocabulary and special tokens**
  - `vocab.special_tokens`: definitions of `<pad>`, `<bos>`, `<eos>`, `<unk>`, etc.
  - `vocab.unk_token`: the unknown token name for the underlying tokenizer.
  - `vocab.extra_tokens`: additional domain-specific symbols.
  - `misc.unk_token`: unknown token name at the IO layer (kept consistent with vocab).

Using this configuration, `IOPipeline.build()` constructs a `PreTrainedTokenizerFast`,
tokenised datasets (`train_dataset`, `test_dataset`), and a `StandardDataCollator`.
These are returned as a dictionary (`io_dict`) and fed into the model and trainer
pipelines.

### `train.yaml` – model and training (`ModelPipeline`, `TrainerPipeline`)

`configs/train.yaml` is the main control file for experiments. It typically contains:

- a `data` block – options for `IOPipeline` (paths to datasets and `lexer.yaml`, etc.)
- a `model` block – architecture and checkpoint configuration for `ModelPipeline`
- a `train` block – training hyperparameters for `TrainerPipeline`
- a `wandb` block – logging configuration for Weights & Biases

The high-level training flow in the examples is:

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

#### `train` block – training hyperparameters

`TrainerPipeline` converts `cfg.train` into a `transformers.TrainingArguments` instance.
Typical keys include:

- `per_device_train_batch_size`, `per_device_eval_batch_size`
- `num_train_epochs`, `max_steps`, `gradient_accumulation_steps`
- `learning_rate`, `weight_decay`, `warmup_steps`, `warmup_ratio`
- `logging_steps`, `eval_steps`, `save_steps`
- `evaluation_strategy`, `save_strategy`
- `metric_for_best_model`, `load_best_model_at_end`
- `output_dir`, `seed`, `fp16`, `bf16`

#### `wandb` block – experiment tracking

`cfg.wandb` is passed to `TrainerPipeline` as `wandb_config` and used to configure
Weights & Biases:

- `project`: project name on WandB.
- `group`: logical experiment group (e.g. task name).
- `run_name`: run identifier shown in the UI.
- `tags`: optional list of tags.

The trainer pipeline sets the corresponding environment variables and ensures
`TrainingArguments.report_to` includes `"wandb"`, so that logging is enabled with a
single configuration block.

