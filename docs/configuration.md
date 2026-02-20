# Configuration

Example tasks in `calt/examples/*` share a common configuration pattern based on
three YAML files:

- [configs/data.yaml](#datayaml-dataset-generation-datasetpipeline) – controls dataset generation via `DatasetPipeline`.
- [configs/train.yaml](#trainyaml-model-and-training-modelpipeline-trainerpipeline) – main control file for the model, training loop, and WandB logging via
  `ModelPipeline` and `TrainerPipeline` (references `lexer.yaml` in its `data` block).
- [configs/lexer.yaml](#lexeryaml-io-and-vocabulary-iopipeline) – controls tokenisation and vocabulary via `IOPipeline`; path set in `train.yaml`’s `data.lexer_config`.

All three are loaded with `OmegaConf.load` and passed around as `omegaconf.DictConfig`
objects, so they support dot-style access (e.g. `cfg.data`, `cfg.model`, `cfg.train`).
WandB configuration can be included in `cfg.train.wandb` or passed separately as `cfg.wandb`.

## `data.yaml` – dataset generation (`DatasetPipeline`)

Example tasks under `calt/examples/*` use `configs/data.yaml` to drive dataset generation
through [DatasetPipeline](dataset_generator.md#calt.dataset.pipeline.DatasetPipeline). Typical usage:

```python
from omegaconf import OmegaConf
from calt.dataset import DatasetPipeline

cfg = OmegaConf.load("configs/data.yaml")
pipeline = DatasetPipeline.from_config(
    cfg.dataset,
    instance_generator=my_instance_generator,
    statistics_calculator=None,
)
pipeline.run()
```

The `dataset` block in `data.yaml` controls all dataset-generation behaviour. Example:

```yaml
dataset:
  save_dir: "./data"
  num_train_samples: 100000
  num_test_samples: 1000
  batch_size: 10000
  n_jobs: 4
  root_seed: 42
  verbose: true
  backend: "sagemath"
  save_text: true
  save_json: false
```

??? "`dataset` — Passed to DatasetPipeline.from_config"
    | Name | Description |
    |--------|-------------|
    | `save_dir` | Base directory where all splits (train/test/…) are written. |
    | `num_train_samples` | Number of training samples to generate (size of the `"train"` split). |
    | `num_test_samples` | Number of test samples to generate (size of the `"test"` split). |
    | `batch_size` | Batch size passed to `DatasetGenerator.run` for efficient multiprocessing. |
    | `n_jobs` | Number of worker processes used by the generator (`backend="multiprocessing"`). |
    | `root_seed` | Global seed used to derive per-sample seeds in the backend. |
    | `verbose` | Whether the generator prints progress information. |
    | `backend` | Which implementation to use: `"sagemath"` → [SageMath backend](dataset_sagemath.md), `"sympy"` → [SymPy backend](dataset_sympy.md). |
    | `save_text` | Whether to write human-readable `.txt` files (`text_raw.txt`, etc.). |
    | `save_json` | Whether to write `.jsonl` files preserving the nested Python structure. |

Under the hood, [DatasetPipeline](dataset_generator.md#calt.dataset.pipeline.DatasetPipeline) resolves the appropriate `DatasetGenerator` for the chosen `backend` and uses the common [DatasetWriter](dataset_generator.md#datasetwriter); both are configured from the same options.

## `train.yaml` – model and training (`ModelPipeline`, `TrainerPipeline`)

Example tasks under `calt/examples/*` use `configs/train.yaml` as the main control file for experiments. It is loaded with `OmegaConf.load` and passed to [IOPipeline](io_pipeline.md), [ModelPipeline](model_pipeline.md), and [TrainerPipeline](trainer.md). Typical usage:

```python
from omegaconf import OmegaConf
from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

cfg = OmegaConf.load("configs/train.yaml")

io_dict = IOPipeline.from_config(cfg.data).build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

trainer_pipeline.train()
success_rate = trainer_pipeline.evaluate_and_save_generation()
print(f"Success rate: {100 * success_rate:.1f}%")
```

The top-level blocks in `train.yaml` are `data`, `model`, `train`, and optionally `wandb` (under `train` or separate). Example:

```yaml
data:
  train_dataset_path: ./data/train_raw.txt
  test_dataset_path: ./data/test_raw.txt
  lexer_config: ./configs/lexer.yaml
  num_train_samples: -1
  num_test_samples: -1
  validate_train_tokens: true
  validate_test_tokens: true
  display_samples: 5

model:
  model_type: generic
  num_encoder_layers: 6
  num_encoder_heads: 8
  num_decoder_layers: 6
  num_decoder_heads: 8
  d_model: 512
  encoder_ffn_dim: 2048
  decoder_ffn_dim: 2048
  max_sequence_length: 512

train:
  save_dir: ./results
  num_train_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_ratio: 0.1
  batch_size: 16
  test_batch_size: 16
  seed: 42
  wandb:
    project: calt
    group: gf17_addition
    name: gf17_addition
```

??? "`data` — Passed to IOPipeline.from_config"
    | Name | Description |
    |--------|-------------|
    | `lexer_config` | Path to `lexer.yaml` (required). |
    | `train_dataset_path` | Path to training raw text file. |
    | `test_dataset_path` | Path to test raw text file. |
    | `num_train_samples` | Number of samples to load for training (-1 = all). |
    | `num_test_samples` | Number of samples to load for evaluation (-1 = all). |
    | `validate_train_tokens` | Whether to validate that all training tokens are in vocab (default false). |
    | `validate_test_tokens` | Whether to validate test tokens (default true). |
    | `display_samples` | Number of sample lines to print when loading (0 to disable). |
    | `use_jsonl` | If true, load from JSONL instead of raw text. |
    | `use_pickle` | If true, load from pickle. |
    | `train_dataset_jsonl`, `test_dataset_jsonl` | Paths when using JSONL. |
    | `train_dataset_pickle`, `test_dataset_pickle` | Paths when using pickle. |
    | `dataset_load_preprocessor` | Optional preprocessor for custom loading. |

??? "`model` — Passed to ModelPipeline.from_io_dict"
    | Name | Description |
    |--------|-------------|
    | `model_type` | Model architecture (e.g. `generic`, `bart`). |
    | `num_encoder_layers`, `num_encoder_heads` | Encoder depth and attention heads. |
    | `num_decoder_layers`, `num_decoder_heads` | Decoder depth and attention heads. |
    | `d_model` | Hidden size (embedding dimension). |
    | `encoder_ffn_dim`, `decoder_ffn_dim` | Feed-forward dimension in encoder/decoder. |
    | `max_sequence_length` | Maximum sequence length. |

??? "`train` — Converted to TrainingArguments by TrainerPipeline"
    | Name | Description |
    |------|-------------|
    | `save_dir` | Output directory for checkpoints and logs. Passed to HuggingFace `TrainingArguments.output_dir`. If omitted, `output_dir` is used; if both are missing, defaults to `"./tmp"`. |
    | `output_dir` | Alias for `save_dir`. Used when `save_dir` is not set. |
    | `num_train_epochs` | Number of training epochs. |
    | `learning_rate` | Learning rate for the optimizer. |
    | `weight_decay` | Weight decay (L2 penalty) coefficient. |
    | `warmup_ratio` | Fraction of training steps used for a linear warmup from 0 to `learning_rate` (0 to 1). |
    | `batch_size` | Per-device training batch size. With multiple GPUs, this is divided by the number of devices and passed as `per_device_train_batch_size`. |
    | `test_batch_size` | Per-device evaluation batch size. Similarly passed as `per_device_eval_batch_size`. |
    | `lr_scheduler_type` | Learning rate schedule. Use `"linear"` or `"constant"`. Defaults to `"linear"` if not set. |
    | `max_grad_norm` | Maximum gradient norm for clipping. |
    | `optimizer` | Optimizer name (passed to HuggingFace `optim`). |
    | `num_workers` | Number of DataLoader worker processes (`dataloader_num_workers`). |
    | `dataloader_pin_memory` | DataLoader `pin_memory` option. Defaults to `true`. |
    | `eval_strategy` | When to run evaluation (e.g. `"steps"`, `"epoch"`). Defaults to `"steps"`. |
    | `eval_steps` | Run evaluation every this many steps when `eval_strategy` is `"steps"`. Defaults to `1000`. |
    | `save_strategy` | When to save checkpoints (e.g. `"steps"`, `"epoch"`). Defaults to `"steps"`. |
    | `save_steps` | Save a checkpoint every this many steps when `save_strategy` is `"steps"`. Defaults to `1000`. |
    | `save_total_limit` | Maximum number of checkpoints to keep; older ones are removed. Defaults to `1`. |
    | `save_safetensors` | If `true`, save model weights in safetensors format. Defaults to `false`. |
    | `label_names` | List of label keys used by the Trainer. Defaults to `["labels"]`. |
    | `logging_strategy` | When to log (e.g. `"steps"`, `"epoch"`). Defaults to `"steps"`. |
    | `logging_steps` | Log every this many steps when `logging_strategy` is `"steps"`. Defaults to `50`. |
    | `seed` | Random seed for reproducibility. |
    | `remove_unused_columns` | Whether to drop dataset columns not used by the model. Defaults to `false`. |
    | `disable_tqdm` | Whether to disable the progress bar. Defaults to `true`. |

??? "`train.wandb` (or top-level `wandb`) — Used by TrainerPipeline for Weights & Biases"
    | Name | Description |
    |--------|-------------|
    | `project` | Project name on WandB. |
    | `group` | Logical experiment group (e.g. task name). |
    | `name` | Run identifier shown in the UI. |
    | `tags` | Optional list of tags. |
    | `no_wandb` | If true, disable WandB logging. |

The trainer pipeline sets the corresponding environment variables and ensures `TrainingArguments.report_to` includes `"wandb"` when WandB is configured.

## `lexer.yaml` – IO and vocabulary (`IOPipeline`)

The `data` block in `train.yaml` points to `lexer.yaml` via `data.lexer_config`. That file controls tokenisation and vocabulary and is loaded by [IOPipeline.from_config](io_pipeline.md). Typical usage:

```python
from omegaconf import OmegaConf
from calt.io import IOPipeline

cfg = OmegaConf.load("configs/train.yaml")
io_pipeline = IOPipeline.from_config(cfg.data)  # loads lexer_config from cfg.data.lexer_config
```

The top-level keys in `lexer.yaml` control vocabulary and number tokenisation. Example:

```yaml
vocab:
  range:
    numbers: ["", 0, 16]
  misc: ["+", "*", "^", "(", ")","|", "-"]
  special_tokens: {}
  flags:
    include_base_vocab: true
    include_base_special_tokens: true

number_policy:
  attach_sign: true
  digit_group: 0
  allow_float: false

strict: true
include_base_vocab: true
```

??? "`vocab` — Passed to VocabConfig.from_config"
    | Name | Description |
    |--------|-------------|
    | `range` | Dict of arbitrary key → `[prefix, min, max]` (inclusive). Each entry expands to tokens `prefix+str(i)` for `i` in `min..max`. For example, `numbers: ["", 0, 16]`, `coefficients: ["C", -50, 50]`, `exponents: ["E", 0, 20]`, `variables: ["x", 0, 2]`. |
    | `misc` | List of extra tokens (e.g. `["+", "=", ","]`). |
    | `special_tokens` | Dict of special token names. Base special tokens are defined in code. |
    | `flags` | Optional. `include_base_vocab`, `include_base_special_tokens` (both default true). |

??? "`number_policy` — Builds NumberPolicy for UnifiedLexer"
    | Name | Description |
    |--------|-------------|
    | `attach_sign` | bool (default true). true = sign is part of the number token; false = sign is a separate token. |
    | `digit_group` | int (default 0). 0 = no digit grouping; d ≥ 1 = split number into tokens of d digits. |
    | `allow_float` | bool (default true). Whether to allow decimal numbers (adds `"."` to vocab if needed). |

??? "`strict`, `include_base_vocab` — Passed to UnifiedLexer"
    | Name | Description |
    |--------|-------------|
    | `strict` | bool (default true). If true, raise an error on unknown characters; if false, emit the unknown token (e.g. `<unk>`) and continue. |
    | `include_base_vocab` | bool (default true). If true, add built-in tokens (separators, operators `+`, `-`, `*`, brackets, etc.) to the lexer’s reserved set; if false, only tokens from `vocab` are used. |

Under the hood, [IOPipeline](io_pipeline.md) instantiates `UnifiedLexer` and `VocabConfig` from this configuration, then builds a HuggingFace-compatible tokenizer, tokenised datasets, and a `StandardDataCollator`. See [Lexer and vocabulary](io_lexer.md) for the API of `UnifiedLexer`, `NumberPolicy`, and `VocabConfig`.

