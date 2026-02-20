# Overview

A convenient extension of the HuggingFace `Trainer` and utility helpers for training and evaluation. It streamlines device placement, metrics computation, and generation result saving.

- [Model pipeline](../model_pipeline/) — builds the model from configuration; `cfg.model` and supported model types.
- [Configuration](../configuration/) — `data.yaml`, `lexer.yaml`, and `train.yaml`.

## Trainer

```
Trainer(*args, **kwargs)
```

Bases: `Trainer`

Extension of *HuggingFace* :class:`~transformers.Trainer`.

The trainer adds task-specific helpers that simplify training generative Transformer models. It accepts all the usual `HTrainer` keyword arguments and does not introduce new parameters - the default constructor is therefore forwarded verbatim.

Source code in `src/calt/trainer/trainer.py`

```
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Keeps a chronological list of metric dictionaries that WandB has
    # seen.  This enables the caller to inspect the *complete* training
    # history after the run has finished without having to query WandB.
    self.log_history = []

    if self.compute_metrics is None:
        self.compute_metrics = self._compute_metrics
```

### evaluate

```
evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix='eval')
```

Override evaluate to also save generation results during training.

This method is called during training evaluation steps and after training. It runs the standard evaluation and then saves generation results.

Source code in `src/calt/trainer/trainer.py`

```
def evaluate(
    self,
    eval_dataset=None,
    ignore_keys=None,
    metric_key_prefix="eval",
):
    """Override evaluate to also save generation results during training.

    This method is called during training evaluation steps and after training.
    It runs the standard evaluation and then saves generation results.
    """
    # Run standard evaluation
    metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    # Also save generation results during evaluation
    # Get current step number if available
    step = getattr(self.state, "global_step", None)
    logger.info(
        f"Running evaluate_and_save_generation (step={step}, metric_key_prefix={metric_key_prefix})"
    )
    try:
        # Pass step number to evaluate_and_save_generation
        success_rate = self.evaluate_and_save_generation(step=step)
        # Add generation metrics to the returned metrics dict
        generation_metrics = {
            f"{metric_key_prefix}_generation_success_rate": success_rate,
        }
        if step is not None:
            generation_metrics[f"{metric_key_prefix}_generation_step"] = step
        # Update the metrics dict
        metrics.update(generation_metrics)
        # Explicitly log only the generation metrics to ensure they are recorded
        self.log(generation_metrics)
        logger.info(
            f"Successfully saved generation results (step={step}, success_rate={success_rate:.4f})"
        )
    except Exception as e:
        # Log error but don't fail the evaluation
        logger.warning(
            f"Failed to save generation results during evaluation: {e}",
            exc_info=True,
        )

    return metrics
```

### evaluate_and_save_generation

```
evaluate_and_save_generation(max_length: int = 512, step: int | None = None)
```

Run greedy/beam-search generation on the evaluation set.

The helper decodes the model outputs into strings, stores the results in `eval_results.json` inside the trainer's output directory and finally computes exact-match accuracy between the generated and reference sequences.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `max_length` | `int` | Maximum generation length. Defaults to 512. | `512` | | `step` | `int` | Current training step number. If None, tries to get from self.state. | `None` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `float` | | Exact-match accuracy in the [0, 1] interval. |

Source code in `src/calt/trainer/trainer.py`

```
def evaluate_and_save_generation(
    self, max_length: int = 512, step: int | None = None
):
    """Run greedy/beam-search generation on the evaluation set.

    The helper decodes the model outputs into strings, stores the results in
    ``eval_results.json`` inside the trainer's output directory and finally computes
    exact-match accuracy between the generated and reference sequences.

    Args:
        max_length (int, optional): Maximum generation length. Defaults to 512.
        step (int, optional): Current training step number. If None, tries to get from self.state.

    Returns:
        float: Exact-match accuracy in the [0, 1] interval.
    """
    if self.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")

    if len(self.eval_dataset) == 0:
        logger.warning(
            "eval_dataset is empty; skipping evaluate_and_save_generation."
        )
        return 0.0

    all_generated_texts = []
    all_reference_texts = []

    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

    self.model.eval()
    tokenizer = self.processing_class

    for batch in eval_dataloader:
        if batch is None:
            continue
        inputs = self._prepare_inputs(batch)
        if inputs is None:
            continue
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        if input_ids is None:
            continue

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                # Optional: specify ``pad_token_id`` / ``eos_token_id`` as
                # keyword arguments if the model configuration requires.
            )

        # generated_ids shape (batch_size, sequence_length)
        current_generated_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        all_generated_texts.extend(current_generated_texts)

        if labels is not None:
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            current_reference_texts = tokenizer.batch_decode(
                labels_for_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            all_reference_texts.extend(current_reference_texts)
        else:
            # Keep placeholder when reference labels are missing.
            all_reference_texts.extend(["" for _ in current_generated_texts])

    # Include step number in filename if available during training
    if step is None:
        step = getattr(self.state, "global_step", None)
    if step is not None:
        # Save step-wise results in a subdirectory
        eval_results_dir = os.path.join(self.args.output_dir, "eval_results")
        os.makedirs(eval_results_dir, exist_ok=True)
        output_eval_file = os.path.join(
            eval_results_dir,
            f"step_{step}.json",
        )
    else:
        output_eval_file = os.path.join(
            self.args.output_dir,
            "eval_results.json",
        )
    results = []
    for gen_text, ref_text in zip(all_generated_texts, all_reference_texts):
        results.append(
            {
                "generated": gen_text,
                "reference": ref_text,
            }
        )

    with open(output_eval_file, "w") as writer:
        json.dump(
            results,
            writer,
            indent=4,
            ensure_ascii=False,
        )

    correct_predictions = 0
    total_predictions = len(all_generated_texts)

    if total_predictions == 0:
        return 0.0

    for gen_text, ref_text in zip(all_generated_texts, all_reference_texts):
        if gen_text.strip() == ref_text.strip():
            correct_predictions += 1

    success_rate = correct_predictions / total_predictions

    return success_rate
```

## TrainerPipeline

The main entry point for building a trainer from config is `TrainerPipeline`. Use `TrainerPipeline.from_io_dict(cfg.train, model, io_dict)` then `.build()` to obtain a pipeline; call `.train()` to run training and `.evaluate_and_save_generation()` for evaluation.

Pipeline for creating trainers from configuration.

Similar to IOPipeline, this class provides a simple interface for creating trainer instances from config files. It automatically selects the appropriate TrainerLoader based on the config.

Examples:

```
>>> from omegaconf import OmegaConf
>>> from calt.trainer import TrainerPipeline
>>>
>>> cfg = OmegaConf.load("config/train.yaml")
>>> model = ...  # Get model from ModelPipeline
>>> tokenizer = ...  # Get tokenizer from IOPipeline
>>> train_dataset = ...  # Get from IOPipeline
>>> eval_dataset = ...  # Get from IOPipeline
>>> data_collator = ...  # Get from IOPipeline
>>>
>>> trainer_pipeline = TrainerPipeline(
...     cfg.train,
...     model=model,
...     tokenizer=tokenizer,
...     train_dataset=train_dataset,
...     eval_dataset=eval_dataset,
...     data_collator=data_collator,
... )
>>> trainer = trainer_pipeline.build()
```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `config` | `DictConfig` | Training configuration from cfg.train (OmegaConf). | *required* | | `model` | `PreTrainedModel | None` | Model instance. | `None` | | `tokenizer` | `PreTrainedTokenizerFast | None` | Tokenizer instance. | `None` | | `train_dataset` | `Dataset | None` | Training dataset. | `None` | | `eval_dataset` | `Dataset | None` | Evaluation dataset. | `None` | | `data_collator` | `StandardDataCollator | None` | Data collator. | `None` |

Source code in `src/calt/trainer/pipeline.py`

```
def __init__(
    self,
    config: DictConfig,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    data_collator: Optional[StandardDataCollator] = None,
    wandb_config: Optional[DictConfig] = None,
    io_dict: Optional[dict] = None,
):
    """Initialize the trainer pipeline.

    Args:
        config (DictConfig): Training configuration from cfg.train (OmegaConf).
        model (PreTrainedModel | None): Model instance.
        tokenizer (PreTrainedTokenizerFast | None): Tokenizer instance.
        train_dataset (Dataset | None): Training dataset.
        eval_dataset (Dataset | None): Evaluation dataset.
        data_collator (StandardDataCollator | None): Data collator.
    """
    self.config = config
    self.model = model
    # Prefer explicit arguments, but allow filling from io_dict when provided
    if io_dict is not None:
        tokenizer = tokenizer or io_dict.get("tokenizer")
        train_dataset = train_dataset or io_dict.get("train_dataset")
        eval_dataset = eval_dataset or io_dict.get("test_dataset")
        data_collator = data_collator or io_dict.get("data_collator")

    self.tokenizer = tokenizer
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.data_collator = data_collator
    self.wandb_config = wandb_config
    self.trainer: Optional[Trainer] = None
    self._loader = None
```

### build

```
build() -> TrainerPipeline
```

Build the trainer from configuration.

Returns:

| Name | Type | Description | | --- | --- | --- | | `TrainerPipeline` | `TrainerPipeline` | Returns self for method chaining. |

Source code in `src/calt/trainer/pipeline.py`

```
def build(self) -> "TrainerPipeline":
    """Build the trainer from configuration.

    Returns:
        TrainerPipeline: Returns self for method chaining.
    """
    # Configure wandb before building TrainingArguments / Trainer
    self._configure_wandb()

    # Import here to avoid circular import
    from .loader import StandardTrainerLoader

    # Create trainer loader
    self._loader = StandardTrainerLoader(
        calt_config=self.config,
        model=self.model,
        tokenizer=self.tokenizer,
        train_dataset=self.train_dataset,
        eval_dataset=self.eval_dataset,
        data_collator=self.data_collator,
    )

    # Load the trainer
    self.trainer = self._loader.load()
    return self
```

### train

```
train(resume_from_checkpoint: str | bool | None = None) -> None
```

Train the model.

This method calls trainer.train(). If resume_from_checkpoint is provided, training will resume from the specified checkpoint.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `resume_from_checkpoint` | `str | bool | None` | If True, resume from the latest checkpoint in output_dir. If str, resume from the specified checkpoint path. If None, start training from scratch. | `None` |

Source code in `src/calt/trainer/pipeline.py`

```
def train(self, resume_from_checkpoint: str | bool | None = None) -> None:
    """Train the model.

    This method calls trainer.train(). If resume_from_checkpoint is provided,
    training will resume from the specified checkpoint.

    Args:
        resume_from_checkpoint: If True, resume from the latest checkpoint in output_dir.
                               If str, resume from the specified checkpoint path.
                               If None, start training from scratch.
    """
    if self.trainer is None:
        raise ValueError("Trainer not built. Call build() first.")

    # Train the model
    self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

### save_model

```
save_model(output_dir: str | None = None) -> None
```

Save the model and tokenizer.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_dir` | `str | None` | Directory to save the model and tokenizer. If None, uses trainer's output_dir. | `None` |

Source code in `src/calt/trainer/pipeline.py`

```
def save_model(self, output_dir: str | None = None) -> None:
    """Save the model and tokenizer.

    Args:
        output_dir: Directory to save the model and tokenizer. If None, uses trainer's output_dir.
    """
    if self.trainer is None:
        raise ValueError("Trainer not built. Call build() first.")
    if output_dir is None:
        output_dir = self.trainer.args.output_dir
    self.trainer.save_model(output_dir=output_dir)
    # Also save tokenizer
    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)
```

### evaluate_and_save_generation

```
evaluate_and_save_generation(max_length: int = 512) -> float
```

Evaluate and save generation results.

Source code in `src/calt/trainer/pipeline.py`

```
def evaluate_and_save_generation(self, max_length: int = 512) -> float:
    """Evaluate and save generation results."""
    if self.trainer is None:
        raise ValueError("Trainer not built. Call build() first.")
    return self.trainer.evaluate_and_save_generation(max_length=max_length)
```

### from_io_dict

```
from_io_dict(
    config: DictConfig,
    model: PreTrainedModel,
    io_dict: dict,
    wandb_config: Optional[DictConfig] = None,
) -> TrainerPipeline
```

Create a TrainerPipeline from a dict returned by IOPipeline.build().

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `config` | `DictConfig` | Training configuration (cfg.train). May contain wandb config as config.wandb. | *required* | | `model` | `PreTrainedModel` | Model instance from ModelPipeline. | *required* | | `io_dict` | `dict` | IOPipeline.build() result. | *required* | | `wandb_config` | `Optional[DictConfig]` | Optional wandb configuration block. If None, tries to get from config.wandb. | `None` |

Source code in `src/calt/trainer/pipeline.py`

```
@classmethod
def from_io_dict(
    cls,
    config: DictConfig,
    model: PreTrainedModel,
    io_dict: dict,
    wandb_config: Optional[DictConfig] = None,
) -> "TrainerPipeline":
    """Create a TrainerPipeline from a dict returned by IOPipeline.build().

    Args:
        config: Training configuration (cfg.train). May contain wandb config as config.wandb.
        model: Model instance from ModelPipeline.
        io_dict: IOPipeline.build() result.
        wandb_config: Optional wandb configuration block. If None, tries to get from config.wandb.
    """
    # Get wandb_config from config.wandb if not provided
    if wandb_config is None and hasattr(config, "wandb"):
        wandb_config = config.wandb

    instance = cls(
        config=config,
        model=model,
        tokenizer=io_dict["tokenizer"],
        train_dataset=io_dict["train_dataset"],
        eval_dataset=io_dict["test_dataset"],
        data_collator=io_dict["data_collator"],
        wandb_config=wandb_config,
        io_dict=io_dict,
    )
    return instance
```

### resume_from_checkpoint

```
resume_from_checkpoint(
    save_dir: str, resume_from_checkpoint: bool = True
) -> TrainerPipeline
```

Resume training from a saved checkpoint directory.

This method loads train.yaml from save_dir, reconstructs IOPipeline, ModelPipeline, and TrainerPipeline, and optionally loads the saved model and tokenizer.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `save_dir` | `str` | Directory containing train.yaml, model/, and tokenizer/. | *required* | | `resume_from_checkpoint` | `bool` | If True, load saved model and tokenizer. If False, create new model. | `True` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `TrainerPipeline` | `TrainerPipeline` | TrainerPipeline instance ready for training continuation. |

Examples:

```
>>> from calt.trainer import TrainerPipeline
>>>
>>> # Load from checkpoint and continue training
>>> trainer_pipeline = TrainerPipeline.resume_from_checkpoint("./results")
>>> trainer_pipeline.build()
>>> trainer_pipeline.train()  # Continue training
```

Source code in `src/calt/trainer/pipeline.py`

```
@classmethod
def resume_from_checkpoint(
    cls,
    save_dir: str,
    resume_from_checkpoint: bool = True,
) -> "TrainerPipeline":
    """Resume training from a saved checkpoint directory.

    This method loads train.yaml from save_dir, reconstructs IOPipeline, ModelPipeline,
    and TrainerPipeline, and optionally loads the saved model and tokenizer.

    Args:
        save_dir: Directory containing train.yaml, model/, and tokenizer/.
        resume_from_checkpoint: If True, load saved model and tokenizer. If False, create new model.

    Returns:
        TrainerPipeline: TrainerPipeline instance ready for training continuation.

    Examples:
        >>> from calt.trainer import TrainerPipeline
        >>>
        >>> # Load from checkpoint and continue training
        >>> trainer_pipeline = TrainerPipeline.resume_from_checkpoint("./results")
        >>> trainer_pipeline.build()
        >>> trainer_pipeline.train()  # Continue training
    """
    from .utils import load_from_checkpoint

    _, _, trainer_pipeline = load_from_checkpoint(save_dir, resume_from_checkpoint)
    return trainer_pipeline
```

## Pipelines and configuration

High-level example scripts (under `calt/examples/*`) use class-based pipelines to keep configuration and wiring simple:

- :class:`calt.io.IOPipeline` – builds datasets, tokenizer, and collator.
- :class:`calt.models.ModelPipeline` – builds the model from configuration.
- :class:`calt.trainer.TrainerPipeline` – builds the HuggingFace `Trainer`.

A typical training script looks like:

```
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

```
from calt.trainer import TrainerPipeline

# Resume training from a saved directory
trainer_pipeline = TrainerPipeline.resume_from_checkpoint(
    save_dir="./results/my_experiment",
    resume_from_checkpoint=True  # Load saved model weights
)
trainer_pipeline.build()
trainer_pipeline.train()  # Continue training
```

For details on the three configuration files used in these examples (`data.yaml`, `lexer.yaml`, `train.yaml`), see [Configuration](../configuration/).
