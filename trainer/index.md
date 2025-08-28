# Trainer

A convenient extension of the HuggingFace `Trainer` and utility helpers for training and evaluation. It streamlines device placement, metrics computation, and generation result saving.

### Class

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

## evaluate_and_save_generation

```
evaluate_and_save_generation(max_length: int = 512)

```

Run greedy/beam-search generation on the evaluation set.

The helper decodes the model outputs into strings, stores the results in `eval_results.json` inside the trainer's output directory and finally computes exact-match accuracy between the generated and reference sequences.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `max_length` | `int` | Maximum generation length. Defaults to 512. | `512` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `float` | | Exact-match accuracy in the [0, 1] interval. |

Source code in `src/calt/trainer/trainer.py`

```
def evaluate_and_save_generation(self, max_length: int = 512):
    """Run greedy/beam-search generation on the evaluation set.

    The helper decodes the model outputs into strings, stores the results in
    ``eval_results.json`` inside the trainer's output directory and finally computes
    exact-match accuracy between the generated and reference sequences.

    Args:
        max_length (int, optional): Maximum generation length. Defaults to 512.

    Returns:
        float: Exact-match accuracy in the [0, 1] interval.
    """
    if self.eval_dataset is None:
        raise ValueError("Trainer: evaluation requires an eval_dataset.")

    all_generated_texts = []
    all_reference_texts = []

    eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

    self.model.eval()
    tokenizer = self.processing_class

    for batch in eval_dataloader:
        inputs = self._prepare_inputs(batch)
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
            labels[labels == -100] = tokenizer.pad_token_id
            current_reference_texts = tokenizer.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            all_reference_texts.extend(current_reference_texts)
        else:
            # Keep placeholder when reference labels are missing.
            all_reference_texts.extend(["" for _ in current_generated_texts])

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

### Utilities

Count the number of CUDA devices visible to the current process.

The function first inspects the environment variable `CUDA_VISIBLE_DEVICES`. When set, only the GPU indices listed there are considered visible and contribute to the count. When not set, the function falls back to `torch.cuda.device_count` and returns the total number of devices detected by the NVIDIA runtime.

Returns:

| Name | Type | Description | | --- | --- | --- | | `int` | `int` | Number of GPUs that the current process is allowed to use. 0 indicates no GPU | | | `int` | is available or that PyTorch was compiled without CUDA support. |

Source code in `src/calt/trainer/utils.py`

```
def count_cuda_devices() -> int:
    """Count the number of CUDA devices visible to the current process.

    The function first inspects the environment variable ``CUDA_VISIBLE_DEVICES``. When set,
    only the GPU indices listed there are considered visible and contribute to the count.
    When not set, the function falls back to ``torch.cuda.device_count`` and returns the
    total number of devices detected by the NVIDIA runtime.

    Returns:
        int: Number of GPUs that the current process is allowed to use. ``0`` indicates no GPU
        is available or that PyTorch was compiled without CUDA support.
    """

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible_devices is not None:
        # ``CUDA_VISIBLE_DEVICES`` is set – split on commas to extract the
        # list of allowed GPU indices (empty strings are filtered out).
        visible_devices = [d for d in cuda_visible_devices.split(",") if d]
        return len(visible_devices)

    # Variable not set – fall back to the total number detected by PyTorch.
    return torch.cuda.device_count()

```

Initialise a Weights & Biases tracking session.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `project` | `str` | Project name under which runs will appear in the WandB dashboard. Defaults to "transformer-algebra". | `'transformer-algebra'` | | `entity` | `str | None` | WandB entity (user or team) that owns the project. When None, the default entity configured in local WandB settings is used. | `None` | | `**extra_config` | | Additional key-value pairs inserted into the run configuration. Useful for hyper-parameter sweeps or ad-hoc experiments. | `{}` |

Source code in `src/calt/trainer/utils.py`

```
def setup_wandb(
    project: str = "transformer-algebra",
    entity: str | None = None,
    **extra_config,
) -> None:
    """Initialise a Weights & Biases tracking session.

    Args:
        project (str, optional): Project name under which runs will appear in the WandB dashboard.
            Defaults to ``"transformer-algebra"``.
        entity (str | None, optional): WandB entity (user or team) that owns the project.
            When ``None``, the default entity configured in local WandB settings is used.
        **extra_config: Additional key-value pairs inserted into the run configuration.
            Useful for hyper-parameter sweeps or ad-hoc experiments.
    """
    # Initialize wandb
    import wandb

    wandb.init(
        project=project,
        entity=entity,
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        },
    )

```
