# Model pipeline

`ModelPipeline` builds a sequence-to-sequence model from the `model` block of your config and the tokenizer produced by [IOPipeline](io_pipeline.md). It is used after `IOPipeline.build()` and before [TrainerPipeline](trainer.md).

- [Overview](trainer.md) — how the three pipelines (IO, Model, Trainer) fit together.
- [Configuration](configuration.md) — the `model` block in `train.yaml` and its keys.

## ModelPipeline

Use `ModelPipeline.from_io_dict(cfg.model, io_dict)` to create a pipeline from the result of `IOPipeline.from_config(cfg.data).build()`. The tokenizer is taken from `io_dict["tokenizer"]`. Call `.build()` to obtain the `PreTrainedModel` instance.

::: calt.models.pipeline.ModelPipeline
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3

## Supported model types

Models are created via an internal `ModelRegistry`. The following types are registered by default:

| `model_type` | Description |
|--------------|-------------|
| `generic`, `transformer`, `calt` | CALT generic Transformer (encoder–decoder). |
| `bart` | HuggingFace BART for conditional generation. |

Set `model_type` in the `model` block of `train.yaml` (e.g. `model_type: generic`). Other keys in the `model` block (e.g. `num_encoder_layers`, `d_model`, `max_sequence_length`) are documented under [Configuration — `model`](configuration.md#trainyaml--model-and-training-modelpipeline-trainerpipeline).


## ModelRegistry

To create a model without using the pipeline (e.g. with a custom config), you can use the registry or helpers from `calt.models`: `ModelRegistry`, `get_model_from_config`. See the API reference below.

::: calt.models.base.ModelRegistry
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - __init__
        - create_from_config
        - list_models
        - register
        - register_config_mapping
