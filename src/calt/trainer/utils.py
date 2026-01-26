"""Utility functions for trainer operations."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedModel

from ..io.pipeline import IOPipeline
from ..models import ModelPipeline
from .pipeline import TrainerPipeline


def apply_dryrun_settings(cfg: DictConfig) -> None:
    """Apply dryrun settings to config for quick testing.

    This function modifies the config in-place to reduce training time and resource usage:
    - Sets training epochs to 1
    - Reduces batch sizes to 8 (if larger)
    - Disables multiprocessing (num_workers = 0)
    - Marks wandb run name/group with "_dryrun" suffix (keeps wandb enabled)
    - Limits dataset size (1000 train samples, 100 test samples)

    Args:
        cfg: OmegaConf DictConfig containing train, data, and wandb sections.
    """
    # Reduce training epochs
    cfg.train.num_train_epochs = 1

    # Reduce batch sizes if needed (keep original if already small)
    if not hasattr(cfg.train, "batch_size") or cfg.train.batch_size > 8:
        cfg.train.batch_size = 8
    if not hasattr(cfg.train, "test_batch_size") or cfg.train.test_batch_size > 8:
        cfg.train.test_batch_size = 8

    # Disable multiprocessing for faster startup
    cfg.train.num_workers = 0

    # Adjust eval/save steps for dryrun to ensure evaluation runs during training
    # With 1000 samples and batch_size=8, we get ~125 steps per epoch
    # Set eval_steps and save_steps to a small value (e.g., 10) so evaluation runs
    cfg.train.eval_steps = 10
    cfg.train.save_steps = 10
    # Ensure eval_strategy and save_strategy are set to "steps" if not already set
    if not hasattr(cfg.train, "eval_strategy"):
        cfg.train.eval_strategy = "steps"
    if not hasattr(cfg.train, "save_strategy"):
        cfg.train.save_strategy = "steps"

    # Configure wandb for dryrun
    # wandb config can be in cfg.wandb or cfg.train.wandb
    wandb_cfg = None
    if hasattr(cfg, "wandb"):
        wandb_cfg = cfg.wandb
    elif hasattr(cfg.train, "wandb"):
        wandb_cfg = cfg.train.wandb

    if wandb_cfg is not None:
        # Keep existing project/group but mark as dryrun
        original_name = getattr(wandb_cfg, "name", "dryrun")
        original_group = getattr(wandb_cfg, "group", "dryrun")
        # Add "dryrun-" prefix to name if not already present
        if not original_name.startswith("dryrun-"):
            wandb_cfg.name = f"dryrun-{original_name}"
        else:
            wandb_cfg.name = original_name
        # Add "dryrun-" prefix to group if not already present
        if not original_group.startswith("dryrun-"):
            wandb_cfg.group = f"dryrun-{original_group}"
        else:
            wandb_cfg.group = original_group
        # Keep wandb enabled but mark run name/group as dryrun
        if hasattr(wandb_cfg, "no_wandb"):
            wandb_cfg.no_wandb = False
    else:
        # Create wandb config if it doesn't exist
        if not hasattr(cfg.train, "wandb"):
            cfg.train.wandb = OmegaConf.create({})
        cfg.train.wandb.project = "calt"
        cfg.train.wandb.group = "dryrun"
        cfg.train.wandb.name = "dryrun"
        cfg.train.wandb.no_wandb = False

    # Limit dataset size for quick testing
    # IOPipeline uses -1 to load all samples, so set specific limits for dryrun
    cfg.data.num_train_samples = 1000  # Limit to 1000 training samples
    cfg.data.num_test_samples = 100  # Limit to 100 test samples


def count_cuda_devices() -> int:
    """Count the number of available CUDA devices.

    Returns:
        int: Number of CUDA devices available. Returns 1 if CUDA is not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1
    except ImportError:
        return 1


def load_from_checkpoint(
    save_dir: str,
    resume_from_checkpoint: bool = True,
) -> tuple[IOPipeline, PreTrainedModel, TrainerPipeline]:
    """Load IOPipeline, model, and TrainerPipeline from a saved checkpoint directory.

    This function loads train.yaml from save_dir, reconstructs IOPipeline, ModelPipeline,
    and TrainerPipeline, and optionally loads the saved model and tokenizer.

    Args:
        save_dir: Directory containing train.yaml, model/, and tokenizer/.
        resume_from_checkpoint: If True, load saved model and tokenizer. If False, create new model.

    Returns:
        tuple: (io_pipeline, model, trainer_pipeline) ready for training continuation.

    Example:
        >>> from calt.trainer.utils import load_from_checkpoint
        >>>
        >>> # Load from checkpoint and continue training
        >>> io_pipeline, model, trainer_pipeline = load_from_checkpoint("./results")
        >>> trainer_pipeline.build()
        >>> trainer_pipeline.train()  # Continue training
    """
    save_dir_path = Path(save_dir)
    if not save_dir_path.exists():
        raise ValueError(f"Save directory does not exist: {save_dir}")

    # Load train.yaml
    train_yaml_path = save_dir_path / "train.yaml"
    if not train_yaml_path.exists():
        raise ValueError(f"train.yaml not found in {save_dir}")

    cfg = OmegaConf.load(train_yaml_path)

    # Build IOPipeline from config
    io_pipeline = IOPipeline.from_config(cfg.data)
    io_dict = io_pipeline.build()

    # Load or create model
    if resume_from_checkpoint:
        # Load saved model and tokenizer
        # HuggingFace Trainer saves to output_dir/model and output_dir/tokenizer
        # But we may also save directly to save_dir
        model_dir = save_dir_path / "model"
        tokenizer_dir = save_dir_path / "tokenizer"

        # If model/ and tokenizer/ subdirectories don't exist, try save_dir directly
        if not model_dir.exists():
            # Check if model files are directly in save_dir
            if (
                (save_dir_path / "config.json").exists()
                or (save_dir_path / "pytorch_model.bin").exists()
                or (save_dir_path / "model.safetensors").exists()
            ):
                model_dir = save_dir_path
            else:
                raise ValueError(f"Model directory not found: {model_dir}")
        if not tokenizer_dir.exists():
            # Check if tokenizer files are directly in save_dir
            if (save_dir_path / "tokenizer_config.json").exists() or (
                save_dir_path / "vocab.json"
            ).exists():
                tokenizer_dir = save_dir_path
            else:
                raise ValueError(f"Tokenizer directory not found: {tokenizer_dir}")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)

        # Determine model type from config
        model_type = cfg.model.get("model_type", "bart").lower()
        if model_type == "bart":
            from transformers import BartForConditionalGeneration

            model = BartForConditionalGeneration.from_pretrained(str(model_dir))
        elif model_type in ("generic", "transformer", "calt"):
            from ..models.generic.model import Transformer

            model = Transformer.from_pretrained(str(model_dir))
        else:
            # Fallback: try to load using ModelRegistry
            from ..models.base import ModelRegistry

            registry = ModelRegistry()
            if model_type in registry.list_models():
                model_class, _ = registry._registry[model_type]
                model = model_class.from_pretrained(str(model_dir))
            else:
                supported_types = registry.list_models()
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported types: {supported_types}"
                )

        # Update tokenizer in io_dict
        io_dict["tokenizer"] = tokenizer
    else:
        # Create new model from config
        model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()

    # Create TrainerPipeline
    trainer_pipeline = TrainerPipeline.from_io_dict(
        config=cfg.train,
        model=model,
        io_dict=io_dict,
        wandb_config=cfg.get("wandb"),
    )

    return io_pipeline, model, trainer_pipeline
