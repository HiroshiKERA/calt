"""
Trainer pipeline for creating trainers from config.

This module provides a class-based interface similar to IOPipeline for creating
trainer instances from configuration files.
"""

import importlib.metadata as importlib_metadata
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from ..io.base import StandardDataCollator
from ..models.losses import RESERVED_LOSS_ARGS
from .trainer import Trainer

logger = logging.getLogger(__name__)

# Present on Linux whenever the NVIDIA kernel driver is loaded
_NVIDIA_DRIVER_FILE = Path("/proc/driver/nvidia/version")

# Inside a conda environment, PyPI's CUDA build of torch loads the system
# libstdc++, which can be too old for conda-forge packages (see the README)
_PYPI_TORCH_NOTE = (
    "(PyPI's CUDA build of torch can break conda-forge packages such as "
    "SciPy; see the README)"
)


def _in_conda_env() -> bool:
    """Return True if this Python runs from a conda environment."""
    # sys.prefix rather than $CONDA_PREFIX, which may name another env
    return (Path(sys.prefix) / "conda-meta").is_dir()


def _torch_installer() -> Optional[str]:
    """Return the tool that installed torch ("conda", "pip", "uv", ...)."""
    try:
        installer = importlib_metadata.distribution("torch").read_text("INSTALLER")
    except (importlib_metadata.PackageNotFoundError, OSError):
        return None
    return installer.strip().lower() if installer else None


def _check_training_device(device: torch.device) -> None:
    """Log the training device, and warn if an NVIDIA GPU sits unused.

    torch only emits a one-time UserWarning when it cannot use the driver,
    and the HF Trainer then quietly trains on the CPU. The usual cause is a
    torch build for a newer CUDA than the driver supports: on Linux, PyPI's
    torch 2.11 and later is built for CUDA 13, which needs driver >= 580.
    (WSL2 has no /proc/driver/nvidia, so this check stays silent there.)
    In a conda environment the advice points at conda-forge's PyTorch,
    since a pip-installed CUDA torch can break conda-forge packages there.
    """
    logger.info("Training device: %s", device)
    # With CUDA available, the CPU was chosen on purpose (e.g. use_cpu)
    if device.type != "cpu" or torch.cuda.is_available():
        return
    # GPUs hidden on purpose: "", "-1", SLURM's "NoDevFiles", ...
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (
        visible.strip() == "-1" or not re.search(r"\d|GPU-|MIG-", visible)
    ):
        return
    # A ROCm build is not meant to use the NVIDIA driver
    if getattr(torch.version, "hip", None):
        return
    try:
        driver_info = _NVIDIA_DRIVER_FILE.read_text()
    except OSError:
        return
    # e.g. "NVRM version: NVIDIA UNIX x86_64 Kernel Module  555.42.06  ..."
    # or "... Open Kernel Module for x86_64  570.86.15  ..."
    lines = driver_info.splitlines()
    match = re.search(r"\b(\d{3,}\.\d+(?:\.\d+)?)\b", lines[0] if lines else "")
    driver = match.group(1) if match else "unknown version"
    conda_env = _in_conda_env()
    # pip and uv always record themselves; anything else (conda, a source
    # build, no record) is treated as conda's and left to `conda install`
    pip_torch = conda_env and _torch_installer() in ("pip", "uv")

    if torch.version.cuda is None:
        if not conda_env:
            advice = (
                "Install a CUDA build of torch "
                "(see https://pytorch.org/get-started/locally/)."
            )
        elif pip_torch:
            advice = (
                "In this conda environment, torch was installed from PyPI; "
                "replace it with the CUDA build from conda-forge: "
                "`pip uninstall torch`, then "
                '`conda install -c conda-forge "pytorch>=2.6=cuda*"` '
                f"{_PYPI_TORCH_NOTE}."
            )
        else:
            advice = (
                "In this conda environment, install the CUDA build of PyTorch "
                "from conda-forge: "
                '`conda install -c conda-forge "pytorch>=2.6=cuda*"`.'
            )
        logger.warning(
            "An NVIDIA GPU is present (driver %s), but torch %s is a CPU-only "
            "build, so training runs on the CPU. %s",
            driver,
            torch.__version__,
            advice,
        )
        return
    driver_major = int(driver.split(".")[0]) if match else None
    cuda_major = int(torch.version.cuda.split(".")[0])
    if cuda_major >= 13 and driver_major is not None and driver_major < 580:
        if not conda_env:
            advice = (
                "On Linux, PyPI's torch 2.11 and later is built for CUDA 13. "
                "Upgrade the driver, or install a CUDA 12 build of torch: "
                '`pip install "torch<2.11"` (Python 3.11-3.14) or '
                "`pip install torch==2.14.1 --index-url "
                "https://download.pytorch.org/whl/cu126` (any Python, including 3.15)."
            )
        elif pip_torch:
            advice = (
                "On Linux, PyPI's torch 2.11 and later is built for CUDA 13. "
                "Upgrade the driver, or replace the PyPI torch with a "
                "CUDA 12 build from conda-forge: `pip uninstall torch`, then "
                '`conda install -c conda-forge "pytorch>=2.6=cuda12*"` '
                f"{_PYPI_TORCH_NOTE}."
            )
        else:
            advice = (
                "Upgrade the driver, or install a CUDA 12 build of PyTorch from "
                "conda-forge in this conda environment: "
                '`conda install -c conda-forge "pytorch>=2.6=cuda12*"`.'
            )
        logger.warning(
            "An NVIDIA GPU is present (driver %s), but torch %s is built for "
            "CUDA %s, which needs NVIDIA driver >= 580, so training runs on the "
            "CPU. %s",
            driver,
            torch.__version__,
            torch.version.cuda,
            advice,
        )
        return
    logger.warning(
        "An NVIDIA GPU is present (driver %s), but torch %s (built for CUDA %s) "
        "cannot use it, so training runs on the CPU. Check that `nvidia-smi` "
        "works, and in a container, that the GPU is passed through.",
        driver,
        torch.__version__,
        torch.version.cuda,
    )


def _check_extra_labels_reach_a_loss(model, train_dataset) -> None:
    """Fail before training when extra labels would be dropped silently."""
    extra = getattr(train_dataset, "extra_labels", None)
    if not extra:
        return
    names = sorted(extra[0])
    spec = getattr(model, "calt_loss", None)
    if spec is None:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        raise ValueError(
            f"The data has extra labels {names}, but model type {model_type!r} "
            "does not pass them to a loss; use generic, decoder_only or "
            "encoder_classifier."
        )
    if spec.is_default:
        logger.warning(
            "The data has extra labels %s, but model.loss_type is the default "
            "cross_entropy, which does not use them.",
            names,
        )
    clash = sorted(set(names) & (RESERVED_LOSS_ARGS | set(spec.loss_kwargs)))
    if clash:
        raise ValueError(
            f"Extra labels {clash} clash with the loss arguments or "
            "model.loss_kwargs; rename them."
        )


class TrainerPipeline:
    """Pipeline for creating trainers from configuration.

    Similar to IOPipeline, this class provides a simple interface for creating
    trainer instances from config files. It automatically selects the appropriate
    TrainerLoader based on the config.

    Examples:
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
    """

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

    def _configure_wandb(self) -> None:
        """Configure wandb-related settings (env vars and training config)."""
        if self.wandb_config is None:
            return

        wb = self.wandb_config
        no_wandb = getattr(wb, "no_wandb", False)

        if no_wandb:
            # Import here to avoid circular import
            from .loader import report_to_without_wandb

            # report_to is what keeps the WandbCallback away (transformers v5
            # ignores WANDB_DISABLED); the variable still stops a callback
            # added by hand on v4.
            os.environ["WANDB_DISABLED"] = "true"
            self.config.report_to = report_to_without_wandb(
                self.config.get("report_to")
            )
            return

        os.environ.pop("WANDB_DISABLED", None)
        for key, env in [
            ("project", "WANDB_PROJECT"),
            ("group", "WANDB_RUN_GROUP"),
            ("name", "WANDB_NAME"),
        ]:
            value = getattr(wb, key, None)
            if value is not None:
                os.environ[env] = str(value)

        # Attach wandb config to training config so TrainerLoader can inspect it
        setattr(self.config, "wandb", wb)

        # Make sure Trainer sees wandb settings
        setattr(self.config, "report_to", getattr(self.config, "report_to", "wandb"))

    def build(self) -> "TrainerPipeline":
        """Build the trainer from configuration.

        Returns:
            TrainerPipeline: Returns self for method chaining.
        """
        # Configure wandb before building TrainingArguments / Trainer
        self._configure_wandb()
        _check_extra_labels_reach_a_loss(self.model, self.train_dataset)

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
        _check_training_device(self.trainer.args.device)
        return self

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

    def evaluate_and_save_generation(self, max_length: int | None = None) -> float:
        """Evaluate and save generation results.

        ``max_length`` defaults to 512, capped at the model's positional
        capacity (see ``Trainer.evaluate_and_save_generation``).
        """
        if self.trainer is None:
            raise ValueError("Trainer not built. Call build() first.")
        return self.trainer.evaluate_and_save_generation(max_length=max_length)

    def __getattr__(self, name: str):
        """Delegate attribute access to trainer if not found in TrainerPipeline.

        This allows calling trainer methods directly on TrainerPipeline, e.g.:
        - trainer_pipeline.evaluate_and_save_generation()
        - trainer_pipeline.evaluate()
        - etc.
        """
        if self.trainer is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                "Trainer not built. Call build() first."
            )
        return getattr(self.trainer, name)

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
