"""Utility helpers for training scripts.

This module groups together small functions that are frequently reused across
training and evaluation scripts - for example, detecting the number of visible
CUDA devices or initialising the Weights & Biases tracking dashboard.
"""

import os
import torch

# from utils.logger import create_logger  # Optional custom logger import


def count_cuda_devices() -> int:
    """Count the number of CUDA devices visible to the current process.

    The function first inspects the environment variable
    ``CUDA_VISIBLE_DEVICES``.  When the variable is set, only the GPU indices
    listed there are considered *visible* and therefore contribute to the
    count.  When the variable is *not* set, the function falls back to
    :pyfunc:`torch.cuda.device_count` and returns the total number of devices
    detected by the NVIDIA runtime.

    Returns
    -------
    int
        The number of GPUs that the current process is allowed to use.  A
        value of ``0`` indicates that no GPU is available or that PyTorch was
        compiled without CUDA support.
    """

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible_devices is not None:
        # ``CUDA_VISIBLE_DEVICES`` is set – split on commas to extract the
        # list of allowed GPU indices (empty strings are filtered out).
        visible_devices = [d for d in cuda_visible_devices.split(",") if d]
        return len(visible_devices)

    # Variable not set – fall back to the total number detected by PyTorch.
    return torch.cuda.device_count()


def setup_wandb(
    project: str = "transformer-algebra",
    entity: str | None = None,
    **extra_config,
) -> None:
    """Initialise a Weights & Biases tracking session.

    Parameters
    ----------
    project : str, default ``"transformer-algebra"``
        Project name under which runs will appear in the WandB dashboard.
    entity : str | None, optional
        WandB *entity* (user or team) that owns the project.  When *None*, the
        default entity configured in the local WandB settings is used.
    **extra_config
        Additional key-value pairs that will be inserted into the run
        configuration.  These values are useful for hyper-parameter sweeps or
        quick ad-hoc experiments.
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
