"""Utility helpers for training scripts.

This module groups together small functions that are frequently reused across
training and evaluation scripts – for example, detecting the number of visible
CUDA devices or initialising the Weights & Biases tracking dashboard.
"""

import os
import torch
import re
from typing import Sequence, Union
from sympy import symbols, Symbol, Integer

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
        Additional key–value pairs that will be inserted into the run
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


def parse_poly(tokens: str, var_names: Sequence[Union[str, Symbol]] | None = None):
    """
    Convert an internal token sequence (e.g. ``"C1 E1 E1 C-3 E0 E7"``)
    into a SymPy polynomial.

    Parameters
    ----------
    tokens : str
        Whitespace-separated string where a token starting with ``C`` indicates
        a coefficient and the following ``E`` tokens indicate exponents.
    var_names : Sequence[str | sympy.Symbol] | None, optional
        Variable names (either strings or pre-created Symbol objects).  If
        ``None`` (default), variables are auto-generated as x0, x1, …

    Returns
    -------
    sympy.Expr
        A SymPy expression corresponding to the polynomial.

    Raises
    ------
    ValueError
        If the token sequence is malformed or the number of variables does not
        match ``var_names``.
    """
    parts = tokens.strip().split()
    if not parts or not parts[0].startswith("C"):
        raise ValueError("Token sequence must start with a 'C' coefficient token.")

    # --- Infer the number of variables from the first term ------------------ #
    try:
        next_c_idx = parts.index(next(p for p in parts[1:] if p.startswith("C")))
        n_vars = next_c_idx - 1
    except StopIteration:  # single-term polynomial
        n_vars = len([p for p in parts[1:] if p.startswith("E")])

    # --- Prepare SymPy symbols --------------------------------------------- #
    if var_names is None:
        vars_ = symbols(" ".join(f"x{i}" for i in range(n_vars)))
    else:
        if len(var_names) != n_vars:
            raise ValueError(f"Expected {n_vars} variable name(s), got {len(var_names)}.")
        # If elements are strings, create Symbols; if they are already Symbols, reuse them.
        if all(isinstance(v, str) for v in var_names):
            vars_ = symbols(" ".join(var_names))  # -> tuple(Symbol, …)
        elif all(isinstance(v, Symbol) for v in var_names):
            vars_ = tuple(var_names)
        else:
            raise TypeError("var_names must be all str or all sympy.Symbol.")

    expr = Integer(0)
    i = 0
    while i < len(parts):
        # Read coefficient
        coeff_token = parts[i]
        if not coeff_token.startswith("C"):
            raise ValueError(f"Expected 'C' token at position {i}, got {coeff_token}.")
        coeff = Integer(coeff_token[1:])
        i += 1

        # Read exponents
        exps = []
        for _ in range(n_vars):
            if i >= len(parts) or not parts[i].startswith("E"):
                raise ValueError(f"Missing 'E' token at position {i}.")
            exps.append(Integer(parts[i][1:]))
            i += 1

        # Build term
        term = coeff
        for v, e in zip(vars_, exps):
            term *= v**e
        expr += term

    return expr
