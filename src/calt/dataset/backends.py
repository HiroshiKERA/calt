# backends.py
"""Dataset generation backends: registry and lazy loading of DatasetGenerator and DatasetWriter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BackendSpec:
    """Specification for a backend: a loader that returns (DatasetGenerator, DatasetWriter) classes."""

    load: Callable[[], tuple[type, type]]  # (DatasetGenerator, DatasetWriter)


def _load_sagemath():
    """Load SageMath-backed DatasetGenerator and DatasetWriter (lazy import)."""
    from calt.dataset.sagemath.dataset_generator import DatasetGenerator
    from calt.dataset.utils.dataset_writer import DatasetWriter

    return DatasetGenerator, DatasetWriter


def _load_sympy():
    """Load SymPy-backed DatasetGenerator and DatasetWriter (lazy import)."""
    from calt.dataset.sympy.dataset_generator import DatasetGenerator
    from calt.dataset.utils.dataset_writer import DatasetWriter

    return DatasetGenerator, DatasetWriter


# Registry of backend names to BackendSpec. Keys: "sagemath", "sympy".
BACKENDS: dict[str, BackendSpec] = {
    "sagemath": BackendSpec(load=_load_sagemath),
    "sympy": BackendSpec(load=_load_sympy),
}


def get_backend_classes(name: str) -> tuple[type, type]:
    """Resolve a backend name to (DatasetGenerator, DatasetWriter) class pair.

    Args:
        name: Backend name; must be a key in BACKENDS (e.g. \"sagemath\", \"sympy\").

    Returns:
        Tuple of (DatasetGenerator class, DatasetWriter class) for the backend.
        Classes are loaded lazily on first use.

    Raises:
        ValueError: If name is not a registered backend.
    """
    try:
        return BACKENDS[name].load()
    except KeyError:
        raise ValueError(f"Invalid backend: {name}. Choose from {list(BACKENDS)}")
