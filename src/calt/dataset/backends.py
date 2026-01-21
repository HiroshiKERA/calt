# backends.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BackendSpec:
    load: Callable[[], tuple[type, type]]  # (DatasetGenerator, DatasetWriter)

def _load_sagemath():
    from calt.dataset.sagemath.dataset_generator import DatasetGenerator
    from calt.dataset.sagemath.utils.dataset_writer import DatasetWriter
    return DatasetGenerator, DatasetWriter

def _load_sympy():
    from calt.dataset.sympy.dataset_generator import DatasetGenerator
    from calt.dataset.sympy.utils.dataset_writer import DatasetWriter
    return DatasetGenerator, DatasetWriter

BACKENDS: dict[str, BackendSpec] = {
    "sagemath": BackendSpec(load=_load_sagemath),
    "sympy": BackendSpec(load=_load_sympy),
}

def get_backend_classes(name: str):
    try:
        return BACKENDS[name].load()
    except KeyError:
        raise ValueError(f"Invalid backend: {name}. Choose from {list(BACKENDS)}")
