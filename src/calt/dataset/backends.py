# backends.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Type

@dataclass(frozen=True)
class BackendSpec:
    load: Callable[[], tuple[type, type]]  # (DatasetGenerator, DatasetWriter)

def _load_sagemath():
    from calt.dataset_generator.sagemath import DatasetGenerator, DatasetWriter
    return DatasetGenerator, DatasetWriter

def _load_sympy():
    from calt.dataset_generator.sympy import DatasetGenerator, DatasetWriter
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
