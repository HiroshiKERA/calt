"""Tests for the warning about an NVIDIA GPU that training cannot use."""

import logging

import pytest
import torch

from calt.trainer import pipeline as pipeline_module
from calt.trainer.pipeline import _check_training_device

DRIVER_INFO = (
    "NVRM version: NVIDIA UNIX x86_64 Kernel Module  555.42.06  "
    "Tue Jun  4 00:59:30 UTC 2024\n"
    "GCC version:  gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04.3)\n"
)
OPEN_DRIVER_INFO = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.86.15  "
    "Release Build  (dvs-builder@U16-I3-B03-4-3)\n"
)


@pytest.fixture
def unusable_gpu(tmp_path, monkeypatch):
    """Pretend the NVIDIA driver is loaded but torch cannot use CUDA."""
    driver_file = tmp_path / "version"
    driver_file.write_text(DRIVER_INFO)
    monkeypatch.setattr(pipeline_module, "_NVIDIA_DRIVER_FILE", driver_file)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return driver_file


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warns_when_driver_is_too_old_for_cuda_13(unusable_gpu, caplog):
    with caplog.at_level(logging.INFO, logger=pipeline_module.__name__):
        _check_training_device(torch.device("cpu"))
    (message,) = _warnings(caplog)
    assert "driver 555.42.06" in message
    assert "CUDA 13.0" in message
    assert "driver >= 580" in message
    assert "Training device: cpu" in caplog.text


def test_reads_open_kernel_module_version(unusable_gpu, monkeypatch, caplog):
    unusable_gpu.write_text(OPEN_DRIVER_INFO)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    _check_training_device(torch.device("cpu"))
    (message,) = _warnings(caplog)
    assert "driver 570.86.15" in message
    # CUDA 12 runs on driver 570, so the cause lies elsewhere
    assert "driver >= 580" not in message
    assert "nvidia-smi" in message


def test_new_driver_gets_generic_advice(unusable_gpu, monkeypatch, caplog):
    unusable_gpu.write_text(DRIVER_INFO.replace("555.42.06", "580.65.06"))
    _check_training_device(torch.device("cpu"))
    (message,) = _warnings(caplog)
    assert "driver >= 580" not in message
    assert "nvidia-smi" in message


def test_warns_about_cpu_only_torch(unusable_gpu, monkeypatch, caplog):
    monkeypatch.setattr(torch.version, "cuda", None)
    _check_training_device(torch.device("cpu"))
    (message,) = _warnings(caplog)
    assert "CPU-only build" in message


@pytest.mark.parametrize("visible", ["", "-1", "NoDevFiles", " "])
def test_no_warning_when_gpus_hidden_on_purpose(
    unusable_gpu, monkeypatch, caplog, visible
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    _check_training_device(torch.device("cpu"))
    assert _warnings(caplog) == []


def test_no_warning_when_cuda_works_but_cpu_was_chosen(
    unusable_gpu, monkeypatch, caplog
):
    # e.g. use_cpu=True or ACCELERATE_USE_CPU=1 on a working GPU machine
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _check_training_device(torch.device("cpu"))
    assert _warnings(caplog) == []


def test_no_warning_for_rocm_build(unusable_gpu, monkeypatch, caplog):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "6.2")
    _check_training_device(torch.device("cpu"))
    assert _warnings(caplog) == []


def test_no_warning_without_nvidia_driver(unusable_gpu, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pipeline_module, "_NVIDIA_DRIVER_FILE", tmp_path / "missing")
    _check_training_device(torch.device("cpu"))
    assert _warnings(caplog) == []


def test_no_warning_when_training_on_gpu(unusable_gpu, caplog):
    _check_training_device(torch.device("cuda", 0))
    assert _warnings(caplog) == []
