"""Tests that lr_scheduler_type and lr_scheduler_kwargs reach TrainingArguments."""

import pytest
from omegaconf import OmegaConf
from transformers.trainer_utils import SchedulerType

from calt.trainer.loader import StandardTrainerLoader


def _training_args(tmp_path, **extra):
    cfg = OmegaConf.create(
        {
            "num_train_epochs": 1,
            "learning_rate": 1e-3,
            "batch_size": 8,
            "test_batch_size": 8,
            "save_dir": str(tmp_path),
            **extra,
        }
    )
    return StandardTrainerLoader(calt_config=cfg).translate_config()


def test_default_scheduler_is_linear(tmp_path):
    assert _training_args(tmp_path).lr_scheduler_type == SchedulerType.LINEAR


def test_null_scheduler_is_linear(tmp_path):
    args = _training_args(tmp_path, lr_scheduler_type=None)
    assert args.lr_scheduler_type == SchedulerType.LINEAR


@pytest.mark.parametrize(
    "name",
    [
        "linear",
        "constant",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant_with_warmup",
        "inverse_sqrt",
    ],
)
def test_scheduler_name_is_passed_through(tmp_path, name):
    # Before the fix, every name except "constant" silently became "linear"
    args = _training_args(tmp_path, lr_scheduler_type=name)
    assert args.lr_scheduler_type == SchedulerType(name)


def test_unknown_scheduler_name_raises(tmp_path):
    with pytest.raises(ValueError):
        _training_args(tmp_path, lr_scheduler_type="cosin")


def test_scheduler_kwargs_are_passed_as_dict(tmp_path):
    args = _training_args(
        tmp_path,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
    )
    assert args.lr_scheduler_type == SchedulerType.COSINE_WITH_MIN_LR
    assert args.lr_scheduler_kwargs == {"min_lr_rate": 0.1}
    assert type(args.lr_scheduler_kwargs) is dict


def test_missing_scheduler_kwargs_raise_before_training(tmp_path):
    with pytest.raises(ValueError, match="train.lr_scheduler_kwargs"):
        _training_args(tmp_path, lr_scheduler_type="cosine_with_min_lr")


def test_scheduler_kwargs_must_be_a_mapping(tmp_path):
    with pytest.raises(TypeError, match="must be a mapping"):
        _training_args(tmp_path, lr_scheduler_kwargs=[0.1])
