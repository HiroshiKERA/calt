"""Tests that ``no_wandb`` and ``apply_dryrun_settings`` keep wandb off.

No model is built: the tests stop at TrainingArguments, which is where
``report_to`` decides whether the WandbCallback is attached.
"""

import pytest
from omegaconf import ListConfig, OmegaConf

from calt.trainer import TrainerPipeline, apply_dryrun_settings
from calt.trainer import loader as loader_module
from calt.trainer.loader import StandardTrainerLoader, report_to_without_wandb


def _train_section(tmp_path, **extra):
    return {
        "num_train_epochs": 1,
        "learning_rate": 1e-3,
        "batch_size": 8,
        "test_batch_size": 8,
        "save_dir": str(tmp_path),
        **extra,
    }


@pytest.fixture(autouse=True)
def _restore_wandb_env(monkeypatch):
    # _configure_wandb writes these into os.environ. setenv records the
    # original state so it is restored after each test even when the
    # variable was unset (delenv alone records nothing then).
    for name in ("WANDB_DISABLED", "WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_NAME"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _training_args(cfg, wandb_config):
    """Run the wandb setup of TrainerPipeline and return its TrainingArguments."""
    TrainerPipeline(
        config=cfg.train, model=None, wandb_config=wandb_config
    )._configure_wandb()
    return StandardTrainerLoader(calt_config=cfg.train).translate_config()


def test_dryrun_disables_train_level_wandb():
    cfg = OmegaConf.create(
        {
            "train": {"wandb": {"name": "task", "group": "grp", "no_wandb": False}},
            "data": {},
        }
    )
    apply_dryrun_settings(cfg)
    assert cfg.train.wandb.no_wandb is True
    assert cfg.train.wandb.name == "dryrun-task"
    assert cfg.train.wandb.group == "dryrun-grp"


def test_dryrun_disables_every_wandb_block():
    cfg = OmegaConf.create(
        {
            "wandb": {},
            "train": {"wandb": {"name": "dryrun-task", "no_wandb": False}},
            "data": {},
        }
    )
    apply_dryrun_settings(cfg)
    assert cfg.wandb.no_wandb is True
    assert cfg.train.wandb.no_wandb is True
    # An existing "dryrun-" prefix is not doubled
    assert cfg.train.wandb.name == "dryrun-task"


def test_dryrun_creates_disabled_block_when_missing():
    cfg = OmegaConf.create({"train": {}, "data": {}})
    apply_dryrun_settings(cfg)
    assert cfg.train.wandb.no_wandb is True
    assert cfg.train.wandb.name == "dryrun"


@pytest.mark.parametrize(
    "report_to,expected",
    [
        (None, "none"),
        ("none", "none"),
        ("wandb", "none"),
        ([], "none"),
        ("tensorboard", ["tensorboard"]),
        (["wandb", "tensorboard"], ["tensorboard"]),
        (ListConfig(["wandb", "tensorboard"]), ["tensorboard"]),
    ],
)
def test_report_to_without_wandb(report_to, expected):
    assert report_to_without_wandb(report_to) == expected


def test_report_to_without_wandb_expands_all(monkeypatch):
    monkeypatch.setattr(
        loader_module,
        "get_available_reporting_integrations",
        lambda: ["wandb", "tensorboard"],
    )
    assert report_to_without_wandb("all") == ["tensorboard"]


@pytest.mark.parametrize("location", ["train", "top"])
@pytest.mark.parametrize(
    "report_to", [None, "wandb", ["wandb", "tensorboard"]], ids=str
)
def test_no_wandb_removes_wandb_from_training_args(tmp_path, location, report_to):
    extra = {} if report_to is None else {"report_to": report_to}
    cfg = OmegaConf.create({"train": _train_section(tmp_path, **extra)})
    block = OmegaConf.create({"project": "calt", "no_wandb": True})
    if location == "train":
        cfg.train.wandb = block
        wandb_config = cfg.train.wandb
    else:
        # Top-level blocks reach the pipeline through wandb_config only,
        # as in load_from_checkpoint
        cfg.wandb = block
        wandb_config = cfg.wandb

    report = _training_args(cfg, wandb_config).report_to
    assert "wandb" not in report
    if isinstance(report_to, list):
        assert "tensorboard" in report


def test_enabled_wandb_still_reports_to_wandb(tmp_path):
    cfg = OmegaConf.create(
        {
            "train": _train_section(
                tmp_path,
                wandb={"project": "calt", "name": "task", "no_wandb": False},
            )
        }
    )
    args = _training_args(cfg, cfg.train.wandb)
    assert args.report_to == ["wandb"]
    assert args.run_name == "task"


def test_dryrun_keeps_wandb_out_of_training_args(tmp_path):
    # Same layout as examples/*/configs/train.yaml with no_wandb: False
    cfg = OmegaConf.create(
        {
            "train": _train_section(
                tmp_path,
                wandb={"project": "calt", "name": "task", "no_wandb": False},
            ),
            "data": {},
        }
    )
    apply_dryrun_settings(cfg)
    args = _training_args(cfg, cfg.train.wandb)
    assert "wandb" not in args.report_to
    assert args.run_name == "dryrun-task"


def test_dryrun_with_only_top_level_wandb_block(tmp_path):
    cfg = OmegaConf.create(
        {
            "wandb": {"project": "calt", "no_wandb": False},
            "train": _train_section(tmp_path),
            "data": {},
        }
    )
    apply_dryrun_settings(cfg)
    # examples/*/train.py call TrainerPipeline.from_io_dict(cfg.train, ...),
    # which reads cfg.train.wandb only
    assert "wandb" not in _training_args(cfg, cfg.train.wandb).report_to
