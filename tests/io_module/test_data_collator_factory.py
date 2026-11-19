"""IOPipeline(data_collator_factory=...): a user collator built from the tokenizer."""

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from calt.io import IOPipeline, StandardDataCollator, UserCallableLoadPreprocessor
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline

ROWS = ["1 2 # 3", "4 5 # 9", "2 2 # 4", "3 1 # 4"]


class CountingCollator(StandardDataCollator):
    """StandardDataCollator that records the tokenizer and every call."""

    def __init__(self, tokenizer=None, extra_label_spec=None):
        super().__init__(tokenizer=tokenizer, extra_label_spec=extra_label_spec)
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        return super().__call__(batch)


class PlainCollator:
    """A collator that is not a StandardDataCollator."""

    def __init__(self, tokenizer):
        self.inner = StandardDataCollator(tokenizer=tokenizer)

    def __call__(self, batch):
        return self.inner(batch)


@pytest.fixture
def data_cfg(tmp_path):
    for split in ("train", "test"):
        (tmp_path / f"{split}.txt").write_text("\n".join(ROWS) + "\n")
    lexer = tmp_path / "lexer.yaml"
    lexer.write_text(
        yaml.safe_dump(
            {
                "vocab": {"range": {"numbers": ["", 0, 9]}},
                "number_policy": {"digit_group": 1},
            }
        )
    )
    return OmegaConf.create(
        {
            "lexer_config": str(lexer),
            "train_dataset_path": str(tmp_path / "train.txt"),
            "test_dataset_path": str(tmp_path / "test.txt"),
        }
    )


def test_default_is_standard_data_collator(data_cfg):
    pipeline = IOPipeline.from_config(data_cfg)
    io = pipeline.build()
    assert type(io["data_collator"]) is StandardDataCollator
    assert pipeline.data_collator is io["data_collator"]


def test_from_config(data_cfg):
    pipeline = IOPipeline.from_config(data_cfg, data_collator_factory=CountingCollator)
    io = pipeline.build()
    collator = io["data_collator"]
    assert isinstance(collator, CountingCollator)
    assert pipeline.data_collator is collator
    assert collator.tokenizer is io["tokenizer"]


def test_constructor(data_cfg):
    base = IOPipeline.from_config(data_cfg)
    seen = {}

    def factory(tokenizer):
        seen["tokenizer"] = tokenizer
        return CountingCollator(tokenizer=tokenizer)

    pipeline = IOPipeline(
        train_dataset_path=data_cfg.train_dataset_path,
        test_dataset_path=data_cfg.test_dataset_path,
        vocab_config=base.vocab_config,
        preprocessor=base.preprocessor,
        data_collator_factory=factory,
    )
    io = pipeline.build()
    assert seen["tokenizer"] is io["tokenizer"]
    assert pipeline.data_collator is io["data_collator"]
    assert isinstance(io["data_collator"], CountingCollator)


def test_set_after_from_config(data_cfg):
    # Like dataset_load_preprocessor, the factory can be set before build().
    pipeline = IOPipeline.from_config(data_cfg)
    pipeline.data_collator_factory = CountingCollator
    assert isinstance(pipeline.build()["data_collator"], CountingCollator)


def test_trainer_calls_the_collator(data_cfg, tmp_path):
    io = IOPipeline.from_config(
        data_cfg, data_collator_factory=CountingCollator
    ).build()
    model = ModelPipeline.from_io_dict(
        OmegaConf.create(
            {
                "model_type": "generic",
                "num_encoder_layers": 1,
                "num_encoder_heads": 2,
                "num_decoder_layers": 1,
                "num_decoder_heads": 2,
                "d_model": 16,
                "encoder_ffn_dim": 32,
                "decoder_ffn_dim": 32,
                "max_sequence_length": 16,
            }
        ),
        io,
    ).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(
        OmegaConf.create(
            {
                "save_dir": str(tmp_path / "results"),
                "num_train_epochs": 1,
                "learning_rate": 5e-4,
                "seed": 0,
                "batch_size": 2,
                "test_batch_size": 2,
                "num_workers": 0,
                "wandb": {"no_wandb": True},
            }
        ),
        model,
        io,
    ).build()
    collator = io["data_collator"]
    assert trainer_pipeline.trainer.data_collator is collator
    assert collator.calls == 0
    # The dataloader may prepare a batch ahead, so only check that it was called.
    next(iter(trainer_pipeline.trainer.get_train_dataloader()))
    assert collator.calls >= 1


def _with_extra_labels(pipeline):
    def format_problem(problem, answer):
        return problem, answer, {"length_label": len(answer.split())}

    pipeline.dataset_load_preprocessor = UserCallableLoadPreprocessor(format_problem)
    return pipeline


def test_extra_label_spec_is_given_to_a_subclass(data_cfg):
    pipeline = _with_extra_labels(
        IOPipeline.from_config(data_cfg, data_collator_factory=CountingCollator)
    )
    io = pipeline.build()
    collator = io["data_collator"]
    assert collator.extra_label_spec is not None
    assert "length_label" in collator.extra_label_spec
    batch = collator([io["train_dataset"][0], io["train_dataset"][1]])
    assert batch["extra_targets"]["length_label"].tolist() == [1, 1]


def test_extra_label_spec_set_by_the_factory_is_kept(data_cfg):
    spec = {"length_label": ("scalar", torch.float32)}

    def factory(tokenizer):
        return StandardDataCollator(tokenizer=tokenizer, extra_label_spec=spec)

    pipeline = _with_extra_labels(
        IOPipeline.from_config(data_cfg, data_collator_factory=factory)
    )
    assert pipeline.build()["data_collator"].extra_label_spec is spec


def test_extra_labels_with_a_subclass_that_skips_super_init(data_cfg):
    class NoSuperInit(StandardDataCollator):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    pipeline = _with_extra_labels(
        IOPipeline.from_config(data_cfg, data_collator_factory=NoSuperInit)
    )
    io = pipeline.build()
    batch = io["data_collator"]([io["train_dataset"][0], io["train_dataset"][1]])
    assert batch["extra_targets"]["length_label"].tolist() == [1, 1]


def test_extra_labels_need_a_standard_data_collator(data_cfg):
    pipeline = _with_extra_labels(
        IOPipeline.from_config(data_cfg, data_collator_factory=PlainCollator)
    )
    with pytest.raises(TypeError, match="StandardDataCollator"):
        pipeline.build()


def test_plain_collator_without_extra_labels(data_cfg):
    io = IOPipeline.from_config(data_cfg, data_collator_factory=PlainCollator).build()
    batch = io["data_collator"]([io["train_dataset"][0], io["train_dataset"][1]])
    assert "input_ids" in batch
