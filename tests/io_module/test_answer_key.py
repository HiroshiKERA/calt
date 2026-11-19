"""A falsy answer (0, "", []) in JSONL is a real answer, not a missing one.

The answer is taken from "answer" unless it is missing or null; only then is the
older "solution" key used.
"""

import json

import pytest
import yaml
from omegaconf import OmegaConf

from calt.dataset.utils.dataset_writer import DatasetWriter
from calt.io import IOPipeline, UserCallableLoadPreprocessor
from calt.io.preprocessor.load_preprocessor import (
    JsonlDefaultLoadPreprocessor,
    PickleDefaultLoadPreprocessor,
    _get_answer_from_source,
)
from calt.io.read import read_data_from_jsonl

# (JSONL object, answer expected from it)
CASES = [
    ({"problem": [0, 0], "answer": 0}, 0),
    ({"problem": "a", "answer": ""}, ""),
    ({"problem": "a", "answer": []}, []),
    ({"problem": "a", "answer": 0.0}, 0.0),
    ({"problem": "a", "answer": False}, False),
    ({"problem": "a", "solution": "5"}, "5"),
    ({"problem": "a", "solution": 0}, 0),
    ({"problem": "a", "answer": 0, "solution": "9"}, 0),
    ({"problem": "a", "answer": None, "solution": "5"}, "5"),
]
MISSING = [
    {"problem": "a"},
    {"problem": "a", "answer": None},
    {"problem": "a", "answer": None, "solution": None},
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_get_answer_from_source(source, expected):
    answer = _get_answer_from_source(source)
    # Also compare types: 0 == False == 0.0 would hide a converted value.
    assert answer == expected and type(answer) is type(expected)


@pytest.mark.parametrize("source", MISSING)
def test_get_answer_from_source_missing(source):
    assert _get_answer_from_source(source) is None


def test_read_data_from_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [source for source, _ in CASES] + MISSING)
    samples = read_data_from_jsonl(str(path))
    # Rows without an answer are skipped with a warning, as before.
    assert samples == [(source["problem"], expected) for source, expected in CASES]


def test_dataset_writer_load_dataset_jsonl(tmp_path):
    write_jsonl(
        tmp_path / "train_data.jsonl", [source for source, _ in CASES] + MISSING
    )
    writer = DatasetWriter(save_dir=str(tmp_path), save_text=False, save_json=True)
    samples = writer.load_dataset_jsonl("train")
    assert samples == [(source["problem"], expected) for source, expected in CASES]


@pytest.mark.parametrize(
    "preprocessor",
    [JsonlDefaultLoadPreprocessor(), PickleDefaultLoadPreprocessor()],
    ids=lambda p: type(p).__name__,
)
def test_default_load_preprocessors(preprocessor):
    assert preprocessor.process_sample({"problem": [0, 0], "answer": 0}) == (
        "0 0",
        "0",
    )
    assert preprocessor.process_sample({"problem": "a", "answer": ""}) == ("a", "")
    for source in MISSING:
        with pytest.raises(ValueError):
            preprocessor.process_sample(source)


def test_user_callable_receives_falsy_answer():
    seen = []

    def format_problem(problem, answer):
        seen.append(answer)
        return str(problem), str(answer)

    preprocessor = UserCallableLoadPreprocessor(format_problem)
    assert preprocessor.process_sample({"problem": [0, 0], "answer": 0}) == (
        "[0, 0]",
        "0",
    )
    assert seen == [0]


def test_io_pipeline_reads_answer_zero_from_jsonl(tmp_path):
    rows = [{"problem": [0, 0], "answer": 0}, {"problem": [1, 2], "answer": 3}]
    for split in ("train", "test"):
        write_jsonl(tmp_path / f"{split}.jsonl", rows)
    lexer = tmp_path / "lexer.yaml"
    lexer.write_text(yaml.safe_dump({"vocab": {"range": {"numbers": ["", 0, 9]}}}))
    cfg = OmegaConf.create(
        {
            "train_dataset_path": str(tmp_path / "train.jsonl"),
            "test_dataset_path": str(tmp_path / "test.jsonl"),
            "lexer_config": str(lexer),
        }
    )
    io = IOPipeline.from_config(cfg).build()
    dataset = io["train_dataset"]
    assert len(dataset) == 2
    assert dataset[0]["target"] == "0"
    assert dataset[1]["target"] == "3"
