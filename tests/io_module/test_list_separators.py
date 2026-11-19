"""list_separators: separators between list elements when JSONL/pickle data is read.

Index 0 is used for the outermost list, 1 for the lists inside it, and so on,
each put between spaces. A depth without a separator is joined with a space.
"""

import json
import logging

import pytest
import yaml
from omegaconf import OmegaConf

from calt.dataset.utils.dataset_writer import DatasetWriter
from calt.io import IOPipeline, UserCallableLoadPreprocessor
from calt.io.preprocessor.load_preprocessor import (
    JsonlDefaultLoadPreprocessor,
    PickleDefaultLoadPreprocessor,
    _to_str,
    normalize_list_separators,
)

NESTED = [["1", "2"], ["3", "4"]]


@pytest.mark.parametrize(
    ("value", "separators", "expected"),
    [
        # No separators: every depth is joined with a space, as before.
        (NESTED, None, "1 2 3 4"),
        (["31600", "54662"], None, "31600 54662"),
        # The reply's example: index 0 outermost, index 1 inside it.
        (NESTED, ["||", "|"], "1 | 2 || 3 | 4"),
        (["31600", "54662"], ["|"], "31600 | 54662"),
        # One separator on a nested list: only the outermost list gets it.
        (NESTED, ["|"], "1 2 | 3 4"),
        # Several separators on a flat list: only index 0 is used.
        (["1", "2"], ["||", "|"], "1 || 2"),
        # A depth without a separator stays a space.
        ([[["1", "2"], ["3"]], [["4"]]], ["||", "|"], "1 2 | 3 || 4"),
        # Extra separators are ignored.
        (NESTED, ["||", "|", ";", ","], "1 | 2 || 3 | 4"),
        # Mixed depths, as DatasetWriter allows.
        ([["1", "2"], "3"], ["||", "|"], "1 | 2 || 3"),
        # Edge cases.
        ([], ["|"], ""),
        (["7"], ["|"], "7"),
        ([[]], ["||", "|"], ""),
        ([1, 0], ["|"], "1 | 0"),
        ("1 | 2", ["||"], "1 | 2"),
        ({"a": [1, 2]}, ["|"], "{'a': [1, 2]}"),
    ],
)
def test_to_str(value, separators, expected):
    assert _to_str(value, separators) == expected


def test_normalize_list_separators():
    assert normalize_list_separators(None) is None
    assert normalize_list_separators("|") == ["|"]
    assert normalize_list_separators([" | "]) == ["|"]
    assert normalize_list_separators(("||", "|")) == ["||", "|"]
    assert normalize_list_separators(OmegaConf.create(["||", "|"])) == ["||", "|"]
    for bad in ([""], [" "], ["|", None], [1]):
        with pytest.raises(ValueError):
            normalize_list_separators(bad)


@pytest.mark.parametrize(
    "preprocessor_class",
    [JsonlDefaultLoadPreprocessor, PickleDefaultLoadPreprocessor],
    ids=lambda c: c.__name__,
)
def test_default_loaders_use_separators_for_problem_and_answer(preprocessor_class):
    source = {"problem": NESTED, "answer": ["5", "6"]}
    assert preprocessor_class().process_sample(source) == ("1 2 3 4", "5 6")
    assert preprocessor_class(["||", "|"]).process_sample(source) == (
        "1 | 2 || 3 | 4",
        "5 || 6",
    )
    # Empty and one-element lists through the loaders too. (An empty answer
    # list also needs the falsy-answer fix of the "answer is 0" PR.)
    empty = {"problem": [], "answer": ["7"]}
    assert preprocessor_class(["|"]).process_sample(empty) == ("", "7")


def test_user_callable_still_gets_the_lists():
    seen = []

    def format_problem(problem, answer):
        seen.append((problem, answer))
        return "x", "y"

    UserCallableLoadPreprocessor(format_problem).process_sample(
        {"problem": NESTED, "answer": ["5"]}
    )
    assert seen == [(NESTED, ["5"])]


def write_data(tmp_path, rows, misc):
    for split in ("train", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    lexer = tmp_path / "lexer.yaml"
    lexer.write_text(
        yaml.safe_dump(
            {
                "vocab": {"range": {"numbers": ["", 0, 9]}, "misc": misc},
                "number_policy": {"digit_group": 1},
            }
        )
    )
    return {
        "lexer_config": str(lexer),
        "train_dataset_path": str(tmp_path / "train.jsonl"),
        "test_dataset_path": str(tmp_path / "test.jsonl"),
    }


ROWS = [
    {"problem": [["31600", "54662"], ["1", "2"]], "answer": ["86262", "3"]},
    {"problem": [["5", "6"], ["7", "8"]], "answer": ["11", "15"]},
]


def test_io_pipeline_python_and_yaml_give_the_same_result(tmp_path):
    data = write_data(tmp_path, ROWS, ["|", "||"])
    yaml_cfg = OmegaConf.create(
        yaml.safe_load(yaml.safe_dump(data) + 'list_separators: ["||", "|"]\n')
    )
    from_yaml = IOPipeline.from_config(yaml_cfg).build()["train_dataset"]

    base = IOPipeline.from_config(OmegaConf.create(data))
    from_python = IOPipeline(
        train_dataset_path=data["train_dataset_path"],
        test_dataset_path=data["test_dataset_path"],
        vocab_config=base.vocab_config,
        preprocessor=base.preprocessor,
        list_separators=["||", "|"],
    ).build()["train_dataset"]

    assert from_yaml.input_texts == from_python.input_texts
    assert from_yaml.target_texts == from_python.target_texts
    assert from_yaml.input_texts[0] == "31600 | 54662 || 1 | 2"
    assert from_yaml.target_texts[0] == "86262 || 3"


def test_separators_stay_between_elements_after_tokenization(tmp_path):
    data = OmegaConf.create(write_data(tmp_path, ROWS, ["|", "||"]))
    data.list_separators = ["||", "|"]
    pipeline = IOPipeline.from_config(data)
    dataset = pipeline.build()["train_dataset"]
    tokens = pipeline.preprocessor.tokenize(dataset.input_texts[0])
    assert tokens == ["3", "1", "6", "0", "0", "|", "5", "4", "6", "6", "2"] + [
        "||",
        "1",
        "|",
        "2",
    ]


def test_without_separators_nothing_changes(tmp_path):
    data = OmegaConf.create(write_data(tmp_path, ROWS, []))
    dataset = IOPipeline.from_config(data).build()["train_dataset"]
    assert dataset.input_texts[0] == "31600 54662 1 2"
    assert dataset.target_texts[0] == "86262 3"


def _text_and_jsonl_lines(tmp_path, samples, separators):
    writer = DatasetWriter(save_dir=str(tmp_path), save_text=True, save_json=True)
    writer.save_batch(samples, tag="train")
    writer.close_all()
    text_lines = (tmp_path / "train_raw.txt").read_text().splitlines()
    loader = JsonlDefaultLoadPreprocessor(separators)
    jsonl_lines = [
        " # ".join(loader.process_sample(json.loads(line)))
        for line in (tmp_path / "train_data.jsonl").read_text().splitlines()
    ]
    return text_lines, jsonl_lines


def test_jsonl_with_matching_separators_reads_like_the_text_file(tmp_path):
    # DatasetWriter writes nested lists to text as " | " inside, " || " outside,
    # and a flat list as " | ". With the matching separators, JSONL reads the same.
    nested = [(NESTED, [["5"], ["6", "7"]])]
    text, jsonl = _text_and_jsonl_lines(tmp_path / "nested", nested, ["||", "|"])
    assert jsonl == text == ["1 | 2 || 3 | 4 # 5 || 6 | 7"]

    flat = [(["1", "2"], ["3"])]
    text, jsonl = _text_and_jsonl_lines(tmp_path / "flat", flat, ["|"])
    assert jsonl == text == ["1 | 2 # 3"]


def test_nested_problem_with_flat_answer_differs_from_the_text_file(tmp_path):
    # One set of separators serves both: the flat answer's outermost list gets
    # index 0 ("||"), while the text file joins a flat list with " | ".
    samples = [(NESTED, ["5", "6"])]
    text, jsonl = _text_and_jsonl_lines(tmp_path, samples, ["||", "|"])
    assert text == ["1 | 2 || 3 | 4 # 5 | 6"]
    assert jsonl == ["1 | 2 || 3 | 4 # 5 || 6"]


def test_warns_on_text_files(tmp_path, caplog):
    for split in ("train", "test"):
        (tmp_path / f"{split}.txt").write_text("1 | 2 # 3\n")
    data = write_data(tmp_path, ROWS, ["|"])
    data["train_dataset_path"] = str(tmp_path / "train.txt")
    data["test_dataset_path"] = str(tmp_path / "test.txt")
    data["list_separators"] = ["|"]
    with caplog.at_level(logging.WARNING, logger="calt.io.pipeline"):
        IOPipeline.from_config(OmegaConf.create(data)).build()
    assert "list_separators" in caplog.text


def test_warns_with_dataset_load_preprocessor(tmp_path, caplog):
    data = OmegaConf.create(write_data(tmp_path, ROWS, ["|", "||"]))
    data.list_separators = ["||", "|"]
    pipeline = IOPipeline.from_config(data)
    pipeline.dataset_load_preprocessor = UserCallableLoadPreprocessor(
        lambda problem, answer: ("1", "2")
    )
    with caplog.at_level(logging.WARNING, logger="calt.io.pipeline"):
        pipeline.build()
    assert "list_separators" in caplog.text


def test_no_warning_for_jsonl(tmp_path, caplog):
    data = OmegaConf.create(write_data(tmp_path, ROWS, ["|", "||"]))
    data.list_separators = ["||", "|"]
    with caplog.at_level(logging.WARNING, logger="calt.io.pipeline"):
        IOPipeline.from_config(data).build()
    assert "list_separators" not in caplog.text
