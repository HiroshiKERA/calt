"""Tests for the offline pre-tokenization cache feature.

Covers:
  - calt.preprocess: hashing, strings cache, pre-tokenized (input_ids) cache,
    round-trip equality with online tokenization, and cache invalidation.
  - IOPipeline.build: detection of pre-tokenized JSONL and the classic-JSONL
    non-confusion (a {"problem","answer"} JSONL must NOT be treated as pretok).
  - StandardDataCollator: padding of list[int] batches without re-tokenization,
    and the unchanged string-batch path (backward compatibility).
"""

import json

import pytest
import torch
from omegaconf import OmegaConf

from calt.io import IOPipeline, StandardDataCollator
from calt.io import preprocess as pp

# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

LEXER_YAML = {
    "vocab": {
        "range": {"numbers": ["", -100, 100]},
        "misc": ["+", "-", "*", "^", "x", "y", "|", "/"],
        "special_tokens": {},
        "flags": {"include_base_vocab": True, "include_base_special_tokens": True},
    },
    "number_policy": {"attach_sign": True, "digit_group": 1, "allow_float": False},
    "strict": True,
    "include_base_vocab": True,
}


@pytest.fixture
def lexer_yaml_file(tmp_path):
    """Write a raw-format lexer.yaml and return its path."""
    p = tmp_path / "lexer.yaml"
    OmegaConf.save(OmegaConf.create(LEXER_YAML), str(p))
    return str(p)


@pytest.fixture
def raw_data(tmp_path):
    """Write tiny train/test raw .txt files; return (train_path, test_path)."""
    train = tmp_path / "train_raw.txt"
    test = tmp_path / "test_raw.txt"
    train.write_text("x^2 + y # y^2 - x\nx*y # x\n")
    test.write_text("x + 1 # 1\n")
    return str(train), str(test)


@pytest.fixture
def data_cfg():
    """Minimal data.yaml-like config with a sampler section (feeds the hash)."""
    return OmegaConf.create(
        {
            "sampler": {
                "symbols": "x,y",
                "field_str": "QQ",
                "order": "degrevlex",
                "max_degree": 4,
            }
        }
    )


@pytest.fixture
def cfg(lexer_yaml_file, raw_data):
    """Minimal train.yaml-like config (only the data section is needed here)."""
    train_path, test_path = raw_data
    return OmegaConf.create(
        {
            "data": {
                "lexer_config": lexer_yaml_file,
                "train_dataset_path": train_path,
                "test_dataset_path": test_path,
            }
        }
    )


@pytest.fixture
def tokenizer(lexer_yaml_file):
    _lexer, tok = pp._build_lexer_and_tokenizer(lexer_yaml_file)
    return tok


# --------------------------------------------------------------------------- #
# Hashing                                                                       #
# --------------------------------------------------------------------------- #


def test_config_hash_deterministic(cfg, data_cfg):
    h1 = pp.compute_config_hash(cfg, data_cfg, "degrevlex")
    h2 = pp.compute_config_hash(cfg, data_cfg, "degrevlex")
    assert h1 == h2


def test_config_hash_changes_with_order(cfg, data_cfg):
    assert pp.compute_config_hash(cfg, data_cfg, "degrevlex") != pp.compute_config_hash(
        cfg, data_cfg, "lex"
    )


def test_config_hash_changes_with_extra_bytes(cfg, data_cfg):
    base = pp.compute_config_hash(cfg, data_cfg, "degrevlex")
    with_extra = pp.compute_config_hash(
        cfg, data_cfg, "degrevlex", extra_hash_bytes=b"hook"
    )
    assert base != with_extra


def test_pretok_hash_differs_from_config_hash(cfg, data_cfg, tokenizer):
    a = pp.compute_config_hash(cfg, data_cfg, "degrevlex")
    b = pp.compute_pretok_hash(cfg, data_cfg, "degrevlex", tokenizer)
    assert a != b


# --------------------------------------------------------------------------- #
# Round-trip: offline tokenization == online tokenization                       #
# --------------------------------------------------------------------------- #


def test_tokenize_to_ids_matches_online(lexer_yaml_file):
    lexer, tok = pp._build_lexer_and_tokenizer(lexer_yaml_file)
    for text in ["x^2 + y", "x*y", "y^2 - x", "x + 1"]:
        online = tok(lexer(text), add_special_tokens=True)["input_ids"]
        offline = pp._tokenize_to_ids(text, lexer, tok)
        assert online == offline, f"mismatch for {text!r}"


# --------------------------------------------------------------------------- #
# Strings cache                                                                 #
# --------------------------------------------------------------------------- #


def test_run_preprocess_creates_cache(cfg, data_cfg):
    cache_dir = pp.run_preprocess(
        cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None
    )
    assert cache_dir.exists()
    assert (cache_dir / "train_processed.jsonl").exists()
    assert (cache_dir / "test_processed.jsonl").exists()
    assert (cache_dir / "_hash.txt").exists()
    # First record must be {"problem", "answer"}
    first = json.loads(
        (cache_dir / "train_processed.jsonl").read_text().splitlines()[0]
    )
    assert "problem" in first and "answer" in first


def test_maybe_use_processed_cache_hit_and_stale(cfg, data_cfg):
    pp.run_preprocess(cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None)
    # Hit
    cfg_hit = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert pp.maybe_use_processed_cache(cfg_hit, data_cfg, "degrevlex", "raw") is True
    assert "train_processed.jsonl" in cfg_hit.data.train_dataset_jsonl
    # Stale after sampler change
    data_cfg2 = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    data_cfg2.sampler.max_degree = 999
    cfg_stale = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert (
        pp.maybe_use_processed_cache(cfg_stale, data_cfg2, "degrevlex", "raw") is False
    )


# --------------------------------------------------------------------------- #
# Pre-tokenized cache                                                           #
# --------------------------------------------------------------------------- #


def test_preprocess_to_ids_creates_cache(cfg, data_cfg):
    cache_dir = pp.preprocess_to_ids(
        cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None
    )
    assert cache_dir.exists()
    assert (cache_dir / "train_ids.jsonl").exists()
    first = json.loads((cache_dir / "train_ids.jsonl").read_text().splitlines()[0])
    assert "input_ids" in first and "target_ids" in first
    assert isinstance(first["input_ids"], list)
    assert all(isinstance(x, int) for x in first["input_ids"])


def test_maybe_use_pretokenized_cache_hit_sets_flag(cfg, data_cfg):
    pp.preprocess_to_ids(cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None)
    cfg_hit = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert (
        pp.maybe_use_pretokenized_cache(cfg_hit, data_cfg, "degrevlex", "raw") is True
    )
    assert "_ids" in cfg_hit.data.train_dataset_jsonl
    assert cfg_hit.data.get("_pretokenized") is True


def test_maybe_use_pretokenized_cache_stale_on_lexer_change(
    cfg, data_cfg, lexer_yaml_file
):
    pp.preprocess_to_ids(cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None)
    # Mutate the lexer.yaml on disk → hash must change → STALE
    modified = dict(LEXER_YAML)
    modified["number_policy"] = {**LEXER_YAML["number_policy"], "digit_group": 2}
    OmegaConf.save(OmegaConf.create(modified), lexer_yaml_file)
    cfg_stale = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert (
        pp.maybe_use_pretokenized_cache(cfg_stale, data_cfg, "degrevlex", "raw")
        is False
    )


# --------------------------------------------------------------------------- #
# IOPipeline.build detection                                                    #
# --------------------------------------------------------------------------- #


def test_iopipeline_loads_pretokenized(cfg, data_cfg):
    pp.preprocess_to_ids(cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None)
    cfg_hit = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert pp.maybe_use_pretokenized_cache(cfg_hit, data_cfg, "degrevlex", "raw")

    io = IOPipeline.from_config(cfg_hit.data)
    io_dict = io.build()
    ds = io_dict["train_dataset"]
    # Pre-tokenized: input_texts holds list[int], not str
    assert isinstance(ds.input_texts[0], list)
    assert isinstance(ds.input_texts[0][0], int)
    # And the preprocessor (lexer) is disabled so __getitem__ returns ids as-is.
    assert ds.preprocessor is None


def test_classic_jsonl_not_treated_as_pretokenized(tmp_path, cfg):
    """A {"problem","answer"} JSONL must go through the normal path."""
    classic = tmp_path / "classic.jsonl"
    classic.write_text(json.dumps({"problem": "x^2", "answer": "y"}) + "\n")

    cfg_classic = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_classic.data.train_dataset_jsonl = str(classic)
    cfg_classic.data.test_dataset_jsonl = str(classic)

    io = IOPipeline.from_config(cfg_classic.data)
    io_dict = io.build()
    ds = io_dict["train_dataset"]
    # Classic path: input_texts hold strings (after JsonlDefault preprocessing)
    assert isinstance(ds.input_texts[0], str)


def test_corrupt_pretokenized_jsonl_raises(tmp_path, cfg):
    """A JSONL that looks pretok (has input_ids) but misses target_ids errors clearly."""
    bad = tmp_path / "bad_ids.jsonl"
    bad.write_text(json.dumps({"input_ids": [1, 2, 3]}) + "\n")  # no target_ids

    cfg_bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_bad.data.train_dataset_jsonl = str(bad)
    cfg_bad.data.test_dataset_jsonl = str(bad)

    io = IOPipeline.from_config(cfg_bad.data)
    with pytest.raises(ValueError, match="target_ids"):
        io.build()


# --------------------------------------------------------------------------- #
# StandardDataCollator: pretok vs string batches                                #
# --------------------------------------------------------------------------- #


def test_collator_pads_pretokenized_batch(tokenizer):
    coll = StandardDataCollator(tokenizer=tokenizer)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    batch = [
        {"input": [bos, 5, 6, eos], "target": [bos, 7, eos]},
        {"input": [bos, 8, eos], "target": [bos, 9, 10, eos]},
    ]
    out = coll(batch)
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].shape == (2, 4)  # padded to longest (4)
    assert "labels" in out and "decoder_input_ids" in out
    # 4 + 3 = 7 real input tokens
    assert out["attention_mask"].sum().item() == 7


def test_collator_string_batch_unchanged(tokenizer):
    """Backward-compat: string batches still go through tokenizer()."""
    coll = StandardDataCollator(tokenizer=tokenizer)
    batch = [{"input": "x ^ 2", "target": "y"}, {"input": "x", "target": "y ^ 2"}]
    out = coll(batch)
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].shape[0] == 2
    assert "labels" in out


# --------------------------------------------------------------------------- #
# build_io_pipeline_with_cache: the cache cascade                               #
# --------------------------------------------------------------------------- #


def test_cascade_picks_pretokenized_first(cfg, data_cfg):
    pp.preprocess_to_ids(cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None)
    cfg_run = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    io, kind = pp.build_io_pipeline_with_cache(
        cfg_run, data_cfg, "degrevlex", "raw", load_preprocessor=None
    )
    assert kind == "pretokenized"
    assert isinstance(io, IOPipeline)


def test_cascade_falls_back_to_strings(cfg, data_cfg):
    # Only the strings cache exists (no *_ids cache) → cascade must pick "strings".
    pp.run_preprocess(cfg, data_cfg, "degrevlex", "raw", load_preprocessor=None)
    cfg_run = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    io, kind = pp.build_io_pipeline_with_cache(
        cfg_run, data_cfg, "degrevlex", "raw", load_preprocessor=None
    )
    assert kind == "strings"
    assert isinstance(io, IOPipeline)


def test_cascade_no_cache_returns_none(cfg, data_cfg):
    # No cache built at all → "none"; load_preprocessor must be wired onto pipeline.
    sentinel = object()
    cfg_run = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    io, kind = pp.build_io_pipeline_with_cache(
        cfg_run, data_cfg, "degrevlex", "raw", load_preprocessor=sentinel
    )
    assert kind == "none"
    assert io.dataset_load_preprocessor is sentinel


# --------------------------------------------------------------------------- #
# Source-ring reconstruction (requires SageMath)                                #
# --------------------------------------------------------------------------- #


def test_parse_field_and_ring(data_cfg):
    pytest.importorskip("sage.all")
    from sage.all import GF, QQ

    assert pp.parse_field("QQ") is QQ
    assert pp.parse_field("GF7") == GF(7)
    with pytest.raises(ValueError):
        pp.parse_field("nonsense")

    ring = pp.build_ring_from_sampler(data_cfg.sampler)
    assert ring.ngens() == 2  # symbols "x,y"
    assert ring.base_ring() is QQ
