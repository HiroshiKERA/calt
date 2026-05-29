"""
Offline preprocessing of raw text datasets into a tokenizer-ready JSONL cache.

Why this module exists
----------------------
CALT's IOPipeline applies the `dataset_load_preprocessor` (e.g. TextToSage →
FGLM → ExpandedForm) once at training startup. For the Gröbner / border tasks
this involves SageMath, which is slow on 100k samples and runs every time you
restart training. We move that work offline:

    generate.py   → data/<F>/train_raw.txt
    preprocess.py → data/<F>/processed_<order>_<format>/{train,test}_processed.jsonl
                                                          + _hash.txt
    train.py      → if cache valid: read JSONL (no SageMath); else: fall back

The cache is invalidated by `_hash.txt`: any change to lexer.yaml, data.yaml
sampler config, or training_order produces a new hash and forces regeneration.

Files written per cache directory
---------------------------------
    train_processed.jsonl   one record per line: {"problem": str, "answer": str}
    test_processed.jsonl    same format
    _hash.txt               SHA256 hash of (lexer.yaml + sampler + order + version)
    _meta.yaml              human-readable record of how/when this cache was built

Building the cache offline (run once, e.g. from a preprocess script)
--------------------------------------------------------------------
    from calt.preprocess import run_preprocess, preprocess_to_ids

    # `load_preprocessor` is the same DatasetLoadPreprocessor you would pass to
    # IOPipeline.dataset_load_preprocessor (or None). `lexer_format` is "raw" or
    # "expanded"; `training_order` is any string identifying the variant.

    # Option A — strings cache (skip the load-preprocessor work, e.g. SageMath):
    run_preprocess(cfg, data_cfg, training_order, lexer_format, load_preprocessor)

    # Option B — pre-tokenized cache (also skip lexer + tokenizer at train time):
    preprocess_to_ids(cfg, data_cfg, training_order, lexer_format, load_preprocessor)

Using the cache at training time (auto-detection)
-------------------------------------------------
    from calt.preprocess import maybe_use_pretokenized_cache, maybe_use_processed_cache

    # Prefer the most aggressive cache available, then build the pipeline.
    if maybe_use_pretokenized_cache(cfg, data_cfg, training_order, lexer_format):
        pass  # cfg.data.{train,test}_dataset_jsonl now point at *_ids.jsonl
    elif maybe_use_processed_cache(cfg, data_cfg, training_order, lexer_format):
        pass  # cfg.data.{train,test}_dataset_jsonl now point at *_processed.jsonl
    io_pipeline = IOPipeline.from_config(cfg.data)  # reads the cache automatically
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

# Bump this if the on-disk format or hashing logic changes; old caches will be
# invalidated (mismatched hash → regen).
PREPROCESS_VERSION = "v1"

# Bump independently for the pre-tokenized (input_ids) on-disk format.
PRETOK_VERSION = "v1"


# --------------------------------------------------------------------------- #
# Hash & cache-path computation                                                #
# --------------------------------------------------------------------------- #


def _read_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def compute_config_hash(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str,
    extra_hash_bytes: bytes = b"",
) -> str:
    """
    SHA256 over everything that affects the post-load-preprocessor strings.

    Inputs to the hash
    ------------------
    - PREPROCESS_VERSION    : forces invalidation on format changes
    - lexer.yaml bytes      : if vocab/number_policy changes, retokenize
    - data.yaml sampler dict: if field/order/vars change, ring & data change
    - training_order        : "degrevlex" or "lex" changes the chain
    - extra_hash_bytes      : caller-supplied bytes (e.g. user post-processor
                              source code) — invalidates cache when the user's
                              `base_postprocessor.py` / `<task>/core/postprocessor.py`
                              changes. Empty bytes ⇒ no contribution.
    """
    h = hashlib.sha256()
    h.update(PREPROCESS_VERSION.encode())
    h.update(b"\x00")

    lexer_path = Path(cfg.data.lexer_config)
    if not lexer_path.is_absolute():
        # Resolve relative to CWD (matches IOPipeline.from_config behavior).
        lexer_path = lexer_path.resolve()
    h.update(_read_bytes(lexer_path))
    h.update(b"\x00")

    if data_cfg is not None and "sampler" in data_cfg:
        sampler_dict = OmegaConf.to_container(data_cfg.sampler, resolve=True)
        h.update(json.dumps(sampler_dict, sort_keys=True).encode())
    h.update(b"\x00")

    h.update(training_order.encode())
    h.update(b"\x00")
    h.update(extra_hash_bytes)
    return h.hexdigest()


def get_cache_dir(
    raw_train_path: str | Path,
    training_order: str,
    lexer_format: str,
) -> Path:
    """
    Cache lives in a sibling directory of the raw .txt, named to make the
    (order, format) variant obvious in `ls`.

        data/QQ/train_raw.txt
        data/QQ/processed_degrevlex_raw/   ← cache for (degrevlex, raw)
        data/QQ/processed_lex_raw/         ← cache for (lex, raw)
        data/QQ/processed_degrevlex_expanded/
    """
    raw_train_path = Path(raw_train_path)
    return raw_train_path.parent / f"processed_{training_order}_{lexer_format}"


# --------------------------------------------------------------------------- #
# Cache writer (called by experiments/*/scripts/preprocess.py)                #
# --------------------------------------------------------------------------- #


def _iter_raw_lines(raw_path: Path):
    """Yield (input_text, target_text) pairs from a CALT raw .txt file."""
    with open(raw_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "#" not in line:
                continue
            inp, tgt = line.split("#", 1)
            yield inp.strip(), tgt.strip()


def _write_jsonl(records, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for problem, answer in records:
            f.write(json.dumps({"problem": problem, "answer": answer}) + "\n")
            n += 1
    return n


def _preprocess_file(
    raw_path: Path,
    out_path: Path,
    load_preprocessor: Any | None,
) -> tuple[int, float]:
    """
    Apply load_preprocessor (if any) to every (input, target) pair in raw_path
    and write the resulting strings to out_path as JSONL.

    If load_preprocessor is None, copies the raw pairs through unchanged — still
    useful so train.py can take the JSONL fast path uniformly.
    """
    t0 = time.time()
    pairs = []
    for inp, tgt in _iter_raw_lines(raw_path):
        if load_preprocessor is not None:
            # CALT's load preprocessors expect the joined "input # target" line,
            # but some (e.g. ExpandedFormLoadPreprocessor) accept (problem, answer)
            # via .process_sample. We use process_sample for compatibility with
            # both the chain wrapper and individual preprocessors.
            problem, answer = load_preprocessor.process_sample(f"{inp}#{tgt}")
        else:
            problem, answer = inp, tgt
        pairs.append((problem, answer))
    n = _write_jsonl(pairs, out_path)
    return n, time.time() - t0


def run_preprocess(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str,
    lexer_format: str,
    load_preprocessor: Any | None,
    force: bool = False,
    extra_hash_bytes: bytes = b"",
) -> Path:
    """
    Materialize the JSONL cache for this (cfg, training_order, lexer_format).

    Returns the cache directory. Skips work if a valid cache already exists
    (hash match), unless force=True.
    """
    raw_train = Path(cfg.data.train_dataset_path)
    raw_test = Path(cfg.data.test_dataset_path)
    cache_dir = get_cache_dir(raw_train, training_order, lexer_format)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_hash = compute_config_hash(cfg, data_cfg, training_order, extra_hash_bytes)
    hash_file = cache_dir / "_hash.txt"

    if (
        not force
        and hash_file.exists()
        and hash_file.read_text().strip() == expected_hash
    ):
        print(
            f"[preprocess] cache up to date at {cache_dir} (hash {expected_hash[:12]}...)"
        )
        return cache_dir

    print(f"[preprocess] writing cache → {cache_dir}")
    print(f"[preprocess]   order={training_order}, lexer_format={lexer_format}")
    print(
        f"[preprocess]   load_preprocessor={type(load_preprocessor).__name__ if load_preprocessor else None}"
    )

    n_train, dt_train = _preprocess_file(
        raw_train, cache_dir / "train_processed.jsonl", load_preprocessor
    )
    print(f"[preprocess]   train: {n_train} samples in {dt_train:.1f}s")
    n_test, dt_test = _preprocess_file(
        raw_test, cache_dir / "test_processed.jsonl", load_preprocessor
    )
    print(f"[preprocess]   test:  {n_test} samples in {dt_test:.1f}s")

    meta = {
        "preprocess_version": PREPROCESS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_order": training_order,
        "lexer_format": lexer_format,
        "lexer_config": str(cfg.data.lexer_config),
        "load_preprocessor": type(load_preprocessor).__name__
        if load_preprocessor
        else None,
        "n_train": n_train,
        "n_test": n_test,
    }
    (cache_dir / "_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    hash_file.write_text(expected_hash)
    print(f"[preprocess] done. hash={expected_hash[:12]}...")
    return cache_dir


# --------------------------------------------------------------------------- #
# Cache reader (called by train.py)                                            #
# --------------------------------------------------------------------------- #


def maybe_use_processed_cache(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str,
    lexer_format: str,
    extra_hash_bytes: bytes = b"",
) -> bool:
    """
    If a valid processed JSONL cache exists for this configuration, mutate cfg.data
    to point IOPipeline at the JSONL files (skipping the SageMath/FGLM/Expanded
    load step) and return True. Otherwise return False (caller should fall back
    to the raw .txt + load_preprocessor path).

    Cache is valid iff `_hash.txt` matches `compute_config_hash(...)`.
    """
    raw_train = Path(cfg.data.train_dataset_path)
    cache_dir = get_cache_dir(raw_train, training_order, lexer_format)
    hash_file = cache_dir / "_hash.txt"
    train_jsonl = cache_dir / "train_processed.jsonl"
    test_jsonl = cache_dir / "test_processed.jsonl"

    if not (hash_file.exists() and train_jsonl.exists() and test_jsonl.exists()):
        return False

    expected = compute_config_hash(cfg, data_cfg, training_order, extra_hash_bytes)
    actual = hash_file.read_text().strip()
    if actual != expected:
        print(
            f"[preprocess] cache at {cache_dir} is STALE "
            f"(expected {expected[:12]}, found {actual[:12]}). Falling back to raw."
        )
        return False

    # Swap cfg paths so IOPipeline picks up the JSONL files. We DON'T clear
    # train_dataset_path — IOPipeline.build() prefers jsonl over .txt when both
    # are set, and the original path is useful for evaluate.py.
    cfg.data.train_dataset_jsonl = str(train_jsonl)
    cfg.data.test_dataset_jsonl = str(test_jsonl)
    return True


# =========================================================================== #
# Pre-tokenized cache (stores input_ids, NOT strings)                          #
# =========================================================================== #
#
# A second, more aggressive level of offline preprocessing: in addition to the
# load_preprocessor work (SageMath/FGLM/ExpandedForm), we ALSO apply UnifiedLexer
# and the HF tokenizer offline and store integer `input_ids` on disk. Training
# then only pads + tensorizes — zero tokenization at runtime.
#
# Cache layout
# ------------
#     data/QQ/processed_<order>_<format>_ids/
#         train_ids.jsonl    {"input_ids": [int, ...], "target_ids": [int, ...]}
#         test_ids.jsonl     same
#         _hash.txt          SHA256 over lexer + sampler + order + tokenizer vocab
#         _meta.yaml         human-readable
#
# Both input_ids and target_ids INCLUDE the bos/eos tokens that the HF tokenizer
# would have wrapped at training time (TemplateProcessing applies "{BOS} $A {EOS}").
# At collate time we just pad to longest and feed the existing slice logic
# (decoder_input_ids = ids[:, :-1], labels = ids[:, 1:]).


def _build_lexer_and_tokenizer(lexer_yaml_path):
    """
    Rebuild the SAME UnifiedLexer and PreTrainedTokenizerFast that
    IOPipeline.from_config would produce. We mirror its construction exactly so
    the offline IDs match what online tokenization would have produced.
    """
    from .io.preprocessor.lexer import NumberPolicy, UnifiedLexer
    from .io.tokenizer import get_tokenizer
    from .io.vocabulary.config import VocabConfig

    with open(lexer_yaml_path) as f:
        lexer_config = yaml.safe_load(f)

    vocab_config_dict = lexer_config.get("vocab", {})
    vocab_config = VocabConfig([], {}).from_config(vocab_config_dict)

    number_policy_dict = lexer_config.get("number_policy", {})
    number_policy = NumberPolicy(
        sign=number_policy_dict.get("attach_sign", True),
        digit_group=number_policy_dict.get("digit_group", 0),
        allow_float=number_policy_dict.get("allow_float", True),
    )

    lexer = UnifiedLexer(
        vocab_config=vocab_config,
        number_policy=number_policy,
        strict=lexer_config.get("strict", True),
        include_base_vocab=lexer_config.get("include_base_vocab", True),
    )
    tokenizer = get_tokenizer(vocab_config=lexer.vocab_config)
    return lexer, tokenizer


def compute_pretok_hash(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str,
    tokenizer,
    extra_hash_bytes: bytes = b"",
) -> str:
    """
    Hash for the pre-tokenized cache. Same inputs as `compute_config_hash` plus
    the tokenizer vocabulary (so any change to the effective vocab — including
    auto-added float tokens that the lexer adds beyond the user's lexer.yaml —
    invalidates the cache).
    """
    h = hashlib.sha256()
    h.update(PRETOK_VERSION.encode())
    h.update(b"\x00")

    lexer_path = Path(cfg.data.lexer_config)
    if not lexer_path.is_absolute():
        lexer_path = lexer_path.resolve()
    h.update(_read_bytes(lexer_path))
    h.update(b"\x00")

    if data_cfg is not None and "sampler" in data_cfg:
        sampler_dict = OmegaConf.to_container(data_cfg.sampler, resolve=True)
        h.update(json.dumps(sampler_dict, sort_keys=True).encode())
    h.update(b"\x00")

    h.update(training_order.encode())
    h.update(b"\x00")

    vocab = tokenizer.get_vocab()
    h.update(json.dumps(vocab, sort_keys=True).encode())
    h.update(b"\x00")
    h.update(extra_hash_bytes)
    return h.hexdigest()


def get_pretok_cache_dir(
    raw_train_path: str | Path,
    training_order: str,
    lexer_format: str,
) -> Path:
    raw_train_path = Path(raw_train_path)
    return raw_train_path.parent / f"processed_{training_order}_{lexer_format}_ids"


def _tokenize_to_ids(text: str, lexer, tokenizer) -> list[int]:
    """Apply UnifiedLexer then HF tokenizer. Returns int IDs incl. BOS/EOS."""
    lexed = lexer(text)
    return tokenizer(lexed, add_special_tokens=True)["input_ids"]


def _pretokenize_file(
    raw_path: Path,
    out_path: Path,
    load_preprocessor: Any | None,
    lexer,
    tokenizer,
) -> tuple[int, float]:
    """
    For every (input, target) pair: apply load_preprocessor (if any), then
    UnifiedLexer + HF tokenizer, write {"input_ids": [...], "target_ids": [...]}.
    """
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for inp, tgt in _iter_raw_lines(raw_path):
            if load_preprocessor is not None:
                problem, answer = load_preprocessor.process_sample(f"{inp}#{tgt}")
            else:
                problem, answer = inp, tgt
            input_ids = _tokenize_to_ids(problem, lexer, tokenizer)
            target_ids = _tokenize_to_ids(answer, lexer, tokenizer)
            f.write(
                json.dumps({"input_ids": input_ids, "target_ids": target_ids}) + "\n"
            )
            n += 1
    return n, time.time() - t0


def preprocess_to_ids(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str,
    lexer_format: str,
    load_preprocessor: Any | None,
    force: bool = False,
    extra_hash_bytes: bytes = b"",
) -> Path:
    """
    Materialize a pre-tokenized cache: input_ids + target_ids stored on disk so
    training does zero tokenization. Skips work when hash matches (unless force).
    """
    raw_train = Path(cfg.data.train_dataset_path)
    raw_test = Path(cfg.data.test_dataset_path)
    cache_dir = get_pretok_cache_dir(raw_train, training_order, lexer_format)
    cache_dir.mkdir(parents=True, exist_ok=True)

    lexer, tokenizer = _build_lexer_and_tokenizer(cfg.data.lexer_config)
    expected_hash = compute_pretok_hash(
        cfg, data_cfg, training_order, tokenizer, extra_hash_bytes
    )
    hash_file = cache_dir / "_hash.txt"

    if (
        not force
        and hash_file.exists()
        and hash_file.read_text().strip() == expected_hash
    ):
        print(
            f"[preprocess-ids] cache up to date at {cache_dir} (hash {expected_hash[:12]}...)"
        )
        return cache_dir

    print(f"[preprocess-ids] writing cache → {cache_dir}")
    print(f"[preprocess-ids]   order={training_order}, lexer_format={lexer_format}")
    print(
        f"[preprocess-ids]   load_preprocessor={type(load_preprocessor).__name__ if load_preprocessor else None}"
    )
    print(f"[preprocess-ids]   tokenizer vocab size={len(tokenizer.get_vocab())}")

    n_train, dt_train = _pretokenize_file(
        raw_train, cache_dir / "train_ids.jsonl", load_preprocessor, lexer, tokenizer
    )
    print(f"[preprocess-ids]   train: {n_train} samples in {dt_train:.1f}s")
    n_test, dt_test = _pretokenize_file(
        raw_test, cache_dir / "test_ids.jsonl", load_preprocessor, lexer, tokenizer
    )
    print(f"[preprocess-ids]   test:  {n_test} samples in {dt_test:.1f}s")

    meta = {
        "pretok_version": PRETOK_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_order": training_order,
        "lexer_format": lexer_format,
        "lexer_config": str(cfg.data.lexer_config),
        "load_preprocessor": type(load_preprocessor).__name__
        if load_preprocessor
        else None,
        "tokenizer_vocab_size": len(tokenizer.get_vocab()),
        "n_train": n_train,
        "n_test": n_test,
    }
    (cache_dir / "_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    hash_file.write_text(expected_hash)
    print(f"[preprocess-ids] done. hash={expected_hash[:12]}...")
    return cache_dir


def maybe_use_pretokenized_cache(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str,
    lexer_format: str,
    extra_hash_bytes: bytes = b"",
) -> bool:
    """
    If a valid pre-tokenized cache exists, swap cfg.data paths to the *_ids.jsonl
    files and set cfg.data._pretokenized = True (read by IOPipeline.build).
    Returns True on cache hit.

    Validates the hash by reconstructing the tokenizer; mismatch → False (caller
    falls back to the strings cache or the raw path).
    """
    raw_train = Path(cfg.data.train_dataset_path)
    cache_dir = get_pretok_cache_dir(raw_train, training_order, lexer_format)
    hash_file = cache_dir / "_hash.txt"
    train_jsonl = cache_dir / "train_ids.jsonl"
    test_jsonl = cache_dir / "test_ids.jsonl"

    if not (hash_file.exists() and train_jsonl.exists() and test_jsonl.exists()):
        return False

    _lexer, tokenizer = _build_lexer_and_tokenizer(cfg.data.lexer_config)
    expected = compute_pretok_hash(
        cfg, data_cfg, training_order, tokenizer, extra_hash_bytes
    )
    actual = hash_file.read_text().strip()
    if actual != expected:
        print(
            f"[preprocess-ids] cache at {cache_dir} is STALE "
            f"(expected {expected[:12]}, found {actual[:12]}). Falling back."
        )
        return False

    cfg.data.train_dataset_jsonl = str(train_jsonl)
    cfg.data.test_dataset_jsonl = str(test_jsonl)
    # OmegaConf accepts arbitrary keys via merge; set the flag IOPipeline reads.
    cfg.data._pretokenized = True
    return True
