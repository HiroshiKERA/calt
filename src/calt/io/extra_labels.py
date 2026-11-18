"""Extra labels returned by a load preprocessor, checked and batched.

A load preprocessor may return ``(input_text, target_text, {name: value})``.
Each name becomes a keyword argument of the model's loss (see
``calt.models.register_loss``). A value is either

* a **scalar**: one number per sample, batched as a ``(B,)`` tensor, or
* a **sequence**: one number per answer token, batched as ``(B, T)`` like
  ``labels``. "Answer tokens" are the lexer's tokens of the target plus the
  closing EOS, i.e. exactly the positions of ``labels``. A list one shorter
  gets one value appended at the EOS position.

Positions past a row's answer and the appended value are filled with 0 (0.0 for
float labels); -100 is never used, because it can be a real label value. Which
positions hold answer tokens is given by ``labels != -100``; which of them a
loss should ignore is up to the loss.

Integers and bools become ``torch.long``, and a label with any float becomes
``torch.float``. Every sample must return the same names.
"""

import numbers
from typing import Any, Iterable, Sequence

import torch

SCALAR = "scalar"
SEQUENCE = "sequence"
FILL_VALUE = 0

# Key of the extra labels in dataset items, batches and the models' forward().
# It must not contain "label": transformers treats every forward() argument
# whose name does as a label (find_labels), which would change Trainer's
# label_names and drop eval_loss for batches without extra labels.
BATCH_KEY = "extra_targets"

# Batch keys and loss arguments an extra label must not shadow.
RESERVED_NAMES = frozenset(
    {
        "input",
        "target",
        "input_ids",
        "attention_mask",
        "labels",
        "logits",
        "hidden",
        BATCH_KEY,
        "return_dict",
        "num_items_in_batch",
        "d_model",
        "vocab_size",
    }
)
RESERVED_PREFIXES = ("decoder_", "target_")


def split_sample(result: Any, where: str) -> tuple[str, str, dict | None]:
    """Unpack a load preprocessor's result into (input, target, extra_labels)."""
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise TypeError(
            f"{where}: a load preprocessor must return (input_text, target_text) "
            "or (input_text, target_text, extra_labels), got "
            f"{type(result).__name__}"
        )
    if len(result) == 2:
        return result[0], result[1], None
    extra = result[2]
    if not isinstance(extra, dict):
        raise TypeError(
            f"{where}: the third element (extra labels) must be a dict, got "
            f"{type(extra).__name__}"
        )
    return result[0], result[1], extra


def _number(value: Any) -> int | float | None:
    """``value`` as a Python number, or None if it is not one."""
    if isinstance(value, torch.Tensor):
        if value.dim() != 0:
            return None
        value = value.item()
    elif hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        value = value.item()  # numpy scalar or 0-d array
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def normalize(name: str, value: Any, where: str) -> tuple[str, Any]:
    """Return ``(kind, python value)`` for one extra label of one sample."""
    if not isinstance(name, str):
        raise TypeError(f"{where}: extra label names must be strings, got {name!r}")
    if name in RESERVED_NAMES or name.startswith(RESERVED_PREFIXES):
        raise ValueError(
            f"{where}: extra label name {name!r} is reserved; rename it "
            f"(reserved: {sorted(RESERVED_NAMES)} and names starting with "
            f"{', '.join(RESERVED_PREFIXES)})."
        )
    number = _number(value)
    if number is not None:
        return SCALAR, number
    if isinstance(value, torch.Tensor) or hasattr(value, "tolist"):
        if getattr(value, "ndim", None) != 1:
            raise ValueError(
                f"{where}: extra label {name!r} must be a number or a 1-D "
                f"sequence of numbers, got shape {tuple(value.shape)}"
            )
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        items = [_number(item) for item in value]
        if any(item is None for item in items):
            raise ValueError(
                f"{where}: extra label {name!r} must be a flat sequence of "
                f"numbers, got {value!r}"
            )
        return SEQUENCE, items
    raise ValueError(
        f"{where}: extra label {name!r} must be a number or a sequence of "
        f"numbers, got {type(value).__name__}"
    )


def normalize_samples(samples: list[dict], source: str) -> list[dict]:
    """Normalize every sample's labels to plain numbers and lists."""
    normalized = []
    for index, labels in enumerate(samples):
        where = f"{source}, sample {index}"
        normalized.append(
            {name: normalize(name, value, where)[1] for name, value in labels.items()}
        )
    return normalized


def build_spec(datasets: Iterable[tuple[str, Sequence[dict]]]) -> dict[str, tuple]:
    """Decide ``{name: (kind, dtype)}`` over all samples of all ``datasets``.

    ``datasets`` yields ``(source, samples)``; every sample of every dataset must
    have the same names, and each name the same kind.
    """
    names = None
    first = None
    kinds: dict[str, str] = {}
    floats: set[str] = set()
    for source, samples in datasets:
        for index, labels in enumerate(samples):
            where = f"{source}, sample {index}"
            if names is None:
                names, first = set(labels), where
            elif set(labels) != names:
                missing = sorted(names - set(labels))
                extra = sorted(set(labels) - names)
                raise ValueError(
                    f"{where} returns extra labels {sorted(labels)}, but {first} "
                    f"returns {sorted(names)} (missing {missing}, unexpected "
                    f"{extra}). Every sample must return the same names; put a "
                    "value of your choice where a sample has no label."
                )
            for name, value in labels.items():
                kind = SEQUENCE if isinstance(value, list) else SCALAR
                if kinds.setdefault(name, kind) != kind:
                    raise ValueError(
                        f"{where}: extra label {name!r} is a {kind} here but a "
                        f"{kinds[name]} in other samples."
                    )
                values = value if isinstance(value, list) else [value]
                if any(isinstance(item, float) for item in values):
                    floats.add(name)
    return {
        name: (kind, torch.float if name in floats else torch.long)
        for name, kind in kinds.items()
    }


def answer_length(preprocessor: Any, target: str) -> int:
    """Number of answer tokens of ``target``: the lexer's tokens plus EOS."""
    tokenize = getattr(preprocessor, "tokenize", None)
    tokens = tokenize(target) if tokenize is not None else preprocessor(target).split()
    return len(tokens) + 1


def check_length(name: str, length: int, answer_len: int, where: str) -> None:
    """Raise unless a sequence label has ``answer_len`` or ``answer_len - 1`` values."""
    if length not in (answer_len, answer_len - 1):
        raise ValueError(
            f"{where}: extra label {name!r} has {length} values, but the answer "
            f"has {answer_len} tokens including EOS, so it must have "
            f"{answer_len} or {answer_len - 1} values."
        )


def check_lengths(
    target_texts: Sequence[str],
    samples: Sequence[dict],
    spec: dict[str, tuple],
    preprocessor: Any,
    source: str,
) -> None:
    """Check every sequence label against its sample's answer length."""
    seq_names = [name for name, (kind, _) in spec.items() if kind == SEQUENCE]
    if not seq_names or preprocessor is None:
        return
    for index, (target, labels) in enumerate(zip(target_texts, samples)):
        where = f"{source}, sample {index} (answer {target!r})"
        try:
            answer_len = answer_length(preprocessor, target)
        except Exception as e:
            raise ValueError(
                f"{where}: the lexer cannot tokenize the answer: {e}"
            ) from e
        for name in seq_names:
            check_length(name, len(labels[name]), answer_len, where)


def collate(
    items: list[dict],
    labels: torch.Tensor,
    spec: dict[str, tuple] | None,
) -> dict[str, torch.Tensor]:
    """Batch the extra labels of ``items`` to match ``labels``."""
    width = labels.size(1)
    answer_lens = (labels != -100).sum(dim=1).tolist()
    names = list(items[0][BATCH_KEY])
    batch = {}
    for name in names:
        values = [item[BATCH_KEY][name] for item in items]
        if spec is not None and name in spec:
            kind, dtype = spec[name]
        else:
            kind = SEQUENCE if isinstance(values[0], list) else SCALAR
            flat = [
                v for value in values for v in (value if kind == SEQUENCE else [value])
            ]
            dtype = (
                torch.float if any(isinstance(v, float) for v in flat) else torch.long
            )
        if kind == SCALAR:
            batch[name] = torch.tensor(values, dtype=dtype)
            continue
        rows = []
        for row, (value, answer_len) in enumerate(zip(values, answer_lens)):
            check_length(
                name,
                len(value),
                answer_len,
                f"batch row {row} (answer {items[row].get('target')!r})",
            )
            rows.append(list(value) + [FILL_VALUE] * (width - len(value)))
        batch[name] = torch.tensor(rows, dtype=dtype)
    return batch
