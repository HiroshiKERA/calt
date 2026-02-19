from __future__ import annotations

import json
import logging
import os
import pickle
from typing import TYPE_CHECKING

# Set up logger for this module
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .preprocessor.load_preprocessor import DatasetLoadPreprocessor


def read_data_from_file(
    data_path: str, max_samples: int | None = None
) -> tuple[list[str], list[str]]:
    """Read input and target texts from a file.

    Args:
        data_path (str): Path to the data file.
        max_samples (int | None, optional): Maximum number of samples to read.
            Use -1 or None to load all samples. Defaults to None.

    Returns:
        tuple[list[str], list[str]]: Two lists of strings for inputs and targets.
    """
    input_texts = []
    target_texts = []

    # Convert -1 to None to load all samples
    if max_samples == -1:
        max_samples = None

    # Validate max_samples parameter
    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            f"max_samples must be positive or -1 (to load all samples), got {max_samples}"
        )

    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load and parse the data file
    with open(data_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue  # Skip empty lines

            # Split input and target expressions using "#" delimiter
            if "#" not in line:
                continue  # Skip lines with unexpected format (no delimiter)

            input_part, target_part = line.split("#", 1)
            input_texts.append(input_part.strip())
            target_texts.append(target_part.strip())

            # Stop loading if max_samples is reached
            if max_samples is not None and len(input_texts) >= max_samples:
                break

    # Log information about loaded samples
    if max_samples is not None and len(input_texts) < max_samples:
        logger.warning(
            f"WARNING Requested {max_samples} samples but only {len(input_texts)} samples found in {data_path}"
        )
    elif max_samples is not None:
        logger.info(
            f"Loaded {len(input_texts)} samples (limited to {max_samples}) from {data_path}"
        )
    else:
        logger.info(f"Loaded {len(input_texts)} samples from {data_path}")

    return input_texts, target_texts


def read_data_from_jsonl(
    data_path: str, max_samples: int | None = None
) -> list[tuple[object, object]]:
    """Read problem-answer pairs from a JSON Lines file.

    Each line must be a JSON object with "problem" and "answer" keys (or "solution"
    for backward compatibility). Returns raw parsed data (problem/answer may be str,
    list, or dict) for use with a DatasetLoadPreprocessor.

    Args:
        data_path: Path to the .jsonl file.
        max_samples: Maximum number of samples to read. Use -1 or None for all.

    Returns:
        List of (problem, answer) pairs as parsed from JSON (not yet stringified).
    """
    if max_samples == -1:
        max_samples = None
    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            f"max_samples must be positive or -1 (to load all samples), got {max_samples}"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                problem = data.get("problem")
                answer = data.get("answer") or data.get("solution")
                if problem is None or answer is None:
                    logger.warning(
                        f"Skip line {line_num} in {data_path}: missing 'problem' or 'answer'/'solution' key"
                    )
                    continue
                samples.append((problem, answer))
            except json.JSONDecodeError as e:
                logger.warning(f"Skip line {line_num} in {data_path}: {e}")
                continue
            if max_samples is not None and len(samples) >= max_samples:
                break

    if max_samples is not None and len(samples) < max_samples:
        logger.warning(
            f"Requested {max_samples} samples but only {len(samples)} found in {data_path}"
        )
    elif max_samples is not None:
        logger.info(
            f"Loaded {len(samples)} samples (limited to {max_samples}) from {data_path}"
        )
    else:
        logger.info(f"Loaded {len(samples)} samples from {data_path}")

    return samples


def read_data_from_pickle(
    data_path: str, max_samples: int | None = None
) -> list[tuple[object, object]]:
    """Read problem-answer pairs from a pickle file.

    Pickle preserves original Python/SageMath objects, so mathematical
    preprocessing can be applied in a DatasetLoadPreprocessor.

    Args:
        data_path: Path to the .pkl file.
        max_samples: Maximum number of samples to read. Use -1 or None for all.

    Returns:
        List of (problem, answer) pairs as loaded from pickle.
    """
    if max_samples == -1:
        max_samples = None
    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            f"max_samples must be positive or -1 (to load all samples), got {max_samples}"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list):
        raise ValueError(
            f"Pickle file must contain a list of (problem, answer), got {type(samples)}"
        )
    if max_samples is not None:
        samples = samples[:max_samples]
        logger.info(
            f"Loaded {len(samples)} samples (limited to {max_samples}) from {data_path}"
        )
    else:
        logger.info(f"Loaded {len(samples)} samples from {data_path}")

    return samples


def load_dataset_texts(
    data_path: str,
    max_samples: int | None = None,
    use_jsonl: bool = False,
    use_pickle: bool = False,
    dataset_load_preprocessor: "DatasetLoadPreprocessor | None" = None,
) -> tuple[list[str], list[str]]:
    """Load dataset and apply load-time preprocessor to get (input_texts, target_texts).

    Args:
        data_path: Path to .txt, .jsonl, or .pkl file.
        max_samples: Maximum number of samples. -1 or None for all.
        use_jsonl: If True, read as JSONL.
        use_pickle: If True, read as pickle (original math objects).
        dataset_load_preprocessor: Optional DatasetLoadPreprocessor. If None,
            uses TextDefaultLoadPreprocessor, JsonlDefaultLoadPreprocessor,
            or PickleDefaultLoadPreprocessor according to source.

    Returns:
        (input_texts, target_texts) for StandardDataset.
    """
    from .preprocessor.load_preprocessor import (
        JsonlDefaultLoadPreprocessor,
        PickleDefaultLoadPreprocessor,
        TextDefaultLoadPreprocessor,
    )

    if dataset_load_preprocessor is None:
        if use_pickle:
            dataset_load_preprocessor = PickleDefaultLoadPreprocessor()
        elif use_jsonl:
            dataset_load_preprocessor = JsonlDefaultLoadPreprocessor()
        else:
            dataset_load_preprocessor = TextDefaultLoadPreprocessor()

    if use_pickle:
        raw_samples = read_data_from_pickle(data_path, max_samples)
        input_texts = []
        target_texts = []
        for problem, answer in raw_samples:
            inp, tgt = dataset_load_preprocessor.process_sample(
                {"problem": problem, "answer": answer}
            )
            input_texts.append(inp)
            target_texts.append(tgt)
        return input_texts, target_texts

    if use_jsonl:
        raw_samples = read_data_from_jsonl(data_path, max_samples)
        input_texts = []
        target_texts = []
        for problem, answer in raw_samples:
            inp, tgt = dataset_load_preprocessor.process_sample(
                {"problem": problem, "answer": answer}
            )
            input_texts.append(inp)
            target_texts.append(tgt)
        return input_texts, target_texts

    # Text: read lines and apply preprocessor to each line
    if max_samples == -1:
        max_samples = None
    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            f"max_samples must be positive or -1 (to load all samples), got {max_samples}"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    input_texts = []
    target_texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            inp, tgt = dataset_load_preprocessor.process_sample(line)
            input_texts.append(inp)
            target_texts.append(tgt)
            if max_samples is not None and len(input_texts) >= max_samples:
                break

    if max_samples is not None and len(input_texts) < max_samples:
        logger.warning(
            f"Requested {max_samples} samples but only {len(input_texts)} found in {data_path}"
        )
    elif max_samples is not None:
        logger.info(
            f"Loaded {len(input_texts)} samples (limited to {max_samples}) from {data_path}"
        )
    else:
        logger.info(f"Loaded {len(input_texts)} samples from {data_path}")

    return input_texts, target_texts
