import logging
import os

# Set up logger for this module
logger = logging.getLogger(__name__)


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
