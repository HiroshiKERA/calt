import logging
import os

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast as Tokenizer

from .preprocessor import AbstractPreprocessor

# Set up logger for this module
logger = logging.getLogger(__name__)


def _read_data_from_file(
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


class StandardDataset(Dataset):
    @classmethod
    def load_file(
        cls,
        data_path: str,
        preprocessor: AbstractPreprocessor,
        max_samples: int | None = None,
    ) -> "StandardDataset":
        """Load data from a file and create a ``StandardDataset`` instance.

        This method maintains backward compatibility with the previous file-based initialization.

        Args:
            data_path (str): Path to the data file.
            preprocessor (AbstractPreprocessor): Preprocessor instance.
            max_samples (int | None, optional): Maximum number of samples to load.
                Use -1 or None to load all samples. Defaults to None.

        Returns:
            StandardDataset: Loaded dataset instance.
        """
        input_texts, target_texts = _read_data_from_file(data_path, max_samples)
        return cls(
            input_texts=input_texts,
            target_texts=target_texts,
            preprocessor=preprocessor,
        )

    def __init__(
        self,
        input_texts: list[str],
        target_texts: list[str],
        preprocessor: AbstractPreprocessor,
        **extra_fields,
    ) -> None:
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.preprocessor = preprocessor
        self.extra_fields = extra_fields

        num_samples = len(self.input_texts)
        if len(self.target_texts) != num_samples:
            raise ValueError(
                "input_texts and target_texts must have the same number of samples."
            )

        for name, data in self.extra_fields.items():
            if len(data) != num_samples:
                raise ValueError(
                    f"Extra field '{name}' has {len(data)} samples, but {num_samples} were expected."
                )

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get dataset item and convert to internal representation.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict[str, str]: A mapping with keys ``"input"`` and ``"target"``.
        """
        src = self.preprocessor(self.input_texts[idx])
        tgt = self.preprocessor(self.target_texts[idx])
        return {"input": src, "target": tgt}

    def __len__(self) -> int:
        return len(self.input_texts)


class StandardDataCollator:
    def __init__(self, tokenizer: Tokenizer = None) -> None:
        self.tokenizer = tokenizer

    def _pad_sequences(self, sequences, padding_value=0):
        """Pad a list of sequences and convert them to a tensor.

        Args:
            sequences (list[list[int]]): Sequences to pad.
            padding_value (int, optional): Value used for padding. Defaults to 0.

        Returns:
            torch.Tensor: Padded tensor with BOS/EOS room allocated.
        """
        # Calculate the maximum length of the sequences.
        max_length = max(len(seq) for seq in sequences)

        # Apply padding.
        padded_sequences = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            # Pad the sequence with the specified padding value.
            padded_seq = seq + [padding_value] * padding_length
            padded_sequences.append(padded_seq)

        # '+2' for bos/eos tokens.
        # Initialize a tensor of zeros with the appropriate shape.
        padded = torch.zeros(len(sequences), max_length + 2, dtype=torch.long)
        # Fill the tensor with the padded sequences, leaving space for BOS/EOS tokens.
        padded[:, 1 : max_length + 1] = torch.tensor(padded_sequences)

        return padded

    def __call__(self, batch):
        """Collate a batch of data samples.

        If a tokenizer is provided, it tokenizes ``input`` and ``target`` attributes.
        Other attributes starting with ``target_`` are prefixed with ``decoder_`` and padded.

        Args:
            batch (list[dict[str, Any]]): Mini-batch samples.

        Returns:
            dict[str, torch.Tensor | list[str]]: Batched tensors and/or lists.
        """
        batch_dict = {}

        # Get the attributes from the first item in the batch.
        attributes = batch[0].keys()

        if self.tokenizer is None:
            # If no tokenizer is provided, return the batch as is.
            for attribute in attributes:
                attribute_batch = [item[attribute] for item in batch]
                batch_dict[attribute] = attribute_batch

            return batch_dict

        for attribute in attributes:
            attribute_batch = [item[attribute] for item in batch]

            if attribute == "input":
                # Tokenize the input sequences.
                inputs = self.tokenizer(
                    attribute_batch, padding="longest", return_tensors="pt"
                )
                batch_dict["input_ids"] = inputs["input_ids"]
                batch_dict["attention_mask"] = inputs["attention_mask"]

            elif attribute == "target":
                # Tokenize the target sequences.
                targets = self.tokenizer(
                    attribute_batch, padding="longest", return_tensors="pt"
                )
                # Prepare decoder input ids (remove the last token, usually EOS).
                batch_dict["decoder_input_ids"] = targets["input_ids"][
                    :, :-1
                ].contiguous()
                # Prepare decoder attention mask accordingly.
                batch_dict["decoder_attention_mask"] = targets["attention_mask"][
                    :, :-1
                ].contiguous()

                # Prepare labels for the loss calculation (shift by one, usually remove BOS).
                labels = targets["input_ids"][:, 1:].contiguous()
                label_attention_mask = (
                    targets["attention_mask"][:, 1:].contiguous().bool()
                )
                # Set padding tokens in labels to -100 to be ignored by the loss function.
                labels[~label_attention_mask] = -100
                batch_dict["labels"] = labels

            else:
                # For other attributes, if they start with 'target_',
                # prefix them with 'decoder_' (e.g., 'target_aux' becomes 'decoder_aux').
                if attribute.startswith("target_"):
                    attribute_key = (
                        "decoder_" + attribute[7:]
                    )  #  Corrected key for batch_dict
                else:
                    attribute_key = (
                        attribute  # Use original attribute name if no prefix
                    )
                # Pad the sequences for these attributes.
                batch_dict[attribute_key] = self._pad_sequences(
                    attribute_batch, padding_value=0
                )

        return batch_dict
