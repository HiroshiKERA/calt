import logging

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast as Tokenizer

from . import extra_labels as extra_labels_module
from .preprocessor import AbstractPreProcessor
from .read import load_dataset_samples, read_data_from_file

# Set up logger for this module
logger = logging.getLogger(__name__)


class StandardDataset(Dataset):
    @classmethod
    def load_file(
        cls,
        data_path: str,
        preprocessor: AbstractPreProcessor | None = None,
        max_samples: int | None = None,
        use_jsonl: bool = False,
        use_pickle: bool = False,
        dataset_load_preprocessor=None,
    ) -> "StandardDataset":
        """Load data from a file and create a ``StandardDataset`` instance.

        When use_jsonl, use_pickle, and dataset_load_preprocessor are all falsy/None,
        behaves as before (text file with input#target per line).

        Args:
            data_path: Path to the data file (.txt, .jsonl, or .pkl).
            preprocessor: Lexer/preprocessor applied in __getitem__. If None, text is returned as-is.
            max_samples: Maximum number of samples. Use -1 or None to load all.
            use_jsonl: If True, read as JSONL.
            use_pickle: If True, read as pickle (original math objects).
            dataset_load_preprocessor: Optional DatasetLoadPreprocessor. If None, uses default per source.

        Returns:
            StandardDataset: Loaded dataset instance.
        """
        extra_labels = None
        if use_jsonl or use_pickle or dataset_load_preprocessor is not None:
            input_texts, target_texts, extra_labels = load_dataset_samples(
                data_path,
                max_samples=max_samples,
                use_jsonl=use_jsonl,
                use_pickle=use_pickle,
                dataset_load_preprocessor=dataset_load_preprocessor,
            )
        else:
            input_texts, target_texts = read_data_from_file(data_path, max_samples)
        return cls(
            input_texts=input_texts,
            target_texts=target_texts,
            preprocessor=preprocessor,
            extra_labels=extra_labels,
        )

    def __init__(
        self,
        input_texts: list[str],
        target_texts: list[str],
        preprocessor: AbstractPreProcessor | None = None,
        extra_labels: list[dict] | None = None,
        **extra_fields,
    ) -> None:
        """Hold the samples of one split.

        Args:
            input_texts: Problem texts (or pre-tokenized id lists).
            target_texts: Answer texts (or pre-tokenized id lists).
            preprocessor: Lexer applied in ``__getitem__``; None returns the
                samples as they are.
            extra_labels: One dict of extra labels per sample, from a load
                preprocessor that returns them (see ``calt.io.extra_labels``),
                or None.
            **extra_fields: Other per-sample fields, kept for compatibility.
        """
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.preprocessor = preprocessor
        self.extra_labels = extra_labels
        self.extra_fields = extra_fields

        num_samples = len(self.input_texts)
        if len(self.target_texts) != num_samples:
            raise ValueError(
                "input_texts and target_texts must have the same number of samples."
            )
        if extra_labels is not None and len(extra_labels) != num_samples:
            raise ValueError(
                f"extra_labels has {len(extra_labels)} samples, but {num_samples} "
                "were expected."
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
            dict[str, str]: A mapping with keys ``"input"`` and ``"target"``,
            plus ``"extra_targets"`` (the extra labels) when the dataset has
            them.
        """
        if self.preprocessor is not None:
            src = self.preprocessor(self.input_texts[idx])
            tgt = self.preprocessor(self.target_texts[idx])
        else:
            # If no preprocessor, return text as-is
            src = self.input_texts[idx]
            tgt = self.target_texts[idx]
        item = {"input": src, "target": tgt}
        if self.extra_labels is not None:
            item[extra_labels_module.BATCH_KEY] = self.extra_labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.input_texts)


class StandardDataCollator:
    def __init__(
        self, tokenizer: Tokenizer = None, extra_label_spec: dict | None = None
    ) -> None:
        """Collate samples into model inputs.

        Args:
            tokenizer: Tokenizer for ``input`` and ``target``.
            extra_label_spec: ``{name: (kind, dtype)}`` decided over the whole
                data by ``IOPipeline.build``, so that every batch gives a label
                the same dtype. Without it, each batch infers its own.
        """
        self.tokenizer = tokenizer
        self.extra_label_spec = extra_label_spec

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
            if attribute == extra_labels_module.BATCH_KEY:
                continue  # needs the labels; batched after this loop
            attribute_batch = [item[attribute] for item in batch]

            # Detect pre-tokenized inputs: each sample is already a list[int]
            # produced by an offline pre-tokenization step (see calt.preprocess).
            # In that case we skip the tokenizer entirely and only pad+tensorize.
            pretokenized = (
                len(attribute_batch) > 0
                and isinstance(attribute_batch[0], list)
                and (
                    len(attribute_batch[0]) == 0
                    or isinstance(attribute_batch[0][0], int)
                )
            )

            if attribute == "input":
                if pretokenized:
                    # Pad to longest using the existing tokenizer's pad_token_id;
                    # this produces the SAME tensor as tokenizer(strings) would
                    # have, given that BOS/EOS were already added at preprocess time.
                    inputs = self.tokenizer.pad(
                        {"input_ids": attribute_batch},
                        padding="longest",
                        return_tensors="pt",
                    )
                else:
                    inputs = self.tokenizer(
                        attribute_batch, padding="longest", return_tensors="pt"
                    )
                batch_dict["input_ids"] = inputs["input_ids"]
                batch_dict["attention_mask"] = inputs["attention_mask"]

            elif attribute == "target":
                if pretokenized:
                    targets = self.tokenizer.pad(
                        {"input_ids": attribute_batch},
                        padding="longest",
                        return_tensors="pt",
                    )
                else:
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

        key = extra_labels_module.BATCH_KEY
        if key in attributes:
            if "labels" not in batch_dict:
                raise ValueError("Extra labels need a target in every sample.")
            # One dict, so the model can tell the labels from other forward()
            # keyword arguments such as the Trainer's num_items_in_batch.
            batch_dict[key] = extra_labels_module.collate(
                batch, batch_dict["labels"], self.extra_label_spec
            )

        return batch_dict
