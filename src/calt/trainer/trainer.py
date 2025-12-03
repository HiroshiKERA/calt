"""Custom HuggingFace Trainer tailored for symbolic computation tasks.

This module introduces `Trainer`, an extension of
:class:`transformers.Trainer` that adds project-specific helpers:

* Device-aware input preparation via :pymeth:`Trainer._prepare_inputs`.
* Automatic metrics computation via :pymeth:`Trainer._compute_metrics`.
* Exact-match generation evaluation with
  :pymeth:`Trainer.evaluate_and_save_generation`.
"""

import json
import os

import numpy as np
import torch
from transformers import Trainer as HTrainer


class Trainer(HTrainer):
    """Extension of *HuggingFace* :class:`~transformers.Trainer`.

    The trainer adds task-specific helpers that simplify training generative
    Transformer models. It accepts all the usual ``HTrainer`` keyword arguments
    and does not introduce new parameters - the default constructor is therefore forwarded verbatim.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keeps a chronological list of metric dictionaries that WandB has
        # seen.  This enables the caller to inspect the *complete* training
        # history after the run has finished without having to query WandB.
        self.log_history = []

        if self.compute_metrics is None:
            self.compute_metrics = self._compute_metrics

    def _prepare_inputs(self, inputs):
        """Move every tensor in "inputs" onto ``self.args.device``.

        Args:
            inputs (dict[str, Any]): Batch dict returned by the data loader.

        Returns:
            dict[str, Any]: The same dictionary with all tensors on the target device.
        """

        return {
            k: (v.to(self.args.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

    def _compute_metrics(self, eval_preds, ignore_index=-100):
        """Compute metrics at each prediction step.

        Args:
            eval_preds (tuple): (predictions, labels) where
                - predictions: shape (batch_size, seq_len)
                - labels: shape (batch_size, seq_len)
            ignore_index (int, optional): Label id to ignore. Defaults to -100.

        Returns:
            dict: Dictionary with accuracy metrics.
        """
        predictions, labels = eval_preds

        # Convert to tensors since inputs are often numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        # Mask tokens with ignore_index
        mask = labels != ignore_index
        correct = (predictions == labels) & mask
        acc = correct.sum().item() / mask.sum().item()

        return {"token_accuracy": acc}

    def evaluate_and_save_generation(self, max_length: int = 512):
        """Run greedy/beam-search generation on the evaluation set.

        The helper decodes the model outputs into strings, stores the results in
        ``eval_results.json`` inside the trainer's output directory and finally computes
        exact-match accuracy between the generated and reference sequences.

        Args:
            max_length (int, optional): Maximum generation length. Defaults to 512.

        Returns:
            float: Exact-match accuracy in the [0, 1] interval.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        all_generated_texts = []
        all_reference_texts = []

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        self.model.eval()
        tokenizer = self.processing_class

        for batch in eval_dataloader:
            inputs = self._prepare_inputs(batch)
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            labels = inputs.get("labels")

            if input_ids is None:
                continue

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    # Optional: specify ``pad_token_id`` / ``eos_token_id`` as
                    # keyword arguments if the model configuration requires.
                )

            # generated_ids shape (batch_size, sequence_length)
            current_generated_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            all_generated_texts.extend(current_generated_texts)

            if labels is not None:
                labels[labels == -100] = tokenizer.pad_token_id
                current_reference_texts = tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_reference_texts.extend(current_reference_texts)
            else:
                # Keep placeholder when reference labels are missing.
                all_reference_texts.extend(["" for _ in current_generated_texts])

        output_eval_file = os.path.join(
            self.args.output_dir,
            "eval_results.json",
        )
        results = []
        for gen_text, ref_text in zip(all_generated_texts, all_reference_texts):
            results.append(
                {
                    "generated": gen_text,
                    "reference": ref_text,
                }
            )

        with open(output_eval_file, "w") as writer:
            json.dump(
                results,
                writer,
                indent=4,
                ensure_ascii=False,
            )

        correct_predictions = 0
        total_predictions = len(all_generated_texts)

        if total_predictions == 0:
            return 0.0

        for gen_text, ref_text in zip(all_generated_texts, all_reference_texts):
            if gen_text.strip() == ref_text.strip():
                correct_predictions += 1

        success_rate = correct_predictions / total_predictions

        return success_rate
