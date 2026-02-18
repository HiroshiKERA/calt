"""Custom HuggingFace Trainer tailored for symbolic computation tasks.

This module introduces `Trainer`, an extension of
:class:`transformers.Trainer` that adds project-specific helpers:

* Device-aware input preparation via :pymeth:`Trainer._prepare_inputs`.
* Automatic metrics computation via :pymeth:`Trainer._compute_metrics`.
* Exact-match generation evaluation with
  :pymeth:`Trainer.evaluate_and_save_generation`.
"""

import json
import logging
import os

import numpy as np
import torch
from transformers import Trainer as HTrainer

logger = logging.getLogger(__name__)


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
            eval_preds (tuple or EvalPrediction): (predictions, labels) or EvalPrediction object where
                - predictions: shape (batch_size, seq_len) or (batch_size, seq_len, vocab_size) if logits
                - labels: shape (batch_size, seq_len)
            ignore_index (int, optional): Label id to ignore. Defaults to -100.

        Returns:
            dict: Dictionary with accuracy metrics including:
                - token_accuracy: Token-level accuracy (fraction of correct tokens)
                - success_rate: Sequence-level accuracy (fraction of sequences that match exactly)
        """
        # Handle both tuple and EvalPrediction object formats
        if hasattr(eval_preds, "predictions") and hasattr(eval_preds, "label_ids"):
            # EvalPrediction object
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
        else:
            # Tuple format
            predictions, labels = eval_preds

        # Handle tuple predictions (e.g., (logits, ...) from model output)
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # Take first element (usually logits)

        # Convert to tensors since inputs are often numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        # If predictions are logits (3D: batch_size, seq_len, vocab_size), convert to token IDs
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)

        # Mask tokens with ignore_index
        mask = labels != ignore_index
        correct = (predictions == labels) & mask
        token_acc = correct.sum().item() / mask.sum().item()

        # Compute success rate (exact sequence match)
        # For each sequence, check if all non-ignored tokens match exactly
        batch_size = predictions.shape[0]
        exact_matches = 0

        for i in range(batch_size):
            # Get valid tokens (non-ignored) for this sequence
            seq_mask = mask[i]
            if seq_mask.sum().item() == 0:
                # Skip sequences with no valid tokens
                continue

            # Compare only the valid tokens
            pred_seq = predictions[i][seq_mask]
            label_seq = labels[i][seq_mask]

            # Check if sequences match exactly
            if pred_seq.shape == label_seq.shape and torch.equal(pred_seq, label_seq):
                exact_matches += 1

        success_rate = exact_matches / batch_size if batch_size > 0 else 0.0

        return {
            "token_accuracy": token_acc,
            "success_rate": success_rate,
        }

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override evaluate to also save generation results during training.

        This method is called during training evaluation steps and after training.
        It runs the standard evaluation and then saves generation results.
        """
        # Run standard evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Also save generation results during evaluation
        # Get current step number if available
        step = getattr(self.state, "global_step", None)
        logger.info(
            f"Running evaluate_and_save_generation (step={step}, metric_key_prefix={metric_key_prefix})"
        )
        try:
            # Pass step number to evaluate_and_save_generation
            success_rate = self.evaluate_and_save_generation(step=step)
            # Add generation metrics to the returned metrics dict
            generation_metrics = {
                f"{metric_key_prefix}_generation_success_rate": success_rate,
            }
            if step is not None:
                generation_metrics[f"{metric_key_prefix}_generation_step"] = step
            # Update the metrics dict
            metrics.update(generation_metrics)
            # Explicitly log only the generation metrics to ensure they are recorded
            self.log(generation_metrics)
            logger.info(
                f"Successfully saved generation results (step={step}, success_rate={success_rate:.4f})"
            )
        except Exception as e:
            # Log error but don't fail the evaluation
            logger.warning(
                f"Failed to save generation results during evaluation: {e}",
                exc_info=True,
            )

        return metrics

    def evaluate_and_save_generation(
        self, max_length: int = 512, step: int | None = None
    ):
        """Run greedy/beam-search generation on the evaluation set.

        The helper decodes the model outputs into strings, stores the results in
        ``eval_results.json`` inside the trainer's output directory and finally computes
        exact-match accuracy between the generated and reference sequences.

        Args:
            max_length (int, optional): Maximum generation length. Defaults to 512.
            step (int, optional): Current training step number. If None, tries to get from self.state.

        Returns:
            float: Exact-match accuracy in the [0, 1] interval.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if len(self.eval_dataset) == 0:
            logger.warning(
                "eval_dataset is empty; skipping evaluate_and_save_generation."
            )
            return 0.0

        all_generated_texts = []
        all_reference_texts = []

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        self.model.eval()
        tokenizer = self.processing_class

        for batch in eval_dataloader:
            if batch is None:
                continue
            inputs = self._prepare_inputs(batch)
            if inputs is None:
                continue
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
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
                current_reference_texts = tokenizer.batch_decode(
                    labels_for_decode,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_reference_texts.extend(current_reference_texts)
            else:
                # Keep placeholder when reference labels are missing.
                all_reference_texts.extend(["" for _ in current_generated_texts])

        # Include step number in filename if available during training
        if step is None:
            step = getattr(self.state, "global_step", None)
        if step is not None:
            # Save step-wise results in a subdirectory
            eval_results_dir = os.path.join(self.args.output_dir, "eval_results")
            os.makedirs(eval_results_dir, exist_ok=True)
            output_eval_file = os.path.join(
                eval_results_dir,
                f"step_{step}.json",
            )
        else:
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
