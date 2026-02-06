"""Load a trained checkpoint and re-run generation evaluation to regenerate eval_results.

Run from the task directory (e.g. arithmetic_addition) so that paths in train.yaml
(./data/, ./configs/) resolve correctly.

Example:
  cd issac2026_experiments/arithmetic_addition
  python ../run_eval.py --checkpoint_dir results/GF7_full/checkpoint-50048
  python ../run_eval.py --checkpoint_dir results/GF7_full  # use final model in save_dir
  python ../run_eval.py --checkpoint_dir results/GF7_full/checkpoint-50048 --step 50048
"""

import json
import os
import re
from pathlib import Path

import click

from calt.trainer.utils import load_from_checkpoint


def _infer_save_dir_and_model_dir(checkpoint_dir: str) -> tuple[str, str | None]:
    """Resolve save_dir (where train.yaml lives) and optional model_dir (checkpoint subdir)."""
    p = Path(checkpoint_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_dir}")

    if (p / "train.yaml").exists():
        save_dir = str(p)
        return save_dir, None  # load model from save_dir
    # Checkpoint subdir (e.g. results/GF7_full/checkpoint-50048)
    save_dir = str(p.parent)
    if not (Path(save_dir) / "train.yaml").exists():
        raise FileNotFoundError(
            f"train.yaml not found in {checkpoint_dir} or parent {save_dir}. "
            "Run this script from the task directory (e.g. arithmetic_addition)."
        )
    return save_dir, str(p)


def _read_step_from_checkpoint(checkpoint_dir: str) -> int | None:
    """Read global_step from trainer_state.json in checkpoint dir if present."""
    path = Path(checkpoint_dir) / "trainer_state.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("global_step")
    except Exception:
        return None


def _step_from_dir_name(checkpoint_dir: str) -> int | None:
    """e.g. checkpoint-50048 -> 50048."""
    name = Path(checkpoint_dir).name
    m = re.match(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def run_eval_for_checkpoint(
    checkpoint_dir: str,
    step: int | None = None,
    max_length: int = 512,
) -> tuple[float, int | None]:
    """Load checkpoint, run evaluate_and_save_generation, return (success_rate, step_used).

    Must be called with cwd set to the task directory (so paths in train.yaml resolve).
    This is a plain function (no Click); use main() for the CLI.
    """
    save_dir, model_dir = _infer_save_dir_and_model_dir(checkpoint_dir)

    io_pipeline, model, trainer_pipeline = load_from_checkpoint(
        save_dir,
        resume_from_checkpoint=True,
        model_dir=model_dir,
        load_train_dataset=False,
        load_eval_dataset=True,
    )
    trainer_pipeline.build()

    if step is None and model_dir is not None:
        step = _read_step_from_checkpoint(model_dir) or _step_from_dir_name(model_dir)
    if step is None:
        step = getattr(trainer_pipeline.trainer.state, "global_step", None)

    success_rate = trainer_pipeline.trainer.evaluate_and_save_generation(
        max_length=max_length,
        num_generation_batches=None,
        step=step,
    )
    return success_rate, step


@click.command()
@click.option(
    "--checkpoint_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to save_dir (e.g. results/GF7_full) or checkpoint subdir (e.g. results/GF7_full/checkpoint-50048).",
)
@click.option(
    "--step",
    type=int,
    default=None,
    help="Step number for output filename step_<step>.json. Default: inferred from checkpoint dir or trainer_state.json.",
)
@click.option(
    "--max_length",
    type=int,
    default=512,
    help="Max generation length.",
)
def main(checkpoint_dir: str, step: int | None, max_length: int) -> None:
    """Load a trained model and re-run evaluate_and_save_generation to regenerate eval_results."""
    os.environ["WANDB_DISABLED"] = "true"  # eval-only: do not log to wandb
    success_rate, step_used = run_eval_for_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        max_length=max_length,
    )
    print(f"Success rate: {100 * success_rate:.1f}%")
    save_dir, _ = _infer_save_dir_and_model_dir(checkpoint_dir)
    out_dir = Path(save_dir) / "eval_results"
    if step_used is not None:
        print(f"Results written to {out_dir / f'step_{step_used}.json'}")
    else:
        print(f"Results written to {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
