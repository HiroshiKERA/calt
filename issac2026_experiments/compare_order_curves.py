"""Compare forward and reverse training curves for an ordering experiment.

Table 4 asks whether the order in which intermediate steps are written changes
what the model learns.  A final success rate answers that only when the two
orders are actually separated; when both saturate, the table stops carrying the
comparison and the curves are what remain informative — how fast the loss falls,
and how long each order takes to reach a given accuracy.

Reads the ``trainer_state.json`` that each run writes into its last checkpoint,
so nothing has to be re-run to produce the comparison.

Usage (from issac2026_experiments):

    python3 compare_order_curves.py \\
        --run "seq2seq forward:relu_recurrence/results" \\
        --run "seq2seq reverse:relu_recurrence/results/reversed" \\
        --run "decoder forward:relu_recurrence/results_decoder" \\
        --run "decoder reverse:relu_recurrence/results_decoder/reversed"
"""

import glob
import json
import os

import click


def _load_history(run_dir: str) -> list:
    """Return the log history of the latest checkpoint under ``run_dir``."""
    checkpoints = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    state_path = os.path.join(checkpoints[-1], "trainer_state.json")
    with open(state_path) as fh:
        return json.load(fh)["log_history"]


def _series(history: list, key: str) -> list:
    """Extract [(epoch, value)] for one logged key."""
    return [(e["epoch"], e[key]) for e in history if key in e]


def _at_epoch(series: list, epoch: float):
    """Last value logged at or before ``epoch``; None if nothing was logged yet."""
    seen = [v for ep, v in series if ep <= epoch]
    return seen[-1] if seen else None


def _first_reaching(series: list, threshold: float):
    """Epoch at which the series first reaches ``threshold``; None if never."""
    for ep, v in series:
        if v >= threshold:
            return ep
    return None


def _fmt(value, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


@click.command()
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    help="LABEL:PATH of a run directory, repeatable.",
)
@click.option(
    "--epochs",
    default="1,2,4,8,16,32,64",
    help="Comma-separated epochs to report.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Optional path to write the comparison as JSON.",
)
def main(runs: tuple, epochs: str, output: str) -> None:
    checkpoints = [float(e) for e in epochs.split(",")]
    parsed = []
    for spec in runs:
        label, _, path = spec.partition(":")
        history = _load_history(path)
        parsed.append(
            {
                "label": label,
                "path": path,
                "train_loss": _series(history, "loss"),
                "eval_loss": _series(history, "eval_loss"),
                "token_accuracy": _series(history, "eval_token_accuracy"),
                "generation": _series(history, "eval_generation_success_rate"),
            }
        )

    width = max(len(r["label"]) for r in parsed) + 2

    for metric, spec in (
        ("train_loss", ".4f"),
        ("eval_loss", ".4f"),
        ("token_accuracy", ".4f"),
        ("generation", ".3f"),
    ):
        print(f"\n{metric} by epoch")
        header = "run".ljust(width) + "".join(f"{e:>10g}" for e in checkpoints)
        print(header)
        print("-" * len(header))
        for run in parsed:
            row = run["label"].ljust(width)
            row += "".join(
                f"{_fmt(_at_epoch(run[metric], e), spec):>10}" for e in checkpoints
            )
            print(row)

    print("\nepoch at which generation accuracy first reaches")
    header = "run".ljust(width) + "".join(f"{t:>10.0%}" for t in (0.5, 0.9, 0.99, 1.0))
    print(header)
    print("-" * len(header))
    for run in parsed:
        row = run["label"].ljust(width)
        row += "".join(
            f"{_fmt(_first_reaching(run['generation'], t), '.1f'):>10}"
            for t in (0.5, 0.9, 0.99, 1.0)
        )
        print(row)

    print("\nfinal")
    for run in parsed:
        gen = run["generation"][-1][1] if run["generation"] else None
        loss = run["eval_loss"][-1][1] if run["eval_loss"] else None
        last_epoch = run["generation"][-1][0] if run["generation"] else None
        print(
            f"  {run['label'].ljust(width)} generation {_fmt(gen, '.3f')}"
            f"  eval_loss {_fmt(loss, '.4f')}  at epoch {_fmt(last_epoch, '.1f')}"
        )

    if output:
        with open(output, "w") as fh:
            json.dump(parsed, fh, indent=2)
        print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
