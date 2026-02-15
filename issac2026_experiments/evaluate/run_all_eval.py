"""
Run evaluate_and_save_generation for all tasks and all result patterns, then output a success_rate table.

Use the calt-env conda environment. Execute from issac2026_experiments/:

  conda activate calt-env
  python evaluate/run_all_eval.py

Or use the wrapper script:  bash evaluate/run_all_eval.sh

- Discovers all (task, save_dir) pairs: directories under each task that contain train.yaml.
- For each, uses the latest checkpoint (or final model in save_dir) and runs generation eval.
- Writes success_rate_table.csv and success_rate_table.md under evaluate/.
- Marks 訓練途中 if checkpoint epoch < num_train_epochs; otherwise 完了.
- Copies eval_results to evaluate/<task>/<pattern>/eval_results/ (original location is also written).
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

# Run from issac2026_experiments; ensure calt and run_eval are importable
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # issac2026_experiments
CALT_SRC = ROOT.parent / "src"  # calt-dev/calt/src
for _ in (CALT_SRC, ROOT):
    if str(_) not in sys.path:
        sys.path.insert(0, str(_))

os.environ["WANDB_DISABLED"] = "true"

TASK_DIRS = [
    "arithmetic_addition",
    "arithmetic_factorization",
    "digit_product",
    "groebner",
    "polynomial_multiplication",
    "polynomial_reduction",
    "relu_recurrence",
]


def _discover_save_dirs(task_dir: Path) -> list[tuple[str, str]]:
    """Return [(relative_path, pattern_name), ...] for this task."""
    out = []
    # results/ (single save_dir or subdirs like results/GF7_full, results/reversed)
    results = task_dir / "results"
    if results.exists():
        if (results / "train.yaml").exists():
            out.append(("results", "default"))
        for d in sorted(results.iterdir()):
            if d.is_dir() and (d / "train.yaml").exists():
                out.append((str(d.relative_to(task_dir)), d.name))
    # results_reversed (legacy top-level, e.g. arithmetic_factorization)
    res_rev = task_dir / "results_reversed"
    if res_rev.exists() and (
        (res_rev / "train.yaml").exists() or (res_rev / "pytorch_model.bin").exists()
    ):
        out.append(("results_reversed", "results_reversed"))
    # results_* (groebner)
    for p in sorted(task_dir.glob("results_*")):
        if p.is_dir() and (
            (p / "train.yaml").exists() or (p / "pytorch_model.bin").exists()
        ):
            out.append((p.name, p.name))
    return out


def _latest_checkpoint(save_dir: Path) -> Path | None:
    """Return path to checkpoint with largest step, or None if no checkpoint-*."""
    checkpoints = list(save_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    def step(p: Path) -> int:
        m = re.match(r"checkpoint-(\d+)$", p.name)
        return int(m.group(1)) if m else 0

    return max(checkpoints, key=step)


def _choose_eval_path(task_dir: Path, rel_path: str) -> str:
    """Return path to use for eval: latest checkpoint if any, else save_dir (relative to task_dir)."""
    save_dir = task_dir / rel_path
    if not save_dir.exists():
        return rel_path
    ckpt = _latest_checkpoint(save_dir)
    if ckpt is not None:
        return str(ckpt.relative_to(task_dir))
    return rel_path


def _is_training_complete(task_dir: Path, save_dir_rel: str, eval_path: str) -> bool:
    """True if the checkpoint used is from a finished training (epoch >= num_train_epochs)."""
    save_dir = task_dir / save_dir_rel
    train_yaml = save_dir / "train.yaml"
    if not train_yaml.exists():
        return False
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(train_yaml)
        num_epochs = float(getattr(cfg.train, "num_train_epochs", 0) or 0)
    except Exception:
        return False
    ckpt_path = task_dir / eval_path
    state_file = ckpt_path / "trainer_state.json"
    if not state_file.exists():
        return False
    try:
        with open(state_file) as f:
            state = json.load(f)
        epoch = float(state.get("epoch", 0))
    except Exception:
        return False
    return epoch >= num_epochs - 0.01


def _write_tables(
    rows: list[tuple[str, str, float | None, int | None, bool]],
    script_dir: Path,
) -> None:
    """Write success_rate_table.csv and success_rate_table.md (incremental update)."""
    csv_path = script_dir / "success_rate_table.csv"
    with open(csv_path, "w") as f:
        f.write("task,pattern,success_rate,checkpoint_step,training_status\n")
        for task, pattern, rate, ckpt, complete in rows:
            val = f"{rate:.4f}" if rate is not None else ""
            status = "完了" if complete else "訓練途中"
            ckpt_str = str(ckpt) if ckpt is not None else ""
            f.write(f"{task},{pattern},{val},{ckpt_str},{status}\n")
    md_path = script_dir / "success_rate_table.md"
    with open(md_path, "w") as f:
        f.write("# Success rate (generation exact match)\n\n")
        f.write("| task | pattern | success_rate | checkpoint | training_status |\n")
        f.write("|------|---------|-------------|------------|----------------|\n")
        for task, pattern, rate, ckpt, complete in rows:
            val = f"{100 * rate:.1f}%" if rate is not None else "—"
            status = "完了" if complete else "訓練途中"
            ckpt_str = str(ckpt) if ckpt is not None else "—"
            f.write(f"| {task} | {pattern} | {val} | {ckpt_str} | {status} |\n")


def run_one_eval(
    task_name: str, rel_path: str, pattern: str, max_length: int
) -> tuple[float | None, int | None]:
    """Run eval for one (task, save_dir). cwd must be task dir. Returns (success_rate, step_used) or (None, None) on error."""
    import run_eval  # run_eval is in ROOT (issac2026_experiments)

    task_dir = Path(".").resolve()
    eval_path = _choose_eval_path(task_dir, rel_path)
    # save_dir = where train.yaml lives (eval_results will be written under it)
    save_dir_rel = (
        rel_path if "checkpoint-" not in eval_path else str(Path(eval_path).parent)
    )
    out_dir = task_dir / save_dir_rel / "eval_results"

    print(
        f"  [{pattern}] モデル: {eval_path} をロードし、全評価データで生成評価を実行します。"
    )
    print(f"  [{pattern}] 結果の保存先: {out_dir}/")
    try:
        rate, step_used = run_eval.run_eval_for_checkpoint(
            checkpoint_dir=eval_path,
            step=None,
            max_length=max_length,
        )
        return rate, step_used
    except Exception as e:
        print(f"  ERROR {task_name} / {pattern}: {e}", file=sys.stderr)
        return None, None


def main(max_length: int = 512) -> None:
    experiments_root = ROOT

    print("=" * 60)
    print(
        "全タスク・全パターンで生成評価 (evaluate_and_save_generation) を実行します。"
    )
    print(f"実行ディレクトリ: {experiments_root}")
    print(f"max_length: {max_length}")
    print(f"対象タスク: {TASK_DIRS}")
    print("=" * 60)

    # (task, pattern, success_rate, checkpoint_step, training_complete)
    rows: list[tuple[str, str, float | None, int | None, bool]] = []
    for task_name in TASK_DIRS:
        task_dir = experiments_root / task_name
        if not task_dir.is_dir():
            print(f"\n[Skip] タスクディレクトリがありません: {task_name}")
            continue
        pairs = _discover_save_dirs(task_dir)
        if not pairs:
            print(f"\n[Skip] 保存ディレクトリが見つかりません: {task_name}")
            continue
        print(f"\n--- Task: {task_name} ({len(pairs)} パターン) ---")
        print(f"  検出したパターン: {[p[1] for p in pairs]}")
        for rel_path, pattern in pairs:
            save_dir = task_dir / rel_path
            # save_dir may be a checkpoint subdir (no train.yaml); then parent has train.yaml
            if not save_dir.exists():
                print(f"  [Skip] {pattern}: パスが存在しません ({rel_path})")
                rows.append((task_name, pattern, None, None, False))
                _write_tables(rows, SCRIPT_DIR)
                continue
            os.chdir(task_dir)
            eval_path = _choose_eval_path(task_dir, rel_path)
            save_dir_rel = (
                rel_path
                if "checkpoint-" not in eval_path
                else str(Path(eval_path).parent)
            )
            rate, step_used = run_one_eval(task_name, rel_path, pattern, max_length)
            training_complete = (
                _is_training_complete(task_dir, save_dir_rel, eval_path)
                if step_used is not None
                else False
            )
            rows.append((task_name, pattern, rate, step_used, training_complete))
            if rate is not None:
                status = "完了" if training_complete else "訓練途中"
                print(
                    f"  [{pattern}] success_rate: {100 * rate:.1f}% (checkpoint-{step_used}, {status})"
                )
            # eval_results を evaluate/<task>/<pattern>/eval_results/ にもコピー（上書き）
            if step_used is not None and rate is not None:
                src = (
                    task_dir / save_dir_rel / "eval_results" / f"step_{step_used}.json"
                )
                if src.exists():
                    dst_dir = SCRIPT_DIR / task_name / pattern / "eval_results"
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / f"step_{step_used}.json"
                    shutil.copy2(src, dst)
                    print(f"  [{pattern}] コピー: {dst}")
            _write_tables(rows, SCRIPT_DIR)
            os.chdir(experiments_root)

    print("\n" + "=" * 60)
    print(
        f"success_rate の表: {SCRIPT_DIR / 'success_rate_table.csv'}, {SCRIPT_DIR / 'success_rate_table.md'}（逐次更新済み）"
    )
    print("=" * 60)
    print(
        f"完了: {len(rows)} 件処理（成功 {sum(1 for r in rows if r[2] is not None)} 件）"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    main(max_length=args.max_length)
