"""Smoke test for DatasetGenerator's multiprocessing-backed parallel path.

No prior test exercised ``n_jobs > 1`` with ``backend="multiprocessing"``, so
this path was unverified across Python versions. Python 3.14 changed the
default ``multiprocessing`` start method on Linux from "fork" to
"forkserver" (see dataset_generator.py's ``_FORK_CONTEXT`` handling).

The ``instance_generator`` here is a closure (captures ``multiplier`` from
its enclosing scope) specifically because closures aren't picklable via the
stdlib ``pickle`` module. Under "fork", worker processes share the parent's
memory directly, so this works regardless. Under "forkserver", joblib must
pickle the generator to hand it to the worker, which fails (the exact
exception type is Python-version-dependent, e.g. ``PicklingError`` on 3.14)
unless the start method is pinned back to "fork" (as ``_FORK_CONTEXT``
does). A plain module-level generator function would not catch this
regression, since it's trivially picklable either way.
"""

from calt.dataset.sympy.dataset_generator import DatasetGenerator


def _make_instance_generator(multiplier: int):
    def instance_generator(seed: int):
        import random

        rng = random.Random(seed)
        problem = rng.randint(1, 1000)
        answer = problem * multiplier
        return problem, answer

    return instance_generator


def test_dataset_generator_runs_with_multiprocessing_backend(tmp_path):
    generator = DatasetGenerator(
        backend="multiprocessing", n_jobs=2, verbose=False, root_seed=0
    )

    generator.run(
        dataset_sizes={"train": 10},
        instance_generator=_make_instance_generator(multiplier=2),
        save_dir=str(tmp_path),
        save_text=True,
        save_json=True,
        batch_size=4,
    )

    raw_path = tmp_path / "train_raw.txt"
    json_path = tmp_path / "train_data.jsonl"
    assert raw_path.exists()
    assert json_path.exists()
    assert len(raw_path.read_text().strip().splitlines()) == 10
    assert len(json_path.read_text().strip().splitlines()) == 10
