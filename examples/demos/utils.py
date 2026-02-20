import os

from calt.io.visualization.comparison_vis import load_eval_results


def _resolve_eval_results_path(eval_results_path=None):
    """Resolve path to eval results JSON. Prefer step_*.json in eval_results/ if single file missing."""
    if eval_results_path is not None:
        return eval_results_path
    path = "results/eval_results.json"
    if os.path.isfile(path):
        return path
    eval_dir = "results/eval_results"
    if os.path.isdir(eval_dir):
        steps = [
            f
            for f in os.listdir(eval_dir)
            if f.startswith("step_") and f.endswith(".json")
        ]
        if steps:
            steps.sort(key=lambda f: int(f.replace("step_", "").replace(".json", "")))
            return os.path.join(eval_dir, steps[-1])
    return path  # fallback (will raise FileNotFoundError with clear path)


def showcase(dataset, success_cases=True, num_show=5, eval_results_path=None):
    if success_cases:

        def indicator_fn(gen, ref):
            return gen == ref

        tag = "success"
    else:

        def indicator_fn(gen, ref):
            return gen != ref

        tag = "failure"

    path = _resolve_eval_results_path(eval_results_path)
    gen_texts, ref_texts = load_eval_results(path)
    cases = [
        (i, gen, ref)
        for i, (gen, ref) in enumerate(zip(gen_texts, ref_texts))
        if indicator_fn(gen, ref)
    ]

    print("-------------------------")
    print(f""" {tag} cases """)
    print("-------------------------")
    for i, gen, ref in cases[:num_show]:
        gen_expr = dataset.preprocessor.decode(gen)
        ref_expr = dataset.preprocessor.decode(ref)
        print(f"  [{i}] gen: {gen_expr}  |  ref: {ref_expr}")
