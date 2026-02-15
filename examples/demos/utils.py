from calt.io.visualization.comparison_vis import load_eval_results


def showcase(dataset, success_cases=True, num_show=5):
    if success_cases:

        def indicator_fn(gen, ref):
            return gen == ref

        tag = "success"
    else:

        def indicator_fn(gen, ref):
            return gen != ref

        tag = "failure"

    gen_texts, ref_texts = load_eval_results("results/eval_results.json")
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
