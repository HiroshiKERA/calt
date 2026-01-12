from calt.utils.display import display_with_diff
from calt.utils.data_loader import load_eval_results

def showcase(dataset, 
             success_cases=True,
             num_show=5):
    
    if success_cases:
        indicator_fn = lambda gen, ref: gen == ref
        tag = 'success'
    else:
        indicator_fn = lambda gen, ref: gen != ref
        tag = 'failure'
    
    gen_texts, ref_texts = load_eval_results("results/eval_results.json")
    cases = [(i, gen, ref) for i, (gen, ref) in enumerate(zip(gen_texts, ref_texts)) if indicator_fn(gen, ref)]

    print('-------------------------')
    print(f''' {tag} cases ''')
    print('-------------------------')
    for (i, gen, ref) in cases[:num_show]:
        gen_expr = dataset.preprocessor.decode(gen)
        ref_expr = dataset.preprocessor.decode(ref)
