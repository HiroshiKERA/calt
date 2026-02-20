# Visualization

Utilities to quickly render visual diffs between model predictions and references. Useful after evaluation to inspect outputs.

## display_with_diff

```
display_with_diff(
    gold: Expr | str,
    pred: Expr | str,
    var_order: Sequence[Symbol] | None = None,
) -> None
```

Render "gold" vs. "pred" with strikethrough on mistakes in "pred".

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `gold` | `Expr | str` | Ground-truth expression. If a string, it will be parsed as a token sequence (e.g., "C1 E1 E1 C-3 E0 E7") via parse_poly. | *required* | | `pred` | `Expr | str` | Model-predicted expression. If a string, it will be parsed as a token sequence via parse_poly. | *required* | | `var_order` | `Sequence[Symbol] | None` | Variable ordering (important for >2 variables). Inferred if None. Also passed to parse_poly if inputs are strings. Defaults to None. | `None` |

Source code in `src/calt/io/visualization/comparison_vis.py`

```
def display_with_diff(
    gold: Expr | str,
    pred: Expr | str,
    var_order: Sequence[Symbol] | None = None,
) -> None:
    """Render "gold" vs. "pred" with strikethrough on mistakes in "pred".

    Args:
        gold (sympy.Expr | str):
            Ground-truth expression. If a string, it will be parsed as a token
            sequence (e.g., "C1 E1 E1 C-3 E0 E7") via ``parse_poly``.
        pred (sympy.Expr | str):
            Model-predicted expression. If a string, it will be parsed as a token
            sequence via ``parse_poly``.
        var_order (Sequence[sympy.Symbol] | None, optional):
            Variable ordering (important for >2 variables). Inferred if None. Also
            passed to ``parse_poly`` if inputs are strings. Defaults to None.
    """

    # --- input conversion ------------------------------------------------- #
    if isinstance(gold, str):
        gold = parse_poly(gold, var_names=var_order)
    if isinstance(pred, str):
        pred = parse_poly(pred, var_names=var_order)

    # --- normalize -------------------------------------------------------- #
    if var_order is None:
        var_order = sorted(
            gold.free_symbols.union(pred.free_symbols), key=lambda s: s.name
        )
    gold_poly = Poly(gold.expand(), *var_order)
    pred_poly = Poly(pred.expand(), *var_order)

    gdict = _poly_to_dict(gold_poly)
    pdict = _poly_to_dict(pred_poly)

    # --- diff detection --------------------------------------------------- #
    diff: dict[tuple[int, ...], str] = {}
    for exps in set(gdict) | set(pdict):
        gcoeff = gdict.get(exps, 0)
        pcoeff = pdict.get(exps, 0)
        if pcoeff == 0 and gcoeff != 0:
            continue  # missing term (not highlighted)
        if gcoeff == 0 and pcoeff != 0:
            diff[exps] = "extra"
        elif gcoeff != pcoeff:
            diff[exps] = "coeff_wrong"

    # --- render ----------------------------------------------------------- #
    gold_tex = latex(gold.expand())
    pred_tex = _build_poly_latex(pdict, var_order, diff)

    display(
        Math(
            r"""\begin{aligned}
        \text{Ground truth\,:}\; & {}"""
            + gold_tex
            + r"""\\
        \text{Prediction\,:}\;   & {}"""
            + pred_tex
            + r"""
        \end{aligned}"""
        )
    )
```

## load_eval_results

```
load_eval_results(file_path: str) -> tuple[list[str], list[str]]
```

Load evaluation results from a JSON file.

The JSON file should contain a list of objects with "generated" and "reference" keys.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str` | Path to the JSON file. | *required* |

Returns:

| Type | Description | | --- | --- | | `tuple[list[str], list[str]]` | tuple\[list[str], list[str]\]: A tuple containing two lists: - List of generated texts. - List of reference texts. |

Source code in `src/calt/io/visualization/comparison_vis.py`

```
def load_eval_results(file_path: str) -> tuple[list[str], list[str]]:
    """Load evaluation results from a JSON file.

    The JSON file should contain a list of objects with "generated" and "reference" keys.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - List of generated texts.
            - List of reference texts.
    """
    generated_texts = []
    reference_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        generated_texts.append(item.get("generated", ""))
        reference_texts.append(item.get("reference", ""))

    return generated_texts, reference_texts
```
