"""Measure Groebner-basis computation time, lex vs degrevlex, with slimgb.

Context: in computer algebra, degrevlex is preferred over lex for a large
computational-efficiency gain, which is why the standard route to a lex basis is
to compute in degrevlex and convert with FGLM.  Table 5 of the paper asks
whether the same preference shows up in *learning*.  This script measures the
classical side of that comparison on exactly the systems the Table 5 datasets
are drawn from, so the two gaps can be put side by side.

Both orders are timed with ``libsingular:slimgb`` on the same ideal.  stdfglm is
deliberately not used: it is the degrevlex+FGLM route, so timing it would
measure the conversion strategy rather than a direct lex computation.

Usage (from issac2026_experiments/groebner):

    sage -python measure_gb_timing.py --config_path configs/data_GF7.yaml
    sage -python measure_gb_timing.py --config_path configs/data.yaml

Results are appended to ``gb_timing_results.json`` next to this file.
"""

import json
import os
import time

import click

# Initialize Sage's polynomial ring stack first, as generate_dataset.py does.
import sage.all  # noqa: F401
import sage.misc.randstate as randstate  # type: ignore
from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm  # type: ignore
from omegaconf import OmegaConf
from sage.all import GF, QQ, ZZ, PolynomialRing  # type: ignore

from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler

ALGORITHM = "libsingular:slimgb"


def _has_large_rational_coefficients(polynomials, threshold: int = 100) -> bool:
    """Same acceptance test as generate_dataset.py, so the systems match."""
    for p in polynomials:
        for c in p.coefficients():
            try:
                num = c.numerator()
                den = c.denominator()
            except Exception:
                continue
            if abs(int(num)) >= threshold or abs(int(den)) >= threshold:
                return True
    return False


def _base_ring(field_str: str):
    if field_str == "QQ":
        return QQ
    if field_str == "ZZ":
        return ZZ
    if field_str.startswith("GF"):
        return GF(int(field_str[2:]))
    raise ValueError(f"unsupported field_str: {field_str}")


def _time_groebner(ring, F, repeats: int = 1) -> tuple[float, int]:
    """Time one Groebner basis computation in ``ring``; return (seconds, size).

    A fresh ideal is built for every call so Singular cannot return a cached
    basis from a previous timing.  On the paper's 2-variable systems a single
    computation lands around 0.1 ms, close enough to the timer floor that the
    fastest of ``repeats`` runs is the honest estimate.
    """
    best = None
    size = 0
    for _ in range(repeats):
        F_ring = [ring(f) for f in F]
        ideal = ring.ideal(F_ring)
        start = time.perf_counter()
        G = ideal.groebner_basis(algorithm=ALGORITHM)
        elapsed = time.perf_counter() - start
        size = len(list(G))
        best = elapsed if best is None else min(best, elapsed)
    return best, size


def _summarize(times: list[float]) -> dict:
    ordered = sorted(times)
    n = len(ordered)
    return {
        "n": n,
        "total_s": sum(ordered),
        "mean_s": sum(ordered) / n,
        "median_s": ordered[n // 2],
        "p90_s": ordered[int(0.9 * n)],
        "max_s": ordered[-1],
    }


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Data config of the Table 5 dataset to measure (same sampler settings).",
)
@click.option(
    "--num_samples",
    type=int,
    default=1000,
    help="Number of systems F=[f1,f2] to time.",
)
@click.option(
    "--root_seed",
    type=int,
    default=None,
    help="Seed offset; defaults to the dataset's root_seed.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="gb_timing_results.json",
    help="Where to append the measurement.",
)
@click.option(
    "--repeats",
    type=int,
    default=5,
    help="Timed repetitions per system; the fastest is kept.",
)
@click.option(
    "--symbols",
    type=str,
    default=None,
    help="Override the sampler's variables, e.g. 'x,y,z'. The paper's Table 5 "
    "uses two; sweeping this shows where the classical lex penalty appears.",
)
@click.option(
    "--max_degree",
    type=int,
    default=None,
    help="Override the sampler's max_degree.",
)
@click.option(
    "--num_polynomials",
    "num_polynomials_opt",
    type=int,
    default=None,
    help="Override the number of polynomials in F.",
)
@click.option(
    "--timeout_s",
    type=float,
    default=60.0,
    help="Skip (and count) systems whose degrevlex basis exceeds this budget.",
)
def main(
    config_path: str,
    num_samples: int,
    root_seed: int,
    output: str,
    repeats: int,
    symbols: str,
    max_degree: int,
    num_polynomials_opt: int,
    timeout_s: float,
) -> None:
    cfg = OmegaConf.load(config_path)
    sampler_cfg = dict(OmegaConf.to_container(cfg.sampler, resolve=True))
    if symbols is not None:
        sampler_cfg["symbols"] = symbols
    if max_degree is not None:
        sampler_cfg["max_degree"] = max_degree
    field_str = sampler_cfg["field_str"]
    symbols = sampler_cfg["symbols"]
    num_polynomials = num_polynomials_opt or int(
        cfg.problem_generator.get("num_polynomials", 2)
    )
    if root_seed is None:
        root_seed = int(cfg.dataset.get("root_seed", 42))

    sampler = PolynomialSampler(**sampler_cfg)
    base = _base_ring(field_str)
    names = [s.strip() for s in symbols.split(",")]
    R_degrevlex = PolynomialRing(base, names, order="degrevlex")
    R_lex = PolynomialRing(base, names, order="lex")
    is_QQ = base == QQ

    per_system = []
    seed = root_seed
    accepted = 0
    timed_out = 0
    while accepted < num_samples:
        randstate.set_random_seed(seed)
        seed += 1
        F = sampler.sample(num_samples=num_polynomials)

        # QQ rejects systems with 3-digit numerators/denominators, as the
        # dataset generator does; the check needs a basis, computed untimed.
        if is_QQ:
            G_check = list(
                R_degrevlex.ideal([R_degrevlex(f) for f in F]).groebner_basis()
            )
            if _has_large_rational_coefficients(list(F) + G_check, threshold=100):
                continue

        # Order of the two timings alternates so neither one systematically
        # benefits from a warm cache or from the other having run first.
        try:
            alarm(timeout_s)
            if accepted % 2 == 0:
                t_dr, n_dr = _time_groebner(R_degrevlex, F, repeats)
                t_lex, n_lex = _time_groebner(R_lex, F, repeats)
            else:
                t_lex, n_lex = _time_groebner(R_lex, F, repeats)
                t_dr, n_dr = _time_groebner(R_degrevlex, F, repeats)
            cancel_alarm()
        except AlarmInterrupt:
            cancel_alarm()
            timed_out += 1
            continue

        per_system.append(
            {
                "degrevlex_s": t_dr,
                "lex_s": t_lex,
                "degrevlex_size": n_dr,
                "lex_size": n_lex,
            }
        )
        accepted += 1
        if accepted % 100 == 0:
            print(f"{accepted}/{num_samples} systems timed", flush=True)

    dr_times = [d["degrevlex_s"] for d in per_system]
    lex_times = [d["lex_s"] for d in per_system]
    dr_stats = _summarize(dr_times)
    lex_stats = _summarize(lex_times)
    # Per-system ratio, so one pathological instance cannot dominate the figure
    # the way it does in the ratio of totals.
    ratios = sorted(d["lex_s"] / d["degrevlex_s"] for d in per_system)

    record = {
        "config_path": config_path,
        "field": field_str,
        "symbols": symbols,
        "num_polynomials": num_polynomials,
        "algorithm": ALGORITHM,
        "max_degree": sampler_cfg["max_degree"],
        "num_variables": len(names),
        "repeats": repeats,
        "num_samples": len(per_system),
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "root_seed": root_seed,
        "degrevlex": dr_stats,
        "lex": lex_stats,
        "ratio_of_totals": dr_stats["total_s"]
        and lex_stats["total_s"] / dr_stats["total_s"],
        "ratio_of_means": lex_stats["mean_s"] / dr_stats["mean_s"],
        "ratio_median": ratios[len(ratios) // 2],
        "ratio_p90": ratios[int(0.9 * len(ratios))],
        "ratio_max": ratios[-1],
        "mean_basis_size": {
            "degrevlex": sum(d["degrevlex_size"] for d in per_system) / len(per_system),
            "lex": sum(d["lex_size"] for d in per_system) / len(per_system),
        },
    }

    print(json.dumps(record, indent=2))

    existing = []
    if os.path.exists(output):
        with open(output) as fh:
            existing = json.load(fh)
    existing.append(record)
    with open(output, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"appended to {output}")


if __name__ == "__main__":
    main()
