from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def rational_factor_generator(seed: int):
    """Generate a QQ-polynomial and its factored form using PolynomialSampler."""
    import sage.misc.randstate as randstate  # type: ignore

    randstate.set_random_seed(seed)

    sampler = PolynomialSampler(
        symbols="x",
        field_str="QQ",
        order="grevlex",
        max_num_terms=4,
        max_degree=4,
        min_degree=1,
        num_bound=10,
    )
    p = sampler.sample(1)[0]
    factored = p.factor()

    return p, factored


def poly_factor_stats_calc(problem, answer) -> dict[str, dict[str, int | float]]:
    return {"problem": _poly_stats(problem), "answer": _factor_stats(answer)}


def _poly_stats(poly) -> dict[str, int | float]:
    if not poly:
        raise ValueError("Polynomial is empty")
    if poly.parent().ngens() == 1:
        degree = int(
            max(poly.degree(), 0)
        )  # if polynomial is zero, then poly.degree() is -1, so we need to set it to 0
    else:
        raise ValueError("Polynomial has multiple variables")

    return {
        "num_terms": len(poly.monomials()),
        "max_degree": degree,
        "min_degree": degree,
    }


def _factor_stats(factor) -> dict[str, int | float]:
    # factor is a Factorization object, list(factor) returns [(factor, exponent), ...]
    factor_list = list(factor)
    # Total number of factors counting multiplicity (e.g., for x^2 * (x+1), it's 2 + 1 = 3)
    total_factors = sum(exp for _, exp in factor_list)

    return {
        "num_distinct_factors": len(factor_list),  # Number of distinct factors
        "total_factors": total_factors,  # Total number of factors counting multiplicity
    }


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=rational_factor_generator,
        statistics_calculator=poly_factor_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")
