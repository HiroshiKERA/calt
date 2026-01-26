import math

from omegaconf import OmegaConf
from sage.all import QQ, RR, ZZ

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def integer_poly_factor_generator(seed: int):
    """Generate an integer-coefficient quadratic and its factored form."""
    import sage.misc.randstate as randstate

    randstate.set_random_seed(seed)

    sampler = PolynomialSampler(
        symbols="x",
        field_str="ZZ",
        order="grevlex",
        max_num_terms=4,
        max_degree=4,
        min_degree=1,
    )
    ps = sampler.sample(num_samples=3)
    p = math.prod(ps)
    factored = p.factor()
    return p, factored


def poly_factor_stats_calc(problem, answer) -> dict[str, dict[str, int | float]]:
    return {"problem": _poly_stats(problem), "answer": _factor_stats(answer)}


def _poly_stats(poly) -> dict[str, int | float]:
    if not poly:
        raise ValueError("Polynomial is empty")

    coeffs = _extract_coefficients(poly)
    if poly.parent().ngens() == 1:
        degree = int(
            max(poly.degree(), 0)
        )  # if polynomial is zero, then poly.degree() is -1, so we need to set it to 0
    else:
        degree = int(
            max(poly.total_degree(), 0)
        )  # if polynomial is zero, then poly.total_degree() is -1, so we need to set it to 0

    return {
        "num_terms": len(poly.monomials()),
        "max_degree": degree,
        "min_degree": degree,
        "max_coeff": max(coeffs),
        "min_coeff": min(coeffs),
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


def _extract_coefficients(poly) -> list[float | int]:
    """Extract coefficients from polynomial based on field type."""
    coeff_field = poly.parent().base_ring()
    if coeff_field == QQ:
        return [abs(float(c.numerator())) for c in poly.coefficients()] + [
            abs(float(c.denominator())) for c in poly.coefficients()
        ]
    elif coeff_field in (RR, ZZ):
        return [abs(float(c)) for c in poly.coefficients()]
    elif coeff_field.is_field() and coeff_field.characteristic() > 0:
        return [int(c) for c in poly.coefficients()]
    return []


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=integer_poly_factor_generator,
        statistics_calculator=poly_factor_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")
