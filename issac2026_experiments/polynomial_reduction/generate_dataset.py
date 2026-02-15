"""Generate polynomial_reduction dataset: univariate f, g; target (q, r) with f = g*q + r.

Input: dividend f | divisor g.
Output (full): quotient q | remainder r.
Output (last_element at train time): remainder r only.
"""

# Initialize Sage's polynomial ring stack first to avoid ImportError: PolynomialRing_generic
import click
import sage.all  # noqa: F401
import sage.misc.randstate as randstate
from omegaconf import OmegaConf
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


class UnivariateQuoRemGenerator:
    """Problem generator: (dividend f, divisor g) -> (quotient q, remainder r) with f = g*q + r."""

    def __init__(self, sampler: PolynomialSampler, min_divisor_degree: int = 1):
        self.sampler = sampler
        self.min_divisor_degree = min_divisor_degree

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], list[MPolynomial_libsingular]]:
        randstate.set_random_seed(seed)
        R = self.sampler.get_ring()
        if R.ngens() != 1:
            raise ValueError(
                "polynomial_reduction expects univariate ring (symbols='x')"
            )
        # Sampler is configured with min_degree >= min_divisor_degree and nonzero_instance=True,
        # so g is never 0 and has degree >= min_divisor_degree; no retry loop needed.
        f, g = self.sampler.sample(2)

        if f.degree() < g.degree():
            f, g = g, f  # swap f and g if f has lower degree

        q, r = f.quo_rem(g)

        return (f, g), (q, r)


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to data config YAML (sampler, problem_generator, dataset).",
)
def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    sampler_cfg = dict(OmegaConf.to_container(cfg.sampler, resolve=True))
    min_divisor_degree = int(cfg.problem_generator.get("min_divisor_degree", 1))
    sampler_cfg["min_degree"] = (
        min_divisor_degree  # ensures g has degree >= min_divisor_degree
    )
    sampler = PolynomialSampler(**sampler_cfg)
    problem_generator = UnivariateQuoRemGenerator(
        sampler=sampler,
        min_divisor_degree=min_divisor_degree,
    )
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=problem_generator,
    )
    pipeline.run()
    print("Dataset generation completed.")


if __name__ == "__main__":
    main()
