
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular
import click

from statistics_calculator import PolyStatisticsCalculator


class CumulativePolyMultiplicationGenerator:
    """
    Problem generator for polynomial product problems.

    This generator creates problems in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the solution is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 * f_2 * ... * f_i.
    """

    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        """
        Initialize polynomial product generator.

        Args:
            sampler: Polynomial sampler
            min_polynomials: Minimum number of polynomials in F
            max_polynomials: Maximum number of polynomials in F
        """

        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], list[MPolynomial_libsingular]]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial system F
        - Solution: polynomial system G (partial products of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, G)
        """
        # Get ring from sampler
        ring = self.sampler.get_ring()

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial products for solution
        G = []
        current_prod = ring(1)
        for f in F:
            current_prod *= f
            G.append(current_prod)

        return F, G


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Path to data config YAML (sampler, problem_generator, dataset).",
)
def main(config_path):
    cfg = OmegaConf.load(config_path)

    sampler = PolynomialSampler(**OmegaConf.to_container(cfg.sampler, resolve=True))
    problem_generator = CumulativePolyMultiplicationGenerator(
        sampler=sampler,
        **OmegaConf.to_container(cfg.problem_generator, resolve=True),
    )

    statistics_calculator = PolyStatisticsCalculator()

    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=problem_generator,
        statistics_calculator=statistics_calculator,
    )
    pipeline.run()
    print("Dataset generation completed")
if __name__ == "__main__":
    main()
