from typing import Any
from sage.all import ZZ, QQ, RR
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular
from calt.dataset_generator.sagemath import (
    PolynomialSampler,
    DatasetGenerator,
    DatasetWriter,
)


class PartialSumProblemGenerator:
    """
    Problem generator for partial sum problems involving polynomials.

    This generator creates problems in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the solution is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 + f_2 + ... + f_i.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial partial sum sampler.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """

        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], list[MPolynomial_libsingular]]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial system F
        - Solution: polynomial system G (partial sums of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, G)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial sums for solution
        G = [sum(F[: i + 1]) for i in range(len(F))]

        return F, G


def main():
    save_dir = "dataset/sagemath/partial_sum_problem/GF7_n=3"

    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        symbols="x, y, z",  # "x, y, z, ... " or "x0, x1, x2, ... "
        field_str="GF2",  # "QQ", "RR", "ZZ", "GF(p)", "GFp", where p is a prime number
        order="degrevlex",  # "lex", "degrevlex", "deglex"
        max_num_terms=None,
        max_degree=5,
        min_degree=5,
        degree_sampling="fixed",  # "uniform" or "fixed"
        term_sampling="fixed",  # "uniform" or "fixed"
        max_coeff=None,  # Used for RR and ZZ
        num_bound=None,  # Used for QQ
        strictly_conditioned=False,
        nonzero_instance=True,
    )

    f = sampler.sample(1)
    print(f)
    print(len(f[0].monomials()))

    # # Initialize problem generator
    # problem_generator = PartialSumProblemGenerator(
    #     sampler=sampler,
    #     max_polynomials=5,
    #     min_polynomials=2,
    # )

    # # Initialize statistics calculator
    # statistics_calculator = PolyStatisticsCalculator()

    # # Initialize dataset generator
    # dataset_generator = DatasetGenerator(
    #     backend="multiprocessing",
    #     n_jobs=-1,
    #     verbose=True,
    #     root_seed=100,
    # )

    # # Initialize writer
    # dataset_writer = DatasetWriter(
    #     save_dir=save_dir,
    #     save_text=True,  # whether to save raw text files
    #     save_json=True,  # whether to save JSON files
    # )

    # # Generate datasets with batch processing
    # dataset_generator.run(
    #     dataset_sizes={"train": 100000, "test": 1000},
    #     batch_size=100000,  # set batch size
    #     problem_generator=problem_generator,
    #     statistics_calculator=statistics_calculator,
    #     dataset_writer=dataset_writer,
    # )


if __name__ == "__main__":
    main()