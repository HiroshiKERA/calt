from typing import Any
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular
from calt.dataset_generator.sagemath import PolynomialSampler


class SumProblemGenerator:
    """
    Problem generator for polynomial sum problems.

    This generator creates problems in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the solution is a single polynomial g = f_1 + f_2 + ... + f_n.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial sum generator.

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
    ) -> tuple[list[MPolynomial_libsingular], MPolynomial_libsingular]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial system F
        - Solution: polynomial g (sum of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, g)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate solution polynomial g (sum of F)
        g = sum(F)

        return F, g


class GCDProblemGenerator:
    """
    Problem generator for polynomial GCD problems.

    This generator creates problems in which the problem is a pair of polynomials F = [f_1, f_2],
    and the solution is a single polynomial g = GCD(f_1, f_2).
    """

    def __init__(self, sampler: PolynomialSampler):
        """
        Initialize polynomial GCD generator.

        Args:
            sampler: Polynomial sampler
        """

        self.sampler = sampler

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], MPolynomial_libsingular]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial system F
        - Solution: polynomial g (GCD of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, g)
        """
        # Get ring from sampler
        ring = self.sampler.get_ring()

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Generate problem polynomials using sampler
        gcd, q1, q2 = self.sampler.sample(num_samples=3)

        # Generate solution polynomial g (GCD of F)
        _gcd = q1.gcd(q2)
        gcd, q1, q2 = gcd * _gcd, ring(q1 / _gcd), ring(q2 / _gcd)
        F = [gcd * q1, gcd * q2]
        g = ring(gcd / gcd.lc())

        return F, g


class ProductProblemGenerator:
    """
    Problem generator for polynomial product problems.

    This generator creates problems in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the solution is a single polynomial g = f_1 * f_2 * ... * f_n.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial product generator.

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
    ) -> tuple[list[MPolynomial_libsingular], MPolynomial_libsingular]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial system F
        - Solution: polynomial g (product of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, g)
        """
        # Get ring from sampler
        ring = self.sampler.get_ring()

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate solution polynomial g (product of F)
        g = ring(1)
        for f in F:
            g *= f

        return F, g


class PartialProdProblemGenerator:
    """
    Problem generator for polynomial product problems.

    This generator creates problems in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the solution is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 * f_2 * ... * f_i.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial product generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """

        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> tuple[list[Any], list[Any]]:
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
