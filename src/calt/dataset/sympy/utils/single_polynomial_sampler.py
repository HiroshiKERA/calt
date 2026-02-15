import math
import random
from typing import Any

from sympy import QQ, RR, ZZ
from sympy.polys.domains.domain import Domain
from sympy.polys.rings import PolyElement, PolyRing


class SinglePolynomialSampler:
    """Sampler for single polynomial with specific constraints"""

    def _precomp_counts(self, n: int, d: int) -> tuple[list[int], int]:
        """
        Given a number of variables n and a degree d return a tuple (C,t)
        such that C is a list of the cardinalities of the sets of
        monomials up to degree d (including) in n variables and t is the
        sum of these cardinalities.

        Args:
            n: number of variables
            d: degree

        Returns:
            Tuple containing:
            - List of cardinalities for each degree
            - Total sum of cardinalities
        """
        C = [1]  # d = 0
        for dbar in range(1, d + 1):
            C.append(math.comb(n + dbar - 1, dbar))
        return C, sum(C)

    def _combination_from_rank(self, r: int, n: int, k: int) -> tuple[int, ...]:
        """
        Generate the k-combination of rank r in the lexicographic ordering of
        all k-combinations of the integers 0..n-1.

        Args:
            r: rank of the combination
            n: size of the set
            k: size of the combination

        Returns:
            Tuple of integers representing the combination

        Raises:
            ValueError: If any of the input parameters are invalid
            TypeError: If any of the input parameters are not integers
        """
        # Type checking
        if not isinstance(r, int) or not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("r, n, and k must be int")

        # Basic value checking
        if n < 0:
            raise ValueError("n must be >= 0")
        if k < 0:
            raise ValueError("k must be >= 0")
        if k > n:
            raise ValueError("k must be <= n")
        if r < 0:
            raise ValueError("r must be >= 0")

        # Special cases
        if n == 0 or k == 0:
            return ()

        # Rank range checking
        max_rank = math.comb(n, k)
        if r >= max_rank:
            raise ValueError(
                f"Rank {r} is out of range. Maximum rank is {max_rank - 1}"
            )

        result = []
        x = 0
        for i in range(k):
            while math.comb(n - x - 1, k - i - 1) <= r:
                r -= math.comb(n - x - 1, k - i - 1)
                x += 1
            result.append(x)
            x += 1
        return tuple(result)

    def _to_monomial(self, i: int, n: int, d: int) -> tuple[int, ...]:
        """
        Given an index i, a number of variables n and a degree d return
        the i-th monomial of degree d in n variables.

        Args:
            i: index: 0 <= i < binom(n+d-1,n-1)
            n: number of variables
            d: degree

        Returns:
            Tuple representing the monomial exponents

        Note:
            We do not check if the provided index/rank is within the allowed
            range. If it is not an infinite loop will occur.
        """
        if n < 1:
            raise ValueError("the number of variables must be >= 1")
        if d < 0:
            raise ValueError("the degree must be >= 0")
        if i < 0:
            raise ValueError("index must be >= 0")
        if i >= math.comb(n + d - 1, d):
            raise ValueError(f"index must be < {math.comb(n + d - 1, d)}")

        comb = self._combination_from_rank(i, n + d - 1, n - 1)

        if not comb:
            return (d,)

        monomial = [comb[0]]
        monomial.extend(comb[j + 1] - comb[j] - 1 for j in range(n - 2))
        monomial.append(n + d - 1 - comb[-1] - 1)
        return tuple(monomial)

    def _random_monomial_upto_degree_class(
        self, n: int, degree: int
    ) -> tuple[int, ...]:
        """
        Choose a random exponent tuple for n variables with a random
        degree d, i.e. choose the degree uniformly at random first
        before choosing a random monomial.

        Args:
            n: number of variables
            degree: degree of monomials

        Returns:
            Tuple representing the monomial exponents
        """
        # Select random degree
        d = random.randint(0, degree)
        total = math.comb(n + d - 1, d)

        # Select random monomial of degree d
        random_index = random.randint(0, total - 1)
        # Generate the corresponding monomial
        return self._to_monomial(random_index, n, d)

    def _random_monomial_upto_degree_uniform(
        self,
        n: int,
        degree: int,
        counts: list[int] | None = None,
        total: int | None = None,
    ) -> tuple[int, ...]:
        """
        Choose a random exponent tuple for n variables with a random
        degree up to d, i.e. choose a random monomial uniformly random
        from all monomials up to degree d. This discriminates against
        smaller degrees because there are more monomials of bigger
        degrees.

        Args:
            n: number of variables
            degree: degree of monomials
            counts: list of cardinalities for each degree
            total: total number of monomials

        Returns:
            Tuple representing the monomial exponents
        """
        if counts is None or total is None:
            counts, total = self._precomp_counts(n, degree)

        # Select a random one
        random_index = random.randint(0, total - 1)

        # Figure out which degree it corresponds to
        d = 0
        while random_index >= counts[d]:
            random_index -= counts[d]
            d += 1

        # Generate the corresponding monomial
        return self._to_monomial(random_index, n, d)

    def _integer_vectors(self, d: int, n: int) -> list[list[int]]:
        """
        Generate all possible combinations of n non-negative integers that sum to d.
        For example, if d=2 and n=3, this function returns:
        [[2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]]

        This is a non-recursive implementation that uses a stack to avoid
        the overhead of recursive calls. It's more efficient than the
        recursive version for large inputs.

        Args:
            d: target sum (degree)
            n: number of variables

        Returns:
            List of integer vectors, where each vector is a list of n non-negative
            integers that sum to d. Each vector represents the exponents of a
            monomial of degree d in n variables.
        """
        if n == 1:
            return [[d]]

        result = []
        # Stack stores tuples of (remaining_sum, current_position, current_vector)
        stack = [(d, 0, [0] * n)]

        while stack:
            remaining, pos, vec = stack.pop()

            if pos == n - 1:
                # Last position: fill with remaining sum
                vec[pos] = remaining
                result.append(vec.copy())
            else:
                # Try all possible values for current position
                for i in range(remaining + 1):
                    vec[pos] = i
                    stack.append((remaining - i, pos + 1, vec.copy()))

        return result

    def _pick_random_until_nonzero(self, generator, non_zero: bool) -> Any:
        """
        generator(): a function that returns a single random value in the desired coefficient field
        non_zero: if True, repeat calling generator() until a non-zero value is returned
        """
        if not non_zero:
            # If non_zero is False, just return the first generated value
            return generator()
        # If non_zero is True, loop until generator() returns a non-zero value
        while True:
            value = generator()
            if value != 0:
                return value

    def random_coeff(self, field: Domain, non_zero: bool = False, **kwargs) -> Any:
        """
        Generate a random coefficient in the given field.

        Args:
            field: The coefficient field (e.g., ZZ, QQ, RR, GF)
            non_zero: If True, ensure the coefficient is non-zero
            **kwargs: Additional parameters for coefficient generation
                - min: minimum value (default: -10)
                - max: maximum value (default: 10)
                - num_bound: bound for numerator and denominator in QQ (default: 10)

        Returns:
            Random coefficient in the specified field

        Raises:
            ValueError: If parameter ranges are invalid or non_zero cannot be satisfied
            NotImplementedError: If the field is not supported
        """

        # Integer coefficient
        if field == ZZ:
            a = kwargs.get("min", -10)
            b = kwargs.get("max", 10)

            if a > b:
                raise ValueError("min must be <= max")

            if non_zero and a == 0 and b == 0:
                raise ValueError("Cannot generate non-zero ZZ with min=0 and max=0")

            # Define a generator function that returns a random ZZ in [a, b]
            def gen_int():
                return ZZ(random.randint(a, b))

            return self._pick_random_until_nonzero(gen_int, non_zero)

        # Real number coefficient
        elif field == RR:
            a = kwargs.get("min", -10.0)
            b = kwargs.get("max", 10.0)

            if a > b:
                raise ValueError("min must be <= max")

            if non_zero and a == 0.0 and b == 0.0:
                raise ValueError("Cannot generate non-zero RR with min=0.0 and max=0.0")

            # Define a generator function that returns a random RR in [a, b]
            def gen_real():
                return RR(random.uniform(a, b))

            return self._pick_random_until_nonzero(gen_real, non_zero)

        # Rational number coefficient
        elif field == QQ:
            num_bound = kwargs.get("num_bound", 10)

            if num_bound <= 0:
                raise ValueError("num_bound must be > 0")

            # Define a generator function that returns a random QQ with numerator in [-num_bound, num_bound] and denominator in [1, num_bound]
            def gen_rat():
                numerator = random.randint(-num_bound, num_bound)
                denominator = random.randint(1, num_bound)
                return QQ(numerator, denominator)

            return self._pick_random_until_nonzero(gen_rat, non_zero)

        # Finite field
        elif field.is_FiniteField:
            p = field.characteristic()

            if non_zero and p == 1:
                raise ValueError(
                    "Cannot generate non-zero finite field coefficient with characteristic 1"
                )

            # Define a generator function that returns a random field element in GF(p)
            def gen_gf():
                return field(random.randint(0, p - 1))

            return self._pick_random_until_nonzero(gen_gf, non_zero)

        else:
            raise NotImplementedError(
                f"Random coefficient generation not implemented for field {field}"
            )

    def random_element(
        self,
        R: PolyRing,
        degree: int = 2,
        terms: int | None = None,
        choose_degree: bool = False,
        non_zero_coeff: bool = False,
        **kwargs,
    ) -> PolyElement:
        """
        Return a random polynomial of at most the specified degree and at most the specified number of terms.

        First monomials are chosen uniformly random from the set of all
        possible monomials of degree up to the specified degree (inclusive). This means
        that it is more likely that a monomial of the specified degree appears than
        a monomial of degree (specified degree - 1) because the former class is bigger.

        Exactly the specified number of distinct monomials are chosen this way and each one gets
        a random coefficient (possibly zero) from the base ring assigned.

        The returned polynomial is the sum of this list of terms.

        Args:
            R: Polynomial ring
            degree: Maximum degree of the polynomial
            terms: Number of terms in the polynomial
            choose_degree: Whether to choose degree randomly first
            non_zero_coeff: If True, ensure all coefficients are non-zero
            **kwargs: Additional parameters for coefficient generation
                - min: minimum value (default: -10)
                - max: maximum value (default: 10)
                - num_bound: bound for numerator and denominator in QQ (default: 10)


        Returns:
            Random polynomial in the given ring
        """
        field = R.domain
        n = R.ngens

        counts, total = self._precomp_counts(n, degree)

        if terms is not None and terms < 0:
            raise ValueError("terms must be >= 0")
        if degree < 0:
            raise ValueError("degree must be >= 0")

        # special cases
        if terms == 0:
            return R.zero
        if degree == 0:
            return R(self.random_coeff(field=field, non_zero=non_zero_coeff, **kwargs))

        # adjust terms
        if terms is None:
            terms = min(5, total)
        else:
            terms = min(terms, total)

        # total is 0. Just return
        if total == 0:
            return R.zero
        elif terms < total / 2:
            # we choose random monomials if t < total/2 because then we
            # expect the algorithm to be faster than generating all
            # monomials and picking a random index from the list. if t ==
            # total/2 we expect every second random monomial to be a
            # double such that our runtime is doubled in the worst case.
            M: set[tuple[int, ...]] = set()
            if not choose_degree:
                while terms:
                    m = self._random_monomial_upto_degree_uniform(
                        n, degree, counts, total
                    )
                    if m not in M:
                        M.add(m)
                        terms -= 1
            else:
                while terms:
                    m = self._random_monomial_upto_degree_class(n, degree)
                    if m not in M:
                        M.add(m)
                        terms -= 1
        elif terms <= total:
            # generate a list of all monomials and choose among them
            if not choose_degree:
                M = sum(
                    [list(self._integer_vectors(_d, n)) for _d in range(degree + 1)], []
                )
                # we throw away those we don't need
                for mi in range(total - terms):
                    M.pop(random.randint(0, len(M) - 1))
                M = [tuple(m) for m in M]
            else:
                M = [list(self._integer_vectors(_d, n)) for _d in range(degree + 1)]
                Mbar = []
                for mi in range(terms):
                    # choose degree 'd' and monomial 'm' at random
                    d = random.randint(0, len(M) - 1)
                    m = random.randint(0, len(M[d]) - 1)
                    Mbar.append(M[d].pop(m))  # remove and insert
                    if len(M[d]) == 0:
                        M.pop(d)  # bookkeeping
                M = [tuple(m) for m in Mbar]

        # Generate random coefficients
        C = [
            self.random_coeff(field=field, non_zero=non_zero_coeff, **kwargs)
            for _ in range(len(M))
        ]

        # Create the polynomial using from_dict
        return R.from_dict(dict(zip(M, C)))
