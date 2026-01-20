from calt.dataset.sagemath.dataset_generator import DatasetGenerator
from calt.dataset.sagemath.utils.dataset_writer import DatasetWriter


def integer_poly_factor_generator(seed: int):
    """Generate an integer-coefficient quadratic and its factored form."""
    from sage.all import ZZ, PolynomialRing  # type: ignore
    import sage.misc.randstate as randstate  # type: ignore
    from random import randint

    randstate.set_random_seed(seed)

    R = PolynomialRing(ZZ, "x")
    x = R.gen()

    # Choose two integer roots a, b in a small range
    a = randint(-9, 9)
    b = randint(-9, 9)

    p = (x - a) * (x - b)
    expanded = p          # already expanded by construction
    factored = p.factor()

    return str(expanded), str(factored)


if __name__ == "__main__":
    generator = DatasetGenerator(n_jobs=1, root_seed=123)
    writer = DatasetWriter(save_dir="./data", save_text=True, save_json=False)

    generator.run(
        dataset_sizes={"train": 100000, "test": 1000},
        problem_generator=integer_poly_factor_generator,
        dataset_writer=writer,
    )
    print("Dataset generation completed")

