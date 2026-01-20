from omegaconf import OmegaConf

from calt.dataset.pipeline import DatasetPipeline


def integer_poly_factor_generator(seed: int):
    """Generate an integer-coefficient quadratic and its factored form."""
    from sage.all import ZZ, PolynomialRing  
    import sage.misc.randstate as randstate  
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
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=integer_poly_factor_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")

