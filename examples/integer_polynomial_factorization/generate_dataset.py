from omegaconf import OmegaConf

from calt.dataset.pipeline import DatasetPipeline


def integer_poly_factor_generator(seed: int):
    """Generate an integer-coefficient quadratic and its factored form."""
    import sage.misc.randstate as randstate
    from sage.all import PolynomialSampler, prod

    randstate.set_random_seed(seed)

    sampler = PolynomialSampler(
        symbols="x",
        field_str="ZZ",
        max_num_terms=4,
        max_degree=4,
        min_degree=1,
    )
    ps = sampler.sample(num_samples=3)
    p = prod(ps)
    factored = p.factor()
    return p, factored


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=integer_poly_factor_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")
