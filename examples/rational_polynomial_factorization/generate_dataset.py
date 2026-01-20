from omegaconf import OmegaConf
from calt.dataset.pipeline import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def rational_factor_generator(seed: int):
    """Generate a QQ-polynomial and its factored form using PolynomialSampler."""
    import sage.misc.randstate as randstate  # type: ignore

    randstate.set_random_seed(seed)

    sampler = PolynomialSampler(
        symbols="x",
        field_str="QQ",
        order="grevlex",
        max_num_terms=3,
        max_degree=2,
        min_degree=1,
        num_bound=10,
    )

    p = sampler.sample(1)[0]
    expanded = p.expand()
    factored = p.factor()

    return str(expanded), str(factored)


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=rational_factor_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")
