import random
from functools import partial

import sage.misc.randstate as randstate  # type: ignore
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def polynomial_addition_generator(seed, field="ZZ", num_variables=3):
    random.seed(seed)
    randstate.set_random_seed(seed)

    length = random.randint(3, 6)
    sampler = PolynomialSampler(
        symbols=", ".join(f"x{i}" for i in range(num_variables)),
        field_str=field,
        order="grevlex",
        max_num_terms=3,
        max_degree=3,
        min_degree=1,
    )
    polynomials = sampler.sample(length)
    cumsum_polynomials = [sum(polynomials[:i]) for i in range(1, len(polynomials) + 1)]
    return polynomials, cumsum_polynomials


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")

    _polynomial_addition_generator = partial(
        polynomial_addition_generator, field="ZZ", num_variables=3
    )
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=_polynomial_addition_generator,
    )
    pipeline.run()
    print("Dataset generation completed")
