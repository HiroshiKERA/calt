import random
import math
from omegaconf import OmegaConf

from calt.dataset.pipeline import DatasetPipeline


def integer_factorization_generator(seed, max_number=30):
    random.seed(seed)

    n = 5 
    # sample 10 prime numbers up to max_number
    from sage.all import primes
    prime_list = list(primes(max_number))
    sampled_primes = random.sample(prime_list, n)
    
    input_int = math.prod(sampled_primes)
    output_int = sorted(sampled_primes)

    return input_int, output_int


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=integer_factorization_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")
