import math
import random
import numpy as np

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


def integer_factor_stats_calc(problem, answer) -> dict[str, dict[str, int | float]]:

    return {"problem": _integer_stats(problem), "answer": _integer_list_stats(answer)}


def _integer_stats(data: int | float) -> dict[str, int | float]:
    if not data:
        raise ValueError("Cannot calculate statistics for empty data")
    return {"value": float(data)}
    

def _integer_list_stats(data: list[int | float]) -> dict[str, int | float]:
    if not data:
        raise ValueError("Cannot calculate statistics for empty data list")
    values = [float(n) for n in data] # Convert to float for calculations
    stats = {
        "num_values": len(data),
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": float(np.mean(values)),
        "std_value": float(np.std(values)),
        "sum_value": sum(values),
    }
    return stats


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=integer_factorization_generator,
        statistics_calculator=integer_factor_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")
