import random

import numpy as np
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline


def gf17_addition_generator(seed):
    random.seed(seed)
    p = 17

    length = random.randint(3, 6)
    numbers = [random.randint(0, p - 1) for _ in range(length)]

    cumulative = []
    s = 0
    for n in numbers:
        s = (s + n) % p
        cumulative.append(s)

    input_str = ",".join(map(str, numbers))
    output_str = ",".join(map(str, cumulative))

    return f"{input_str}", f"{output_str}"


def gf17_addition_stats_calc(problem, answer) -> dict[str, dict[str, int | float]]:
    return {
        "problem": _integer_list_stats(problem),
        "answer": _integer_list_stats(answer),
    }


def _integer_list_stats(data: str) -> dict[str, int | float]:
    if not data:
        raise ValueError("Cannot calculate statistics for empty data list")

    data = data.split(",")
    values = [int(n) for n in data]  # Convert to float for calculations
    stats = {
        "num_values": len(data),
        "min_value": min(values),
        "max_value": max(values),
        "mean_value": float(np.mean(values)),
        "std_value": float(np.std(values)),
    }
    return stats


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=gf17_addition_generator,
        statistics_calculator=gf17_addition_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")
