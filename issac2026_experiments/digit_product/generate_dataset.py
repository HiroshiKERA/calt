"""Generate dataset for Prod task: digit-level product of two integers (L=10).

Input: X = [a_1,...,a_{L/2}, b_1,...,b_{L/2}] (left-zero-padded digits of a and b).
Target: [c_1,...,c_L] = left-zero-padded digits of a*b.
L=10: 5 digits for a, 5 for b; target 10 digits.
"""

import numpy as np
import click
from omegaconf import OmegaConf
from sage.misc.randstate import set_random_seed
from sage.misc.prandom import randint

from calt.dataset import DatasetPipeline


def _sample_with_num_digits(num_digits: int, rng_half: int) -> int:
    """Sample an integer with exactly `num_digits` digits (1..rng_half). Left-zero-padded to rng_half later."""
    if num_digits <= 0 or num_digits > rng_half:
        raise ValueError(f"num_digits must be in 1..{rng_half}, got {num_digits}")
    if num_digits == 1:
        low, high = 0, 9
    else:
        low = 10 ** (num_digits - 1)
        high = 10**num_digits - 1
    return randint(low, high)


def _pad_digits(n: int, length: int) -> str:
    """Left-zero-pad integer n to `length` digits."""
    s = str(n)
    if len(s) > length:
        raise ValueError(f"n={n} has more than {length} digits")
    return s.zfill(length)


class DigitProductGenerator:
    """Prod task: input = pad(a,L/2)+pad(b,L/2), target = pad(a*b,L). L even."""

    def __init__(self, length: int = 10, max_digit_value: int = 9):
        """
        Args:
            length: Even length L. Input has L digits (L/2 for a, L/2 for b), target L digits.
            max_digit_value: Not used; digits are 0-9. Kept for config compatibility.
        """
        if length % 2 != 0:
            raise ValueError(f"length must be even, got {length}")
        self.length = length
        self.half = length // 2  # 5 for L=10
        self.max_a_b = 10**self.half - 1  # 99999 for L/2=5

    def __call__(self, seed: int) -> tuple[str, str]:
        set_random_seed(seed)
        # Sample digit counts uniformly (1..half) so all lengths appear equally often
        num_digits_a = randint(1, self.half)
        num_digits_b = randint(1, self.half)
        a = _sample_with_num_digits(num_digits_a, self.half)
        b = _sample_with_num_digits(num_digits_b, self.half)
        product = a * b
        input_str = _pad_digits(a, self.half) + _pad_digits(b, self.half)  # L digits
        target_str = _pad_digits(product, self.length)  # L digits (product up to 10^L-1)
        return input_str, target_str


def digit_product_stats_calc(problem: str, answer: str) -> dict:
    return {"problem_len": len(problem), "answer_len": len(answer)}


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default="configs/data.yaml",
    help="Path to data config YAML.",
)
def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    gen_cfg = OmegaConf.to_container(cfg.get("problem_generator", {}), resolve=True)
    if not gen_cfg:
        raise ValueError("config must have 'problem_generator' with length (even, e.g. 10)")
    problem_generator = DigitProductGenerator(**gen_cfg)
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=problem_generator,
        statistics_calculator=digit_product_stats_calc,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
