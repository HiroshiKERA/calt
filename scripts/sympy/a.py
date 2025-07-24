from typing import Any, List, Tuple, Dict, Union, Callable, Optional
from time import time
from datetime import timedelta
from calt.dataset_generator.sympy import (
    PolynomialSampler,
)
from sympy.polys.orderings import grevlex
import numpy as np


def generate_sample(
    self,
    job_id: int,
    train: bool,
    problem_generator: Callable,
    statistics_calculator: Optional[Callable] = None,
) -> Tuple[Union[List[Any], Any], Union[List[Any], Any], Dict[str, Any], timedelta]:
    # Generate a unique seed for this job
    seed = self._generate_seed(job_id, train)

    start_time = time()
    problem_input, problem_output = problem_generator(seed)
    runtime = time() - start_time

    if statistics_calculator is not None:
        sample_stats = statistics_calculator(problem_input, problem_output)
    else:
        sample_stats = None

    return problem_input, problem_output, sample_stats, runtime


if __name__ == "__main__":
    # R, *gens = ring("x,y", ZZ, order="grevlex")

    # print(dill.dump(domain))

    sampler = PolynomialSampler(
        symbols="x,y,z",
        field_str="GF2",
        order=grevlex,
        max_num_terms=None,
        max_degree=10,
        min_degree=10,
        term_sampling="fixed",

    )

    F = sampler.sample(1)
    f = F[0]
    print(f)
    print(len(f))

    # A = sampler.sample(4)
    # B = sampler.sample(4)

    # _A = np.array(A).reshape(2, 2)
    # _B = np.array(B).reshape(2, 2)

    # print(_A)
    # print(_B)

    # # print(_A @ _B)
    # # print(_B @ _A)

    # print(_A[0, 0])
    # print(_A[0, 1])
    # print(_A[1, 0])
    # print(_A[1, 1])

    # print(_B[0, 0])
    # print(_B[0, 1])

    # print()

    # M = sampler.sample(2, size=(2, 2), density=0.5)
    # print(M)

    # M = sampler.sample(1, size=(2, 2), density=0.5)
    # print(M)

    # problem_generator = PartialSumProblemGenerator(sampler, 5, 2)

    # stats_calculator = PolyStatisticsCalculator(num_vars, domain)

    # print(pickle.dumps(sampler))
    # print()
    # print(pickle.dumps(problem_generator))
    # print()
    # print(pickle.dumps(stats_calculator))

    # F = sampler.sample(1)
    # f = F[0]
    # print(f)
    # print(type(f))
    # print(pickle.dumps(str(f)))
    # print(pickle.dumps(R.domain))
    # print(f.ring.gens)

    # results = Parallel(
    #     n_jobs=-1, backend="multiprocessing", verbose=True
    # )(
    #     delayed(problem_generator)(i)for i in range(10)
    # )

    # print(results)
