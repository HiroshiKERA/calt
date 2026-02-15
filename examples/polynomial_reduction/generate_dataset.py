"""Generate polynomial reduction dataset: f, g1..g3 (ZZ, 3 vars, 2--5 terms).

Gröbner basisは常に存在し、I.reduce(f) で簡約は常に成功する。商は使わず余りだけ保存。
"""

import hashlib
import pickle
from pathlib import Path

import sage.misc.randstate as randstate  # type: ignore
from omegaconf import OmegaConf

from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


def _seed(global_index: int, tag: str, root_seed: int) -> int:
    s = f"{root_seed}_{tag}_{global_index}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def _sample_one(seed: int):
    randstate.set_random_seed(seed)
    sampler = PolynomialSampler(
        symbols="x,y,z",
        field_str="ZZ",
        order="degrevlex",
        max_num_terms=5,
        max_degree=4,
        min_degree=1,
        max_coeff=10,
        term_sampling="uniform",
        degree_sampling="uniform",
    )
    f = sampler.sample(1)[0]
    g1, g2, g3 = sampler.sample(3)
    for p in (f, g1, g2, g3):
        if len(p.monomials()) < 2:
            return None
    R = f.parent()
    ideal = R.ideal([g1, g2, g3])
    G = ideal.groebner_basis()
    if not G:
        return None
    r = ideal.reduce(f)
    return ((f, tuple(G)), (r,))


def main():
    cfg = OmegaConf.load("configs/data.yaml")
    d = cfg.dataset
    save_dir = Path(d.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_train = int(d.num_train_samples)
    n_test = int(d.num_test_samples)
    root_seed = int(d.root_seed)

    def generate(n: int, tag: str):
        samples = []
        idx = 0
        while len(samples) < n:
            s = _sample_one(_seed(idx, tag, root_seed))
            if s is not None:
                samples.append(s)
            idx += 1
        return samples

    train_samples = generate(n_train, "train")
    test_samples = generate(n_test, "test")

    with open(save_dir / "train_data.pkl", "wb") as f:
        pickle.dump(train_samples, f)
    with open(save_dir / "test_data.pkl", "wb") as f:
        pickle.dump(test_samples, f)
    print(f"Saved {len(train_samples)} train, {len(test_samples)} test to {save_dir}")


if __name__ == "__main__":
    main()
