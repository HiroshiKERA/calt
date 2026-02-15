import numpy as np
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline


def eigvec_generator(seed: int):
    rng = np.random.default_rng(seed)
    # Simple 3x3 symmetric PSD matrix M = A^T A, so eigenvalues are real and >= 0
    A = rng.normal(size=(3, 3))
    M = A.T @ A

    # Symmetric PSD → use eigh
    vals, vecs = np.linalg.eigh(M)
    idx = np.argmax(vals)
    v = vecs[:, idx]
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm

    # round to 2 decimals; convert to string to avoid expression like 1.4e-2 etc.
    M = M.tolist()
    M = [[f"{x:.2f}" for x in row] for row in M]
    M_str = ";".join([",".join(row) for row in M])  # rows separated by ';'
    v = v.tolist()
    v_str = ",".join([f"{x:.2f}" for x in v])  # components separated by ','

    return M_str, v_str


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=eigvec_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")
