import numpy as np
from omegaconf import OmegaConf
from calt.dataset.pipeline import DatasetPipeline


def eigvec_generator(seed: int):
    # Simple 3x3 symmetric PSD matrix M = A^T A, so eigenvalues are real and >= 0
    A = np.random.randn(3, 3)
    M = np.round(A.T @ A, 2)

    # Symmetric PSD → use eigh
    vals, vecs = np.linalg.eigh(M)
    idx = np.argmax(vals)
    v = vecs[:, idx]
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    v = np.round(v, 2)

    # round to 2 decimals and format as rows separated by ';'
    rows = ["{:.2f},{:.2f},{:.2f}".format(*row) for row in M]
    matrix_str = ";".join(rows)
    vec_str = "{:.2f},{:.2f},{:.2f}".format(*v)

    return matrix_str, vec_str


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/data.yaml")
    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        problem_generator=eigvec_generator,
        statistics_calculator=None,
    )
    pipeline.run()
    print("Dataset generation completed")
