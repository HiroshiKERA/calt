# CALT: Computer ALgebra with Transformer

CALT is a simple Python library for learning arithmetic and symbolic computation with
Transformer models. It offers a basic Transformer model and training utilities so that
non-experts in deep learning (e.g., mathematicians) can focus on constructing datasets
and defining tasks.

The library is organised around three main pipelines:

- **Dataset pipeline** – generate paired problems/solutions with SageMath or SymPy backends.
- **IO pipeline** – tokenise text and build datasets and collators from configuration.
- **Trainer pipeline** – build and run HuggingFace `Trainer` instances from YAML configs.

For most users, the recommended entry point is to start from one of the example tasks
under `calt/examples/*` and customise only the dataset generator and configuration files.

## Documentation map

- `dataset_generator.md` – dataset generation backends and `DatasetPipeline`.
- `data_loader.md` – IO pipeline, tokenisation, and `lexer.yaml` configuration.
- `trainer.md` – model and trainer pipelines, high-level training flow.
- `configuration.md` – how `data.yaml`, `lexer.yaml`, and `train.yaml` work together.

## Installation

CALT can be installed via `pip`:

```bash
pip install calt-x
```

We highly recommend using the
[CALT codebase](https://github.com/HiroshiKERA/calt-codebase) – a comprehensive template
repository to build your own projects using CALT. The quickstart guide can be found in
the [CALT codebase documentation](https://hiroshikera.github.io/calt-codebase/).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{kera2025calt,
  title={CALT: A Library for Computer Algebra with Transformer},
  author={Hiroshi Kera and Shun Arawaka and Yuta Sato},
  year={2025},
  archivePrefix={arXiv},
  eprint={2506.08600}
}
```

The following is a small list of related studies from our group:

- ["Learning to Compute Gröbner Bases," Kera et al., 2024](https://arxiv.org/abs/2311.12904)
- ["Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms," Kera and Pelleriti et al., 2025](https://arxiv.org/abs/2505.23696)
- ["Geometric Generality of Transformer-Based Gröbner Basis Computation," Kambe et al., 2025](https://arxiv.org/abs/2504.12465)

Refer to our paper
["CALT: A Library for Computer Algebra with Transformer," Kera et al., 2025](https://arxiv.org/abs/2506.08600)
for a comprehensive overview.

