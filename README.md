# CALT: Computer ALgebra with Transformer

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hiroshikera.github.io/calt/)

- [Documentation](https://hiroshikera.github.io/calt/)
- [Practical Codebase (calt-codebase)](https://github.com/HiroshiKERA/calt-codebase)
- [Demo Notebook (Google Colab)](https://colab.research.google.com/github/HiroshiKERA/calt/blob/main/examples/demos/minimal_demo.ipynb)
- [Task Builder (web app)](https://calt.mai2.org/)

CALT is available as the Python package `calt-x`. For practical examples and
project templates built with CALT, see
[calt-codebase](https://github.com/HiroshiKERA/calt-codebase).

## Overview

CALT is a simple Python library for learning arithmetic and symbolic computation
with a Transformer model (a deep neural model to realize sequence-to-sequence
functions).

It offers a basic Transformer model and training pipeline, and non-experts in
deep learning can focus on constructing datasets to train and evaluate the model.

## Quick Start

### Installation

CALT is the Python package `calt-x`, which you can install with conda or with
pip. We recommend conda: it installs CALT in a dedicated environment, with
PyTorch from conda-forge in the same command. CALT supports Python `>=3.11`
(tested in CI on 3.11–3.15).

#### With conda (recommended)

1. Create the environment with CALT and PyTorch. Keep the quotes around the
   package specs. For features that depend on SageMath, add `sage` to the
   `conda create` command.

   With an NVIDIA GPU:

   ```bash
   conda create -n calt-env -c conda-forge "calt-x>=1.5" "pytorch>=2.6=cuda*"
   ```

   Without an NVIDIA GPU (including macOS):

   ```bash
   conda create -n calt-env -c conda-forge "calt-x>=1.5" "pytorch>=2.6"
   ```

2. Activate the environment:

   ```bash
   conda activate calt-env
   ```

3. Check PyTorch. It prints e.g. `12.9 True` with a GPU and `None False`
   without one.

   ```bash
   python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
   ```

The available versions on conda-forge are listed by
`conda search calt-x --channel conda-forge`, and the feedstock is
[conda-forge/calt-x-feedstock](https://github.com/conda-forge/calt-x-feedstock).

#### With pip

1. Create a virtual environment with Python 3.11 or newer, and activate it.
   Check the version with `python3 --version`; if it is older, use e.g.
   `python3.12` instead of `python3`. On Windows, run `py -3 -m venv calt-env`
   and `calt-env\Scripts\activate` instead. On Debian or Ubuntu,
   `python3 -m venv` needs the `python3-venv` package.

   ```bash
   python3 -m venv calt-env
   source calt-env/bin/activate
   ```

2. Install CALT from PyPI. pip installs PyTorch from PyPI along with it.

   ```bash
   pip install calt-x
   ```

3. Check PyTorch. If the second value is `False` on a Linux machine with an
   NVIDIA GPU, see "GPU with pip" below.

   ```bash
   python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
   ```

#### Notes on GPUs and conda

- **NVIDIA driver:** the conda GPU command needs driver 525 or newer
  (`nvidia-smi` shows "CUDA Version" 12.0 or higher). conda-forge ships CUDA 12
  and CUDA 13 builds of PyTorch, and conda picks one that your driver supports.
  With an older driver the command fails to solve, because conda-forge has no
  CUDA 11 build of PyTorch `>=2.6`; upgrade the driver rather than dropping
  `>=2.6`, which would bring back the problem described under "No pip torch in
  a conda environment" below.
- **GPU not visible to conda** (e.g. on an HPC login node): prefix the command
  with `CONDA_OVERRIDE_CUDA=12.9`. Do the same on a Maxwell, Pascal or Volta GPU
  with driver 580 or newer, since the CUDA 13 builds conda would pick do not
  support those GPUs.
- **SageMath in an existing environment:**
  `conda install -c conda-forge sage "pytorch>=2.6=cuda*"` (or
  `sage "pytorch>=2.6"` without a GPU).
- **No pip torch in a conda environment:** do not install torch with pip there,
  and do not run `pip install calt-x` there before PyTorch is installed from
  conda-forge. PyPI's CUDA build of torch loads the system `libstdc++`, and
  conda-forge packages such as SciPy and SageMath then fail to import with
  ``version `CXXABI_1.3.15' not found`` on systems whose `libstdc++` is older
  than conda-forge's (e.g. Ubuntu 22.04). Once PyTorch comes from conda-forge,
  `pip install calt-x` keeps it.
- **GPU with pip:** on Linux, PyPI's torch 2.11 and later
  is built for CUDA 13 and needs NVIDIA driver 580 or newer. With an older
  driver, training falls back to the CPU and CALT logs a warning. Upgrade the
  driver, or install a CUDA 12 build of torch: `pip install "torch<2.11"`
  (Python 3.11–3.14) or
  `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`
  (any Python, including 3.15). This does not apply to conda environments;
  see the notes above.

### Instance Generation

For minimal usage, users only need to implement an instance generator for their
own task. For example:

```python
def int_sum_generator(seed, N=5, lb=-10, ub=10):
    random.seed(seed)

    # get N random integers from [lb, ub]
    problem = [random.randint(lb, ub) for _ in range(N)]
    answer = sum(problem)

    return problem, answer
```

Feeding the generator to `DatasetPipeline` generates training and evaluation
sets. The `data.yaml` configuration file provides full control over the
generation process.

```python
cfg = OmegaConf.load("configs/data.yaml")
pipeline = DatasetPipeline.from_config(
    cfg.dataset,
    instance_generator=int_sum_generator
)
pipeline.run()
```

### Training Script

Then, a short script handles training and evaluation through `IOPipeline`,
`ModelPipeline`, and `TrainerPipeline`. The `train.yaml` configuration file and
the associated `lexer.yaml` file provide control over the training setup.

```python
cfg = OmegaConf.load("configs/train.yaml")

io_pipeline = IOPipeline.from_config(cfg.data)
io_dict = io_pipeline.build()

model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

trainer_pipeline.train()
trainer_pipeline.save_model()
trainer_pipeline.evaluate_and_save_generation()
```

### Examples

See `examples/` for minimal examples. For practical experiments and project
templates built with CALT, see
[calt-codebase](https://github.com/HiroshiKERA/calt-codebase).
The [CALT Task Builder](https://calt.mai2.org/) lets you pick ready-made
example tasks or define your own generator in your browser, then download a
ready-to-run CALT project (data generation and training run on your machine).

### For users without a local GPU

If you do not have a local GPU, you can still try CALT in two ways:

1. **Run the demo on Google Colab**

   Use the demo notebook from your browser:
   <https://colab.research.google.com/github/HiroshiKERA/calt/blob/main/examples/demos/minimal_demo.ipynb>

2. **Use remote jobs on Kaggle**

   Submit and monitor training jobs from your local terminal using
   `calt remote ...`. See the remote job documentation:
   <https://hiroshikera.github.io/calt/remote/>

## Citation

If you use CALT in your project, please cite our paper:

```bibtex
@misc{kera2025calt,
  title={CALT: A Library for Computer Algebra with Transformer},
  author={Hiroshi Kera and Shun Arakawa and Yuta Sato},
  year={2025},
  archivePrefix={arXiv},
  eprint={2506.08600}
}
```

> Note: The current arXiv preprint is based on the previous version of CALT.
> The update will come soon.
