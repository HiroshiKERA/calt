# CALT: Computer ALgebra with Transformer
This project is currently in its initial development phase. The file structure and content are subject to significant changes. Please ensure you are referring to the latest version when using it.

# Environment Setup using Docker

This guide explains how to set up the development environment using Docker.

## Prerequisites

- Docker installed on your system.
- NVIDIA GPU drivers installed if you plan to use GPU acceleration (`--gpus all` option).

## Build the Docker Image

To build the Docker image, navigate to the `calt` directory and run the following command:

```bash
make build
```

Alternatively, you can use the direct Docker command:

```bash
docker build -t ta-sage .
```

## Run the Docker Container

To run the Docker container in detached mode with GPU support, execute:

```bash
make run
```

The direct Docker command is:

```bash
docker run --gpus all -d --name ta-sage-container -v "$(pwd)":/app ta-sage tail -f /dev/null
```
*Note: When running this command directly in your terminal, `$(pwd)` will resolve to your current working directory. The Makefile uses `$(CURDIR)` which serves the same purpose within the Makefile context.*

## Access the Container

Once the container is running, you can access it using:

```bash
docker exec -it ta-sage-container bash
```

## Stop and Remove the Container

To stop and remove the container, you can use:

```bash
make stop
```

Or manually:

```bash
docker stop ta-sage-container
docker rm ta-sage-container
```

## Local Setup (without Docker)

This section describes how to set up the environment locally without using Docker. This assumes you have SageMath installed on your system.

### 1. Install SageMath

You can install SageMath using `apt` on Debian/Ubuntu-based systems. It's not necessary to have the absolute latest version.

Install SageMath:

```bash
sudo apt-get install -y sagemath
```

### 2. Install Dependencies

Once SageMath is installed, you can install the required Python packages using `sage -pip`.

First, upgrade pip:

```bash
sage -pip install --upgrade pip
```

Next, install the Python dependencies:

```bash
sage -pip install --break-system-packages \
    "torch==2.6.0" \
    "transformers>=4.49.0" \
    "omegaconf>=2.3.0" \
    "wandb>=0.15.11" \
    "accelerate>=0.29.0" \
    "joblib>=1.5.0"
```

**For GPU support with PyTorch:**
If you need GPU support, replace the `torch` installation line with the one that specifies the CUDA version compatible with your system. For example, for CUDA 12.4:

```bash
sage -pip install --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0"
```

### 3. Install `transformer_algebra` (Editable)

Finally, install the `transformer_algebra` package in editable mode. Navigate to the root of the `calt` project directory (where this `README.md` and the  `pyproject.toml` for `transformer_algebra` are located) and run:

```bash
sage -pip install -e .
```
This command assumes that the necessary setup files for `transformer_algebra` are in the current directory (`.`). If `transformer_algebra` is a subdirectory (e.g., `/app` as in the Dockerfile context), you would run `sage -pip install -e /path/to/transformer_algebra_directory`.

## Generating Datasets

To generate the default dataset, run the following command from the project root:

```bash
sage scripts/generate.py
```

To generate datasets using a different `ProblemGenerator` class, you will need to modify `scripts/generate.py` by uncommenting the desired `ProblemGenerator` class and commenting out others.

## Running Training

To start training with the default configuration, execute the following command from the project root:

```bash
sage scripts/train.py
```

### Weights & Biases (wandb) Setup

If you are using Weights & Biases (wandb) for the first time to log training progress, you will need to create an account on their website and set up your API key. When you run the training script for the first time, you will be prompted to enter your API key.

https://wandb.ai/site/

### Configuration

Training parameters can be modified by editing the configuration file located at `config/train_example.yaml`.

## Demonstrations

Simple demonstrations for data generation and training are available as Jupyter Notebook files. You can find them in the `notebook` directory (please create this directory and add your notebooks if it doesn't exist yet).

To run these notebooks, you need to start SageMath's Jupyter server using the command `sage -n` and then select the SageMath kernel in the notebook interface.
