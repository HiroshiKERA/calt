# CALT: Computer ALgebra with Transformer

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hiroshikera.github.io/calt/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-View%20Documentation-blue.svg)](https://hiroshikera.github.io/calt/)

> 📖 **📚 [View Full Documentation](https://hiroshikera.github.io/calt/)**

## Overview

CALT is a simple Python library for learning arithmetic and symbolic computation with a Transformer model (a deep neural model to realize sequence-to-sequence functions). 

It offers a basic Transformer model and training pipeline, and non-experts of deep learning can focus on constructing datasets to train and evaluate the model. 

## Quick Start

### Installation

```bash
pip install calt-x
```

### Instance Generation
For minimal usage, users only need to implement an instance generator for their own task. For example:

```python
def int_sum_generator(seed, N=5, lb=-10, ub=10):
    random.seed(seed)

    # get N random integers from [-10, 10]
    problem = [random.randint(lb, ub) for _ in range(N)]
    answer = sum(problem)

    return problem, answer
```

Feeding the generator to CALT `DataPieline` generates training and evaluation sets. The `data.yaml` gives a full control over the generation process. 
```python
cfg = OmegaConf.load("configs/data.yaml")
pipeline = DatasetPipeline.from_config(
    cfg.dataset,
    instance_generator=int_sum_generator
)
pipeline.run()
```

### Training Script
Then, a short script implement the training and evalutation through `IOPipeline`, `ModelPipeline`, and `TrainerPipeline`. The config file `train.yaml` (and associated `lexer.yaml`) gives full control over the training setup. 
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
See `exmamples/` directories. 


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
> Note: The current arXiv preprint is based on the previous version of CALT. The update will come soon. 

