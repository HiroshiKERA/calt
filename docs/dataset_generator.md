# Overview

A unified interface with SageMath and SymPy backends for large-scale dataset generation. It produces paired problems and answers, supports batch writing, and computes incremental statistics.

[DatasetPipeline](#calt.dataset.pipeline.DatasetPipeline) and [DatasetWriter](#datasetwriter) are shared regardless of backend (`sagemath` or `sympy`). For details on each component, see:

- [DatasetWriter](#datasetwriter) — writing samples to disk
- [SageMath backend](dataset_sagemath.md) — `DatasetGenerator` and `PolynomialSampler` for SageMath
- [SymPy backend](dataset_sympy.md) — `DatasetGenerator` and `PolynomialSampler` for SymPy

::: calt.dataset.pipeline.DatasetPipeline
    options:
      heading: "DatasetPipeline"

Configuration for the dataset pipeline is done via the `dataset` block in `data.yaml`. For the option list, usage example, and YAML sample, see [Configuration](configuration.md).


::: calt.dataset.utils.dataset_writer.DatasetWriter
    options:
      heading: "DatasetWriter"
      members:
        - __init__

