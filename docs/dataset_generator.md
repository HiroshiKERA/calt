# Dataset Generator

A unified interface with SageMath and SymPy backends for large-scale dataset generation. It produces paired problems and solutions, supports batch writing, and computes incremental statistics.

## Common (SageMath backend example)

### Generation flow
::: calt.dataset.sagemath.dataset_generator.DatasetGenerator

### Writing and statistics
::: calt.dataset.sagemath.utils.dataset_writer.DatasetWriter
::: calt.dataset.sagemath.utils.statistics_calculator.MemoryEfficientStatisticsCalculator

### Sampling
::: calt.dataset.sagemath.utils.polynomial_sampler.PolynomialSampler

## Common (SymPy backend example)

### Generation flow
::: calt.dataset.sympy.dataset_generator.DatasetGenerator

### Writing and statistics
::: calt.dataset.sympy.utils.dataset_writer.DatasetWriter
::: calt.dataset.sympy.utils.statistics_calculator.MemoryEfficientStatisticsCalculator

### Sampling
::: calt.dataset.sympy.utils.polynomial_sampler.PolynomialSampler
::: calt.dataset.sympy.utils.single_polynomial_sampler.SinglePolynomialSampler
