# Dataset Generator

A unified interface with SageMath and SymPy backends for large-scale dataset generation. It produces paired problems and solutions, supports batch writing, and computes incremental statistics.

## Common (SageMath backend example)

### Generation flow
:::: calt.dataset_generator.sagemath.dataset_generator.DatasetGenerator

### Writing and statistics
:::: calt.dataset_generator.sagemath.utils.dataset_writer.DatasetWriter
:::: calt.dataset_generator.sagemath.utils.statistics_calculator.MemoryEfficientStatisticsCalculator

### Sampling
:::: calt.dataset_generator.sagemath.utils.polynomial_sampler.PolynomialSampler

## Common (SymPy backend example)

### Generation flow
:::: calt.dataset_generator.sympy.dataset_generator.DatasetGenerator

### Writing and statistics
:::: calt.dataset_generator.sympy.utils.dataset_writer.DatasetWriter
:::: calt.dataset_generator.sympy.utils.statistics_calculator.MemoryEfficientStatisticsCalculator

### Sampling
:::: calt.dataset_generator.sympy.utils.polynomial_sampler.PolynomialSampler
:::: calt.dataset_generator.sympy.utils.single_polynomial_sampler.SinglePolynomialSampler
