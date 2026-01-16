"""Backward compatibility module for preprocessor imports.

This module re-exports all preprocessor classes from the new processors module
to maintain backward compatibility with existing code.
"""

# Re-export all classes from the new processors module
from ..processors import (
    AbstractPreprocessor,
    CoefficientPostfixProcessor,
    IntegerToInternalProcessor,
    PolynomialToInternalProcessor,
    ProcessorChain,
    TermParseException,
)

__all__ = [
    "AbstractPreprocessor",
    "CoefficientPostfixProcessor",
    "IntegerToInternalProcessor",
    "PolynomialToInternalProcessor",
    "ProcessorChain",
    "TermParseException",
]
