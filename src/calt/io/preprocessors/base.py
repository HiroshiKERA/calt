"""Base classes for preprocessors."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable

# Basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class TermParseException(Exception):
    """Custom exception raised during term parsing errors."""

    pass

class AbstractPreprocessor(ABC):
    """Base abstract class for all preprocessors."""

    def __init__(self, num_variables: int, max_degree: int, max_coeff: int):
        """Initialize preprocessor parameters.

        Args:
            num_variables (int): Number of variables in the polynomial (e.g., x0, x1, ...).
            max_degree (int): Maximum degree of the polynomial.
            max_coeff (int): Maximum coefficient value in the polynomial.
        """
        if num_variables < 0:
            raise ValueError("num_variables must be positive")
        if max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if max_coeff <= 0:
            raise ValueError("max_coeff must be positive")

        self.num_variables = num_variables
        self.max_degree = max_degree
        self.max_coeff = max_coeff
        self.var_name_to_index = {f"x{i}": i for i in range(num_variables)}

    def __call__(self, text: str) -> str:
        """Process text (convenience wrapper for process method)."""
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> str:
        """Abstract method for text processing to be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: str) -> str:
        """Abstract method for token processing to be implemented by subclasses."""
        raise NotImplementedError

    # For backward compatibility: process is an alias for to_internal
    def process(self, text: str) -> str:
        return self.encode(text)

class ProcessorChain(AbstractPreprocessor):
    """Compose multiple preprocessors and apply them sequentially."""

    def __init__(self, processors: Iterable["AbstractPreprocessor"]) -> None:
        processors = list(processors)
        if not processors:
            raise ValueError("ProcessorChain requires at least one preprocessor.")

        first = processors[0]
        super().__init__(
            num_variables=first.num_variables,
            max_degree=first.max_degree,
            max_coeff=first.max_coeff,
        )
        self.processors = processors

    def encode(self, text: str) -> str:
        for processor in self.processors:
            text = processor.encode(text)
        return text

    def decode(self, tokens: str) -> str:
        for processor in reversed(self.processors):
            tokens = processor.decode(tokens)
        return tokens
