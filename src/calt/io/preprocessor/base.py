"""Base classes for pre-processors."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable

# Basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class TermParseException(Exception):
    """Custom exception raised during term parsing errors."""
    pass

class AbstractPreProcessor(ABC):
    """Base abstract class for pre-processors.
    
    Pre-processors are used to transform text. Lexer is a type of pre-processor
    that converts raw input to tokens. Additional pre-processors can be chained
    together using PreProcessorChain.
    """

    def __init__(self):
        """Initialize pre-processor."""
        pass

    def __call__(self, text: str) -> str:
        """Process text (convenience wrapper for encode method)."""
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> str:
        """Encode text (apply transformation).
        
        Args:
            text: Input text.
        
        Returns:
            Transformed text.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: str) -> str:
        """Decode tokens (reverse transformation).
        
        Args:
            tokens: Transformed text.
        
        Returns:
            Original text.
        """
        raise NotImplementedError

    # For backward compatibility: process is an alias for encode
    def process(self, text: str) -> str:
        return self.encode(text)

class PreProcessorChain(AbstractPreProcessor):
    """Compose multiple post-processors and apply them sequentially."""

    def __init__(self, processors: Iterable["AbstractPreProcessor"]) -> None:
        """Initialize pre-processor chain.
        
        Args:
            processors: Iterable of pre-processors to chain together.
        """
        processors = list(processors)
        if not processors:
            raise ValueError("PreProcessorChain requires at least one pre-processor.")
        
        super().__init__()
        self.processors = processors

    def encode(self, text: str) -> str:
        """Apply all pre-processors in sequence."""
        for processor in self.processors:
            text = processor.encode(text)
        return text

    def decode(self, tokens: str) -> str:
        """Reverse all pre-processors in reverse order."""
        for processor in reversed(self.processors):
            tokens = processor.decode(tokens)
        return tokens

