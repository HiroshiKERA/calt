"""Base classes for post-processors (processors that run after lexer)."""

import logging
from abc import ABC, abstractmethod
from typing import Iterable

# Basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class TermParseException(Exception):
    """Custom exception raised during term parsing errors."""
    pass

class AbstractPostProcessor(ABC):
    """Base abstract class for post-processors that run after lexer tokenization.
    
    Post-processors are used to apply additional transformations to the tokenized
    text after the lexer has converted raw input to tokens. They can be chained
    together using PostProcessorChain.
    """

    def __init__(self):
        """Initialize post-processor."""
        pass

    def __call__(self, text: str) -> str:
        """Process text (convenience wrapper for encode method)."""
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> str:
        """Encode text (apply transformation).
        
        Args:
            text: Input text (token text from lexer).
        
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

class PostProcessorChain(AbstractPostProcessor):
    """Compose multiple post-processors and apply them sequentially."""

    def __init__(self, processors: Iterable["AbstractPostProcessor"]) -> None:
        """Initialize post-processor chain.
        
        Args:
            processors: Iterable of post-processors to chain together.
        """
        processors = list(processors)
        if not processors:
            raise ValueError("PostProcessorChain requires at least one post-processor.")
        
        super().__init__()
        self.processors = processors

    def encode(self, text: str) -> str:
        """Apply all post-processors in sequence."""
        for processor in self.processors:
            text = processor.encode(text)
        return text

    def decode(self, tokens: str) -> str:
        """Reverse all post-processors in reverse order."""
        for processor in reversed(self.processors):
            tokens = processor.decode(tokens)
        return tokens

