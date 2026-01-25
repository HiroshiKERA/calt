"""
Vocabulary validator: Subroutine to apply lexer to a dataset and detect out-of-vocabulary tokens.

This module provides functions to tokenize datasets using a lexer and detect tokens not present in the vocabulary.
"""

from typing import List

from ..vocabulary.config import VocabConfig
from ..preprocessor.lexer import UnifiedLexer


def validate_dataset_tokens(
    lexer: UnifiedLexer,
    vocab_config: VocabConfig,
    input_texts: List[str],
    target_texts: List[str],
    max_samples: int = None,
) -> List[str]:
    """Check a dataset and detect out-of-vocabulary tokens.
    
    Args:
        lexer: UnifiedLexer instance
        vocab_config: VocabConfig instance
        input_texts: List of input text strings
        target_texts: List of target text strings
        max_samples: Maximum number of samples to check (None for all, optional)
        
    Returns:
        List of out-of-vocabulary tokens (unique tokens, sorted)
    """
    
    vocab_set = set(vocab_config.get_vocab())
    special_tokens = vocab_config.get_special_tokens()
    special_token_set = set(special_tokens.values())
    all_valid_tokens = vocab_set | special_token_set
    
    out_of_vocab_tokens = set()
    
    samples_to_check = input_texts[:max_samples] if max_samples else input_texts
    target_samples = target_texts[:max_samples] if max_samples else target_texts
    
    for input_text, target_text in zip(samples_to_check, target_samples):
        # Check both input and target
        for text in [input_text, target_text]:
            try:
                tokens = lexer.tokenize(text)
                
                # Check whether each token is contained in vocab
                for token in tokens:
                    if token not in all_valid_tokens:
                        out_of_vocab_tokens.add(token)
            except Exception:
                pass
    
    return sorted(out_of_vocab_tokens)


