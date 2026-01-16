"""Data loading utilities for the Transformer Algebra project.

This module defines helper functions that build the training and evaluation
`Dataset`, `Tokenizer`, and `DataCollator` objects used throughout the
library.  In particular, the `load_data` factory translates symbolic
polynomial expressions into the internal token representation expected by the
Transformer models.
"""

import logging

import yaml
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerFast as StandardTokenizer

from .utils.data_collator import (
    StandardDataCollator,
    StandardDataset,
    _read_data_from_file,
)
from .processors import (
    AbstractPreprocessor,
    IntegerToInternalProcessor,
    PolynomialToInternalProcessor,
)
from .utils.tokenizer import VocabConfig, get_tokenizer

from .vocabs import get_generic_vocab, get_monomial_vocab

logger = logging.getLogger(__name__)

class IOPipeline():
    def __init__(self, 
                 train_dataset_path: str | None = None,
                 test_dataset_path: str | None = None,
                 num_train_samples: int | None = None,
                 num_test_samples: int | None = None,
                 vocab_config: VocabConfig | dict | str | None = None,
                 preprocessor: AbstractPreprocessor | None = None):
        
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.vocab_config = self.get_vocab_config(vocab_config)
        self.preprocessor = preprocessor
    
    def get_vocab_config(self, vocab_config: VocabConfig | dict | str):
        if not isinstance(vocab_config, VocabConfig):
            vocab_config = VocabConfig.from_config(vocab_config)
        
        return vocab_config
    
    def build(self):
        '''
        Build the data pipeline by loading the raw text data, applying the preprocessor, and setting the collator.
        '''
        # config = self.config
        
        # Step 1: Load raw text data and apply preprocessor (if any)
        # e.g., 
        # raw data: "2*x1^2*x0 + 5*x0 - 3"
        # processed data (by InfixPolynomialProcessor): "C2 E1 E2 + C5 E1 E0 + C-3 E0 E0"
        train_dataset = StandardDataset.load_file(self.train_dataset_path, self.preprocessor, self.num_train_samples)
        test_dataset = StandardDataset.load_file(self.test_dataset_path, self.preprocessor, self.num_test_samples)

        # Step 2: Set collator that will transform the processed data into tokens (or token ids)
        #         This will be called every time at the beginning of each epoch
        # e.g., 
        # processed data: "C2 E1 E2 + C5 E1 E0 + C-3 E0 E0"
        # tokens: ["C2", "E1", "E2", "C5", "E1", "E0", "C-3", "E0", "E0"]
        tokenizer = get_tokenizer(vocab_config=self.vocab_config)
        data_collator = StandardDataCollator(tokenizer=tokenizer)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        
        return {"train_dataset": train_dataset, 
                "test_dataset": test_dataset, 
                "data_collator": data_collator, 
                "tokenizer": tokenizer}
    
