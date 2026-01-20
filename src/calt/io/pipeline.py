"""Data loading utilities for the Transformer Algebra project.

This module defines helper functions that build the training and evaluation
`Dataset`, `Tokenizer`, and `DataCollator` objects used throughout the
library.  In particular, the `load_data` factory translates symbolic
polynomial expressions into the internal token representation expected by the
Transformer models.
"""

import logging
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerFast as StandardTokenizer

from .base import (
    StandardDataCollator,
    StandardDataset,
)
from .preprocessors import (
    AbstractPostProcessor,
    UnifiedLexer,
    NumberPolicy,
)
from .tokenizer import get_tokenizer
from .vocabs.base import VocabConfig

from .vocabs import get_generic_vocab, get_monomial_vocab

logger = logging.getLogger(__name__)


def load_data(
    train_dataset_path: str | None = None,
    test_dataset_path: str | None = None,
    field: str | None = None,
    num_variables: int | None = None,
    max_degree: int | None = None,
    max_coeff: int | None = None,
    max_length: int = 512,
    processor: AbstractPostProcessor | None = None,
    num_train_samples: int | None = None,
    num_test_samples: int | None = None,
    vocab_config: VocabConfig | dict | str | None = None,
):
    """Load data and create dataset, tokenizer, and data collator.
    
    This is a convenience function that wraps IOPipeline for backward compatibility.
    
    Args:
        train_dataset_path: Path to training dataset file
        test_dataset_path: Path to test dataset file
        field: Field specification ("ZZ" for integers, "GF<p>" for finite field)
        num_variables: Number of variables in polynomials
        max_degree: Maximum degree for any variable
        max_coeff: Maximum absolute value for coefficients
        max_length: Maximum sequence length
        processor: Preprocessor instance (optional)
        num_train_samples: Maximum number of training samples to load
        num_test_samples: Maximum number of test samples to load
        vocab_config: VocabConfig, dict, or path to YAML file (optional)
    
    Returns:
        tuple: (dataset_dict, tokenizer, data_collator) where dataset_dict contains
            "train" and "test" keys with StandardDataset instances
    """
    # Create vocab_config if field is provided
    if vocab_config is None and field is not None:
        if num_variables is None or max_degree is None or max_coeff is None:
            raise ValueError("num_variables, max_degree, and max_coeff must be provided when field is specified")
        
        vocab_config = get_monomial_vocab(num_variables, -max_coeff, max_coeff, 0, max_degree)
    
    # Use IOPipeline to build the data pipeline
    pipeline = IOPipeline(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        vocab_config=vocab_config,
        preprocessor=processor,
    )
    
    result = pipeline.build()
    
    # Return in the expected format: (dataset_dict, tokenizer, data_collator)
    dataset_dict = {
        "train": result["train_dataset"],
        "test": result["test_dataset"],
    }
    
    return dataset_dict, result["tokenizer"], result["data_collator"]

class IOPipeline():
    def __init__(self, 
                 train_dataset_path: str | None = None,
                 test_dataset_path: str | None = None,
                 num_train_samples: int | None = None,
                 num_test_samples: int | None = None,
                 vocab_config: VocabConfig | dict | str | None = None,
                 preprocessor: AbstractPostProcessor | None = None):
        """Initialize IOPipeline.
        
        Args:
            train_dataset_path: Path to training dataset file
            test_dataset_path: Path to test dataset file
            num_train_samples: Maximum number of training samples to load
            num_test_samples: Maximum number of test samples to load
            vocab_config: VocabConfig, dict, or path to YAML file
            preprocessor: Preprocessor instance (optional)
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.vocab_config = self.get_vocab_config(vocab_config)
        self.preprocessor = preprocessor
    
    @classmethod
    def from_config(cls, config: DictConfig) -> "IOPipeline":
        """Create IOPipeline from configuration.
        
        Args:
            config (DictConfig): Data configuration from cfg.data (OmegaConf).
                Can include:
                - preprocessor: str ("generic", "none", "null") or "lexer" to use UnifiedLexer
                - lexer_config: str path to lexer.yaml file (required if preprocessor="lexer")
                - vocab_config: str path to vocab.yaml file (if not using lexer_config)
        
        Returns:
            IOPipeline: IOPipeline instance configured from the config.
        
        Example:
            >>> from omegaconf import OmegaConf
            >>> from calt.io.pipeline import IOPipeline
            >>> 
            >>> cfg = OmegaConf.load("config/train.yaml")
            >>> io_pipeline = IOPipeline.from_config(cfg.data)
        """
        preprocessor = config.get("preprocessor")
        lexer_config_path = config.get("lexer_config")
        vocab_config_path = config.get("vocab_config")
        
        # Handle lexer config
        if isinstance(preprocessor, str) and preprocessor.lower() == "lexer":
            if lexer_config_path is None:
                raise ValueError("lexer_config must be provided when preprocessor='lexer'")
            
            # Resolve lexer config path (support relative paths)
            lexer_config_path_obj = Path(lexer_config_path)
            if not lexer_config_path_obj.is_absolute():
                # Try to resolve relative to current working directory first
                if not lexer_config_path_obj.exists():
                    # If not found, try relative to config file location if available
                    # (This is a best-effort approach)
                    pass
            lexer_config_path = str(lexer_config_path_obj.resolve())
            
            # Load lexer config
            with open(lexer_config_path, 'r') as f:
                lexer_config = yaml.safe_load(f)
            
            # Create VocabConfig from lexer config
            vocab_config_dict = lexer_config.get("vocab", {})
            vocab_config = VocabConfig([], {}).from_config(vocab_config_dict)
            
            # Create NumberPolicy from lexer config
            number_policy_dict = lexer_config.get("number_policy", {})
            number_policy = NumberPolicy(
                sign=number_policy_dict.get("sign", "separate"),
                digit_group=number_policy_dict.get("digit_group", 0),
                allow_float=number_policy_dict.get("allow_float", True),
                dot_token=number_policy_dict.get("dot_token", "."),
            )
            
            # Create UnifiedLexer
            preprocessor = UnifiedLexer(
                vocab_config=vocab_config,
                number_policy=number_policy,
                strict=lexer_config.get("strict", True),
                include_base_vocab=lexer_config.get("include_base_vocab", True),
            )
            
            # Use vocab_config from lexer config
            vocab_config_path = vocab_config
        elif isinstance(preprocessor, str):
            # Convert string preprocessor name to None (generic means no preprocessing)
            if preprocessor.lower() in ["generic", "none", "null"]:
                preprocessor = None
            else:
                # If other preprocessor types are needed, add them here
                raise ValueError(
                    f"Unsupported preprocessor type: {preprocessor}. "
                    f"Supported types: 'generic', 'none', 'null', 'lexer'"
                )
        
        return cls(
            train_dataset_path=config.get("train_dataset_path"),
            test_dataset_path=config.get("test_dataset_path"),
            num_train_samples=config.get("num_train_samples", -1),
            num_test_samples=config.get("num_test_samples", -1),
            vocab_config=vocab_config_path,
            preprocessor=preprocessor,
        )
    
    def get_vocab_config(self, vocab_config: VocabConfig | dict | str | None):
        if vocab_config is None:
            return None
        if not isinstance(vocab_config, VocabConfig):
            vocab_config_obj = VocabConfig([], {})
            vocab_config = vocab_config_obj.from_config(vocab_config)
        
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
        if self.vocab_config is None:
            raise ValueError("vocab_config must be provided to build the tokenizer")
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
    
