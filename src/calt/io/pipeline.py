"""Data loading utilities for the Transformer Algebra project.

This module defines :class:`IOPipeline`, which builds the training and evaluation
`Dataset`, `Tokenizer`, and `DataCollator` objects used throughout the library.
"""

import logging
from pathlib import Path

import yaml
from omegaconf import DictConfig

from .base import (
    StandardDataCollator,
    StandardDataset,
)
from .preprocessor import (
    AbstractPreProcessor,
    DatasetLoadPreprocessor,
    NumberPolicy,
    UnifiedLexer,
)
from .read import read_data_from_file
from .tokenizer import get_tokenizer
from .validation.vocab_validator import validate_dataset_tokens
from .vocabulary.config import VocabConfig

logger = logging.getLogger(__name__)


class IOPipeline:
    def __init__(
        self,
        train_dataset_path: str | None = None,
        test_dataset_path: str | None = None,
        num_train_samples: int | None = None,
        num_test_samples: int | None = None,
        vocab_config: VocabConfig | dict | str | None = None,
        preprocessor: AbstractPreProcessor | None = None,
        validate_train_tokens: bool = False,
        validate_test_tokens: bool = False,
        use_jsonl: bool = False,
        use_pickle: bool = False,
        train_dataset_jsonl: str | None = None,
        test_dataset_jsonl: str | None = None,
        train_dataset_pickle: str | None = None,
        test_dataset_pickle: str | None = None,
        dataset_load_preprocessor: DatasetLoadPreprocessor | None = None,
        display_samples: int | None = None,
    ):
        """Initialize IOPipeline.

        Args:
            train_dataset_path: Path to training dataset file (.txt, .jsonl, or .pkl)
            test_dataset_path: Path to test dataset file
            num_train_samples: Maximum number of training samples to load
            num_test_samples: Maximum number of test samples to load
            vocab_config: VocabConfig, dict, or path to YAML file
            preprocessor: Lexer/preprocessor instance (optional)
            use_jsonl: If True, read train/test as JSONL when path or jsonl path is set
            use_pickle: If True, read train/test as pickle (original math objects)
            train_dataset_jsonl: Optional path to training JSONL
            test_dataset_jsonl: Optional path to test JSONL
            train_dataset_pickle: Optional path to training pickle
            test_dataset_pickle: Optional path to test pickle
            dataset_load_preprocessor: Optional load-time preprocessor (user-provided or library)
            display_samples: If set and > 0, print this many train samples: raw (before load
                preprocessor), which preprocessor is applied (if any), and after preprocessor.
                0 or None to disable.
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.vocab_config = self.get_vocab_config(vocab_config)
        self.preprocessor = preprocessor
        self.validate_train_tokens = validate_train_tokens
        self.validate_test_tokens = validate_test_tokens
        self.use_jsonl = use_jsonl
        self.use_pickle = use_pickle
        self.train_dataset_jsonl = train_dataset_jsonl
        self.test_dataset_jsonl = test_dataset_jsonl
        self.train_dataset_pickle = train_dataset_pickle
        self.test_dataset_pickle = test_dataset_pickle
        self.dataset_load_preprocessor = dataset_load_preprocessor
        self.display_samples = display_samples
        # Store config dicts for checkpoint saving
        self.lexer_config_dict: dict | None = None
        self.vocab_config_dict: dict | None = None

    @classmethod
    def from_config(cls, config: DictConfig) -> "IOPipeline":
        """Create IOPipeline from configuration.

        Args:
            config (DictConfig): Data configuration from cfg.data (OmegaConf).
                Must include:
                - lexer_config: str path to lexer.yaml file (required)

        Returns:
            IOPipeline: IOPipeline instance configured from the config.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> from calt.io import IOPipeline
            >>>
            >>> cfg = OmegaConf.load("config/train.yaml")
            >>> io_pipeline = IOPipeline.from_config(cfg.data)
        """
        lexer_config_path = config.get("lexer_config")
        if lexer_config_path is None:
            raise ValueError("lexer_config must be provided")

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
        with open(lexer_config_path, "r") as f:
            lexer_config = yaml.safe_load(f)

        # Create VocabConfig from lexer config
        vocab_config_dict = lexer_config.get("vocab", {})
        vocab_config = VocabConfig([], {}).from_config(vocab_config_dict)

        # Create NumberPolicy from lexer config
        number_policy_dict = lexer_config.get("number_policy", {})
        # attach_sign: true = attach sign to number, false = separate sign as token
        attach_sign = number_policy_dict.get("attach_sign", True)  # default: attach
        number_policy = NumberPolicy(
            sign=attach_sign,  # sign=True means attach, sign=False means separate
            digit_group=number_policy_dict.get("digit_group", 0),
            allow_float=number_policy_dict.get("allow_float", True),
        )

        # Create UnifiedLexer (vocab extension is handled inside UnifiedLexer.__init__)
        preprocessor = UnifiedLexer(
            vocab_config=vocab_config,
            number_policy=number_policy,
            strict=lexer_config.get("strict", True),
            include_base_vocab=lexer_config.get("include_base_vocab", True),
        )

        # Use the extended vocab_config from lexer (includes auto-added tokens for floats)
        vocab_config_path = preprocessor.vocab_config

        use_jsonl = config.get("use_jsonl", False)
        use_pickle = config.get("use_pickle", False)
        train_jsonl = config.get("train_dataset_jsonl")
        test_jsonl = config.get("test_dataset_jsonl")
        train_pickle = config.get("train_dataset_pickle")
        test_pickle = config.get("test_dataset_pickle")
        dataset_load_preprocessor = config.get("dataset_load_preprocessor")
        display_samples = config.get("display_samples")

        # Create instance
        instance = cls(
            train_dataset_path=config.get("train_dataset_path"),
            test_dataset_path=config.get("test_dataset_path"),
            num_train_samples=config.get("num_train_samples", -1),
            num_test_samples=config.get("num_test_samples", -1),
            vocab_config=vocab_config_path,
            preprocessor=preprocessor,
            validate_train_tokens=config.get("validate_train_tokens", False),
            validate_test_tokens=config.get("validate_test_tokens", True),
            use_jsonl=use_jsonl,
            use_pickle=use_pickle,
            train_dataset_jsonl=train_jsonl,
            test_dataset_jsonl=test_jsonl,
            train_dataset_pickle=train_pickle,
            test_dataset_pickle=test_pickle,
            dataset_load_preprocessor=dataset_load_preprocessor,
            display_samples=display_samples,
        )

        # Store config dicts for checkpoint saving
        # Store the original lexer_config (includes vocab, number_policy, strict, etc.)
        instance.lexer_config_dict = lexer_config
        # Store vocab_config dict (from lexer config, before extension)
        instance.vocab_config_dict = vocab_config_dict

        return instance

    def get_vocab_config(self, vocab_config: VocabConfig | dict | str | None):
        if vocab_config is None:
            return None
        if not isinstance(vocab_config, VocabConfig):
            vocab_config_obj = VocabConfig([], {})
            vocab_config = vocab_config_obj.from_config(vocab_config)

        return vocab_config

    def validate_tokens(self, dataset: StandardDataset):
        """Validate tokens in a dataset and raise error if out-of-vocabulary tokens are found."""

        out_of_vocab_tokens = validate_dataset_tokens(
            lexer=self.preprocessor,
            vocab_config=self.vocab_config,
            input_texts=dataset.input_texts,
            target_texts=dataset.target_texts,
        )

        if out_of_vocab_tokens:
            token_list = ", ".join([f"'{token}'" for token in out_of_vocab_tokens])
            error_msg = (
                "\n--------------------------------\n"
                f"Vocabulary validation errors in dataset.\n"
                f"Out-of-vocabulary tokens: {token_list}\n"
                f"Please check your lexer.yaml configuration and dataset generation."
                "\n--------------------------------\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def build(self):
        """
        Build the data pipeline by loading the raw text data, applying the preprocessor, and setting the collator.
        """
        # config = self.config

        # Step 1: Load data (text, JSONL, or pickle) and apply load-time preprocessor if any
        train_path = (
            self.train_dataset_pickle
            or self.train_dataset_jsonl
            or self.train_dataset_path
        )
        test_path = (
            self.test_dataset_pickle
            or self.test_dataset_jsonl
            or self.test_dataset_path
        )
        train_use_pickle = self.use_pickle or (
            self.train_dataset_pickle is not None
            or (train_path and str(train_path).endswith(".pkl"))
        )
        test_use_pickle = self.use_pickle or (
            self.test_dataset_pickle is not None
            or (test_path and str(test_path).endswith(".pkl"))
        )
        train_use_jsonl = self.use_jsonl or (
            self.train_dataset_jsonl is not None
            or (train_path and str(train_path).endswith(".jsonl"))
        )
        test_use_jsonl = self.use_jsonl or (
            self.test_dataset_jsonl is not None
            or (test_path and str(test_path).endswith(".jsonl"))
        )
        train_preprocessor = self.dataset_load_preprocessor
        if train_preprocessor is None and (train_use_jsonl or train_use_pickle):
            from .preprocessor.load_preprocessor import (
                JsonlDefaultLoadPreprocessor,
                PickleDefaultLoadPreprocessor,
            )

            train_preprocessor = (
                PickleDefaultLoadPreprocessor()
                if train_use_pickle
                else JsonlDefaultLoadPreprocessor()
            )
        test_preprocessor = self.dataset_load_preprocessor
        if test_preprocessor is None and (test_use_jsonl or test_use_pickle):
            from .preprocessor.load_preprocessor import (
                JsonlDefaultLoadPreprocessor,
                PickleDefaultLoadPreprocessor,
            )

            test_preprocessor = (
                PickleDefaultLoadPreprocessor()
                if test_use_pickle
                else JsonlDefaultLoadPreprocessor()
            )
        n_show = self.display_samples if self.display_samples is not None else 0
        # When display_samples > 0 and plain txt: load and show raw (before any load preprocessor)
        if n_show > 0 and not train_use_jsonl and not train_use_pickle and train_path:
            raw_inputs, raw_targets = read_data_from_file(
                train_path, max_samples=self.num_train_samples
            )
            n_raw = min(n_show, len(raw_inputs))
            print(
                f"[Display] Raw (before any load preprocessor): {len(raw_inputs)} samples, "
                f"showing first {n_raw}:"
            )
            for i in range(n_raw):
                inp = (
                    raw_inputs[i]
                    if len(raw_inputs[i]) <= 50
                    else raw_inputs[i][:47] + "..."
                )
                tgt = (
                    raw_targets[i]
                    if len(raw_targets[i]) <= 50
                    else raw_targets[i][:47] + "..."
                )
                print(f"  [{i}] input:  {inp!r}")
                print(f"      target: {tgt!r}")
            print()
        train_dataset = StandardDataset.load_file(
            train_path,
            self.preprocessor,
            self.num_train_samples,
            use_jsonl=train_use_jsonl,
            use_pickle=train_use_pickle,
            dataset_load_preprocessor=train_preprocessor,
        )
        test_dataset = StandardDataset.load_file(
            test_path,
            self.preprocessor,
            self.num_test_samples,
            use_jsonl=test_use_jsonl,
            use_pickle=test_use_pickle,
            dataset_load_preprocessor=test_preprocessor,
        )

        if self.validate_train_tokens:
            print("Validating training dataset tokens...", end=" ")
            self.validate_tokens(train_dataset)
            print("passed!")
        if self.validate_test_tokens:
            print("Validating test dataset tokens...", end=" ")
            self.validate_tokens(test_dataset)
            print("passed!")

        # Display samples: preprocessor description (if any) and after-preprocessor samples
        if n_show > 0:
            if train_preprocessor is not None:
                name = type(train_preprocessor).__name__
                if hasattr(train_preprocessor, "preprocessors"):
                    chain = ", ".join(
                        type(p).__name__ for p in train_preprocessor.preprocessors
                    )
                    name = f"{name}({chain})"
                print(f"[Display] Load preprocessor: {name}")
            else:
                print("[Display] No load preprocessor applied.")
            n_after = min(n_show, len(train_dataset.input_texts))
            print(
                f"[Display] After load preprocessor: {len(train_dataset.input_texts)} samples, "
                f"showing first {n_after}:"
            )
            for i in range(n_after):
                inp = train_dataset.input_texts[i]
                tgt = train_dataset.target_texts[i]
                inp_short = inp if len(inp) <= 50 else inp[:47] + "..."
                tgt_short = tgt if len(tgt) <= 50 else tgt[:47] + "..."
                print(f"  [{i}] input:  {inp_short!r}")
                print(f"      target: {tgt_short!r}")
            print()

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

        self.io_dict = {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "tokenizer": tokenizer,
            "data_collator": data_collator,
        }

        return self.io_dict
