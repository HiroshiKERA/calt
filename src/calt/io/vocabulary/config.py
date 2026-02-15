import yaml

BASE_VOCAB = ["[SEP]"]
BASE_SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "cls_token": "[CLS]",
}


def get_base_vocab():
    return BASE_VOCAB


def get_base_special_tokens():
    return BASE_SPECIAL_TOKENS


class VocabConfig:
    def __init__(
        self,
        vocab: list[str],
        special_tokens: dict[str, str],
        include_base_vocab=True,
        include_base_special_tokens=True,
    ):
        self.vocab = vocab
        self.special_tokens = special_tokens

        if include_base_vocab:
            self.vocab = BASE_VOCAB + self.vocab
        if include_base_special_tokens:
            self.special_tokens = BASE_SPECIAL_TOKENS | self.special_tokens

    def from_config(self, config: dict | str):
        """Load VocabConfig from a YAML file or dictionary.

        Expected format:
            range:
              coefficients: ["C", -50, 50]  # (prefix, min, max_inclusive)
              exponents: ["E", 0, 20]
              variables: ["x", 0, 2]
            misc: ["+", "*", "^", "(", ")"]
            special_tokens: {}
            flags:
              include_base_vocab: true
              include_base_special_tokens: true
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        from .utils import get_range_vocab

        range_config = config.get("range", {})
        misc_vocab = config.get("misc", [])
        special_tokens = config.get("special_tokens", {})

        # Check flags section for include_base settings
        flags = config.get("flags", {})
        include_base_vocab = flags.get(
            "include_base_vocab", config.get("include_base_vocab", True)
        )
        include_base_special_tokens = flags.get(
            "include_base_special_tokens",
            config.get("include_base_special_tokens", True),
        )

        range_vocab = []
        for key, value in range_config.items():
            # value can be a list [prefix, min, max] or tuple (prefix, min, max)
            if isinstance(value, (list, tuple)) and len(value) == 3:
                prefix, min_val, max_val = value
                range_vocab.extend(get_range_vocab(prefix, min_val, max_val))

        self.vocab = range_vocab + misc_vocab
        self.special_tokens = special_tokens

        if include_base_vocab:
            self.vocab = BASE_VOCAB + self.vocab
        if include_base_special_tokens:
            self.special_tokens = BASE_SPECIAL_TOKENS | self.special_tokens

        return self

    def get_vocab(self):
        return self.vocab

    def get_special_tokens(self):
        return self.special_tokens

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.dump({"vocab": self.vocab, "special_tokens": self.special_tokens}, f)
