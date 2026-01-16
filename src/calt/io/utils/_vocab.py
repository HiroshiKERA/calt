from typing import TypedDict

class VocabConfig(TypedDict):
    special_vocab: dict[str, str] = {'pad_token': '[PAD]', 'bos_token': '<s>', 'eos_token': '</s>', 'cls_token': '[CLS]'}
    
    def __init__(self, 
                 vocab: list[str], 
                 special_vocab: dict[str, str] = None):
        self.vocab = vocab
        self.special_vocab = special_vocab if special_vocab is not None else VocabConfig.special_vocab


def set_infix_polynomial_vocab(max_coeff, max_degree, 
                               ops=['+'], 
                               misc_vocab=['[SEP]'], 
                               misc_special_tokens={'pad_token': '[PAD]', 'bos_token': '<s>', 'eos_token': '</s>', 'cls_token': '[CLS]'})
    
    coeffs = [f'C{i}' for i in range(-max_coeff, max_coeff + 1)] 
    exponents = [f'E{i}' for i in range(max_degree + 1)]

    vocab = coeffs + exponents + ops + misc_vocab
    special_vocab = misc_special_tokens

    return VocabConfig(vocab=vocab)

def set_arithmetic_vocab(max_coeff, max_degree, misc=[]):
    coeffs = [f'C{i}' for i in range(-max_coeff, max_coeff + 1)] 
    exponents = [f'E{i}' for i in range(max_degree + 1)]
    ops = [f'{op}' for op in ops]
    misc = [f'{misc}' for misc in misc]
    vocab = coeffs + exponents + ops + misc

    return VocabConfig(vocab=vocab)
    
    
