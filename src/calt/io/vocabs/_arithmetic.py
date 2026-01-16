from .utils import get_vocab

def get_generic_vocab(field, num_variables, min_numbers, max_numbers):
    vocab_list = [("", min_numbers, max_numbers+1),
                  ]
    misc = ['+', '*', '^', '/', '(', ')']

    return get_vocab(vocab_list, misc)

def get_minimal_vocab(min_numbers, max_numbers):
    
    vocab_list = [("", min_numbers, max_numbers+1)]
    misc = []

    return get_vocab(vocab_list, misc)