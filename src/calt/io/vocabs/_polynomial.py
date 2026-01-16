from .base import VocabConfig
from .utils import get_vocab

def get_generic_vocab(num_variables, min_coeff, max_coeff, min_degree, max_degree):
    vocab_list = [("C", min_coeff, max_coeff+1),
                  ('E', min_degree, max_degree+1),
                  ("x", 0, num_variables),
                  ]
    misc = ['+', '*', '^', '(', ')']
    return get_vocab(vocab_list, misc)

def get_monomial_vocab(num_variables, min_coeff, max_coeff, min_degree, max_degree):
    '''
    Expanded form (monomial-based representation)
    Each term is represented independently with coefficient and exponent tokens.
    e.g. 2 x0^2 + 3 x1^2 + 4 x2^2
    -> C2 E2 E0 + C3 E2 E1 + C4 E2 E2
    '''
    
    vocab_list = [('C', min_coeff, max_coeff+1),
                  ('E', min_degree, max_degree+1),
                  ('x', 0, num_variables),
                  ]
    misc = ['+']
    return get_vocab(vocab_list, misc)
