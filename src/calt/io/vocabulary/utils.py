from .config import VocabConfig


# recieve list of (prefix, begin, end)
def get_vocab(vocab_list: list[tuple[str, int, int]], misc=[]):
    
    vocab = []
    for prefix, min, max in vocab_list:
        vocab.extend(get_range_vocab(prefix, min, max))
    
    vocab.extend(misc)
    
    return VocabConfig(vocab=vocab, special_tokens={})

def get_range_vocab(tag, begin, end, centered=False):
    
    shift = (begin + end) // 2 if centered else 0
    if centered: 
        begin = begin - shift
        end = end + shift
    
    return [f"{tag}{i}" for i in range(begin, end+1)]

def get_finite_field_vocab(field, centered=False):
    assert is_finite_field(field)
    p = int(field[2:])
    
    return get_range_vocab("C", -p+1, p, centered=centered)

def is_finite_field(field):
    return field.startswith("GF")