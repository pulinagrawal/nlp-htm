from transformers import AutoTokenizer
from functools import lru_cache
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

token_ids = lambda text: tokenizer.encode(text)
stringify = lambda token_ids: tokenizer.decode(token_ids)
tokens = lambda text: tokenizer.tokenize(text)

@lru_cache(maxsize=1000)
def num_sp(input_num_bits, input_sparsity, input_seed):
    """
    Generate a random sparse distributed representation.

    Args:
        input_num_bits (int): The number of bits.
        input_sparsity (float): The sparsity value.
        input_seed (int): The random seed.

    Returns:
        list: The sparse representation.
    """
    np.random.seed(input_seed)
    return (np.random.rand(input_num_bits) < input_sparsity).astype(int)

# Example usage
def timing_it():
    num_bits = 100
    sparsity = 0.1
    import time
    start = time.time()
    for _ in range(100):
        sparse_representation = num_sp(num_bits, sparsity, np.random.randint(0, 1000))
    end = time.time()
    print("Time taken: ", end-start)
    print(sparse_representation)

timing_it()