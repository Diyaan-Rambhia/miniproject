"""
Phase: Preprocessing and Encoding
Expects: List of domain strings (from data_loader.py) and max sequence length
Outputs: Vocabulary mapping dict and encoded domain integer array (X)
"""

import numpy as np


def build_vocab(domains: list) -> dict:
    """
    Builds a char -> index mapping from all characters seen across
    the domain list. Index 0 reserved for padding, 1 for unknown.
    """
    chars = sorted(set("".join(domains)))
    vocab = {ch: idx + 2 for idx, ch in enumerate(chars)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    print(f"Vocabulary size: {len(vocab)} (includes PAD/UNK)")
    return vocab


def encode_domain(domain: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    Converts a domain string into a fixed-length array of character
    indices, truncating or padding as needed.
    """
    indices = [vocab.get(ch, vocab["<UNK>"]) for ch in domain[:max_len]]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


def encode_all_domains(domains: list, vocab: dict, max_len: int) -> np.ndarray:
    return np.stack([encode_domain(d, vocab, max_len) for d in domains])


build_char_vocab = build_vocab
encode_domains = encode_all_domains
