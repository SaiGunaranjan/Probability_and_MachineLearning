
# -*- coding: utf-8 -*-
"""
Text preprocessing utilities for character- and word-level language models.

Now supports both levels via the `level` argument (defaults to 'char' for backward compatibility).
For word-level tokenization, we use a simple regex-based tokenizer:
    tokens = re.findall(r"\\w+|[^\\w\\s]", text.lower())
This keeps punctuation as separate tokens. You can turn off punctuation retention by passing keep_punct=False.
"""

import re
import numpy as np

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def tokenize_text(text, level='char', lowercase=True, keep_punct=True):
    if level == 'char':
        return list(text)
    # word-level
    if lowercase:
        text = text.lower()
    if keep_punct:
        # words or single non-space punctuation characters
        tokens = re.findall(r"\w+|[^\w\s]", text)
    else:
        tokens = re.findall(r"\w+", text)
    return tokens

def create_vocab(items):
    """
    items: iterable of tokens (chars or words).
    Returns token2idx (dict) and idx2token (list).
    """
    unique = sorted(set(items))
    token2idx = {t: i for i, t in enumerate(unique)}
    idx2token = unique
    return token2idx, idx2token

def items_to_indices(items, token2idx):
    return np.array([token2idx[t] for t in items], dtype=np.int32)

def split_train_val(indices, split_ratio=0.8):
    split_idx = int(len(indices) * split_ratio)
    return indices[:split_idx], indices[split_idx:]

def partition_segments(indices, n_segments):
    segment_len = len(indices) // n_segments
    trimmed_len = segment_len * n_segments
    indices_trimmed = indices[:trimmed_len]
    segments = indices_trimmed.reshape(n_segments, segment_len)
    return segments

def one_hot_encode(indices_slice, vocab_size):
    batch_size, seq_len = indices_slice.shape
    one_hot = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
    rows = np.arange(batch_size)[:, None]
    cols = np.arange(seq_len)[None, :]
    one_hot[rows, cols, indices_slice] = 1
    return one_hot

def get_batch(data_segments, label_segments, batch_step, seq_len, vocab_size):
    batch_start = batch_step * seq_len
    batch_end = batch_start + seq_len
    data_batch_slice = data_segments[:, batch_start:batch_end]
    label_batch_slice = label_segments[:, batch_start:batch_end]
    input_batch = one_hot_encode(data_batch_slice, vocab_size)
    label_batch = one_hot_encode(label_batch_slice, vocab_size)
    return input_batch, label_batch

def prepare_data(file_path, n_segments=64, seq_len=100, split_ratio=0.8,
                 level='char', lowercase=True, keep_punct=True):
    """
    Prepare dataset for LM training.

    Arguments
    ---------
    file_path : str
    n_segments : int
        Number of parallel segments (matched to batch size).
    seq_len : int
        Unroll length.
    split_ratio : float
        Train/val split.
    level : {'char','word'}
    lowercase : bool
        Applies only for level='word'.
    keep_punct : bool
        If False, punctuation is dropped at word level.

    Returns (dict)
    --------------
    {
        "token2idx": dict,
        "idx2token": list,
        # Backward-compat fields for existing code:
        "char2idx": dict (alias of token2idx),
        "idx2char": list (alias of idx2token),
        "vocab_size": int,
        "train_data_segments": np.ndarray [n_segments, T_train],
        "train_label_segments": np.ndarray [n_segments, T_train],
        "val_data_segments": np.ndarray [n_segments, T_val],
        "val_label_segments": np.ndarray [n_segments, T_val],
        "n_train_batches": int,
        "n_val_batches": int,
        "seq_len": int,
        "level": str
    }
    """
    text = load_text_file(file_path)
    items = tokenize_text(text, level=level, lowercase=lowercase, keep_punct=keep_punct)
    token2idx, idx2token = create_vocab(items)
    vocab_size = len(idx2token)

    indices = items_to_indices(items, token2idx)

    # Split indices for training and validation
    train_indices, val_indices = split_train_val(indices, split_ratio)

    # Create labels by shifting by 1 token
    train_labels_indices = train_indices[1:]
    train_indices = train_indices[:-1]

    val_labels_indices = val_indices[1:]
    val_indices = val_indices[:-1]

    # Partition train and val into segments
    train_data_segments = partition_segments(train_indices, n_segments)
    train_label_segments = partition_segments(train_labels_indices, n_segments)

    val_data_segments = partition_segments(val_indices, n_segments)
    val_label_segments = partition_segments(val_labels_indices, n_segments)

    # Compute number of batches per epoch for train and val
    n_train_batches = train_data_segments.shape[1] // seq_len
    n_val_batches = val_data_segments.shape[1] // seq_len

    return {
        "token2idx": token2idx,
        "idx2token": idx2token,
        # Back-compat for existing LSTM class fields:
        "char2idx": token2idx,
        "idx2char": idx2token,
        "vocab_size": vocab_size,
        "train_data_segments": train_data_segments,
        "train_label_segments": train_label_segments,
        "val_data_segments": val_data_segments,
        "val_label_segments": val_label_segments,
        "n_train_batches": n_train_batches,
        "n_val_batches": n_val_batches,
        "seq_len": seq_len,
        "level": level,
    }


if __name__ == "__main__":
    params = prepare_data("harry_potter_small.txt", n_segments=32, seq_len=30, level='word') # seq_len=100, level='char'

    print(f"Vocabulary size: {params['vocab_size']}")
    print(f"Train batches per epoch: {params['n_train_batches']}")
    print(f"Validation batches per epoch: {params['n_val_batches']}")

    # Loop over all training batches
    print("Iterating over training batches:")
    for batch_step in range(params["n_train_batches"]):
        train_input_batch, train_label_batch = get_batch(
            params["train_data_segments"],
            params["train_label_segments"],
            batch_step,
            params["seq_len"],
            params["vocab_size"],
        )
        print(f"Train batch {batch_step+1}/{params['n_train_batches']} shapes: inputs={train_input_batch.shape}, labels={train_label_batch.shape}")
        # Insert training code here or break early in testing
        # break  # uncomment to test first batch only

    # Loop over all validation batches
    print("\nIterating over validation batches:")
    for batch_step in range(params["n_val_batches"]):
        val_input_batch, val_label_batch = get_batch(
            params["val_data_segments"],
            params["val_label_segments"],
            batch_step,
            params["seq_len"],
            params["vocab_size"],
        )
        print(f"Validation batch {batch_step+1}/{params['n_val_batches']} shapes: inputs={val_input_batch.shape}, labels={val_label_batch.shape}")
        # Insert validation code here or break early in testing
        # break  # uncomment to test first batch only