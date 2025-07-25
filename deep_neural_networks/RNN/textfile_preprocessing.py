# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 01:02:47 2025

@author: Sai Gunaranjan
"""

import numpy as np

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_vocab(text):
    unique_chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(unique_chars)}
    idx2char = unique_chars
    return char2idx, idx2char

def text_to_indices(text, char2idx):
    return np.array([char2idx[c] for c in text], dtype=np.int32)

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

def prepare_data(file_path, n_segments=64, seq_len=100, split_ratio=0.8):
    text = load_text_file(file_path)
    char2idx, idx2char = create_vocab(text)
    vocab_size = len(idx2char)

    text_indices = text_to_indices(text, char2idx)

    # Split indices for training and validation
    train_indices, val_indices = split_train_val(text_indices, split_ratio)

    # Create labels by shifting by 1 character
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
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": vocab_size,
        "train_data_segments": train_data_segments,
        "train_label_segments": train_label_segments,
        "val_data_segments": val_data_segments,
        "val_label_segments": val_label_segments,
        "n_train_batches": n_train_batches,
        "n_val_batches": n_val_batches,
        "seq_len": seq_len,
        "n_segments": n_segments
    }

if __name__ == "__main__":
    params = prepare_data("harry_potter_small.txt", n_segments=32, seq_len=100)

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
