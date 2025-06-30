#!/usr/bin/env python3
import os
import argparse
import numpy as np
from gensim.utils import simple_preprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F


def get_optimal_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, dim)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def generate_sliding_windows(tokens, window_size, stride):
    segments = []
    L = len(tokens)
    for start in range(0, L, stride):
        end = start + window_size
        if start != 0 and end > L:
            start, end = max(0, L - window_size), L
        segments.append(tokens[start:end])
        if end >= L:
            break
    return segments


def build_fixed_subdocs(
    texts,  # List[str]
    vocab,  # List[str]
    window_size,
    stride,
    batch_size,
):
    # prepare vocab→idx
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}

    seg_texts = []
    bows = []
    segments_per_doc = []

    # 1) tokenize & segment + build raw bows
    for doc in tqdm(texts, desc="Segmenting docs"):
        tokens = simple_preprocess(doc)
        segs = generate_sliding_windows(tokens, window_size, stride)
        segments_per_doc.append(len(segs))
        for seg in segs:
            seg_texts.append(" ".join(seg))
            vec = np.zeros(V, dtype=np.float32)
            for w in seg:
                idx = w2i.get(w)
                if idx is not None:
                    vec[idx] += 1.0
            bows.append(vec)
    bows = np.stack(bows, axis=0)  # (M, V)

    # 2) pad to fixed Q_max
    Q_max = max(segments_per_doc)
    N = len(texts)
    bow_tensor = np.zeros((N, Q_max, V), dtype=np.float32)

    idx = 0
    for i, q in enumerate(segments_per_doc):
        bow_tensor[i, :q, :] = bows[idx : idx + q]
        idx += q

    return bow_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="20NG")
    parser.add_argument("-w", "--window_size", type=int, default=40)
    parser.add_argument("-s", "--stride", type=int, default=30)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    args = parser.parse_args()

    data_path = f"tm_datasets/{args.dataset_name}"
    output_dir = f"tm_datasets/{args.datase_namet}/dynamic_subdoc"

    # prepare output
    os.makedirs(output_dir, exist_ok=True)

    train_texts = open(os.path.join(data_path, "train_texts.txt")).read().splitlines()
    test_texts = open(os.path.join(data_path, "test_texts.txt")).read().splitlines()
    vocab = open(os.path.join(data_path, "vocab.txt")).read().splitlines()

    # train
    bow_tr = build_fixed_subdocs(
        train_texts,
        vocab,
        args.window_size,
        args.stride,
        args.batch_size,
    )
    np.savez_compressed(os.path.join(output_dir, "train_sub.npz"), data=bow_tr)

    # test
    bow_te = build_fixed_subdocs(
        test_texts,
        vocab,
        args.window_size,
        args.stride,
        args.batch_size,
    )
    np.savez_compressed(os.path.join(output_dir, "test_sub.npz"), data=bow_te)
    print(f"Saved .npz files to {output_dir}")


if __name__ == "__main__":
    main()
