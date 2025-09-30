import os
import argparse
import numpy as np
from tqdm import tqdm
import re
import string
import torch
from reconstruct_raw import reconstruct_raw
from sklearn.datasets import fetch_20newsgroups


def preprocess(doc: str):
    # Lowercase
    doc = doc.lower()

    # Split by some ending puncts
    sentences = re.split(r"(?<=[.?!])\s+", doc)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


def join_sentence(sentences, sentences_per_subdoc=1):
    translator = str.maketrans("", "", string.punctuation)

    # Group every n sentences together
    grouped_sentences = [
        " ".join(sentences[i : i + sentences_per_subdoc])
        for i in range(0, len(sentences), sentences_per_subdoc)
    ]

    return [
        subdoc.translate(translator).split() for subdoc in grouped_sentences if subdoc
    ]


def build_subsentence(docs, vocab, sentences_per_subdoc):
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    docs = [join_sentence(preprocess(doc), sentences_per_subdoc) for doc in docs]

    X = len(docs)
    Q_max = max(len(doc) for doc in docs)  # max subdocs

    bow_tensor = np.zeros((X, Q_max, V), dtype=np.float32)

    for doc_idx, doc in tqdm(enumerate(docs)):
        for subdoc_idx, subdoc in tqdm(enumerate(doc)):
            for token in subdoc:
                if token in w2i:
                    bow_tensor[doc_idx, subdoc_idx, w2i[token]] += 1
    return bow_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="20NG")
    parser.add_argument("-s", "--sentences_per_subdoc", type=int, default=40)
    args = parser.parse_args()

    data_path = f"tm_datasets/{args.dataset_name}"
    output_dir = f"tm_datasets/{args.dataset_name}/sub_sentence"

    # prepare output
    os.makedirs(output_dir, exist_ok=True)

    # An example with the 20NG dataset
    twenty_newsgroup_raw = fetch_20newsgroups(subset="all")
    twenty_newsgroup_raw_texts = twenty_newsgroup_raw.data

    train_texts = open(os.path.join(data_path, "train_texts.txt")).read().splitlines()
    test_texts = open(os.path.join(data_path, "test_texts.txt")).read().splitlines()
    vocab = open(os.path.join(data_path, "vocab.txt")).read().splitlines()

    raw_train_texts, raw_test_texts, _, _ = reconstruct_raw(
        train_texts, test_texts, twenty_newsgroup_raw_texts
    )

    # train
    bow_tr = build_subsentence(raw_train_texts, vocab, args.sentences_per_subdoc)

    np.savez_compressed(os.path.join(output_dir, "train_sub.npz"), data=bow_tr)

    # test
    bow_te = build_subsentence(raw_test_texts, vocab, args.sentences_per_subdoc)

    np.savez_compressed(os.path.join(output_dir, "test_sub.npz"), data=bow_te)
    print(f"Saved .npz files to {output_dir}")


if __name__ == "__main__":
    main()
