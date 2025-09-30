from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy import sparse


class DatasetHandler:
    """
    Loads sparse .npz doc/train/test and sub-doc files and a vocab.txt,
    converts to dense torch.Tensors, and prepares a DataLoader.
    """

    def __init__(self, data_path, batch_size, subdoc_type="sub_span"):
        self.args = SimpleNamespace(data_path=data_path)
        train_doc_path = f"{data_path}/train_bow.npz"
        test_doc_path = f"{data_path}/test_bow.npz"
        
        if subdoc_type == "sub_span":
            train_sub_path = f"{data_path}/dynamic_subdoc/train_sub.npz"
            test_sub_path = f"{data_path}/dynamic_subdoc/test_sub.npz"
        elif subdoc_type == "sub_sentence":
            train_sub_path = f"{data_path}/sub_sentence/updated_train_subsent_1_4.npz"
            test_sub_path = f"{data_path}/sub_sentence/updated_test_subsent_1_4.npz"
        
        vocab_path = f"{data_path}/vocab.txt"

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [w.strip() for w in f if w.strip()]

        # Load document-level sparse data
        train_sp = sparse.load_npz(train_doc_path)
        test_sp = sparse.load_npz(test_doc_path)

        train_sub_np = np.load(train_sub_path)["data"].astype(np.float32)
        test_sub_np = np.load(test_sub_path)["data"].astype(np.float32)

        # Convert docs to dense numpy → torch.Tensor
        X_train = train_sp.toarray().astype(np.float32)
        X_test = test_sp.toarray().astype(np.float32)

        sub_train = torch.from_numpy(train_sub_np)  # [N_train, S, V]
        sub_test = torch.from_numpy(test_sub_np)  # [N_train, S, V]

        # Store labels if available
        self.y_train = np.loadtxt(f"{data_path}/train_labels.txt", dtype=int)
        self.y_test = np.loadtxt(f"{data_path}/test_labels.txt", dtype=int)

        # Torch tensors for docs
        self.train_data = torch.from_numpy(X_train)
        self.test_data = torch.from_numpy(X_test)

        self.train_dataloader = DataLoader(
            TensorDataset(self.train_data, sub_train),
            batch_size=batch_size,
            shuffle=True,
        )
