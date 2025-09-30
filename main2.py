import argparse
import os
from dataset_handler.dataset_handler_2 import DatasetHandler
from scipy import sparse
from types import SimpleNamespace
from models.SubNTM import SubNTM
from trainer.trainer import BasicTrainer
from utils.file_utils import update_args

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser(description="LLM evaluation")
parser.add_argument("-d", "--dataset_name", type=str, default="20NG")
parser.add_argument("-n", "--num_topics", type=int, default=50)
parser.add_argument("-c", "--config")
parser.add_argument("-s", "--subdoc_type", type=str, default="sub_span")
args = parser.parse_args()

dataset = args.dataset_name
num_topics = args.num_topics

dataset_path = f"tm_datasets/{dataset}"
out_dir = f"outputs/{dataset}_{num_topics}"

update_args(args, path=args.config)

os.makedirs(out_dir, exist_ok=True)

ds = DatasetHandler(data_path=dataset_path, batch_size=200)
vocab_size = len(ds.vocab)
W_emb = (
    sparse.load_npz(f"{dataset_path}/word_embeddings.npz").toarray().astype("float32")
)
arguments = SimpleNamespace(
    vocab_size=vocab_size,
    en1_units=200,
    dropout=0.0,
    embed_size=200,
    num_topic=num_topics,
    num_cluster=10,  # for DKM loss
    adapter_alpha=args.adapter_alpha,
    beta_temp=0.2,
    tau=1.0,
    weight_loss_ECR=args.weight_loss_ECR,
    sinkhorn_alpha=20.0,
    sinkhorn_max_iter=1000,
    augment_coef=args.augment_coef,
    data_path=dataset_path,
    word_embeddings=W_emb,
    lambda_doc=args.lambda_doc,
)
model = SubNTM(arguments)
trainer = BasicTrainer(
    model,
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    lr_scheduler="StepLR",
    lr_step_size=125,
    log_interval=5,
)
tw, train_t = trainer.fit_transform(ds, num_top_words=15, verbose=True)
trainer.save_beta(out_dir)
trainer.save_theta(ds, out_dir)

print("Training complete. Outputs saved to:", out_dir)
