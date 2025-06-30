import os
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils.print_topic_words import print_topic_words
from evaluations.topic_coherence import TC_on_wikipedia
from evaluations.topic_diversity import compute_topic_diversity
from evaluations.topic_classification import evaluate_classification
from evaluations.topic_clustering import evaluate_clustering
from evaluations.topic_inverted_bias_overlap import evaluate_irbo


class BasicTrainer:
    def __init__(
        self,
        model,
        epochs: int = 200,
        learning_rate: float = 2e-3,
        batch_size: int = 200,
        lr_scheduler: str = None,
        lr_step_size: int = 125,
        log_interval: int = 5,
    ):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.logger = logging.getLogger("main")
        self.logger.setLevel(logging.INFO)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def make_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            return StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5)
        raise NotImplementedError

    def fit_transform(self, ds, num_top_words=15, verbose=False):
        self.train(ds, verbose)
        top_words = self.export_top_words(ds.vocab, num_top_words)
        train_theta = self.test(ds.train_data)
        return top_words, train_theta

    def train(self, ds, verbose=False):
        opt = self.make_optimizer()
        if self.lr_scheduler:
            sch = self.make_lr_scheduler(opt)

        data_size = len(ds.train_dataloader.dataset)
        for epoch in tqdm(range(1, self.epochs + 1), desc="Epochs"):
            self.model.train()
            loss_acc = defaultdict(float)

            for doc_batch, sub_batch in ds.train_dataloader:
                x = doc_batch.to(self.device)
                x_sub = sub_batch.to(self.device)
                rst = self.model(x, x_sub)
                loss = rst["loss"]

                opt.zero_grad()
                loss.backward()
                opt.step()

                bs = x.size(0)
                for k, v in rst.items():
                    loss_acc[k] += v.item() * bs

            if self.lr_scheduler:
                sch.step()
                
            if verbose and epoch % self.log_interval == 0:
                msg = f"Epoch {epoch:03d}"
                for k, total in loss_acc.items():
                    msg += f" | {k}: {total/data_size:.4f}"
                print(msg)
                self.logger.info(msg)
            
            if epoch % 10 == 0:
                train_t, test_t = self.export_theta(ds)
                clus = evaluate_clustering(test_t, ds.y_test)
                self.logger.info(f"Clustering result: {clus}")
                
                tw = self.model.get_top_words(ds.vocab, num_top_words=15)
                metric_data = evaluate_classification(train_t, test_t, ds.y_train, ds.y_test)
                self.logger.info(f"Classification results: {metric_data}")
                
                _, cv = TC_on_wikipedia(tw, cv_type="C_V"); self.logger.info(f"Coherence Cv: {cv:.4f}")
                td = compute_topic_diversity([' '.join(t) for t in tw]); self.logger.info(f"Diversity TD: {td:.4f}")
                irbo = evaluate_irbo(tw, topk=15)
                
                metric_data["Diversity_TD"] = td
                metric_data["IRBO"] = irbo
                metric_data["CV"] = cv
                
                if isinstance(clus, dict):
                    for ck, cvl in clus.items():
                        metric_data[f"clustering/{ck}"] = cvl

    def test(self, data):
        self.model.eval()
        thetas = []
        with torch.no_grad():
            N = data.shape[0]
            for idx in torch.split(torch.arange(N), self.batch_size):
                batch = data[idx].to(self.device)
                theta = self.model.get_theta(batch)
                thetas.extend(theta.cpu().tolist())
        return np.array(thetas)

    def export_beta(self):
        return self.model.get_beta().detach().cpu().numpy()

    def export_top_words(self, vocab, num_top_words):
        beta = self.export_beta()
        return print_topic_words(beta, vocab, num_top_words)

    def export_theta(self, ds):
        test_doc = ds.test_data
        train_doc = ds.train_data
        return self.test(train_doc), self.test(test_doc)

    def save_beta(self, out):
        np.save(os.path.join(out, f"beta.npy"), self.export_beta())

    def save_theta(self, ds, out):
        tr, te = self.export_theta(ds)
        np.save(os.path.join(out, f"train_theta.npy"), tr)
        np.save(os.path.join(out, f"test_theta.npy"), te)
