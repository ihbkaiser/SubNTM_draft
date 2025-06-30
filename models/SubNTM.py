import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models._ECR import ECR


class SubNTM(nn.Module):
    def __init__(self, args):
        super(SubNTM, self).__init__()
        self.args = args
        self.beta_temp = args.beta_temp
        self.tau = args.tau
        self.num_cluster = args.num_cluster
        self.a = np.ones((1, args.num_topic), dtype=np.float32)
        self.mu2 = nn.Parameter(
            torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), axis=1)).T)
        )
        self.var2 = nn.Parameter(
            torch.as_tensor(
                (
                    ((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T
                    + (1.0 / (args.num_topic * args.num_topic))
                    * np.sum(1.0 / self.a, axis=1)
                ).T
            )
        )
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.local_adapter_mu = nn.Parameter(
            torch.ones_like(self.mu2), requires_grad=False
        )
        self.local_adapter_var = nn.Parameter(
            torch.ones_like(self.var2) * args.adapter_alpha, requires_grad=False
        )

        self.fc11 = nn.Linear(args.vocab_size, args.en1_units)
        self.fc12 = nn.Linear(args.en1_units, args.en1_units)
        self.fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.fc22 = nn.Linear(args.en1_units, args.num_topic)
        self.fc1_dropout = nn.Dropout(args.dropout)
        self.theta_dropout = nn.Dropout(args.dropout)

        self.mean_bn = nn.BatchNorm1d(args.num_topic)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.sub_fc11 = nn.Linear(args.vocab_size, args.en1_units)
        self.sub_fc12 = nn.Linear(args.en1_units, args.en1_units)
        self.sub_fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.sub_fc22 = nn.Linear(args.en1_units, args.num_topic)
        self.sub_fc1_dropout = nn.Dropout(args.dropout)
        self.adapter_alpha2 = args.adapter_alpha**2
        self.sub_mean_bn = nn.BatchNorm1d(args.num_topic)
        self.sub_mean_bn.weight.requires_grad = False
        self.sub_logvar_bn = nn.BatchNorm1d(args.num_topic)
        self.sub_logvar_bn.weight.requires_grad = False
        if args.word_embeddings is not None:
            self.word_embeddings = torch.from_numpy(args.word_embeddings).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(args.vocab_size, args.embed_size)
            )
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
        self.topic_embeddings = torch.empty(
            (args.num_topic, self.word_embeddings.shape[1])
        )
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))
        self.cluster_centers = nn.Parameter(
            torch.randn(args.num_cluster, args.num_topic)
        )
        self.lambda_doc = args.lambda_doc

        self.ECR = ECR(
            args.weight_loss_ECR, args.sinkhorn_alpha, args.sinkhorn_max_iter
        )

    def doc_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def subdoc_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def doc_encode(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.doc_reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        loss_KL = self.compute_loss_doc_KL(mu, logvar)
        return theta, loss_KL, z

    def compute_loss_doc_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * (
            (var_division + diff_term + logvar_division).sum(dim=1)
            - self.args.num_topic
        )
        return KLD.mean()

    def compute_loss_subdoc_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.local_adapter_var
        diff = mu - self.local_adapter_mu
        diff_term = diff * diff / self.local_adapter_var
        logvar_division = self.local_adapter_var.log() - logvar
        KLD = 0.5 * (
            (var_division + diff_term + logvar_division).sum(axis=1)
            - self.args.num_topic
        )
        return KLD.mean()

    def subdoc_encode(self, x):
        e1 = F.softplus(self.sub_fc11(x))
        e1 = F.softplus(self.sub_fc12(e1))
        e1 = self.sub_fc1_dropout(e1)
        mu = self.sub_mean_bn(self.sub_fc21(e1))
        logvar = self.sub_logvar_bn(self.sub_fc22(e1))
        z = self.subdoc_reparameterize(mu, logvar)
        loss_KL = self.compute_loss_subdoc_KL(mu, logvar)
        return z, loss_KL

    def pairwise_euclidean_distance(self, x, y):
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        y_norm = torch.sum(y**2, dim=1)
        return x_norm + y_norm - 2 * torch.matmul(x, y.t())

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings
        )
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def forward(self, x, x_sub=None):
        theta, loss_KL, z = self.doc_encode(x)
        if x_sub is None:
            raise ValueError("x_sub must be provided for sub-document encoding.")
            beta = self.get_beta()
            recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
            recon_loss = -(x * recon.log()).sum(dim=1).mean()
            loss_TM = recon_loss + loss_KL
        else:
            beta = self.get_beta()
            recon_doc = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
            recon_doc_loss = -(x * recon_doc.log()).sum(dim=1).mean()
            B, S, V = x_sub.shape
            flat = x_sub.view(B * S, V)
            z_e, kl_ad = self.subdoc_encode(flat)
            z_e = z_e.view(B, S, -1)
            theta_ds = F.softmax(z_e * theta.unsqueeze(1), dim=-1)
            recon = F.softmax(
                self.decoder_bn((theta_ds @ beta).view(-1, beta.size(1))).view(
                    B, S, -1
                ),
                dim=-1,
            )
            x_sub_augment = x_sub + self.args.augment_coef * x.unsqueeze(1)
            nll = -(x_sub_augment * torch.log(recon + 1e-10)).sum(-1).mean()
            kl_adapter = kl_ad.mean()
            loss_TM = nll + loss_KL + kl_adapter + self.lambda_doc * recon_doc_loss

        cost_matrix = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings
        )
        optimal_transport_loss = self.ECR(cost_matrix)
        loss = loss_TM + optimal_transport_loss
        return {"loss": loss, "loss_TM": loss_TM, "ot_loss": optimal_transport_loss}

    def get_theta(self, x):
        theta, _, _ = self.doc_encode(x)
        return theta

    def get_theta_subdoc(self, x, x_sub):
        """
        Lấy θ của từng sub-document mà không tính toàn bộ loss.
        Trả về tensor shape (B, S, T).
        """
        # Encode document → theta
        theta, _, _ = self.doc_encode(x)  # (B, T)

        # Encode subdoc
        B, S, V = x_sub.shape
        flat = x_sub.view(B * S, V)
        z_e, _ = self.subdoc_encode(flat)  # (B*S, T)
        z_e = z_e.view(B, S, -1)  # (B, S, T)

        # Combine
        theta_ds = F.softmax(z_e * theta.unsqueeze(1), dim=-1)
        return theta_ds

    def get_top_words(self, vocab, num_top_words=15):
        beta = self.get_beta().cpu().detach().numpy()
        top_words = []
        for i, topic_dist in enumerate(beta):
            top_idx = np.argsort(topic_dist)[-num_top_words:][::-1]
            topic_words = np.array(vocab)[top_idx]
            top_words.append(list(topic_words))
        return top_words
