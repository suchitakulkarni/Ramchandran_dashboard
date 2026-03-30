import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import joblib

import src.config as config
from utils.rama_grid import  RamaGrid, build_rama_grid


torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


# --- loss functions (Decoupled) ---

def vae_base_loss(recon, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + config.KL_WEIGHT * kl_loss, recon_loss, kl_loss
    
# --- differentiable GMM log-prob ---

class TorchGMM:
    def __init__(self, gmm: GaussianMixture):
        self.n_components = gmm.n_components
        self.D = gmm.means_.shape[1]
        self.log_weights = torch.tensor(np.log(gmm.weights_ + 1e-12), dtype=torch.float32)
        self.means = torch.tensor(gmm.means_, dtype=torch.float32)

        covs = gmm.covariances_.astype(np.float64)
        precisions = np.linalg.inv(covs)
        _, logdet = np.linalg.slogdet(covs)

        self.precisions = torch.tensor(precisions, dtype=torch.float32)
        self.log_det = torch.tensor(logdet, dtype=torch.float32)
        self.log_2pi = float(self.D * np.log(2.0 * np.pi))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)
        prec_diff = torch.einsum("bkd,kde->bke", diff, self.precisions)
        mahal = torch.einsum("bkd,bkd->bk", prec_diff, diff)
        log_gauss = -0.5 * (self.log_2pi + self.log_det.unsqueeze(0) + mahal)
        log_components = log_gauss + self.log_weights.unsqueeze(0)
        return torch.logsumexp(log_components, dim=1)


def build_torch_gmm(gmm: GaussianMixture) -> TorchGMM:
    return TorchGMM(gmm)


def physics_penalty(decoded_scaled: torch.Tensor, scaler: MinMaxScaler, torch_gmm: TorchGMM) -> torch.Tensor:
    data_min = torch.tensor(scaler.data_min_, dtype=torch.float32)
    data_range = torch.tensor(scaler.data_range_, dtype=torch.float32)
    decoded_raw = (decoded_scaled + 1.0) / 2.0 * data_range + data_min
    log_prob = torch_gmm.log_prob(decoded_raw)
    # Using a soft clamp or high ceil to ensure gradients don't zero out completely
    penalty = torch.clamp(-log_prob, min=0.0, max=config.PHYSICS_LOG_PROB_CEIL)
    return penalty.mean()


# --- VAE architecture ---

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, config.HIDDEN_DIM)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2)
        self.fc_mu = nn.Linear(config.HIDDEN_DIM // 2, config.LATENT_DIM)
        self.fc_log_var = nn.Linear(config.HIDDEN_DIM // 2, config.LATENT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.LATENT_DIM, config.HIDDEN_DIM // 2)
        self.fc2 = nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM)
        self.fc3 = nn.Linear(config.HIDDEN_DIM, 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.tanh(self.fc3(h))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
