import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

if "PROJECT_ROOT" in os.environ:
    root_path = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    # fallback: assume this file is somewhere inside src/
    root_path = Path(__file__).resolve().parents[1]

import src.config as config
from utils.rama_grid import  RamaGrid, build_rama_grid
from utils.utils import get_angles_from_4d

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


# --- loss functions (Decoupled) ---

'''def vae_base_loss(recon, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + config.KL_WEIGHT * kl_loss, recon_loss, kl_loss'''

'''def vae_base_loss(recon, x, mu, log_var):
    # 1. Unscale back to raw Degrees using the original data limits
    data_min = torch.tensor(scaler.data_min_, device=recon.device)
    data_range = torch.tensor(scaler.data_range_, device=recon.device)
    
    recon_deg = (recon + 1.0) / 2.0 * data_range + data_min
    x_deg = (x + 1.0) / 2.0 * data_range + data_min
    
    # 2. Convert Degrees to Radians for the Cosine function
    # Note: We use (pi / 180) because 180 degrees is the physical wrap point,
    # regardless of whether your data reaches it or not.
    recon_rad = recon_deg * (torch.pi / 180.0)
    x_rad = x_deg * (torch.pi / 180.0)
    
    # 3. Periodic Distance: (1 - cos(delta_theta))
    # This is 0 when the degrees are identical, and correctly handles the 
    # jump between -180 and +180.
    loss_phi = 1 - torch.cos(recon_rad[:, 0] - x_rad[:, 0])
    loss_psi = 1 - torch.cos(recon_rad[:, 1] - x_rad[:, 1])
    recon_loss = torch.mean(loss_phi + loss_psi)

    # 4. KL Divergence (Unchanged)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + config.KL_WEIGHT * kl_loss, recon_loss, kl_loss'''

def vae_base_loss(recon_4d, target_4d, mu, log_var):
    # Standard MSE now works perfectly because the 4D space is continuous
    recon_loss = nn.functional.mse_loss(recon_4d, target_4d, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + config.KL_WEIGHT * kl_loss, recon_loss, kl_loss

def physics_penalty(recon_4d: torch.Tensor, torch_gmm) -> torch.Tensor:
    
    angles_2d =  get_angles_from_4d(recon_4d)
    log_prob = torch_gmm.log_prob(angles_2d)
    # Using a soft clamp or high ceil to ensure gradients don't zero out completely
    penalty = torch.clamp(-log_prob, min=0.0, max=config.PHYSICS_LOG_PROB_CEIL)
    return penalty.mean()
    
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


# --- VAE architecture ---

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, config.HIDDEN_DIM)
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
        
        # Change output from 2 to 4 to match [sin_phi, cos_phi, sin_psi, cos_psi]
        self.fc3 = nn.Linear(config.HIDDEN_DIM, 4) 
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        # Tanh ensures every output is strictly in the [-1, 1] range,
        return self.tanh(self.fc3(h))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 1. Use Xavier/Glorot for the hidden layers to maintain gradient flow
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    # Overwrite the specific bottleneck layers in the encoder
    def finalize_bottleneck(self):
        # 2. Force mu and log_var to start near zero.
        # This ensures the KL loss starts near zero and the latent space is 
        # centered, giving the physics prior a stable "clean slate."
        for layer in [self.encoder.fc_mu, self.encoder.fc_log_var]:
            nn.init.constant_(layer.weight, 1e-4)
            nn.init.constant_(layer.bias, 0)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
