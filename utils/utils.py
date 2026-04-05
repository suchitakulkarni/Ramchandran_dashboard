import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import mannwhitneyu
import joblib

import src.config as config
logger = logging.getLogger(__name__)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def setup_logging(level=logging.INFO):
    """
    Call once from main.py. All modules use logging.getLogger(__name__)
    and inherit this configuration automatically.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.RESULTS_DIR, "run.log"), mode="w")
        ]
    )

# --- data loading ---
def load_data(csv_path):
    logger.info(f"loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    for colname in ["phi", "psi", "sphi","cphi", "spsi", "cpsi"]:
        if colname not in df.columns: logging.error(f"column {colname} does not exist in the datafile")
    angles = df[["phi", "psi"]].values.astype(np.float32)
    angles_4D = df[["sphi","cphi", "spsi", "cpsi"]].values.astype(np.float32)
    return angles, angles_4D

def to_4d_torch(angles_deg):
    rad = angles_deg * (torch.pi / 180.0)
    # Stack along the feature dimension (dim=1)
    return torch.stack([
        torch.sin(rad[:, 0]), torch.cos(rad[:, 0]), # Phi
        torch.sin(rad[:, 1]), torch.cos(rad[:, 1])  # Psi
    ], dim=1)
    
def get_angles_from_4d(recon_4d):
    # recon_4d shape: [batch, 4] -> [sin_phi, cos_phi, sin_psi, cos_psi]
    phi_rad = torch.atan2(recon_4d[:, 0], recon_4d[:, 1])
    psi_rad = torch.atan2(recon_4d[:, 2], recon_4d[:, 3])
    
    # Convert to degrees for your RamaGrid
    return torch.stack([phi_rad, psi_rad], dim=1) * (180.0 / torch.pi)


# --- GMM fitting ---

def fit_gmm(angles_raw, gmm_path):
    logger.info(f"fitting gmm now using compnents {config.N_GMM_COMPONENTS} with covariance type full and seed {config.SEED}")
    gmm = GaussianMixture(
        n_components=config.N_GMM_COMPONENTS,
        covariance_type="full",
        random_state=config.SEED
    )
    gmm.fit(angles_raw)
    joblib.dump(gmm, gmm_path)
    logger.info(f"GMM fitted with {config.N_GMM_COMPONENTS} components -> {gmm_path}")
    return gmm

# --- Cohen's d ---
def cohen_d(group1, group2):
    n1, n2   = len(group1), len(group2)
    var1     = np.var(group1, ddof=1)
    var2     = np.var(group2, ddof=1)
    pooled   = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled)


def effect_size_label(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# --- per-sigma entropy from raw samples ---

def compute_entropy(phi_arr, psi_arr):
    phi_std = phi_arr.std()
    psi_std = psi_arr.std()
    return 0.5 * np.log(2 * np.pi * np.e * (phi_std * psi_std + 1e-8))
