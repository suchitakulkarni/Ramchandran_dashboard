import os
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import mannwhitneyu
import joblib

import config

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


# --- data loading ---

def load_data(csv_path, scaler_path):
    df = pd.read_csv(csv_path)
    angles = df[["phi", "psi"]].values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    angles_scaled = scaler.fit_transform(angles)
    joblib.dump(scaler, scaler_path)
    return angles_scaled, scaler, angles


# --- GMM fitting ---

def fit_gmm(angles_raw, gmm_path):
    gmm = GaussianMixture(
        n_components=config.N_GMM_COMPONENTS,
        covariance_type="full",
        random_state=config.SEED
    )
    gmm.fit(angles_raw)
    joblib.dump(gmm, gmm_path)
    print(f"GMM fitted with {config.N_GMM_COMPONENTS} components -> {gmm_path}")
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