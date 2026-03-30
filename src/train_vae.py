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
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path: sys.path.insert(0, root_path)

from utils.utils import load_data, fit_gmm
from src.model import build_torch_gmm, VAE, vae_base_loss, physics_penalty
import src.config as config

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# --- training and evaluation ---

def train_epoch(model, loader, optimizer, scaler, torch_gmm, use_physics, current_phys_weight):
    model.train()
    total_loss, total_recon, total_kl, total_phys = 0.0, 0.0, 0.0, 0.0
    for x_batch, in loader:
        optimizer.zero_grad()
        recon, mu, log_var = model(x_batch)
        
        base_loss, recon_loss, kl_loss = vae_base_loss(recon, x_batch, mu, log_var)
        
        if use_physics:
            phys = physics_penalty(recon, scaler, torch_gmm)
            loss = base_loss + current_phys_weight * phys
            total_phys += phys.item()
        else:
            loss = base_loss
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_phys / n


def eval_epoch(model, loader, scaler, torch_gmm, use_physics, current_phys_weight):
    model.eval()
    total_loss, total_recon, total_kl, total_phys = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for x_batch, in loader:
            recon, mu, log_var = model(x_batch)
            base_loss, recon_loss, kl_loss = vae_base_loss(recon, x_batch, mu, log_var)
            
            if use_physics:
                phys = physics_penalty(recon, scaler, torch_gmm)
                loss = base_loss + current_phys_weight * phys
                total_phys += phys.item()
            else:
                loss = base_loss
                
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_phys / n


def run_training(use_physics, angles_scaled, scaler, torch_gmm, save_path, experiment):
    variant = "physics" if use_physics else "baseline"
    print(f"  Training {variant} VAE...")

    n_total = len(angles_scaled)
    n_val = max(1, int(n_total * config.VAL_SPLIT))
    idx = np.random.permutation(n_total)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    tensor_train = torch.tensor(angles_scaled[train_idx], dtype=torch.float32)
    tensor_val = torch.tensor(angles_scaled[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(tensor_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(tensor_val), batch_size=config.BATCH_SIZE, shuffle=False)

    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    history = []


    for epoch in range(1, config.EPOCHS + 1):
        # Calculate dynamic weight for physics loss
        if use_physics:
            # Linearly ramp from 0 to config.PHYSICS_WEIGHT over warmup_epochs
            current_w = config.PHYSICS_WEIGHT * min(1.0, epoch / config.WARMUP_EPOCHS)
        else:
            current_w = 0.0

        tr_loss, tr_recon, tr_kl, tr_phys = train_epoch(
            model, train_loader, optimizer, scaler, torch_gmm, use_physics, current_w
        )
        val_loss, val_recon, val_kl, val_phys = eval_epoch(
            model, val_loader, scaler, torch_gmm, use_physics, current_w
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_recon": tr_recon, "train_kl": tr_kl, "train_phys": tr_phys,
            "val_loss": val_loss, "val_recon": val_recon, "val_kl": val_kl, "val_phys": val_phys,
            "phys_weight": current_w
        })

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch}/{config.EPOCHS} | "
                f"val_loss: {val_loss:.4f} | recon: {val_recon:.4f} | "
                f"phys_raw: {val_phys:.4f} | weight: {current_w:.2f}"
            )

    torch.save(model.state_dict(), save_path)
    loss_path = os.path.join(config.RESULTS_DIR, f"loss_{experiment}_{variant}.csv")
    pd.DataFrame(history).to_csv(loss_path, index=False)
    return model


def main():
    for experiment, (csv_path, scaler_path, gmm_path) in config.EXPERIMENTS.items():
        if not os.path.exists(csv_path):
            continue

        print(f"\n=== Experiment: {experiment} ===")
        angles_scaled, scaler, angles_raw = load_data(csv_path, scaler_path)
        gmm = fit_gmm(angles_raw, gmm_path)
        torch_gmm = build_torch_gmm(gmm)

        run_training(False, angles_scaled, scaler, torch_gmm, config.model_path(experiment, "baseline"), experiment)
        run_training(True, angles_scaled, scaler, torch_gmm, config.model_path(experiment, "physics"), experiment)

    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
