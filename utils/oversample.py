import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import config

os.makedirs(config.DATA_DIR, exist_ok=True)

PI = 3.141592653589793


# --- density estimation ---

def estimate_density(phi, psi, grid_size=100):
    xy = np.vstack([phi, psi])
    kde = gaussian_kde(xy)
    grid_phi = np.linspace(-180, 180, grid_size)
    grid_psi = np.linspace(-180, 180, grid_size)
    gp, gs = np.meshgrid(grid_phi, grid_psi)
    positions = np.vstack([gp.ravel(), gs.ravel()])
    density_grid = kde(positions).reshape(gp.shape)
    return kde, density_grid, gp, gs


def compute_cv(kde, phi, psi, min_density_percentile=config.KDE_MIN_DENSITY_PERCENTILE):
    point_densities = kde(np.vstack([phi, psi]))
    threshold = np.percentile(point_densities[point_densities > 0], min_density_percentile)
    populated = point_densities[point_densities >= threshold]
    if len(populated) == 0:
        return np.inf
    cv = populated.std() / (populated.mean() + 1e-10)
    return cv


# --- sampling weights ---

def compute_sampling_weights(kde, phi, psi):
    densities = kde(np.vstack([phi, psi]))
    densities = np.clip(densities, 1e-10, None)
    weights = 1.0 / densities
    weights = weights / weights.sum()
    return weights


# --- periodic jitter ---

def apply_periodic_jitter(phi, psi, jitter_std=config.JITTER_STD, n_samples=1):
    phi_jittered = phi + np.random.normal(0, jitter_std, n_samples)
    psi_jittered = psi + np.random.normal(0, jitter_std, n_samples)
    # wrap to [-180, 180] to respect periodicity
    phi_jittered = ((phi_jittered + 180) % 360) - 180
    psi_jittered = ((psi_jittered + 180) % 360) - 180
    return phi_jittered, psi_jittered


# --- main oversampling loop ---

def oversample(df, label="dataset"):
    phi = df["phi"].values.astype(np.float64)
    psi = df["psi"].values.astype(np.float64)

    print(f"\n  [{label}] Starting with {len(phi)} residues")

    kde, _, _, _ = estimate_density(phi, psi)
    cv = compute_cv(kde, phi, psi)
    print(f"  [{label}] Initial CV: {cv:.4f}  (target < {config.CV_THRESHOLD})")

    if cv <= config.CV_THRESHOLD:
        print(f"  [{label}] Already balanced, no oversampling needed")
        return df.copy()

    synthetic_records = []
    iteration = 0

    while cv > config.CV_THRESHOLD and iteration < config.MAX_OVERSAMPLE_ITER:
        iteration += 1

        weights = compute_sampling_weights(kde, phi, psi)
        n_to_add = max(10, len(phi) // 10)
        chosen_indices = np.random.choice(len(phi), size=n_to_add, replace=True, p=weights)

        new_phi_list = []
        new_psi_list = []
        for idx in chosen_indices:
            new_phi, new_psi = apply_periodic_jitter(phi[idx], psi[idx], n_samples=1)
            new_phi_list.append(new_phi[0])
            new_psi_list.append(new_psi[0])

        new_phi_arr = np.array(new_phi_list)
        new_psi_arr = np.array(new_psi_list)

        phi = np.concatenate([phi, new_phi_arr])
        psi = np.concatenate([psi, new_psi_arr])

        for p, q in zip(new_phi_arr, new_psi_arr):
            synthetic_records.append({
                "pdb_id": "synthetic",
                "phi": round(float(p), 4),
                "psi": round(float(q), 4),
                "source": "synthetic"
            })

        kde, _, _, _ = estimate_density(phi, psi)
        cv = compute_cv(kde, phi, psi)

        print(f"  [{label}] Iteration {iteration:02d} | "
              f"total points: {len(phi):5d} | CV: {cv:.4f}")

    if cv <= config.CV_THRESHOLD:
        print(f"  [{label}] Converged at iteration {iteration} with CV={cv:.4f}")
    else:
        print(f"  [{label}] Reached max iterations ({config.MAX_OVERSAMPLE_ITER}) "
              f"with CV={cv:.4f}")

    original_df = df.copy()
    original_df["source"] = "original"
    synthetic_df = pd.DataFrame(synthetic_records)

    balanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f"  [{label}] Final dataset: {len(balanced_df)} residues "
          f"({len(synthetic_records)} synthetic, "
          f"{len(original_df)} original)")
    return balanced_df


# --- main ---

def main():
    pairs = [
        (config.CSV_ORIGINAL,  config.CSV_ORIGINAL_BALANCED,  "original"),
        (config.CSV_AUGMENTED, config.CSV_AUGMENTED_BALANCED, "augmented"),
    ]

    for csv_in, csv_out, label in pairs:
        if not os.path.exists(csv_in):
            print(f"Input CSV not found: {csv_in} -- skipping {label}")
            continue

        print(f"\n=== Oversampling: {label} ===")
        df = pd.read_csv(csv_in)

        if "source" not in df.columns:
            df["source"] = "original"

        balanced_df = oversample(df, label=label)
        balanced_df.to_csv(csv_out, index=False)
        print(f"  Saved -> {csv_out}")

        print(f"\n  Source breakdown for {label}:")
        print(balanced_df["source"].value_counts().to_string())

    print("\nOversampling complete.")


if __name__ == "__main__":
    main()
