"""
visualise.py

Diagnostic plots for the Ramachandran VAE project.

Stage 0 : learning curves (train vs val, all loss components)
Stage 1 : training data overview
Stage 2 : GMM fit quality
Stage 3 : reconstruction quality
Stage 4 : generated samples from prior
Stage 5 : perturbation scatter + entropy summary  (reads perturbation_samples.csv)
Stage 6 : entropy scaling with CI + compliance rate  (reads perturbation_compliance.csv)

All stages loop over config.EXPERIMENTS.
Output files are namespaced as  results/stage{N}_{experiment}_{description}.png
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, mannwhitneyu

from pathlib import Path

if "PROJECT_ROOT" in os.environ:
    root_path = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    # fallback: assume this file is somewhere inside src/
    root_path = Path(__file__).resolve().parents[1]
    
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import src.config as config
from src.train_vae import VAE
from utils.ramachandran_regions import overlay_regions, compliance_lovell
from utils.utils import compute_entropy, effect_size_label, cohen_d, to_4d_torch, get_angles_from_4d

os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(config.RESULTS_DIR,"plots"), exist_ok=True)

# 3. Add the root to sys.path if it's not already there

COLORS = {
    "baseline": "#2196F3",
    "physics":  "#E91E63",
    "data":     "#333333",
    "gmm":      "#FF9800",
}


# --- shared helpers ---

def ramachandran_background(ax):
    ax.axhline(0, color="lightgrey", linewidth=2)
    ax.axvline(0, color="lightgrey", linewidth=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel(r"$\phi$ (degrees)")
    ax.set_ylabel(r"$\psi$ (degrees)")
    ax.set_aspect("equal")


def savefig(fig, filename):
    path = os.path.join(config.RESULTS_DIR, "plots", filename)
    fig.savefig(path, bbox_inches="tight", dpi = 600)
    plt.close(fig)
    print(f"Saved {path}")


def load_experiment_artifacts(experiment):
    _, gmm_path = config.EXPERIMENTS[experiment]
    gmm    = joblib.load(gmm_path)

    model_baseline = VAE()
    model_baseline.load_state_dict(
        torch.load(config.model_path(experiment, "baseline"), weights_only=True)
    )
    model_baseline.eval()

    model_physics = VAE()
    model_physics.load_state_dict(
        torch.load(config.model_path(experiment, "physics"), weights_only=True)
    )
    model_physics.eval()

    return gmm, model_baseline, model_physics


def load_experiment_data(experiment):
    csv_path, _ = config.EXPERIMENTS[experiment]
    return pd.read_csv(csv_path)


# =============================================================================
# Stage 0 : learning curves
# =============================================================================

def plot_learning_curves(experiment):
    """
    Reads loss_{experiment}_{variant}.csv saved during training.
    One figure per experiment with 2 rows (baseline / physics) x 3 cols
    (total loss | recon loss | kl loss), train and val overlaid.
    Physics variant adds a 4th column for the physics penalty.
    """
    variants = ["baseline", "physics"]
    n_cols   = 4  # total | recon | kl | phys  (phys col empty for baseline)

    fig, axes = plt.subplots(2, n_cols, figsize=(18, 8))
    #fig.suptitle(
    #    f"Stage 0: Learning Curves -- {experiment}", fontsize=13, fontweight="bold"
    #)

    for row, variant in enumerate(variants):
        loss_path = os.path.join(
            config.RESULTS_DIR, f"datafiles/loss_{experiment}_{variant}.csv"
        )
        if not os.path.exists(loss_path):
            for ax in axes[row]:
                ax.set_visible(False)
            print(f"  Loss file not found: {loss_path} -- skipping")
            continue

        df = pd.read_csv(loss_path)
        epochs = df["epoch"].values
        color  = COLORS[variant]

        component_cols = [
            ("train_loss",  "val_loss",  "total loss"),
            ("train_recon", "val_recon", "recon loss"),
            ("train_kl",    "val_kl",    "KL loss"),
        ]

        for col, (tr_col, val_col, title) in enumerate(component_cols):
            ax = axes[row, col]
            ax.plot(epochs, df[tr_col],  color=color,   alpha=0.9, linewidth=1.5,
                    label="train")
            ax.plot(epochs, df[val_col], color=color,   alpha=0.5, linewidth=1.5,
                    linestyle="--", label="val")
            ax.set_title(f"{variant} | {title}", fontsize=9)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # physics penalty (only for physics variant)
        ax_phys = axes[row, 3]
        if variant == "physics" and "train_phys" in df.columns:
            ax_phys.plot(epochs, df["train_phys"], color=color,  alpha=0.9,
                         linewidth=1.5, label="train")
            ax_phys.plot(epochs, df["val_phys"],   color=color,  alpha=0.5,
                         linewidth=1.5, linestyle="--", label="val")
            ax_phys.set_title("physics | physics penalty", fontsize=9)
            ax_phys.set_xlabel("epoch")
            ax_phys.set_ylabel("penalty")
            ax_phys.legend(fontsize=7)
            ax_phys.grid(True, alpha=0.3)
        else:
            ax_phys.set_visible(False)

    plt.tight_layout()
    savefig(fig, f"stage0_{experiment}_learning_curves.png")


# =============================================================================
# Stage 1 : training data
# =============================================================================

def plot_training_data(experiment, df):
    fig = plt.figure(figsize=(16, 12))
    #fig.suptitle(
    #    f"Stage 1: Training Data -- {experiment}", fontsize=13, fontweight="bold"
    #)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    pdb_ids   = df["pdb_id"].unique() if "pdb_id" in df.columns else ["all"]
    cmap      = plt.cm.get_cmap("tab10", len(pdb_ids))
    color_map = {pid: cmap(i) for i, pid in enumerate(pdb_ids)}

    phi = df["phi"].values
    psi = df["psi"].values

    # Ramachandran scatter
    ax1 = fig.add_subplot(gs[0, 0])
    if "pdb_id" in df.columns:
        for pid in pdb_ids:
            sub = df[df["pdb_id"] == pid]
            ax1.scatter(sub["phi"], sub["psi"], s=8, alpha=0.6,
                        color=color_map[pid], label=pid)
        ax1.legend(fontsize=7, markerscale=2)
    else:
        ax1.scatter(phi, psi, s=8, alpha=0.6, color=COLORS["data"])
    ramachandran_background(ax1)
    overlay_regions(ax1, show_legend=True)
    ax1.set_title("Ramachandran plot")

    # KDE density
    ax2 = fig.add_subplot(gs[0, 1])
    xy       = np.vstack([phi, psi])
    kde      = gaussian_kde(xy)
    grid_phi = np.linspace(-180, 180, 150)
    grid_psi = np.linspace(-180, 180, 150)
    gp, gps  = np.meshgrid(grid_phi, grid_psi)
    density  = kde(np.vstack([gp.ravel(), gps.ravel()])).reshape(gp.shape)
    ax2.contourf(gp, gps, density, levels=20, cmap="Blues")
    ax2.scatter(phi, psi, s=4, alpha=0.3, color="black")
    ramachandran_background(ax2)
    ax2.set_title("Joint KDE density")

    # phi marginal
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(phi, bins=36, color=COLORS["data"], edgecolor="white", linewidth=0.5)
    ax3.set_xlabel("phi (degrees)")
    ax3.set_ylabel("count")
    ax3.set_title("phi marginal")
    ax3.set_xlim(-180, 180)

    # psi marginal
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(psi, bins=36, color=COLORS["data"], edgecolor="white", linewidth=0.5)
    ax4.set_xlabel("psi (degrees)")
    ax4.set_ylabel("count")
    ax4.set_title("psi marginal")
    ax4.set_xlim(-180, 180)

    # phi vs psi scatter with correlation
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(phi, psi, s=6, alpha=0.4, color=COLORS["data"])
    corr = np.corrcoef(phi, psi)[0, 1]
    ax5.set_title(f"phi vs psi  (r = {corr:.3f})")
    ax5.set_xlabel("phi (degrees)")
    ax5.set_ylabel("psi (degrees)")
    ramachandran_background(ax5)
    overlay_regions(ax5, show_legend=False)

    # residues per PDB
    ax6 = fig.add_subplot(gs[1, 2])
    if "pdb_id" in df.columns:
        counts = df["pdb_id"].value_counts()
        ax6.bar(counts.index, counts.values,
                color=[color_map[p] for p in counts.index])
        ax6.set_xlabel("PDB ID")
    else:
        ax6.bar(["all"], [len(df)], color=COLORS["data"])
    ax6.set_ylabel("Residue count")
    ax6.set_title("Residues per structure")

    plt.tight_layout()
    savefig(fig, f"stage1_{experiment}_training_data.png")

def plot_training_data_streamlit(experiment, df):
    fig = plt.figure(figsize=(8, 4))
    #gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.4)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    pdb_ids   = df["pdb_id"].unique() if "pdb_id" in df.columns else ["all"]
    cmap      = plt.cm.get_cmap("tab10", len(pdb_ids))
    color_map = {pid: cmap(i) for i, pid in enumerate(pdb_ids)}

    phi = df["phi"].values
    psi = df["psi"].values

    # Ramachandran scatter
    ax1 = fig.add_subplot(gs[0, 0])
    if "pdb_id" in df.columns:
        for pid in pdb_ids:
            sub = df[df["pdb_id"] == pid]
            ax1.scatter(sub["phi"], sub["psi"], s=5, alpha=0.6,
                        color=color_map[pid], label=pid)
        ax1.legend(fontsize=10, markerscale=3)
    else:
        ax1.scatter(phi, psi, s=5, alpha=0.6, color=COLORS["data"])
    ramachandran_background(ax1)
    overlay_regions(ax1, show_legend=True)
    ax1.set_title("Ramachandran plot")

    # residues per PDB
    ax6 = fig.add_subplot(gs[0, 1])
    if "pdb_id" in df.columns:
        counts = df["pdb_id"].value_counts()
        ax6.bar(counts.index, counts.values,
                color=[color_map[p] for p in counts.index])
        ax6.set_xlabel("Protein ID")
    else:
        ax6.bar(["all"], [len(df)], color=COLORS["data"])
    ax6.set_ylabel("Frequency")
    ax6.set_title("Protein types in the training dataset")

    plt.tight_layout()
    savefig(fig, f"stage1_{experiment}_training_data_st.png")


# =============================================================================
# Stage 2 : GMM fit quality
# =============================================================================

def plot_gmm_quality(experiment, df, gmm):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"GMM Fit Quality -- {experiment}", fontsize=13, fontweight="bold"
    )

    phi = df["phi"].values
    psi = df["psi"].values

    grid_phi    = np.linspace(-180, 180, 200)
    grid_psi    = np.linspace(-180, 180, 200)
    gp, gps     = np.meshgrid(grid_phi, grid_psi)
    grid_points = np.vstack([gp.ravel(), gps.ravel()]).T
    log_prob    = gmm.score_samples(grid_points)
    density     = np.exp(log_prob).reshape(gp.shape)

    for ax, show_scatter in zip(axes, [True, False]):
        cf = ax.contourf(gp, gps, density, levels=25, cmap="Oranges", alpha=0.8)
        ax.contour(gp, gps, density, levels=10, colors="darkorange",
                   linewidths=0.5, alpha=0.6)
        if show_scatter:
            ax.scatter(phi, psi, s=6, alpha=0.4, color="black", label="training data")
            ax.legend(fontsize=8)
            ax.set_title("GMM density + training data")
        else:
            ax.set_title("GMM density only")
        ramachandran_background(ax)
        overlay_regions(ax, show_legend=False)
        fig.colorbar(cf, ax=ax, label="probability density")

    savefig(fig, f"stage2_{experiment}_gmm_quality.png")


# =============================================================================
# Stage 3 : reconstruction quality
# =============================================================================

def plot_reconstruction(experiment, df, model_baseline, model_physics):
    angles_raw  = df[["phi", "psi"]].values.astype(np.float32)
    angles_tensor = torch.from_numpy(angles_raw)
    angles_4d = to_4d_torch(angles_tensor)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Reconstruction Quality -- {experiment}", fontsize=13, fontweight="bold"
    )

    for row, (model, label, color) in enumerate([
        (model_baseline, "Baseline VAE", COLORS["baseline"]),
        (model_physics,  "Physics VAE",  COLORS["physics"]),
    ]):
        with torch.no_grad():
            recon, _, _ = model(angles_4d)
        recon_raw = get_angles_from_4d(recon).numpy()

        phi_orig  = angles_raw[:, 0]
        psi_orig  = angles_raw[:, 1]
        phi_recon = recon_raw[:, 0]
        psi_recon = recon_raw[:, 1]

        ax = axes[row, 0]
        ax.scatter(phi_orig,  psi_orig,  s=6, alpha=0.4, color="grey",  label="original")
        ax.scatter(phi_recon, psi_recon, s=6, alpha=0.4, color=color,   label="reconstructed")
        ramachandran_background(ax)
        overlay_regions(ax, show_legend=False)
        lov_fav, lov_allow = compliance_lovell(phi_recon, psi_recon)
        ax.set_title(
            f"{label}: original vs reconstructed\n"
            f"Lovell favoured={lov_fav:.2f}  allowed={lov_allow:.2f}"
        )
        ax.legend(fontsize=7)

        phi_res = phi_recon - phi_orig
        ax = axes[row, 1]
        ax.hist(phi_res, bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("phi residual (degrees)")
        ax.set_ylabel("count")
        ax.set_title(f"{label}: phi residuals  (std={phi_res.std():.2f})")

        psi_res = psi_recon - psi_orig
        ax = axes[row, 2]
        ax.hist(psi_res, bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("psi residual (degrees)")
        ax.set_ylabel("count")
        ax.set_title(f"{label}: psi residuals  (std={psi_res.std():.2f})")

    plt.tight_layout()
    savefig(fig, f"stage3_{experiment}_reconstruction.png")


# =============================================================================
# Stage 4 : generated samples
# =============================================================================

def plot_generated_samples(experiment, df, model_baseline, model_physics,
                           n_samples=500):
    phi_orig = df["phi"].values
    psi_orig = df["psi"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Generated Samples from Prior -- {experiment}",
        fontsize=13, fontweight="bold"
    )

    axes[0].scatter(phi_orig, psi_orig, s=6, alpha=0.5, color="grey")
    ramachandran_background(axes[0])
    overlay_regions(axes[0], show_legend=True)
    fav_data, allow_data = compliance_lovell(phi_orig, psi_orig)
    axes[0].set_title(
        f"Training data\nLovell favoured={fav_data:.2f}  allowed={allow_data:.2f}"
    )

    for ax, model, label, color in [
        (axes[1], model_baseline, "Baseline VAE", COLORS["baseline"]),
        (axes[2], model_physics,  "Physics VAE",  COLORS["physics"]),
    ]:
        z = torch.randn(n_samples, config.LATENT_DIM)
        with torch.no_grad():
            generated = model.decoder(z)
        generated_raw = get_angles_from_4d(generated)

        ax.scatter(phi_orig, psi_orig, s=4, alpha=0.2, color="grey", label="training")
        ax.scatter(generated_raw[:, 0], generated_raw[:, 1], s=8, alpha=0.5,
                   color=color, label="generated")
        ramachandran_background(ax)
        overlay_regions(ax, show_legend=False)
        lov_fav, lov_allow = compliance_lovell(generated_raw[:, 0], generated_raw[:, 1])
        ax.set_title(
            f"{label}\nLovell favoured={lov_fav:.2f}  allowed={lov_allow:.2f}"
        )
        ax.legend(fontsize=7)

    savefig(fig, f"stage4_{experiment}_generated_samples.png")


# =============================================================================
# Stage 5 : perturbation scatter (reads perturbation_samples.csv)
# =============================================================================

def plot_perturbation_scatter(experiment, df_samples):
    """
    Scatter plots of decoded (phi, psi) clouds per sigma, for this experiment.
    One figure: 2 rows (baseline / physics) x len(sigmas) columns.
    """
    sub = df_samples[df_samples["experiment"] == experiment]
    if sub.empty:
        print(f"  No perturbation samples found for {experiment} -- skipping stage 5")
        return

    sigmas   = sorted(sub["sigma"].unique())
    variants = ["baseline", "physics"]
    n_cols   = len(sigmas)

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))
    fig.suptitle(
        f"Perturbation Scatter -- {experiment}", fontsize=13, fontweight="bold"
    )

    for row, variant in enumerate(variants):
        color = COLORS[variant]
        for col, sigma in enumerate(sigmas):
            ax = axes[row, col]
            pts = sub[(sub["variant"] == variant) & (sub["sigma"] == sigma)]
            ax.scatter(pts["phi"], pts["psi"], s=6, alpha=0.4, color=color)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            overlay_regions(ax, show_legend=False,
                            alpha_favoured=0.12, alpha_allowed=0.06)
            ax.set_title(f"{variant}\nsigma={sigma}", fontsize=8)
            if col == 0:
                ax.set_ylabel(r"$\psi$ (degrees)", fontsize=7)
            ax.set_xlabel(r"$\phi$ (degrees)", fontsize=7)
            ax.tick_params(labelsize=6)

            phi_s = pts["phi"].std()
            psi_s = pts["psi"].std()
            ent   = 0.5 * np.log(2 * np.pi * np.e * (phi_s * psi_s + 1e-8))
            lov_fav, _ = compliance_lovell(pts["phi"].values, pts["psi"].values)
            ax.text(0.05, 0.92, f"H={ent:.2f}", transform=ax.transAxes,
                    fontsize=7, color=color)
            ax.text(0.05, 0.82, f"fav={lov_fav:.2f}", transform=ax.transAxes,
                    fontsize=7, color=color)

    plt.tight_layout()
    savefig(fig, f"stage5_{experiment}_perturbation_scatter.png")


# =============================================================================
# Stage 6 : entropy scaling with CI + compliance rate
# =============================================================================

def plot_entropy_and_compliance(experiment, df_samples, df_compliance):
    """
    Panel 1 : entropy with 95% CI bands (bootstrap), baseline vs physics
    Panel 2 : GMM compliance rate vs sigma  (data-dependent)
    Panel 3 : Lovell favoured and allowed rates vs sigma  (data-independent)
    """
    print(f'plotting for experiment {experiment}')
    sub_s = df_samples[df_samples["experiment"] == experiment]
    sub_c = df_compliance[df_compliance["experiment"] == experiment]

    if sub_s.empty or sub_c.empty:
        print(f"  No data found for {experiment} -- skipping stage 6")
        return

    sigmas = sorted(sub_s["sigma"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Entropy Scaling and Compliance -- {experiment}",
        fontsize=13, fontweight="bold"
    )

    for variant in ["baseline", "physics"]:
        color   = COLORS[variant]
        ent_means, ent_lo, ent_hi = [], [], []
        gmm_vals   = []
        lov_fav_vals   = []
        lov_allow_vals = []

        for sigma in sigmas:
            pts = sub_s[(sub_s["variant"] == variant) & (sub_s["sigma"] == sigma)]
            phi_arr = pts["phi"].values
            psi_arr = pts["psi"].values

            # bootstrap 95% CI on entropy
            n = len(phi_arr)
            boot_ents = []
            rng = np.random.default_rng(config.SEED)
            for _ in range(500):
                idx  = rng.integers(0, n, size=n)
                ph_b = phi_arr[idx].std()
                ps_b = psi_arr[idx].std()
                boot_ents.append(
                    0.5 * np.log(2 * np.pi * np.e * (ph_b * ps_b + 1e-8))
                )
            boot_ents = np.array(boot_ents)
            ent_means.append(boot_ents.mean())
            ent_lo.append(np.percentile(boot_ents, 2.5))
            ent_hi.append(np.percentile(boot_ents, 97.5))

            crow = sub_c[
                (sub_c["variant"] == variant) & (sub_c["sigma"] == sigma)
            ]
            if len(crow):
                gmm_vals.append(crow["gmm_compliance"].values[0]
                                if "gmm_compliance" in crow.columns
                                else crow["compliance_rate"].values[0])
                lov_fav_vals.append(
                    crow["lovell_favoured"].values[0]
                    if "lovell_favoured" in crow.columns else np.nan
                )
                lov_allow_vals.append(
                    crow["lovell_allowed"].values[0]
                    if "lovell_allowed" in crow.columns else np.nan
                )
            else:
                gmm_vals.append(np.nan)
                lov_fav_vals.append(np.nan)
                lov_allow_vals.append(np.nan)

        sigmas_arr = np.array(sigmas)
        ent_means  = np.array(ent_means)
        ent_lo     = np.array(ent_lo)
        ent_hi     = np.array(ent_hi)

        # panel 1 : entropy
        axes[0].plot(sigmas_arr, ent_means, "o-", color=color,
                     label=variant, linewidth=2)
        axes[0].fill_between(sigmas_arr, ent_lo, ent_hi, color=color, alpha=0.15)

        # panel 2 : GMM compliance
        axes[1].plot(sigmas_arr, gmm_vals, "o-", color=color,
                     label=variant, linewidth=2)

        # panel 3 : Lovell compliance -- favoured solid, allowed dashed
        axes[2].plot(sigmas_arr, lov_fav_vals, "o-", color=color,
                     label=f"{variant} favoured", linewidth=2)
        axes[2].plot(sigmas_arr, lov_allow_vals, "o--", color=color,
                     label=f"{variant} allowed", linewidth=1.5, alpha=0.7)

    axes[0].set_xlabel("perturbation sigma")
    axes[0].set_ylabel("entropy (nats)")
    axes[0].set_title("Entropy vs sigma  (95% CI, bootstrap)")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("perturbation sigma")
    axes[1].set_ylabel("compliance rate")
    axes[1].set_title(
        f"GMM compliance vs sigma\n(data-dependent, threshold={config.GMM_COMPLIANCE_THRESHOLD:.1f})"
    )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("perturbation sigma")
    axes[2].set_ylabel("compliance rate")
    axes[2].set_title(
        "Lovell compliance vs sigma\n(data-independent, Lovell et al. 2003)"
    )
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    savefig(fig, f"stage6_{experiment}_entropy_compliance.png")

def run_stats():

    """
    Statistical analysis of physics vs baseline VAE performance.
    No plots are produced here -- visualisations live in plot_diagnostics.py.

    Reads
    -----
    results/perturbation_samples.csv    (long-form, one row per decoded sample)
    results/perturbation_compliance.csv (aggregated compliance rates)

    Outputs
    -------
    results/physics_wins_stats.csv
        Per (experiment, sigma, metric): Mann-Whitney U statistic, p-value,
        Cohen's d, win direction, and effect size label.

    results/physics_wins_summary.csv
        Per (experiment, metric): win rate across sigmas, mean Cohen's d,
        fraction of sigmas with p < 0.05.

    Printed summary table to stdout.

    Notes on statistical power
    --------------------------
    Tests are run per sigma level with n = N_PERTURBATION per group.
    With the default N=30, Mann-Whitney U has ~80% power to detect a medium
    effect (Cohen's d ~ 0.5) at alpha=0.05.  Increase N_PERTURBATION
    in config for higher power.  With N < 10, results should be treated as
    exploratory only.
    """
    samples_path    = os.path.join(config.RESULTS_DIR,  "datafiles/perturbation_samples.csv")
    compliance_path = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_compliance.csv")

    if not os.path.exists(samples_path):
        print(
            "perturbation_samples.csv not found. "
            "Run perturbation_analysis.py first."
        )
        return

    df_s = pd.read_csv(samples_path)
    df_c = pd.read_csv(compliance_path) if os.path.exists(compliance_path) else None

    # compute per-sample entropy proxy: we use phi and psi directly as the
    # two variables; the per-group entropy is derived from their joint std.
    # For per-sample Mann-Whitney we compare the distribution of |phi| + |psi|
    # (total angular displacement from origin) as a scalar summary, plus
    # phi_std, psi_std, and compliance tested separately.

    rows_detail  = []
    rows_summary = []

    experiments = df_s["experiment"].unique()
    sigmas      = sorted(df_s["sigma"].unique())

    # metrics drawn from per-sample data
    sample_metrics = {
        "phi_abs":  lambda sub: sub["phi"].abs().values,
        "psi_abs":  lambda sub: sub["psi"].abs().values,
    }

    for experiment in experiments:
        exp_sub = df_s[df_s["experiment"] == experiment]

        # per sigma
        for sigma in sigmas:
            sig_sub = exp_sub[exp_sub["sigma"] == sigma]
            b_sub   = sig_sub[sig_sub["variant"] == "baseline"]
            p_sub   = sig_sub[sig_sub["variant"] == "physics"]

            if b_sub.empty or p_sub.empty:
                continue

            metrics_to_test = dict(sample_metrics)

            # add compliance metrics if available -- report all three columns
            if df_c is not None:
                _compliance_cols = [
                    c for c in ["gmm_compliance", "lovell_favoured", "lovell_allowed"]
                    if c in df_c.columns
                ]
                b_comp = df_c[
                    (df_c["experiment"] == experiment) &
                    (df_c["variant"]    == "baseline") &
                    (df_c["sigma"]      == sigma)
                ]
                p_comp = df_c[
                    (df_c["experiment"] == experiment) &
                    (df_c["variant"]    == "physics") &
                    (df_c["sigma"]      == sigma)
                ]
                if not b_comp.empty and not p_comp.empty:
                    for _col in _compliance_cols:
                        b_val = float(b_comp[_col].values[0])
                        p_val = float(p_comp[_col].values[0])
                        rows_detail.append({
                            "experiment":   experiment,
                            "sigma":        sigma,
                            "metric":       _col,
                            "baseline_val": round(b_val, 4),
                            "physics_val":  round(p_val, 4),
                            "delta":        round(p_val - b_val, 4),
                            "physics_wins": int(p_val > b_val),
                            "mwu_stat":     np.nan,
                            "p_value":      np.nan,
                            "cohen_d":      np.nan,
                            "effect_label": "n/a (scalar)",
                            "note":         "scalar -- no significance test",
                        })

            for metric_name, extractor in metrics_to_test.items():
                b_vals = extractor(b_sub)
                p_vals = extractor(p_sub)

                n = min(len(b_vals), len(p_vals))
                if n < 3:
                    continue

                # Mann-Whitney U: two-sided, then determine direction
                stat, p_two = mannwhitneyu(p_vals, b_vals, alternative="two-sided")
                # physics wins if it has lower values (tighter/more constrained)
                physics_wins = int(np.median(p_vals) < np.median(b_vals))

                d   = cohen_d(b_vals, p_vals)  # positive d => baseline > physics
                eff = effect_size_label(d)

                rows_detail.append({
                    "experiment":   experiment,
                    "sigma":        sigma,
                    "metric":       metric_name,
                    "baseline_val": round(float(np.median(b_vals)), 4),
                    "physics_val":  round(float(np.median(p_vals)), 4),
                    "delta":        round(float(np.median(b_vals) - np.median(p_vals)), 4),
                    "physics_wins": physics_wins,
                    "mwu_stat":     round(float(stat), 2),
                    "p_value":      round(float(p_two), 4),
                    "cohen_d":      round(d, 4),
                    "effect_label": eff,
                    "note":         f"n={n} per group",
                })

        # summary across sigmas for this experiment
        # exclude scalar compliance rows (no MWU test, cohen_d is NaN)
        det_exp = [r for r in rows_detail if r["experiment"] == experiment
                   and r["metric"] in sample_metrics]
        for metric_name in sample_metrics:
            metric_rows = [r for r in det_exp if r["metric"] == metric_name]
            if not metric_rows:
                continue
            win_rate   = np.mean([r["physics_wins"] for r in metric_rows])
            mean_d     = np.mean([r["cohen_d"]      for r in metric_rows])
            sig_frac   = np.mean([r["p_value"] < 0.05 for r in metric_rows])
            rows_summary.append({
                "experiment":        experiment,
                "metric":            metric_name,
                "win_rate":          round(win_rate, 3),
                "mean_cohen_d":      round(mean_d, 3),
                "frac_p_lt_0.05":   round(sig_frac, 3),
                "mean_delta":        np.nan,
                "n_sigmas_tested":   len(metric_rows),
            })

        # compliance summary: mean delta across sigmas, win rate, no cohen_d
        _compliance_cols = [
            c for c in ["gmm_compliance", "lovell_favoured", "lovell_allowed"]
            if any(r["metric"] == c for r in rows_detail if r["experiment"] == experiment)
        ]
        for _col in _compliance_cols:
            col_rows = [
                r for r in rows_detail
                if r["experiment"] == experiment and r["metric"] == _col
            ]
            if not col_rows:
                continue
            rows_summary.append({
                "experiment":        experiment,
                "metric":            _col,
                "win_rate":          round(np.mean([r["physics_wins"] for r in col_rows]), 3),
                "mean_cohen_d":      np.nan,
                "frac_p_lt_0.05":   np.nan,
                "mean_delta":        round(np.mean([r["delta"] for r in col_rows]), 4),
                "n_sigmas_tested":   len(col_rows),
            })

    df_detail  = pd.DataFrame(rows_detail)
    df_summary = pd.DataFrame(rows_summary)

    detail_path  = os.path.join(config.RESULTS_DIR, "datafiles/physics_wins_stats.csv")
    summary_path = os.path.join(config.RESULTS_DIR, "datafiles/physics_wins_summary.csv")

    df_detail.to_csv(detail_path,   index=False)
    df_summary.to_csv(summary_path, index=False)

    _compliance_metric_names = {"gmm_compliance", "lovell_favoured", "lovell_allowed"}

    print("=== Per-sigma statistics ===")
    print(df_detail.to_string(index=False))

    print("\n=== Summary: distributional metrics (phi_abs, psi_abs) ===")
    df_dist = df_summary[~df_summary["metric"].isin(_compliance_metric_names)].copy()
    print(
        df_dist[["experiment", "metric", "win_rate", "mean_cohen_d",
                 "frac_p_lt_0.05", "n_sigmas_tested"]].to_string(index=False)
    )

    print("\n=== Summary: compliance metrics (scalar delta, no significance test) ===")
    df_comp = df_summary[df_summary["metric"].isin(_compliance_metric_names)].copy()
    print(
        df_comp[["experiment", "metric", "win_rate", "mean_delta",
                 "n_sigmas_tested"]].to_string(index=False)
    )

    print(f"\nSaved -> {detail_path}")
    print(f"Saved -> {summary_path}")

    # plain-language summary
    print("\n=== Plain-language summary ===")
    for _, row in df_summary.iterrows():
        if row["metric"] in _compliance_metric_names:
            print(
                f"  {row['experiment']} | {row['metric']}: "
                f"physics wins {row['win_rate']*100:.0f}% of sigma levels, "
                f"mean delta = {row['mean_delta']:+.4f} "
                f"(positive = physics more compliant)"
            )
        else:
            direction = "lower" if row["mean_cohen_d"] > 0 else "higher"
            print(
                f"  {row['experiment']} | {row['metric']}: "
                f"physics wins {row['win_rate']*100:.0f}% of sigma levels, "
                f"mean Cohen's d = {row['mean_cohen_d']:.3f} ({effect_size_label(row['mean_cohen_d'])} effect, "
                f"physics {direction}), "
                f"{row['frac_p_lt_0.05']*100:.0f}% of tests p < 0.05"
            )

    return df_detail, df_summary


# =============================================================================
# main
# =============================================================================

def main():
    samples_path    = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_samples.csv")
    compliance_path = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_compliance.csv")

    df_samples    = pd.read_csv(samples_path)    if os.path.exists(samples_path)    else None
    df_compliance = pd.read_csv(compliance_path) if os.path.exists(compliance_path) else None

    for experiment, (csv_path, gmm_path) in config.EXPERIMENTS.items():
        if not os.path.exists(csv_path):
            print(f"CSV not found for '{experiment}' -- skipping")
            continue
        

        print(f"\n=== Plotting: {experiment} ===")
        df                                            = load_experiment_data(experiment)
        gmm, model_baseline, model_physics    = load_experiment_artifacts(experiment)

        print("  Stage 0: learning curves")
        plot_learning_curves(experiment)

        print("  Stage 1: training data")
        plot_training_data(experiment, df)
        plot_training_data_streamlit(experiment, df)

        print("  Stage 2: GMM quality")
        plot_gmm_quality(experiment, df, gmm)

        print("  Stage 3: reconstruction")
        plot_reconstruction(experiment, df, model_baseline, model_physics)

        print("  Stage 4: generated samples")
        plot_generated_samples(experiment, df, model_baseline, model_physics)

        if df_samples is not None:
            print("  Stage 5: perturbation scatter")
            plot_perturbation_scatter(experiment, df_samples)

            if df_compliance is not None:
                print("  Stage 6: entropy + compliance")
                plot_entropy_and_compliance(experiment, df_samples, df_compliance)
        else:
            print("  Stages 5-6: perturbation_samples.csv not found -- run perturbation_analysis.py")

    run_stats()
    print("\nAll plots saved to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
