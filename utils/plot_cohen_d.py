"""
plot_cohen_d.py

Visualises the Cohen d results from the perturbation analysis.
Produces a two-panel figure:
  Top row : violin plots of phi_abs and psi_abs distributions per sigma,
            colored by variant (baseline vs physics)
  Bottom  : Cohen d bar chart across sigmas for both metrics,
            bars colored by direction (physics wins / loses)

Reads
-----
results/datafiles/perturbation_samples.csv

Outputs
-------
results/cohen_d_explanation.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

from pathlib import Path

if "PROJECT_ROOT" in os.environ:
    root_path = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    # fallback: assume this file is somewhere inside src/
    root_path = Path(__file__).resolve().parents[1]
    
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import src.config as config

# color palette consistent with existing plots
C_BASELINE = "#4C9BE8"
C_PHYSICS  = "#E84C7D"
C_WIN      = "#2E8B57"
C_LOSE     = "#C0392B"
C_ZERO     = "#AAAAAA"


def cohen_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2.0)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def plot_cohen_d_explanation(samples_path: str, out_path: str) -> None:
    df = pd.read_csv(samples_path)

    sigmas   = sorted(df["sigma"].unique())
    n_sigmas = len(sigmas)

    metrics = [
        ("phi_abs", r"$\phi$ (degrees)", r"$|\phi|$ absolute value"),
        ("psi_abs", r"$\psi$ (degrees)", r"$|\psi|$ absolute value"),
    ]

    fig = plt.figure(figsize=(18, 12))
    #fig.suptitle(
    #    "Cohen d Explanation: Physics vs Baseline VAE under Latent Perturbation",
    #    fontsize=14, fontweight="bold", y=0.98
    #)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.35,
        height_ratios=[1.6, 1.0]
    )

    # top row: violin plots per metric
    for col_idx, (metric, angle_label, metric_title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col_idx])

        positions_b = []
        positions_p = []
        data_b      = []
        data_p      = []

        x_ticks     = []
        x_labels    = []

        for i, sigma in enumerate(sigmas):
            sub = df[df["sigma"] == sigma]
            b_vals = sub[sub["variant"] == "baseline"]["phi" if metric == "phi_abs" else "psi"].abs().values
            p_vals = sub[sub["variant"] == "physics"]["phi"  if metric == "phi_abs" else "psi"].abs().values

            pos_b = i * 3.0
            pos_p = i * 3.0 + 1.0

            positions_b.append(pos_b)
            positions_p.append(pos_p)
            data_b.append(b_vals)
            data_p.append(p_vals)

            x_ticks.append(i * 3.0 + 0.5)
            x_labels.append(f"{sigma:.2g}")

        vb = ax.violinplot(data_b, positions=positions_b, widths=0.8,
                           showmedians=True, showextrema=False)
        vp = ax.violinplot(data_p, positions=positions_p, widths=0.8,
                           showmedians=True, showextrema=False)

        for body in vb["bodies"]:
            body.set_facecolor(C_BASELINE)
            body.set_alpha(0.6)
        vb["cmedians"].set_color(C_BASELINE)
        vb["cmedians"].set_linewidth(2)

        for body in vp["bodies"]:
            body.set_facecolor(C_PHYSICS)
            body.set_alpha(0.6)
        vp["cmedians"].set_color(C_PHYSICS)
        vp["cmedians"].set_linewidth(2)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=15)
        ax.tick_params(labelsize = 15)
        ax.set_xlabel(r"perturbation $\sigma$", fontsize=15)
        #ax.set_ylabel(f"|{angle_label}| (degrees)", fontsize=15)
        ax.set_ylabel(f"|{angle_label}|", fontsize=15)
        ax.set_title(
            f"{metric_title}\n"
            rf"{'physics tighter at small sigma, converges at large $\sigma$' if metric == 'phi_abs' else 'physics more dispersed than baseline at small $\sigma$'}",
            fontsize=15, pad=6
        )
        ax.grid(axis="y", alpha=0.3)

        patch_b = mpatches.Patch(color=C_BASELINE, alpha=0.7, label="baseline")
        patch_p = mpatches.Patch(color=C_PHYSICS,  alpha=0.7, label="physics")
        ax.legend(handles=[patch_b, patch_p], fontsize=15, loc="upper left")

    # bottom row: Cohen d bar chart across sigmas
    '''ax_d = fig.add_subplot(gs[1, :])

    x     = np.arange(n_sigmas)
    width = 0.35

    d_phi = []
    d_psi = []

    for sigma in sigmas:
        sub    = df[df["sigma"] == sigma]
        b_phi  = sub[sub["variant"] == "baseline"]["phi"].abs().values
        p_phi  = sub[sub["variant"] == "physics"]["phi"].abs().values
        b_psi  = sub[sub["variant"] == "baseline"]["psi"].abs().values
        p_psi  = sub[sub["variant"] == "physics"]["psi"].abs().values
        d_phi.append(cohen_d(b_phi, p_phi))
        d_psi.append(cohen_d(b_psi, p_psi))

    d_phi = np.array(d_phi)
    d_psi = np.array(d_psi)

    def bar_colors(d_arr, base_win, base_lose):
        return [base_win if v > 0 else base_lose for v in d_arr]

    bars_phi = ax_d.bar(
        x - width / 2, d_phi, width,
        color=bar_colors(d_phi, C_WIN, C_LOSE),
        alpha=0.85, label="phi_abs"
    )
    bars_psi = ax_d.bar(
        x + width / 2, d_psi, width,
        color=bar_colors(d_psi, C_WIN, C_LOSE),
        alpha=0.55, label="psi_abs"
    )

    # value labels on bars
    for bar in bars_phi:
        h = bar.get_height()
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.1 if h >= 0 else -0.25),
            f"{h:.2f}", ha="center", va="bottom", fontsize=12
        )
    for bar in bars_psi:
        h = bar.get_height()
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.1 if h >= 0 else -0.25),
            f"{h:.2f}", ha="center", va="bottom", fontsize=12
        )

    ax_d.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # reference lines for effect size thresholds
    for thresh, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax_d.axhline( thresh, color=C_ZERO, linewidth=0.6, linestyle=":")
        ax_d.axhline(-thresh, color=C_ZERO, linewidth=0.6, linestyle=":")
        ax_d.text(n_sigmas - 0.5, thresh + 0.05, label, fontsize=12,
                  color=C_ZERO, ha="right")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"{s:.2g}" for s in sigmas], fontsize=15)
    ax_d.tick_params(labelsize = 15)
    ax_d.set_xlabel("perturbation sigma", fontsize=15)
    ax_d.set_ylabel("Cohen d\n(positive = physics lower = physics wins)", fontsize=15)
    ax_d.set_title(
        "Cohen d per sigma -- positive bars: physics wins (tighter)   "
        "negative bars: physics loses (more dispersed)\n"
        "solid = phi_abs   faded = psi_abs",
        fontsize=15
    )
    ax_d.grid(axis="y", alpha=0.3)

    win_patch  = mpatches.Patch(color=C_WIN,  alpha=0.85, label="physics wins (d > 0)")
    lose_patch = mpatches.Patch(color=C_LOSE, alpha=0.85, label="physics loses (d < 0)")
    phi_patch  = mpatches.Patch(color="gray", alpha=0.85, label="phi_abs (solid)")
    psi_patch  = mpatches.Patch(color="gray", alpha=0.55, label="psi_abs (faded)")
    ax_d.legend(
        handles=[win_patch, lose_patch, phi_patch, psi_patch],
        fontsize=15, loc="lower right", ncol=2
    )'''

    x     = np.arange(n_sigmas)
    width = 0.35

    d_phi = []
    d_psi = []

    for sigma in sigmas:
        sub    = df[df["sigma"] == sigma]
        b_phi  = sub[sub["variant"] == "baseline"]["phi"].abs().values
        p_phi  = sub[sub["variant"] == "physics"]["phi"].abs().values
        b_psi  = sub[sub["variant"] == "baseline"]["psi"].abs().values
        p_psi  = sub[sub["variant"] == "physics"]["psi"].abs().values
        d_phi.append(cohen_d(b_phi, p_phi))
        d_psi.append(cohen_d(b_psi, p_psi))

    d_phi = np.array(d_phi)
    d_psi = np.array(d_psi)

    def bar_colors(d_arr, base_win, base_lose):
        return [base_win if v > 0 else base_lose for v in d_arr]


    # bottom row: two stacked Cohen d bar charts, one per metric
    ax_phi = fig.add_subplot(gs[1, 0])
    ax_psi = fig.add_subplot(gs[1, 1])

    for ax, d_vals, metric_label in [
        (ax_phi, d_phi, r"$|\phi|$"),
        (ax_psi, d_psi, r"$|\psi$|"),
    ]:
        colors = [C_WIN if v > 0 else C_LOSE for v in d_vals]
        bars = ax.bar(x, d_vals, width=0.5, color=colors, alpha=0.85)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.1 if h >= 0 else -0.35),
                f"{h:.2f}", ha="center", va="bottom", fontsize=15
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        #for thresh, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        #    ax.axhline( thresh, color=C_ZERO, linewidth=0.6, linestyle=":")
        #    ax.axhline(-thresh, color=C_ZERO, linewidth=0.6, linestyle=":")
        #    ax.text(n_sigmas - 0.6, thresh + 0.05, label, fontsize=15, color=C_ZERO)

        ax.set_xticks(x)
        ax.tick_params(labelsize = 15)
        ax.set_xticklabels([f"{s:.2g}" for s in sigmas], fontsize=15)
        ax.set_xlabel(rf"perturbation $\sigma$", fontsize=15)
        ax.set_ylabel("Cohen d\n(positive = physics wins)", fontsize=15)
        ax.set_title(rf"Cohen d per $\sigma$, {metric_label} dimension", fontsize=15)
        ax.grid(axis="y", alpha=0.3)

        win_patch  = mpatches.Patch(color=C_WIN,  alpha=0.85, label="physics wins (d > 0)")
        lose_patch = mpatches.Patch(color=C_LOSE, alpha=0.85, label="physics loses (d < 0)")
        if ax == ax_phi:
            ax.legend(handles=[win_patch, lose_patch], fontsize=15, loc="upper right")
        else:
            ax.legend(handles=[win_patch, lose_patch], fontsize=15, loc="lower right")

        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    samples_path = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_samples.csv")
    out_path     = os.path.join(config.RESULTS_DIR, "cohen_d_explanation.png")
    plot_cohen_d_explanation(samples_path, out_path)
