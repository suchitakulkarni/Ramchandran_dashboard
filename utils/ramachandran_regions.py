"""
ramachandran_regions.py

Loads the Lovell et al. (2003) Top500 general-case density grid shipped
with the pyrama package and provides:

  - overlay_regions(ax)
        draws favoured / allowed contours on an existing matplotlib axes

  - compliance_lovell(phi_arr, psi_arr)
        returns (frac_favoured, frac_allowed) for arrays of angles in degrees

Thresholds (normalised density values, matching RAMPAGE conventions):
  FAVOURED_THRESHOLD  : core basins, ~98% of well-refined residues
  ALLOWED_THRESHOLD   : outer boundary, ~99.95% of well-refined residues

These thresholds are data-independent from your training set, which is
what makes them useful as a physics compliance metric.

Reference
---------
Lovell SC et al. (2003). Structure validation by Calpha geometry:
phi, psi and Cbeta deviation. Proteins 50(3):437-450.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyrama


from pathlib import Path

if "PROJECT_ROOT" in os.environ:
    root_path = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    # fallback: assume this file is somewhere inside src/
    root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import src.config as config
# --- load grid once at import time ---

_DATA_PATH = os.path.join(
    os.path.dirname(pyrama.__file__), "data", "pref_general.data"
)

def _load_grid():
    raw = np.loadtxt(_DATA_PATH, comments="#")
    # raw columns: phi, psi, density  (2-degree bins, phi varies fastest)
    phi_vals = np.unique(raw[:, 0])   # -179 .. 179, step 2
    psi_vals = np.unique(raw[:, 1])
    grid = raw[:, 2].reshape(len(phi_vals), len(psi_vals))
    return phi_vals, psi_vals, grid

PHI_VALS, PSI_VALS, DENSITY_GRID = _load_grid()

# RAMPAGE-equivalent thresholds on the normalised [0,1] density
FAVOURED_THRESHOLD = 0.02   # core basins  (~98% of residues in well-refined structures)
ALLOWED_THRESHOLD  = 0.002  # outer shell  (~99.95%)


# --- public API ---

def overlay_regions(ax, alpha_favoured=0.15, alpha_allowed=0.08,
                    color_favoured="#4CAF50", color_allowed="#FFC107",
                    show_legend=True):
    """
    Draw filled Lovell favoured and allowed regions on a matplotlib axes.
    Call this AFTER setting xlim/ylim on the axes.

    Parameters
    ----------
    ax              : matplotlib Axes
    alpha_favoured  : fill transparency for favoured region
    alpha_allowed   : fill transparency for allowed region
    color_favoured  : fill colour for favoured region (green)
    color_allowed   : fill colour for allowed region (amber)
    show_legend     : whether to add legend entries
    """
    # meshgrid for contourf -- note: contourf expects (y, x) orientation
    gp, gps = np.meshgrid(PHI_VALS, PSI_VALS, indexing="ij")

    # allowed (outer) -- draw first so favoured overlays it
    ax.contourf(
        gp, gps, DENSITY_GRID,
        levels=[ALLOWED_THRESHOLD, FAVOURED_THRESHOLD],
        colors=[color_allowed],
        alpha=alpha_allowed,
    )
    # favoured (core)
    ax.contourf(
        gp, gps, DENSITY_GRID,
        levels=[FAVOURED_THRESHOLD, 1.01],
        colors=[color_favoured],
        alpha=alpha_favoured,
    )
    # contour lines for crisp boundaries
    ax.contour(
        gp, gps, DENSITY_GRID,
        levels=[ALLOWED_THRESHOLD],
        colors=[color_allowed],
        linewidths=0.8,
        alpha=0.6,
    )
    ax.contour(
        gp, gps, DENSITY_GRID,
        levels=[FAVOURED_THRESHOLD],
        colors=[color_favoured],
        linewidths=0.8,
        alpha=0.8,
    )

    if show_legend:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=color_favoured, alpha=0.5, label="Lovell favoured"),
                Patch(facecolor=color_allowed,  alpha=0.5, label="Lovell allowed"),
            ],
            fontsize=6,
            loc="lower right",
        )


def compliance_lovell(phi_arr, psi_arr):
    """
    Compute fraction of (phi, psi) samples in the Lovell favoured and
    allowed regions, using grid interpolation.

    Parameters
    ----------
    phi_arr, psi_arr : array-like, angles in degrees in [-180, 180]

    Returns
    -------
    frac_favoured : float
    frac_allowed  : float  (includes favoured)
    """
    phi_arr = np.asarray(phi_arr, dtype=float)
    psi_arr = np.asarray(psi_arr, dtype=float)

    # nearest-neighbour lookup on the 2-degree grid
    phi_idx = np.round((phi_arr - PHI_VALS[0]) / 2).astype(int).clip(0, len(PHI_VALS) - 1)
    psi_idx = np.round((psi_arr - PSI_VALS[0]) / 2).astype(int).clip(0, len(PSI_VALS) - 1)

    densities = DENSITY_GRID[phi_idx, psi_idx]

    frac_favoured = float((densities >= FAVOURED_THRESHOLD).mean())
    frac_allowed  = float((densities >= ALLOWED_THRESHOLD).mean())
    return frac_favoured, frac_allowed

def compliance_rate(samples_raw, gmm):
    """
    Fraction of decoded samples whose GMM log-prob exceeds the threshold.
    """
    log_probs = gmm.score_samples(samples_raw)
    return float((log_probs >= config.GMM_COMPLIANCE_THRESHOLD).mean())