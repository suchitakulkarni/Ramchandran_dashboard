import torch
import numpy as np
from pathlib import Path
import pyrama

import joblib



class RamaGrid:
    """
    Differentiable Ramachandran energy prior using the Top500 reference
    density grid shipped with pyrama (Richardson lab, Lovell et al. 2003).

    Replaces TorchGMM with an identical log_prob(x) interface.
    Input x is expected in raw degrees, shape (batch, 2), phi first psi second.

    The grid covers (-180, 180) x (-180, 180) in 2-degree bins (180 x 180).
    log_prob is evaluated via bilinear interpolation, fully differentiable
    with respect to x.
    """

    def __init__(self, data_file: str | None = None):
        grid, phi_coords, psi_coords = self._load_grid(data_file)

        # grid shape: (180, 180), axes: phi (rows), psi (cols)
        log_grid = np.log(grid + 1e-12)

        # store as tensors, shape (1, 1, H, W) for grid_sample
        self.log_grid = torch.tensor(
            log_grid, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        self.phi_min = float(phi_coords[0])
        self.phi_max = float(phi_coords[-1])
        self.psi_min = float(psi_coords[0])
        self.psi_max = float(psi_coords[-1])

    def _load_grid(self, data_file: str | None):
        if data_file is None:
            
            pkg_dir = Path(pyrama.__file__).parent
            data_file = pkg_dir / "data" / "pref_general.data"

        phi_vals, psi_vals, density_vals = [], [], []

        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                phi_vals.append(float(parts[0]))
                psi_vals.append(float(parts[1]))
                density_vals.append(float(parts[2]))

        phi_unique = sorted(set(phi_vals))
        psi_unique = sorted(set(psi_vals))
        n_phi = len(phi_unique)
        n_psi = len(psi_unique)

        phi_idx = {v: i for i, v in enumerate(phi_unique)}
        psi_idx = {v: i for i, v in enumerate(psi_unique)}

        grid = np.zeros((n_phi, n_psi), dtype=np.float64)
        for ph, ps, d in zip(phi_vals, psi_vals, density_vals):
            grid[phi_idx[ph], psi_idx[ps]] = d

        return grid, np.array(phi_unique), np.array(psi_unique)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 2) in raw degrees, phi first psi second.
        returns: (batch,) log probability under the Top500 Ramachandran density.
        """
        phi = x[:, 0]
        psi = x[:, 1]

        # normalize to [-1, 1] for grid_sample
        # grid_sample convention: x maps to width (psi), y maps to height (phi)
        phi_norm = 2.0 * (phi - self.phi_min) / (self.phi_max - self.phi_min) - 1.0
        psi_norm = 2.0 * (psi - self.psi_min) / (self.psi_max - self.psi_min) - 1.0

        # grid_sample expects (N, 1, 1, 2) grid coords in (x, y) = (psi, phi) order
        grid_coords = torch.stack([psi_norm, phi_norm], dim=1)
        grid_coords = grid_coords.unsqueeze(1).unsqueeze(1)

        log_grid = self.log_grid.to(x.device)

        # bilinear interpolation, fully differentiable
        sampled = torch.nn.functional.grid_sample(
            log_grid.expand(x.shape[0], -1, -1, -1),
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return sampled.squeeze(1).squeeze(1).squeeze(1)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            log_probs = self.log_prob(x_tensor)
        return log_probs.numpy()
    
    def score(self, x: np.ndarray) -> float:
        return float(self.score_samples(x).mean())


def save_rama_grid(grid: RamaGrid, grid_path: str) -> RamaGrid:
    joblib.dump(grid, grid_path)
    print(f"RamaGrid saved -> {grid_path}")
    return grid

def load_rama_grid(grid_path: str) -> RamaGrid:
    return joblib.load(grid_path)


def build_rama_grid(data_file: str | None = None) -> RamaGrid:
    return RamaGrid(data_file)

def build_and_save_rama_grid(grid_path: str) -> RamaGrid:
    grid = build_rama_grid()
    joblib.dump(grid, grid_path)
    print(f"RamaGrid built from Top500 -> {grid_path}")
    return grid
