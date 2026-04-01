"""
perturbation_analysis.py

Loads trained VAE models and probes the decoder by perturbing around
the latent prior mean (z=0).  For each (experiment, variant, sigma),
N_PERTURBATION_SAMPLES decoded (phi, psi) pairs are saved in long form.

Two compliance metrics are computed per group:
  - GMM compliance   : fraction above config.GMM_COMPLIANCE_THRESHOLD
                       (data-dependent, reflects training distribution)
  - Lovell favoured  : fraction in Lovell et al. (2003) favoured region
  - Lovell allowed   : fraction in Lovell et al. (2003) allowed region
                       (data-independent, true physics prior)

Outputs
-------
results/perturbation_samples.csv
    Long-form table: experiment, variant, sigma, sample_idx, phi, psi

results/perturbation_compliance.csv
    Aggregated: experiment, variant, sigma, n_samples,
                phi_std, psi_std, entropy,
                gmm_compliance, lovell_favoured, lovell_allowed
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import joblib

from pathlib import Path
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
from utils.ramachandran_regions import compliance_lovell
from utils.utils import get_angles_from_4d

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

os.makedirs(config.RESULTS_DIR, exist_ok=True)


# --- loaders ---

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


# --- perturbation sampling ---

def sample_perturbations(model, sigma, n_samples):
    """
    Perturb z=0 with Gaussian noise of std=sigma, decode, inverse-transform.
    Returns array of shape (n_samples, 2) in raw angle space (degrees).
    """
    z_center = torch.zeros(1, config.LATENT_DIM)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            z_p = z_center + sigma * torch.randn(1, config.LATENT_DIM)
            decoded = model.decoder(z_p)
            decoded_raw = get_angles_from_4d(decoded).numpy()
            samples.append(decoded_raw[0])
    return np.array(samples)  # (n_samples, 2)


def compliance_rate(samples_raw, gmm):
    """
    Fraction of decoded samples whose GMM log-prob exceeds the threshold.
    """
    log_probs = gmm.score_samples(samples_raw)
    return float((log_probs >= config.GMM_COMPLIANCE_THRESHOLD).mean())


# --- main ---

def run_perturbation_analysis():
    all_samples    = []
    all_compliance = []

    for experiment in config.EXPERIMENTS:

        print(f"\n=== Perturbation analysis: {experiment} ===")
        gmm, model_baseline, model_physics = load_experiment_artifacts(experiment)

        for model, variant in [(model_baseline, "baseline"), (model_physics, "physics")]:
            for sigma in config.PERTURBATION_SIGMAS:
                samples = sample_perturbations(
                    model, sigma, config.N_PERTURBATIONS
                )

                # long-form rows
                for idx, (phi, psi) in enumerate(samples):
                    all_samples.append({
                        "experiment": experiment,
                        "variant":    variant,
                        "sigma":      sigma,
                        "sample_idx": idx,
                        "phi":        phi,
                        "psi":        psi,
                    })

                # aggregate stats for compliance table
                phi_std = samples[:, 0].std()
                psi_std = samples[:, 1].std()
                # differential entropy of a 2D Gaussian approximation
                entropy = 0.5 * np.log(
                    2 * np.pi * np.e * (phi_std * psi_std + 1e-8)
                )
                gmm_comp = compliance_rate(samples, gmm)
                lov_fav, lov_allow = compliance_lovell(samples[:, 0], samples[:, 1])

                all_compliance.append({
                    "experiment":      experiment,
                    "variant":         variant,
                    "sigma":           sigma,
                    "n_samples":       config.N_PERTURBATIONS,
                    "phi_std":         round(phi_std, 4),
                    "psi_std":         round(psi_std, 4),
                    "entropy":         round(entropy, 4),
                    "gmm_compliance":  round(gmm_comp, 4),
                    "lovell_favoured": round(lov_fav, 4),
                    "lovell_allowed":  round(lov_allow, 4),
                })

                print(
                    f"  {variant:8s}  sigma={sigma:.3f}  "
                    f"entropy={entropy:.3f}  "
                    f"gmm={gmm_comp:.3f}  "
                    f"lovell_fav={lov_fav:.3f}  lovell_allow={lov_allow:.3f}"
                )

    # save long-form samples
    samples_path = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_samples.csv")
    pd.DataFrame(all_samples).to_csv(samples_path, index=False)
    print(f"\nSaved long-form samples -> {samples_path}")

    # save compliance summary
    compliance_path = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_compliance.csv")
    df_comp = pd.DataFrame(all_compliance)
    df_comp.to_csv(compliance_path, index=False)
    print(f"Saved compliance summary -> {compliance_path}")
    print(df_comp.to_string(index=False))

    return pd.DataFrame(all_samples), df_comp


if __name__ == "__main__":
    run_perturbation_analysis()
