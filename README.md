# Ramachandran Physics-Informed VAE

Comparison of a standard VAE vs a physics-informed VAE for generating protein
backbone dihedral angles (phi, psi) consistent with Ramachandran constraints.

Built as a demonstration of understanding conformational sampling problems in
enzyme space, using five structurally diverse proteins from the PDB (https://www.rcsb.org/):
1BRS (Barnase-Barstar), 1TIM (TIM-barrel), 2LZM (T4 lysozyme),
1UBQ (Ubiquitin), 1VII (Villin headpiece).

We have about 2000 data samples in the training.

---

## Core idea

The Top500 dataset (https://onlinelibrary.wiley.com/doi/10.1002/prot.10286) serves as a proxy for the protein energy landscape. 
We use the log-probability in the physics VAE as a penalty that discourages decoded
samples from landing in low-probability (high-energy) Ramachandran regions.

The penalty is implemented as a **fully differentiable PyTorch GMM** so the
gradient flows end-to-end from the energy landscape back through the decoder
weights. The GMM is used only for fitting; all inference runs in PyTorch using
the fitted parameters (means, precision matrices, log-determinants).

Compliance is evaluated against two independent criteria:

- **GMM compliance**: data-dependent, reflects training distribution
- **Lovell compliance**: data-independent, uses Lovell et al. (2003) Top500
  favoured/allowed region boundaries shipped with `pyrama`

---

## Key findings

- Baseline VAE suffers **posterior collapse** on unbalanced data, generating
  along a 1D curve through Ramachandran space
- Physics VAE with a differentiable penalty **recovers all three Ramachandran
  islands**, matching training data Lovell favoured compliance (0.82)
- Physics VAE shows a **large, statistically significant phi stability win**
  under latent space perturbation (100% win rate, Cohen's d = 0.905,
  83% of sigma levels p < 0.05)
- **Lovell compliance is the stronger claim**: it uses boundaries from a
  completely independent curated dataset and directly measures physical
  plausibility rather than proximity to training data

---

## File structure

```
project
  dashboard.py                -- Streamlit dashboard (5 tabs)
  requirements.txt            -- Python dependencies
  src/
  |__ __init__.py
  |__ config.py
  |__ model.py
  |__ train_vae.py
  |__ perturbation_analysis.py
  utils/
  |__ __init__.py
  |__ oversample.py
  |__ plot_cohen_d.py
  |__ prepare_data.py
  |__ ramachandran_regions.py
  |__ rama_grid.py
  |__ utils.py
  |__ visualise.py
  data/                       -- CSV files, one per experiment
  models/                     -- saved .pt models, scalers, GMMs
  results/                    -- all output CSVs and PNGs
  |__ plots
  |__ datafiles
```

---

## Some of the important Config parameters

```python
# core architecture
SEED            = set on system clock
LATENT_DIM      = 4
HIDDEN_DIM      = 64
EPOCHS          = 300
BATCH_SIZE      = 64
LR              = 1e-3
KL_WEIGHT       = 0.5
PHYSICS_WEIGHT  = 1.0

# data
VAL_SPLIT       = 0.2

# GMM
N_GMM_COMPONENTS         = 8
GMM_COMPLIANCE_THRESHOLD = -5.0   # log-prob cutoff; tune from Stage 2 plots

# physics penalty
PHYSICS_LOG_PROB_CEIL    = 10.0   # clamp ceiling on -log_prob per sample
                                   # prevents outlier batches dominating early

# perturbation analysis
PERTURBATION_SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
N_PERTURBATIONS     = 200         # samples per (experiment, variant, sigma)
N_GENERATED_SAMPLES = 500         # samples for Stage 4 prior sampling
```

---

## Run order

```bash
# 1. extract dihedral angles from PDB structures
python utils/prepare_data.py

# 2. train and run models
python main.py

# 6. launch dashboard
streamlit run dashboard.py
```

Open http://localhost:8501

---

## Plot stages

| Stage | File pattern | Content |
|-------|-------------|---------|
| 0 | `stage0_{exp}_learning_curves.png` | Train vs val loss, all components |
| 1 | `stage1_{exp}_training_data.png` | Ramachandran scatter, KDE, marginals |
| 2 | `stage2_{exp}_gmm_quality.png` | GMM density vs training data |
| 3 | `stage3_{exp}_reconstruction.png` | Original vs reconstructed + residuals |
| 4 | `stage4_{exp}_generated_samples.png` | Prior samples + Lovell compliance |
| 5 | `stage5_{exp}_perturbation_scatter.png` | Decoded clouds per sigma |
| 6 | `stage6_{exp}_entropy_compliance.png` | Entropy CI + GMM + Lovell compliance |

---
## Dependencies

```
torch >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
scikit-learn >= 1.3.0
joblib >= 1.3.0
biopython >= 1.81
pyrama >= 0.4.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
streamlit >= 1.32.0
```

---

## Design decisions

**Why a differentiable GMM rather than sklearn.score_samples?**
sklearn requires `.detach().numpy()` which breaks the autograd graph. The
physics penalty then appears in the reported loss but its gradient never
reaches the decoder weights -- both models train identically. Reimplementing
the GMM log-prob as a logsumexp over quadratic forms in PyTorch fixes this
with ~30 lines of code.

**Why not oversample to balance classes?**
Ramachandran imbalance is physical reality. GMM oversampling generates density
in sterically forbidden inter-cluster voids, corrupting both the training signal
and the physics penalty. Confirmed empirically: balanced experiments showed no
physics advantage and worse Lovell compliance.

**Why fixed epochs rather than early stopping?**
VAE loss sums reconstruction + KL + physics penalty. Early stopping on the
total conflates annealing dynamics with convergence. Loss curves (Stage 0)
diagnose overfitting post-hoc.

**Why latent space perturbation rather than input space?**
The probe tests decoder stability and physical regularity, not input
sensitivity. Perturbing around z=0 asks: as I move away from the prior mean,
does the decoder stay in physically plausible regions?

**Why two compliance metrics?**
GMM compliance is circular -- it measures proximity to training data, which
the reconstruction loss already encourages. Lovell compliance uses boundaries
from an independent curated dataset (Top500 high-resolution structures) and
is the stronger claim for a physics-informed model.

---
