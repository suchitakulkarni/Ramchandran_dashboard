import os

# --- paths ---
DATA_DIR    = "data"
MODEL_DIR   = "models"
RESULTS_DIR = "results"
PDB_DIR     = os.path.join(DATA_DIR, "pdb_files")

# --- data ---
ORIGINAL_PDB_IDS  = ["1UBQ", "1VII", "2LZM", "1BRS", "1TIM"]
AUGMENTED_PDB_IDS = ["1UBQ", "1VII", "2LZM", "1BRS", "1TIM", "1IGT", "2ACE", "1TEN", "1POH"]

CSV_ORIGINAL  = os.path.join(DATA_DIR, "ramachandran_original.csv")
CSV_AUGMENTED = os.path.join(DATA_DIR, "ramachandran_augmented.csv")

CSV_ORIGINAL_BALANCED  = os.path.join(DATA_DIR, "ramachandran_original_balanced.csv")
CSV_AUGMENTED_BALANCED = os.path.join(DATA_DIR, "ramachandran_augmented_balanced.csv")

SCALER_ORIGINAL  = os.path.join(MODEL_DIR, "scaler_original.pkl")
SCALER_AUGMENTED = os.path.join(MODEL_DIR, "scaler_augmented.pkl")
SCALER_ORIGINAL_BALANCED  = os.path.join(MODEL_DIR, "scaler_original_balanced.pkl")
SCALER_AUGMENTED_BALANCED = os.path.join(MODEL_DIR, "scaler_augmented_balanced.pkl")

GMM_ORIGINAL  = os.path.join(MODEL_DIR, "gmm_original.pkl")
GMM_AUGMENTED = os.path.join(MODEL_DIR, "gmm_augmented.pkl")
GMM_ORIGINAL_BALANCED  = os.path.join(MODEL_DIR, "gmm_original_balanced.pkl")
GMM_AUGMENTED_BALANCED = os.path.join(MODEL_DIR, "gmm_augmented_balanced.pkl")

# -- training --
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 20

# ceiling on the -log_prob penalty per sample
# in-distribution log_probs are typically in [-4, -2]
# anything below -10 is deep in forbidden space -- cap it there
# to prevent a single outlier batch dominating early training
PHYSICS_LOG_PROB_CEIL = 10.0

# --- oversampling ---
# CV_THRESHOLD: coefficient of variation stopping criterion for density-based oversampling.
# Computed only over populated Ramachandran regions (see KDE_MIN_DENSITY_PERCENTILE).
# Lower = more uniform density = more synthetic points generated.
# Recommended range: 0.2 - 0.5. Default: 0.3
CV_THRESHOLD = 0.3

# KDE_MIN_DENSITY_PERCENTILE: grid cells below this percentile of nonzero KDE density
# are treated as forbidden/unpopulated and excluded from CV calculation.
# Default: 10
KDE_MIN_DENSITY_PERCENTILE = 10

# JITTER_STD: standard deviation in degrees of Gaussian jitter applied to synthetic points.
# Should be small enough not to generate physically implausible angles.
# Default: 2.0 degrees
JITTER_STD = 2.0

# MAX_OVERSAMPLE_ITER: safety cap on oversampling iterations to prevent infinite loops.
MAX_OVERSAMPLE_ITER = 50

# --- GMM ---
N_GMM_COMPONENTS = 5
GMM_COMPLIANCE_THRESHOLD = -5.0   # GMM log-prob cutoff for "allowed" region
                                  # -- tune this by looking at Stage 2 plots
# KT: Boltzmann temperature parameter for physics constraint.
# Lower = sharper penalty on implausible configurations.
# Default: 1.0
KT = 1

# --- VAE architecture ---
LATENT_DIM  = 2
HIDDEN_DIM  = 32

# --- VAE training ---
EPOCHS     = 2000
BATCH_SIZE = 64
LR         = 5e-4
KL_WEIGHT  = 0.01

# PHYSICS_WEIGHT: weight of the Boltzmann penalty term in the physics-constrained VAE loss.
# Higher = stronger physics constraint. Default: 0.5
PHYSICS_WEIGHT = 0.005 # 0.005
WARMUP_EPOCHS = 200
SEED = 42

# --- perturbation analysis ---
PERTURBATION_SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
N_PERTURBATIONS     = 200
N_GENERATED_SAMPLES = 500

# --- experiment registry ---
# Maps experiment name to (csv_path, scaler_path, gmm_path)
EXPERIMENTS = {
    "original":           (CSV_ORIGINAL,           SCALER_ORIGINAL,           GMM_ORIGINAL),
    #"original_balanced":  (CSV_ORIGINAL_BALANCED,   SCALER_ORIGINAL_BALANCED,  GMM_ORIGINAL_BALANCED),
    #"augmented":          (CSV_AUGMENTED,            SCALER_AUGMENTED,          GMM_AUGMENTED),
    #"augmented_balanced": (CSV_AUGMENTED_BALANCED,   SCALER_AUGMENTED_BALANCED, GMM_AUGMENTED_BALANCED),
}

# Model filename convention: models/vae_{experiment}_{variant}.pt
# variant is "baseline" or "physics"
def model_path(experiment, variant):
    return os.path.join(MODEL_DIR, f"vae_{experiment}_{variant}.pt")
