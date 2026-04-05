import os,sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from pathlib import Path

try:
    root_path = Path(os.environ["PROJECT_ROOT"])
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
except: 
    print("please set the PROJECT_ROOT variable corresponding to the project root directory in the enviornment")
    sys.exit()

from src.perturbation_analysis import run_perturbation_analysis
from src.train_vae import run_training
from utils.utils import setup_logging
from utils.utils import load_data, fit_gmm
from utils.rama_grid import  RamaGrid, build_and_save_rama_grid
from src.model import build_torch_gmm, vae_base_loss
from src.perturbation_analysis import load_experiment_artifacts, sample_perturbations, compliance_rate
from utils.visualise import (load_experiment_data, plot_learning_curves, 
                            plot_training_data, plot_gmm_quality, plot_reconstruction, 
                            plot_generated_samples, plot_perturbation_scatter, plot_entropy_and_compliance, run_stats)
import src.config as config

logger = logging.getLogger(__name__)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


def main():
    logger.info("main started")
    for experiment, (csv_path, gmm_path) in config.EXPERIMENTS.items():
        if not os.path.exists(csv_path):
            logger.warning(f"\n {csv_path} does not exist, will skip the experiment")
            continue

        logger.info(f"=== Experiment: {experiment} ===")
        angles, angles_4D = load_data(csv_path)
        #gmm = fit_gmm(angles_raw, gmm_path)
        #torch_gmm = build_torch_gmm(gmm)
        torch_gmm = build_and_save_rama_grid(gmm_path)

        run_training(False, angles_4D, torch_gmm, config.model_path(experiment, "baseline"), experiment)
        run_training(True, angles_4D, torch_gmm, config.model_path(experiment, "physics"), experiment)

    logger.info("All experiments done.")
    
    run_perturbation_analysis()

    samples_path    = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_samples.csv")
    compliance_path = os.path.join(config.RESULTS_DIR, "datafiles/perturbation_compliance.csv")

    df_samples    = pd.read_csv(samples_path)    if os.path.exists(samples_path)    else None
    df_compliance = pd.read_csv(compliance_path) if os.path.exists(compliance_path) else None

    for experiment, (csv_path, gmm_path) in config.EXPERIMENTS.items():
        logger.info(f"Running experiment %s  with csv path %s and gmm path %s", experiment, csv_path, gmm_path)
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found for '%s' -- skipping", experiment)
            continue
        

        logger.info(f"=== Plotting: {experiment} ===")
        df                                            = load_experiment_data(experiment)
        gmm, model_baseline, model_physics    = load_experiment_artifacts(experiment)

        logger.info("  Stage 0: learning curves")
        plot_learning_curves(experiment)

        logger.info("  Stage 1: training data")
        plot_training_data(experiment, df)

        logger.info("  Stage 2: GMM quality")
        plot_gmm_quality(experiment, df, gmm)

        logger.info("  Stage 3: reconstruction")
        plot_reconstruction(experiment, df,  model_baseline, model_physics)

        logger.info("  Stage 4: generated samples")
        plot_generated_samples(experiment, df,  model_baseline, model_physics)

        if df_samples is not None:
            logger.info("  Stage 5: perturbation scatter")
            plot_perturbation_scatter(experiment, df_samples)

            if df_compliance is not None:
                logger.info("  Stage 6: entropy + compliance")
                plot_entropy_and_compliance(experiment, df_samples, df_compliance)
        else:
            logger.info("  Stages 5-6: perturbation_samples.csv not found -- run perturbation_analysis.py")

    run_stats()
    logger.info(f"All plots saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    main()
