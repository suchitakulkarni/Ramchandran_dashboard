"""
dashboard.py

Streamlit dashboard for the Ramachandran Physics-Informed VAE project.

Tabs
----
1. Introduction   -- Ramachandran problem explainer
2. Training       -- pre-computed loss curves + generated sample PNGs
3. Perturbation   -- live: user sets sigma + n_samples, reruns, shows compliance
4. Statistics     -- Cohen's d table + plain-language summary

Run
---
    streamlit run dashboard.py
"""

import os, sys
import glob
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from pathlib import Path

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if root_path not in sys.path: sys.path.insert(0, root_path)

from utils.visualise import overlay_regions
import src.config as config
from src.train_vae import VAE
from src.model import TorchGMM, build_torch_gmm
from utils.rama_grid import  RamaGrid, build_and_save_rama_grid
from utils.ramachandran_regions import compliance_lovell, compliance_rate
from src.perturbation_analysis import sample_perturbations
import src.config as config

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ramachandran Physics VAE",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── minimal custom CSS ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* clean monospace accent for numbers */
    .metric-value { font-family: 'Courier New', monospace; font-size: 2rem;
                    font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 0.78rem; color: #666; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-top: 0.2rem; }
    .win-badge    { background: #d4edda; color: #155724; padding: 2px 8px;
                    border-radius: 4px; font-size: 0.82rem; font-weight: 600; }
    .lose-badge   { background: #f8d7da; color: #721c24; padding: 2px 8px;
                    border-radius: 4px; font-size: 0.82rem; font-weight: 600; }
    .neutral-badge{ background: #e2e3e5; color: #383d41; padding: 2px 8px;
                    border-radius: 4px; font-size: 0.82rem; font-weight: 600; }
    .section-rule { border: none; border-top: 2px solid #e8e8e8;
                    margin: 1.5rem 0; }
    /* tighten tab labels */
    button[data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

EXPERIMENT = "original"   # augmented results excluded by design decision

# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models_and_artifacts():
    """Load saved models, scalers, GMMs once and cache them."""
    csv_path, gmm_path = config.EXPERIMENTS[EXPERIMENT]

    if not os.path.exists(gmm_path):
        return None

    gmm       = joblib.load(gmm_path)
    #torch_gmm = build_torch_gmm(gmm)
    torch_gmm = build_and_save_rama_grid(gmm_path)

    models = {}
    for variant in ("baseline", "physics"):
        model_path = config.model_path(EXPERIMENT, variant)
        if os.path.exists(model_path):
            m = VAE()
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
            m.eval()
            models[variant] = m

    return {"gmm": gmm, "torch_gmm": torch_gmm,
            "models": models}


def run_perturbation(models, torch_gmm, sigma, n_samples):
    """
    Perturb around z=0 (prior mean) for each variant.
    Returns a dict with phi, psi arrays and compliance scalars per variant.
    """
    results = {}
    z_center = torch.zeros(1, config.LATENT_DIM)

    for variant, model in models.items():
        angles = sample_perturbations(model, sigma, n_samples)
        #with torch.no_grad():
        #    noise = torch.randn(n_samples, config.LATENT_DIM) * sigma
        #    z     = z_center + noise
        #    recon = model.decoder(z)   # (n_samples, 2) in [-1, 1]

        # inverse transform to raw angles
        #data_min   = torch.tensor(scaler.data_min_,   dtype=torch.float32)
        #data_range = torch.tensor(scaler.data_range_, dtype=torch.float32)
        #raw = ((recon + 1.0) / 2.0 * data_range + data_min).numpy()

        phi = angles[:, 0]
        psi = angles[:, 1]

        # GMM compliance
        gmm_compliance = compliance_rate(angles, torch_gmm)
        lovell_favoured, lovell_allowed = compliance_lovell(angles[:, 0], angles[:, 1])

        results[variant] = {
            "phi": phi, "psi": psi,
            "gmm_compliance":   gmm_compliance,
            "lovell_favoured":  lovell_favoured,
            "lovell_allowed":   lovell_allowed,
        }

    return results


def ramachandran_scatter(results, title_suffix=""):
    """Return a matplotlib figure with baseline | physics side by side."""
    colors = {"baseline": "#4C9BE8", "physics": "#E8594C"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, variant in zip(axes, ("baseline", "physics")):
        r   = results[variant]
        ax.scatter(r["phi"], r["psi"], s=12, alpha=0.55,
                   color=colors[variant], label=variant, rasterized=True)
        ax.axhline(0, color="#ccc", lw=0.7, ls="--")
        ax.axvline(0, color="#ccc", lw=0.7, ls="--")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("phi (degrees)", fontsize=9)
        ax.set_ylabel("psi (degrees)", fontsize=9)
        ax.set_title(f"{variant.capitalize()} VAE", fontsize=10, fontweight="bold",
                     color=colors[variant])

        # annotation
        lines = [f"GMM compliance: {r['gmm_compliance']:.2f}"]
        if r["lovell_favoured"] is not None:
            lines.append(f"Lovell favoured: {r['lovell_favoured']:.2f}")
            lines.append(f"Lovell allowed:  {r['lovell_allowed']:.2f}")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                fontsize=7.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        overlay_regions(ax, show_legend=True)

    fig.suptitle(f"Perturbation Samples -- {title_suffix}", fontsize=11,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def png_or_message(path):
    if os.path.exists(path):
        st.image(path, width='stretch')
    else:
        st.info(f"File not found: `{path}`  \nRun the pipeline first.")


# ── tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Introduction",
    "Training Results",
    "Live Perturbation",
    "Statistics",
    "Future developments"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Introduction
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.title("Teaching Machines the Grammar of Proteins")
    st.header("Can physics-informed generative models learn what data alone cannot?")
    st.markdown("""
    Proteins fold into shapes governed by physics, not just statistics. Yet generative models 
    treat protein structure as a pure data problem and fail in predictable ways.

    This dashboard asks a simple question: what happens when you build the physics in from the start?

    We compare a standard data-driven generative model against a physics-informed approach on a concrete task: 
    generating protein backbone conformations that respect known structural constraints. The results show
    a measurable, statistically significant difference — and reveal exactly where and why data alone is not enough.
    """)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("The Ramachandran Problem")

    col_text, col_img = st.columns([1, 1])
    with col_text:
        st.markdown("""
        Every amino acid in a protein backbone has two rotatable bonds described
        by dihedral angles **phi** and **psi**. Not all angle combinations are
        possible -- large regions of (phi, psi) space are forbidden
        because atoms would clash.

        The **Ramachandran plot** maps these angles for a set of high-resolution
        crystal structures. It reveals three main allowed regions:

        - **Alpha-helix** region: phi ~ -60, psi ~ -40
        - **Beta-sheet** region: phi ~ -120, psi ~ +130
        - **Left-handed helix**: phi ~ +60, psi ~ +40 (rare, glycine only)

        A generative model for protein conformations must respect these boundaries.
        A naive model trained on angle data will often collapse to a single region
        or generate angles in forbidden space.
        """)
    st.subheader("Proteins in the Dataset")
    st.markdown("Five structurally diverse proteins spanning the major Ramachandran regions.")

    proteins = [
        ("1BRS", "Barnase-Barstar", "Protein-protein recognition complex, alpha/beta"),
        ("1TIM", "Triosephosphate isomerase", "Canonical TIM-barrel enzyme fold"),
        ("2LZM", "T4 phage lysozyme", "Alpha-helical, protein engineering workhorse"),
        ("1UBQ", "Ubiquitin", "Compact mixed alpha/beta, 76 residues"),
        ("1VII", "Villin headpiece", "35-residue fast-folding three-helix bundle"),
    ]

    cols = st.columns(len(proteins))
    for col, (pdb_id, name, description) in zip(cols, proteins):
        url = f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg"
        col.image(url, width='stretch')
        col.markdown(f"**{pdb_id}** — {name}")
        col.caption(description)

    with col_img:
        stage1_path = os.path.join(
            config.RESULTS_DIR, f"plots/stage1_{EXPERIMENT}_training_data_st.png"
        )
        png_or_message(stage1_path)
        st.caption("Training data distribution in Ramachandran space.")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Our Approach")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**GMM Energy Landscape**")
        st.markdown("""
        We use the Richardson Lab Top500 reference density (Lovell et al. 2003), general case
        (non-Gly, non-Pro, non-pre-Pro), B-factor < 30, available via pyrama v2.0.2 which ships
        the pre-computed density grid directly. The Ramachandran energy landscape is defined via
        the log-probability of this density, accessed through bilinear interpolation at decoded
        (phi, psi) coordinates. High log-prob = low energy = physically favourable region.
        """)
    with col_b:
        st.markdown("**Differentiable Physics Penalty**")
        latext = r'''
        $$ 
        \mathcal{L}_{physics}​= \frac{1}{N}\sum clip(−logpTop500​(\phi_i​,\psi_i​), 0, C)
        $$ 
        
        $KT$ is a hyperparameter and $C$ is a hyperparameter for gradient clipping 
        '''
        st.markdown(
        """
        The GMM is reimplemented in PyTorch so the gradient flows end-to-end
        through the decoder. The penalty is the mean clamped negative
        log-probability of decoded samples under the GMM.
        """)
        st.write(latext)
    with col_c:
        st.markdown("**Validation**")
        st.markdown("""
        Compliance is evaluated against **Lovell et al. (2003) Top500**
        favoured/allowed region boundaries providing an unbiased physical plausibility score.
        """)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Key Findings")
    st.markdown("""
    | Finding | Result |
    |---------|--------|
    | Physics VAE phi stability | **100% win rate**, Cohen's d = 0.905 (large effect) |
    | Physics VAE psi behaviour | Destabilised -- anisotropic landscape effect |
    | GMM fitting to trainiing data | Anchors the physics prior to data, leads to bias in generation |
    | Lovell allowed compliance | **95%** on original data |
    | Posterior collapse | Observed in physics inspired approach, better tuning is needed|
    | GMM oversampling | Harmful -- generates density in sterically forbidden voids |

    The asymmetric phi/psi result is a genuine physical observation: phi is
    subject to steeper steric clash gradients, so the physics penalty anchors
    phi while the decoder uses psi as a high-variance residual for reconstruction.
    Fixing this requires decoupled, per-angle penalties -- a clear next step.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Training Results
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Training Results")
    st.markdown("""
    This tab contains four plots corresponding to the four stages of the 
    training and validation cycle.

    The first plot shows contour levels of the Top500 Ramachandran reference 
    density (Richardson Lab, Lovell et al. 2003). Darker colors correspond to 
    higher probability of occurrence in nature. The overlaid training data shows 
    a slight shift to larger phi angles in the lower cluster, demonstrating the 
    dataset-specific bias that a purely data-driven method would inherit.

    The second plot shows the loss curves during training and validation.

    The third plot shows the capability of the trained models to reconstruct 
    the training data. The data-driven baseline achieves smaller coordinate 
    residuals (phi std=12.55 deg, psi std=11.72 deg) but 11% of reconstructed 
    points fall outside the Lovell favoured region and 6% are outside the 
    allowed region. The physics-informed model stays entirely within the Lovell 
    allowed region (favoured=1.00, allowed=1.00), at the cost of larger 
    coordinate residuals (phi std=45.07 deg, psi std=30.20 deg). This tradeoff 
    reflects the model prioritising physical validity over positional fidelity.

    The fourth plot shows generated samples, new predicted backbone 
    conformations corresponding to plausible protein structures. The 
    data-driven baseline places 25% of generated points outside the Lovell 
    favoured region and 15% outside the allowed region. The physics-informed 
    model generates 0% outliers under both Lovell criteria. 
    """)
    st.caption("All plots are pre-computed. Re-run `train_vae.py` to regenerate.")

    st.subheader("GMM Energy Landscape")
    png_or_message(
        os.path.join(config.RESULTS_DIR, f"plots/stage2_{EXPERIMENT}_gmm_quality.png")
    )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    st.subheader("Learning Curves")
    png_or_message(
        os.path.join(config.RESULTS_DIR, f"plots/stage0_{EXPERIMENT}_learning_curves.png")
    )


    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Reconstruction Quality")
    png_or_message(
        os.path.join(config.RESULTS_DIR, f"plots/stage3_{EXPERIMENT}_reconstruction.png")
    )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.subheader("Generated Samples from Prior")
    st.markdown("""
    500 samples drawn from N(0, I) and decoded. Lovell compliance scores are
    independent of the training data.
    """)
    png_or_message(
        os.path.join(config.RESULTS_DIR, f"plots/stage4_{EXPERIMENT}_generated_samples.png")
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Live Perturbation
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Live Perturbation Analysis")
    st.markdown("""
    Perturb around the prior mean (z = 0) and decode. Controls how far into
    latent space you move and how many samples you draw. Both variants run
    simultaneously so results are directly comparable.
    """)

    artifacts = load_models_and_artifacts()

    if artifacts is None or not artifacts["models"]:
        st.error(
            "Saved models not found. Run `train_vae.py` first to generate "
            f"`models/vae_{EXPERIMENT}_baseline.pt` and "
            f"`models/vae_{EXPERIMENT}_physics.pt`."
        )
    else:
        col_ctrl, col_info = st.columns([1, 2])
        with col_ctrl:
            sigma = st.slider(
                "Perturbation sigma",
                min_value=0.01, max_value=2.0,
                value=0.1, step=0.01,
                help="Standard deviation of Gaussian noise added to z=0. "
                     "Small sigma = local neighbourhood; large sigma = prior sampling."
            )
            n_samples = st.slider(
                "Number of samples",
                min_value=50, max_value=1000,
                value=200, step=50,
                help="More samples = smoother scatter, slower render."
            )
            run_btn = st.button("Run perturbation", type="primary")

        with col_info:
            st.markdown("""
            **What this shows**

            Starting from the prior mean (the most probable latent point),
            we add Gaussian noise of magnitude sigma and ask: does the decoder
            stay in physically plausible Ramachandran regions?

            - Small sigma tests **local decoder stability**
            - Large sigma approaches **prior sampling** (same as Stage 4)
            - Compliance scores update with each run
            """)

        if run_btn:
            with st.spinner("Running perturbation..."):
                results = run_perturbation(
                    models    = artifacts["models"],
                    torch_gmm = artifacts["torch_gmm"],
                    sigma     = sigma,
                    n_samples = n_samples,
                )
            # compliance metric cards
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.subheader("Compliance Scores")

            cols = st.columns(4)
            metrics = [
                ("GMM compliance",    "gmm_compliance"),
                ("Lovell favoured",   "lovell_favoured"),
                ("Lovell allowed",    "lovell_allowed"),
            ]


            # distribution cards for phi and psi
            with cols[3]:
                b_phi_std = float(results["baseline"]["phi"].std())
                p_phi_std = float(results["physics"]["phi"].std())
                delta_phi = p_phi_std - b_phi_std
                badge = "win-badge" if delta_phi < 0 else "lose-badge"
                st.markdown(f"""
                <div>
                    <div class="metric-label">phi std (physics lower = win)</div>
                    <div class="metric-value">{p_phi_std:.1f} deg</div>
                    <div style="font-size:0.8rem; color:#666;">
                        baseline: {b_phi_std:.1f} deg &nbsp;
                        <span class="{badge}">
                            {'+ ' if delta_phi >= 0 else ''}{delta_phi:.1f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # scatter plot
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.subheader("Ramachandran Scatter")
            fig = ramachandran_scatter(
                results,
                title_suffix=f"sigma={sigma:.2f}, n={n_samples}"
            )
            st.pyplot(fig)
            plt.close(fig)

            # phi / psi distribution comparison
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.subheader("Angle Distributions")
            fig2, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            colors = {"baseline": "#4C9BE8", "physics": "#E8594C"}

            for angle, col_idx in [("phi", 0), ("psi", 1)]:
                ax = axes[col_idx]
                for variant in ("baseline", "physics"):
                    vals = results[variant][angle]
                    ax.hist(vals, bins=40, alpha=0.55, color=colors[variant],
                            label=variant, density=True)
                ax.set_xlabel(f"{angle} (degrees)", fontsize=9)
                ax.set_ylabel("density", fontsize=9)
                ax.set_title(angle, fontsize=10, fontweight="bold")
                ax.legend(fontsize=8)

            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        else:
            st.info("Set sigma and n_samples, then click **Run perturbation**.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Statistics
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.header("Statistical Summary (from offline results)")
    
        stage1_path = os.path.join(
            config.RESULTS_DIR, f"plots/compliance_original.png"
        )
        png_or_message(stage1_path)
        st.caption("Compliance figures over five runs")

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # ── compliance summary ─────────────────────────────────────────────
        st.subheader("Compliance Metrics (scalar delta)")
        st.markdown("""
        For a dataset of about 2k samples, the physics informed appraoch tolerates perturbations up to sigma=0.5 before Ramachandran quality degrades, versus sigma=0.2 for baseline
        """)

        comp_display = df_comp[["metric", "win_rate", "mean_delta", "n_sigmas_tested"]].copy()
        comp_display.columns = ["Metric", "Win rate", "Mean delta", "Sigma levels"]
        comp_display["Win rate"] = comp_display["Win rate"].apply(lambda x: f"{x*100:.0f}%")
        comp_display["Mean delta"] = comp_display["Mean delta"].apply(
            lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}"
        )
        st.dataframe(comp_display, width='stretch', hide_index=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # ── per-sigma detail ───────────────────────────────────────────────
        if df_det is not None:
            with st.expander("Per-sigma detail table"):
                df_det_orig = df_det[df_det["experiment"] == EXPERIMENT].copy()
                st.dataframe(df_det_orig, width='stretch', hide_index=True)

        # ── plain-language summary ─────────────────────────────────────────
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.subheader("Plain-language summary")

        st.markdown("""
        **phi angle (steric constraint)**
        The physics VAE produces significantly tighter phi distributions under
        perturbation -- 100% win rate across sigma levels with a large effect size
        (Cohen's d = 0.905, 83% of tests p < 0.05). The physics penalty
        effectively anchors the backbone's primary steric degree of freedom.

        **psi angle (hydrogen-bond constraint)**
        The physics VAE destabilises psi relative to baseline. This is an
        anisotropic landscape effect: the Ramachandran energy gradient is steeper
        in phi than psi, so the joint penalty is dominated by phi.
        The decoder compensates by using psi as a high-variance residual channel
        to minimise reconstruction loss. Decoupled per-angle penalties are the
        direct fix.

        **Compliance**
        GMM compliance is unchanged (by construction -- the penalty is the GMM
        energy, so both models learn to stay inside it). Lovell compliance is
        negative for the same reason as psi instability. The 95% allowed rate
        reported for the full generated sample set is the headline number.

        **Conclusion**
        Physics regularisation demonstrably improves conformational stability
        for the sterically-dominated degree of freedom. The asymmetric result
        is a mechanistic finding, not a failure -- it motivates the next
        iteration of the model.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Next steps
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.subheader("Future developments")

    st.markdown(""" **Physics prior improvements**
    The most immediate next step is temperature scaling at inference time, 
    applying T greater than 1 to the learned energy landscape at generation 
    to encourage exploration of low-density but physically valid regions 
    without retraining. This directly addresses the mode collapse and the failure
    to populate the small Lovell islands. Separately, replacing the Top500 reference
    density with a CMAP-based first-principles torsional potential would make 
    the physics constraint genuinely physics-derived rather than empirically grounded. 
    This requires residue-type labels and selective application, only to residues 
    where the alanine dipeptide approximation is valid, and does not fix the generative bias
    without the architectural change described below.
    

    **Architecture**
    Moving to a conditional VAE where residue type is an explicit decoder input would make 
    the generative distribution residue-type-aware, fixing the structural bias that CMAP 
    introduces into generation. This is the prerequisite for using first-principles constraints correctly. 
    Separately, replacing the standard Gaussian prior with a GMM prior in latent space would encourage 
    multimodal latent structure that mirrors the multimodal Ramachandran landscape, potentially addressing 
    mode collapse at a more fundamental level than temperature scaling

    **Network tuning**
    The current results are from an untuned network. A systematic hyperparameter sweep covering latent dimension, 
    beta annealing schedule, physics loss weight, and learning rate would establish whether the performance gap 
    between baseline and physics VAE holds under fair comparison conditions. The Lovell compliance gap is large 
    enough that tuning is unlikely to close it entirely, but the reconstruction residual gap should be characterised 
    under tuned conditions before making strong claims.

    **Evaluation and robustness certificate**
    The small Lovell island population check, whether the physics VAE ever generates samples in 
    the left-handed helix and minor beta regions, is the single most drug-discovery-relevant experiment remaining. 
    Running a latent space grid search rather than random sampling would reveal whether these regions exist 
    anywhere in the learned latent space. The robustness certificate should be extended to report a single 
    robustness radius per generated structure, defined as the sigma at which Lovell compliance first degrades, 
    as a scalar summary suitable for lab decision making.

    **Dataset and generalisation**
    Training on a more structurally diverse dataset that better represents all five Lovell islands would separate 
    the data bias contribution from the physics constraint contribution in the results, directly addressing 
    the strongest skeptical critique. This does not require the Top500 PDB pipeline but rather deliberate 
    curation of structures representing each island rather than adding more helical proteins.

    **Comparison with SOTA**
    A direct comparison with a diffusion or flow matching baseline under the same evaluation framework, 
    covering Lovell compliance, robustness radius, and small island population rate, would position 
    the methodological contribution cleanly relative to current architecture trends and show that 
    the physics prior question is architecture-agnostic.
    """)
