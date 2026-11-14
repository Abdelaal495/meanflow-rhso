"""Configuration: registries, strict validation, sweep expansion, and the run planner.

Carried over from the MPC-Flow notebook (sections 2-4) with two deliberate changes:

  * the configuration lives in `configs/experiments.yaml`, not in a notebook cell;
  * `standard_flow:` / `meanflow:` are no longer user-facing branches.  A model's dynamics
    family is a property of the model, so the user writes `models: {jit: ..., pmf: ...}`
    and the registry decides the family.

Unknown keys are errors rather than silent no-ops, and everything here runs *before* any
checkpoint is downloaded.

Six reconstruction strategies are declared here: SDEdit, MPC-RHC, MPC-delta_t, PnP-Flow,
D-Flow and RHSO.  PnP and D-Flow were added later; see `docs/methods_pnp_dflow.md` for what
is published and what is a research extension, and `docs/schedule_and_rhso.md` for RHSO.
MPC's defaults are deliberately untouched by those additions: `phi_normalization` still
defaults to `half_sum_squared` for MPC, so every Table E2 lambda keeps the meaning it had.

`beta` is a SHARED field alongside `t0`: it is the exponent of the universal power-law time
schedule s_k = s0 + (1 - s0)(k/N)^beta (see src/schedule.py).  It defaults to 1.0, which is
the legacy uniform grid bitwise, so a configuration that never mentions it resolves exactly
as it did before.
"""

from __future__ import annotations

import copy
import datetime
import itertools
import math
import os
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .rhso_diagnostics import (DEFAULT_JACOBIAN_POWER_ITERATIONS, DEFAULT_JACOBIAN_PROBES,
                               DEFAULT_JACOBIAN_SEED)
from .schedule import (DEFAULT_BETA, canonical_time_grid, grid_intervals,
                       nominal_uniform_delta, resolve_beta)
from .utils import (FLOW_ASCENDING, MEANFLOW, MEANFLOW_DESCENDING, STANDARD_FLOW,
                    SEED_RECIPES, as_sweep_list, canonical_params_key, canonical_start_time,
                    deep_merge, native_time, stable_hash)


class ConfigError(ValueError):
    """Raised for configurations that cannot produce a meaningful run."""


# =====================================================================================
# Model registry -- technical definitions, carried over unchanged from the notebooks
# =====================================================================================
MODEL_REGISTRY_DEFAULTS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------ iMeanFlow (JAX/latent)
    "imf": {
        "display_name": "iMF",
        "dynamics_family": MEANFLOW,
        "framework": "jax",
        "state_space": "latent",
        "native_shape": (32, 32, 4),           # H, W, C
        "layout": "BHWC",
        "pixel_resolution": 256,
        "prediction_kind": "mean_velocity",
        "native_time_mapping": MEANFLOW_DESCENDING,

        "repo_url": "https://github.com/Lyy-iiis/imeanflow.git",
        "repo_rev": "bf60cd7cb653f6628e59d48034b333c5eba445e2",
        "repo_dirname": "imeanflow",
        "hf_repo": "Lyy0725/iMF",
        "ckpt_file": "iMF-B-2.zip",
        "model_str": "imfDiT_B_2",             # architecture named in the repository; iMF-XL/2 = imfDiT_XL_2
        "config_yml": "configs/eval_config.yml",
        "use_ema": True,                       # single EMA pytree (contrast with pMF)

        "vae_type": None,                      # None -> config.dataset.vae ("mse")
        "vae_encode_mode": "sample",
        "vae_decode_batch": 8,
        # Per-channel affine normalisation from utils/vae_util.LatentManager -- NOT 0.18215.
        "latent_mean": (0.86488, -0.27787343, 0.21616915, 0.3738409),
        "latent_std": (4.85503674, 5.31922414, 3.93725398, 3.9870003),

        "batch_size": 2,                       # FIXED -> one compiled shape
        "fixed_batch_shape": True,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": 8.0, "interval": (0.40, 0.65), "mode": None},
        "packages": ("jax_stack", "diffusers", "flax", "orbax"),
        "accelerators": ("tpu", "gpu", "cpu"),
        "default_solver": None,
    },

    # ------------------------------------------------------------------ pixel MeanFlow (JAX)
    "pmf": {
        "display_name": "pMF",
        "dynamics_family": MEANFLOW,
        "framework": "jax",
        "state_space": "pixel",
        "native_shape": (256, 256, 3),
        "layout": "BHWC",
        "pixel_resolution": 256,
        "prediction_kind": "mean_velocity",
        "native_time_mapping": MEANFLOW_DESCENDING,

        "repo_url": "https://github.com/Lyy-iiis/pMF.git",
        "repo_rev": "75f6073042c21f7104686261a0c4784db4ede9d1",
        "repo_dirname": "pMF",
        "hf_repo": "Lyy0725/pMF",
        "ckpt_file": "pMF-L-16.zip",
        "config_yml": "configs/pMF_L_16_config.yml",
        "ema_key": 1000,                       # ema_params is a DICT keyed by EMA value

        "noise_scale": 1.0,
        "batch_size": 2,
        "fixed_batch_shape": True,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": 7.0, "interval": (0.20, 0.70), "mode": None},
        "packages": ("jax_stack", "flax", "orbax"),
        "accelerators": ("tpu", "gpu", "cpu"),
        "default_solver": None,
    },

    # ------------------------------------------------------------------ SiT (Torch/latent)
    "sit": {
        "display_name": "SiT",
        "dynamics_family": STANDARD_FLOW,
        "framework": "torch",
        "state_space": "latent",
        "native_shape": (4, 32, 32),            # C, H, W
        "layout": "BCHW",
        "pixel_resolution": 256,
        "prediction_kind": "velocity",
        "native_time_mapping": FLOW_ASCENDING,

        "repo_url": "https://github.com/willisma/SiT.git",
        "repo_rev": None,
        "repo_dirname": "SiT",
        "variant": "SiT-XL/2",
        "checkpoint": "official",
        "vae_id": "stabilityai/sd-vae-ft-mse",
        "vae_encode_mode": "mean",
        "dtype": "auto",                        # bf16 where supported, else fp16, cpu -> fp32

        "batch_size": 2,
        "fixed_batch_shape": False,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": 4.0, "interval": None, "mode": "official_first3"},
        "packages": ("torch_stack", "diffusers", "timm"),
        "accelerators": ("gpu", "cpu"),
        "default_solver": "heun",
        "euler_final_step_for_heun": False,
    },

    # ------------------------------------------------------------------ JiT (Torch/pixel)
    "jit": {
        "display_name": "JiT",
        "dynamics_family": STANDARD_FLOW,
        "framework": "torch",
        "state_space": "pixel",
        "native_shape": (3, 256, 256),
        "layout": "BCHW",
        "pixel_resolution": 256,
        "prediction_kind": "clean",             # predicts x_1; velocity is derived
        "native_time_mapping": FLOW_ASCENDING,

        "repo_url": "https://github.com/LTH14/JiT.git",
        "repo_rev": None,
        "repo_dirname": "JiT",
        "variant": "JiT-B/16",
        "checkpoint_backend": "hf_mirror",      # "hf_mirror" | "original_local"
        "hf_mirror_repo": "BiliSakura/JiT-diffusers",
        "local_checkpoint": None,
        "dtype": "safe_auto",                   # BF16 where supported, else FP32 (never FP16)
        "integration_dtype": "float32",         # solver/MPC state stays FP32 (official)
        "t_eps": 5e-2,                          # velocity denominator stabiliser near t = 1
        "noise_scale": 1.0,
        "recommended_cfg": {"JiT-B/16": 3.0, "JiT-L/16": 2.4, "JiT-H/16": 2.2},

        "batch_size": 2,
        "fixed_batch_shape": False,
        "num_classes": 1000,
        "null_label": 1000,
        "guidance": {"scale": None, "interval": (0.1, 1.0), "mode": None},  # None -> recommended
        "packages": ("torch_stack", "diffusers", "timm"),
        "accelerators": ("gpu", "cpu"),
        "default_solver": "heun",
        "euler_final_step_for_heun": True,      # official JiT sampler policy
    },
}

MODEL_NAMES: Tuple[str, ...] = tuple(MODEL_REGISTRY_DEFAULTS)


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    dynamics_family: str
    framework: str
    state_space: str
    prediction_kind: str
    supports_encoding: bool            # required for any t0 < 1 job (guide -> native space)
    supported_accelerators: Tuple[str, ...]
    fixed_batch_shape: bool
    native_time_mapping: str
    supported_solvers: Tuple[Optional[str], ...] = (None,)
    supported_methods: Tuple[str, ...] = ("sdedit", "mpc_rhc", "mpc_delta_t", "pnp",
                                          "dflow", "rhso")

    def describe(self) -> str:
        solvers = [s for s in self.supported_solvers if s]
        return ("%-4s dynamics=%-13s fw=%-5s space=%-6s predicts=%-13s batch=%-5s%s"
                % (self.name, self.dynamics_family, self.framework, self.state_space,
                   self.prediction_kind, "fixed" if self.fixed_batch_shape else "free",
                   ("  solvers=%s" % solvers) if solvers else ""))


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "imf": ModelCapabilities("imf", MEANFLOW, "jax", "latent", "mean_velocity", True,
                             ("tpu", "gpu", "cpu"), True, MEANFLOW_DESCENDING, (None,)),
    "pmf": ModelCapabilities("pmf", MEANFLOW, "jax", "pixel", "mean_velocity", True,
                             ("tpu", "gpu", "cpu"), True, MEANFLOW_DESCENDING, (None,)),
    "sit": ModelCapabilities("sit", STANDARD_FLOW, "torch", "latent", "velocity", True,
                             ("gpu", "cpu"), False, FLOW_ASCENDING,
                             ("euler", "heun", "rk4")),
    "jit": ModelCapabilities("jit", STANDARD_FLOW, "torch", "pixel", "clean", True,
                             ("gpu", "cpu"), False, FLOW_ASCENDING,
                             ("euler", "heun", "rk4")),
}
# Flower's refinement is a proximal step on the IMAGE, so only the pixel-space priors offer
# it; SiT and iMF (latent) keep their entries exactly as they were.
for _pixel_model in ("pmf", "jit"):
    MODEL_CAPABILITIES[_pixel_model] = replace(
        MODEL_CAPABILITIES[_pixel_model],
        supported_methods=MODEL_CAPABILITIES[_pixel_model].supported_methods + ("flower",))
del _pixel_model


# =====================================================================================
# Inverse-problem declarations
# =====================================================================================
@dataclass(frozen=True)
class ProblemDeclaration:
    name: str
    title: str
    required_params: Tuple[str, ...]
    optional_params: Tuple[str, ...]
    guide_mode: str                      # ONE guide per problem; see README
    default_params: Dict[str, Any]
    description: str = ""


PROBLEM_DECLARATIONS: Dict[str, ProblemDeclaration] = {
    "denoising": ProblemDeclaration(
        "denoising", "Denoising", ("sigma",), (), "identity",
        {"sigma": 0.20},
        "A = I; y = x* + eta.  Guide g(y) = y."),
    "deblur": ProblemDeclaration(
        "deblur", "Gaussian deblurring", ("sigma", "blur_sigma"),
        ("kernel_size", "padding"), "observed",
        {"sigma": 0.05, "blur_sigma": 1.0, "kernel_size": 7, "padding": "reflect"},
        "Separable Gaussian blur.  Kernel SUPPORT is a choice here, not a paper value."),
    "super_resolution": ProblemDeclaration(
        "super_resolution", "Super-resolution x2", ("sigma", "factor"), (),
        "upsample_bicubic",
        {"sigma": 0.05, "factor": 2},
        "A keeps every second pixel; the guide bicubically lifts y back to 256x256."),
    "box_inpaint": ProblemDeclaration(
        "box_inpaint", "Box inpainting", ("sigma", "box"), (), "zero_fill",
        {"sigma": 0.05, "box": 40},
        "Central box removed.  Guide is the masked observation; no classical prefill."),
    "random_inpaint": ProblemDeclaration(
        "random_inpaint", "Random inpainting", ("sigma", "missing_fraction"), (), "zero_fill",
        {"sigma": 0.01, "missing_fraction": 0.70},
        "Pixels dropped at random.  Guide is the masked observation; no classical prefill."),
    "stroke_painting": ProblemDeclaration(
        "stroke_painting", "Stroke painting -> image", (), ("preset", "sigma", "blur_sigma"),
        "observed",
        {"preset": "medium", "sigma": 0.0, "blur_sigma": 0.8},
        "Frozen SLIC stroke geometry rendered differentiably; y = A_G(x*), no added noise."),
}

PROBLEM_NAMES: Tuple[str, ...] = tuple(PROBLEM_DECLARATIONS)
MASK_PROBLEMS: Tuple[str, ...] = ("box_inpaint", "random_inpaint")
STROKE_PRESETS: Dict[str, Dict[str, int]] = {
    "sparse": dict(n_segments=80, compactness=15, stroke_width=10),
    "medium": dict(n_segments=200, compactness=10, stroke_width=6),
    "dense": dict(n_segments=500, compactness=8, stroke_width=3),
}


# =====================================================================================
# Method declarations
# =====================================================================================
@dataclass(frozen=True)
class MethodDeclaration:
    name: str
    title: str
    is_mpc: bool
    uses_K: bool
    fields: Tuple[str, ...]
    description: str


# `beta` is shared exactly like `t0`: every method places its times on the same power-law
# grid, and beta = 1 reproduces the uniform schedule bitwise.
SHARED_FIELDS: Tuple[str, ...] = ("t0", "beta")
SDEDIT_FIELDS: Tuple[str, ...] = ("steps", "solver")
MPC_COMMON_FIELDS: Tuple[str, ...] = (
    "num_mpc_steps", "lam", "n_ctrl", "lr", "optimizer", "warm_start", "grad_clip",
    "phi_normalization", "control_cost_normalization")
PNP_FIELDS: Tuple[str, ...] = (
    "num_pnp_steps", "gamma0", "alpha", "noise_samples", "phi_normalization")
DFLOW_FIELDS: Tuple[str, ...] = (
    "steps", "solver", "num_opt_steps", "lr", "optimizer", "phi_normalization")
# RHSO has NO lambda, NO control-cost normalisation, NO K and no warm start: the optimised
# variable is the state itself, so there is nothing to trade the fidelity off against.
# `solver` is meaningful only for standard-flow models; the MeanFlow capability table has no
# solver, so the validator rejects it there exactly as it does for the other methods.
# `mu` is RHSO's OWN state-anchor regularisation weight and is deliberately not named
# `lam`: MPC's lambda weights an added control signal, this weights a displacement of the
# state from where the trajectory actually was.  It defaults to 0.0 (penalty off).
# `rhso_terminal_mode` selects WHAT ONE INNER OBJECTIVE PREDICTS, and nothing else:
#   auto    (default) MeanFlow -> direct, standard flow -> suffix.  Exactly the behaviour
#           every configuration written before this field existed already had.
#   direct  ONE terminal prediction from the current state: T_theta(q; s_k -> 1) for a
#           MeanFlow, x_hat_1(q, s_k) for a standard flow.
#   suffix  the legacy standard-flow planner: differentiable integration of [s_k, ..., 1].
#           Rejected for MeanFlow models, which have no ODE to integrate.
# The `rhso_*_diagnostics` fields switch on EXTRA measurements (see src/rhso_diagnostics.py).
# They never change the objective, the gradients or the executed trajectory, and their cost
# is accounted separately from the algorithm's.
RHSO_FIELDS: Tuple[str, ...] = (
    "num_rhso_steps", "num_opt_steps", "lr", "mu", "optimizer", "phi_normalization",
    "solver", "rhso_terminal_mode", "rhso_stage_diagnostics",
    "rhso_consistency_diagnostics", "rhso_jacobian_diagnostics", "rhso_jacobian_probes",
    "rhso_jacobian_power_iters", "rhso_jacobian_seed",
    # TASK-ALIGNED endpoint authority; a separate, independently switchable measurement
    # from the random-direction Jacobian probes above.  Off by default.
    "rhso_gradient_authority_diagnostics",
    # Time dependence of the state-anchor weight: stage k uses mu_k = mu * m(s_k).
    # "constant" (the default, m = 1) is the objective RHSO has always had; "increasing"
    # m(t) = (t^2 + (1-t)^2) / (1-t)^2 and "decreasing" (its reciprocal) are the
    # alternatives.  See src/rhso.py (mu_schedule_multiplier).
    "mu_schedule")

# Flower (src/flower.py).  num_flower_steps is the iteration count N (one network
# evaluation each); flower_gamma is 0 (the paper's reported setting) or 1 (adds a posterior
# sample kappa_t); the CG settings only matter for deblurring.  No phi_normalization: the
# refinement weighs the data term by the true noise level 1/sigma^2 and nothing else.
FLOWER_FIELDS: Tuple[str, ...] = (
    "num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol")

METHOD_DECLARATIONS: Dict[str, MethodDeclaration] = {
    "sdedit": MethodDeclaration(
        "sdedit", "SDEdit", False, False, SHARED_FIELDS + SDEDIT_FIELDS,
        "Ordinary generation from z_t0 to the data endpoint.  No control, no measurement "
        "term: the measurement enters only through the initialisation guide."),
    "mpc_rhc": MethodDeclaration(
        "mpc_rhc", "MPC-RHC", True, True, SHARED_FIELDS + MPC_COMMON_FIELDS + ("K",),
        "Receding-horizon control over [s, 1] discretised into K steps (Algorithms 1 and 3). "
        "K = 1 does not backpropagate through the generative model."),
    "mpc_delta_t": MethodDeclaration(
        "mpc_delta_t", "MPC-delta_t", True, False,
        SHARED_FIELDS + MPC_COMMON_FIELDS + ("delta_t_lambda_scaling",),
        "Delta-t horizon control with a one-step value-function surrogate (Algorithm 2). "
        "K is meaningless here and is rejected."),
    "pnp": MethodDeclaration(
        "pnp", "PnP-Flow", False, False, SHARED_FIELDS + PNP_FIELDS,
        "Plug-and-Play Flow Matching: an initial prior projection of the shared z_t0, then "
        "num_pnp_steps cycles of (data-fidelity gradient step -> stochastic reprojection "
        "onto the flow path -> denoising).  Never backpropagates through the generative "
        "model.  MeanFlow models use the learned transition T(x; s -> 1) as the denoiser, "
        "which is a research extension."),
    "rhso": MethodDeclaration(
        "rhso", "RHSO", False, False, SHARED_FIELDS + RHSO_FIELDS,
        "Receding-Horizon State Optimization: at every scheduled time, optimise the CURRENT "
        "generative state against its own terminal prediction, execute exactly ONE scheduled "
        "interval, then discard the problem and re-optimise from the state actually reached. "
        "No control variable, no control penalty, no lambda and no K. For a standard flow "
        "the terminal planner integrates the remaining suffix of the outer grid; for a "
        "MeanFlow it is the single learned transition T(q; s_k -> 1). Optionally adds "
        "mu * R(q, x_k), a trust-region-like penalty on how far the optimised state has "
        "moved from the state the trajectory actually reached at the start of that stage; "
        "mu = 0 (the default) is the plain terminal-fidelity objective."),
    "dflow": MethodDeclaration(
        "dflow", "D-Flow", False, False, SHARED_FIELDS + DFLOW_FIELDS,
        "Optimises the starting generative state q by differentiating the terminal "
        "measurement loss through the generative map.  q is initialised to the shared "
        "z_t0, so t0 < 1 optimises an INTERMEDIATE state rather than the source "
        "distribution -- an intentional extension of the published method."),
    "flower": MethodDeclaration(
        "flower", "Flower", False, False, SHARED_FIELDS + FLOWER_FIELDS,
        "Flower: num_flower_steps iterations of (destination x_hat = x + (1-t) v, or "
        "T(x; t -> 1) for a MeanFlow -> Gaussian proximal refinement with the true noise "
        "level, closed form or conjugate gradient -> re-noising to the next time).  Pixel "
        "priors and linear operators only; never backpropagates through the model."),
}

METHOD_NAMES: Tuple[str, ...] = tuple(METHOD_DECLARATIONS)

# Strategies that are compared AGAINST SDEdit; each wants a paired SDEdit job at the same t0.
BASELINE_METHOD: str = "sdedit"
COMPARED_METHODS: Tuple[str, ...] = ("mpc_rhc", "mpc_delta_t", "pnp", "dflow", "rhso",
                                     "flower")

# The closed set of sweepable fields.  A list in ANY OTHER field is a literal value
# (e.g. guidance.interval, which is one interval and never a sweep).
SWEEPABLE_FIELDS: Tuple[str, ...] = (
    "t0", "beta", "steps", "solver", "K", "num_mpc_steps", "lam", "n_ctrl", "lr",
    "optimizer", "warm_start", "grad_clip",
    "phi_normalization", "control_cost_normalization", "delta_t_lambda_scaling",
    # PnP-Flow
    "num_pnp_steps", "gamma0", "alpha", "noise_samples",
    # D-Flow
    "num_opt_steps",
    # RHSO
    "num_rhso_steps", "mu",
    # RHSO terminal planner and theory-validation diagnostics
    "rhso_terminal_mode", "rhso_stage_diagnostics", "rhso_consistency_diagnostics",
    "rhso_jacobian_diagnostics", "rhso_jacobian_probes", "rhso_jacobian_power_iters",
    "rhso_jacobian_seed", "rhso_gradient_authority_diagnostics",
    # Flower
    "num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol",
    # RHSO anchor schedule
    "mu_schedule",
)
MODEL_LEVEL_FIELDS: Tuple[str, ...] = ("guidance", "batch_size", "record_loss_history")

# `half_mean_squared_per_measurement` is the PnP / D-Flow default; see problems.make_phi.
VALID_PHI_NORMALIZATIONS = ("half_sum_squared", "sum_squared", "mean_squared",
                            "gaussian_likelihood", "half_mean_squared_per_measurement")
PER_MEASUREMENT_NORMALIZATION = "half_mean_squared_per_measurement"
VALID_CONTROL_COST_NORMALIZATIONS = ("sum_squared", "mean_squared")
VALID_DELTA_T_LAMBDA_SCALINGS = ("none", "inverse_delta")
VALID_OPTIMIZERS = ("adam", "sgd")
# D-Flow is deliberately Adam-only in this iteration (no LBFGS, no line search).
VALID_DFLOW_OPTIMIZERS = ("adam",)
# RHSO likewise: its optimisation problem is rebuilt at every outer stage, so a stateful
# alternative would need its own reset semantics before it could be reported honestly.
VALID_RHSO_OPTIMIZERS = ("adam",)
VALID_SOLVERS = ("euler", "heun", "rk4")
SOLVER_STAGE_EVALUATIONS: Dict[str, int] = {"euler": 1, "heun": 2, "rk4": 4}
VALID_DATA_SOURCES = ("hf_imagenet_val", "local_folder")
VALID_ACCELERATORS = ("auto", "gpu", "tpu", "cpu")


# =====================================================================================
# Table E2 -- paper hyperparameters, used ONLY as fallback defaults
# =====================================================================================
# (lambda, N_ctrl, lr), MPC-Flow Appendix E.2, tuned on a CelebA 128x128 pixel-space U-Net.
# These are NOT tuned JiT/pMF values; every resolved job records where its value came from.
PAPER_E2: Dict[str, Any] = {
    "mpc_delta_t": {
        "denoising":        (15.0,  20, 0.1),
        "deblur":           (7.5,   20, 0.1),
        "super_resolution": (7.5,   20, 0.1),
        "random_inpaint":   (0.5,   20, 0.1),
        "box_inpaint":      (5.0,   20, 0.1),
    },
    # RHC is tabulated only for K = 1 and K = 3.
    "mpc_rhc": {
        1: {
            "denoising":        (0.1,   20, 0.1),
            "deblur":           (0.063, 20, 0.1),
            "super_resolution": (1.0,   20, 0.1),
            "random_inpaint":   (1.0,   20, 0.1),
            "box_inpaint":      (0.040, 20, 0.1),
        },
        3: {
            "denoising":        (0.1,   20, 0.05),
            "deblur":           (0.063, 20, 0.05),
            "super_resolution": (0.063, 20, 0.05),
            "random_inpaint":   (1.0,   20, 0.05),
            "box_inpaint":      (1.0,   20, 0.05),
        },
    },
}

# Stroke painting is NOT in Table E2 (the paper has no such task).  These are repository
# defaults, chosen conservatively, and they are labelled as such in every result row.
REPO_MPC_DEFAULTS: Dict[str, Tuple[float, int, float]] = {
    "mpc_rhc": (1.0, 20, 0.1),
    "mpc_delta_t": (5.0, 20, 0.1),
}

# PnP-Flow and D-Flow defaults.  These are NOT taken from either paper: the PnP-Flow paper
# tunes (alpha, N) per dataset and task on a validation split, and D-Flow uses LBFGS with a
# target-PSNR stopping rule that this repository deliberately does not implement.  Every
# resolved job records `repository_default_untuned` as the provenance so no table can
# present them as published values.
PNP_DEFAULTS: Dict[str, Any] = {
    "num_pnp_steps": 10,
    "gamma0": 1.0,
    "alpha": 1.0,
    "noise_samples": 1,
}
DFLOW_DEFAULTS: Dict[str, Any] = {
    "steps": 1,
    "num_opt_steps": 10,
    "lr": 0.01,
    "optimizer": "adam",
}
# RHSO is this repository's own strategy: it has no published hyperparameters at all, so
# every value below is a starting point recorded as `repository_default_untuned`.
RHSO_DEFAULTS: Dict[str, Any] = {
    "num_rhso_steps": 4,
    "num_opt_steps": 10,
    "lr": 0.01,
    "optimizer": "adam",
    # NOT untuned guesses either: "auto" is "whatever this family already did", and every
    # diagnostic defaults to OFF so an existing configuration runs exactly as before.
    "rhso_terminal_mode": "auto",
    "rhso_stage_diagnostics": False,
    "rhso_consistency_diagnostics": False,
    "rhso_jacobian_diagnostics": False,
    "rhso_gradient_authority_diagnostics": False,
    # NOT an untuned guess but a deliberate OFF switch: 0.0 is the vanilla objective, so
    # every RHSO configuration written before this field existed keeps its exact meaning.
    "mu": 0.0,
    # Same reasoning: "constant" is the time-independent weight RHSO always used.
    "mu_schedule": "constant",
}
UNTUNED = "repository_default_untuned"

# Flower: N is a starting point to be tuned (the paper uses 100 to 500) and is recorded as
# UNTUNED.  gamma = 0 is the setting of all of the paper's reported results, and the CG
# limits are the paper's own (50 iterations, relative residual 1e-5).
FLOWER_DEFAULTS: Dict[str, Any] = {
    "num_flower_steps": 100,
    "flower_gamma": 0,
    "flower_cg_max_iters": 50,
    "flower_cg_tol": 1e-5,
}
# The operators Flower's refinement is defined for (mirrors src/flower.SUPPORTED_PROBLEMS).
FLOWER_PROBLEMS: Tuple[str, ...] = ("denoising", "random_inpaint", "box_inpaint",
                                    "super_resolution", "deblur")


def _flower_seed_recipe() -> str:
    from .flower import FLOWER_SEED_RECIPE        # local: keeps config free of model imports
    return FLOWER_SEED_RECIPE


def _legacy_terminal_mode(dynamics_family: str) -> str:
    """What `rhso_terminal_mode: auto` resolves to -- i.e. the pre-field behaviour."""
    from .rhso import legacy_terminal_mode
    return legacy_terminal_mode(dynamics_family)


def rhso_terminal_planner(terminal_mode: str, model: str) -> str:
    """An honest name for the object one inner RHSO objective differentiates through.

    Resolved from the MODEL REGISTRY (no adapter exists at planning time) and deliberately
    NOT collapsed into one label: a MeanFlow's T_theta is a learned family of
    finite-interval transport maps, a clean-prediction model's x_hat_1 is a direct endpoint
    prediction, and a velocity model's one-step endpoint is an extrapolation. Three
    different objects, three different names, one shared cost per inner objective.
    """
    if terminal_mode == "suffix":
        return "suffix_integration"
    caps = MODEL_CAPABILITIES[model]
    if caps.dynamics_family == MEANFLOW:
        return "learned_finite_interval_map"
    if caps.prediction_kind == "clean":
        return "direct_clean_endpoint_prediction"
    return "velocity_extrapolation"


def resolve_mpc_hyperparameters(problem: str, method: str,
                                K: Optional[int]) -> Tuple[Dict[str, Any], str, List[str]]:
    """Return ({'lam','n_ctrl','lr'}, provenance_label, warnings) for a missing (lam,n_ctrl,lr).

    Provenance is one of
        "paper_E2"                      -- a value tabulated by MPC-Flow Appendix E.2
        "paper_E2_nearest_K(K=a->b)"    -- fallback CHOICE made here, not a published value
        "repository_default"            -- no paper entry exists for this problem
    """
    warnings_: List[str] = []
    table = PAPER_E2.get(method, {})
    if method == "mpc_delta_t":
        if problem in table:
            lam, n_ctrl, lr = table[problem]
            return {"lam": float(lam), "n_ctrl": int(n_ctrl), "lr": float(lr)}, "paper_E2", []
    elif method == "mpc_rhc":
        Kv = int(K if K is not None else 1)
        available = sorted(table)
        if available:
            if Kv in table:
                chosen, label = Kv, "paper_E2"
            else:
                chosen = min(available, key=lambda k: (abs(k - Kv), k))
                label = "paper_E2_nearest_K(K=%d->%d)" % (Kv, chosen)
                warnings_.append(
                    "K=%d is not tabulated in MPC-Flow Table E2 (only K in %s are). "
                    "lam/n_ctrl/lr fall back to K=%d; that fallback is a CHOICE made by this "
                    "repository and must not be reported as a published hyperparameter."
                    % (Kv, available, chosen))
            if problem in table[chosen]:
                lam, n_ctrl, lr = table[chosen][problem]
                return {"lam": float(lam), "n_ctrl": int(n_ctrl), "lr": float(lr)}, label, warnings_
    lam, n_ctrl, lr = REPO_MPC_DEFAULTS[method]
    warnings_.append(
        "%s/%s has no MPC-Flow Table E2 entry; using this repository's conservative default "
        "(lam=%g, n_ctrl=%d, lr=%g). Tune it before drawing conclusions."
        % (problem, method, lam, n_ctrl, lr))
    return {"lam": float(lam), "n_ctrl": int(n_ctrl), "lr": float(lr)}, "repository_default", \
        warnings_


def resolve_model_registry(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Registry defaults with the user's TECHNICAL overrides applied."""
    if name not in MODEL_REGISTRY_DEFAULTS:
        raise ConfigError("Unknown model %r (known: %s)" % (name, list(MODEL_NAMES)))
    return deep_merge(MODEL_REGISTRY_DEFAULTS[name], (config.get("models") or {}).get(name))


# =====================================================================================
# Validation
# =====================================================================================
TOP_LEVEL_KEYS = ("runtime", "data", "models", "defaults", "experiments", "metrics")
RUNTIME_KEYS = ("seed", "accelerator", "output_root", "cache_root",
                "release_model_after_use", "continue_on_experiment_error", "progress_bars",
                "save_individual_images", "verbose", "max_atomic_jobs", "resume",
                "warmup", "replicate", "gpu_memory_profiling", "nvml_sample_interval",
                # OPT-IN (rebuttal Task 1): per-image device-synchronised timing and
                # allocator snapshots, see src/timing.py. Absent = off = unchanged runs.
                "timing_detail")
DATA_KEYS = ("source", "local_folder", "image_size")
DEFAULTS_KEYS = SWEEPABLE_FIELDS + ("record_loss_history",)
EXPERIMENT_KEYS = ("enabled", "problem", "num_images", "degradation", "defaults",
                   "models", "notes",
                   # OPT-IN common-random-number measurement noise; see
                   # utils.paired_measurement_noise_parts.  Absent (the default) reproduces
                   # the repository's existing seeding exactly.
                   "measurement_noise_group")
MODEL_ENTRY_KEYS = ("methods",) + MODEL_LEVEL_FIELDS + SWEEPABLE_FIELDS
METHOD_ENTRY_KEYS = SWEEPABLE_FIELDS + ("record_loss_history",)

REMOVED_KEYS: Dict[str, str] = {
    "standard_flow": "Removed. A model's dynamics family is a property of the model, not a "
                     "user choice: write experiments.<name>.models.{jit,pmf,...} instead.",
    "meanflow": "Removed. See 'standard_flow'.",
    "enabled_models": "Removed. A model participates because it appears under an enabled "
                      "experiment.",
    "enabled_problems": "Removed. Each experiment names exactly one 'problem'.",
    "num_mpc_steps": "Not a global. Put it inside experiments.<name>.models.<model>."
                     "methods.<method>.num_mpc_steps (scalar or list).",
    "hyperparameters": "Removed. Table E2 is built in (PAPER_E2) and used as a fallback.",
    "initialization_guide_mode": "Removed. Each problem defines exactly one guide (see "
                                 "PROBLEM_DECLARATIONS); classical inpainting prefill is "
                                 "deliberately not available.",
    "preflight": "Renamed. Sanity checks are controlled by run.py --check / --no-check.",
}


def _require_mapping(value: Any, where: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("%s must be a mapping, got %s" % (where, type(value).__name__))
    return value


def _reject_unknown(block: Dict[str, Any], valid: Sequence[str], where: str,
                    removed: Optional[Dict[str, str]] = None) -> None:
    """Closed key set: unknown keys are errors, so no option can silently be a no-op."""
    for key in block:
        if removed and key in removed:
            raise ConfigError("%s.%s is no longer supported.\n    %s" % (where, key, removed[key]))
        if key not in valid:
            hint = ""
            close = [v for v in valid if isinstance(v, str)
                     and (v.startswith(str(key)[:3]) or str(key).startswith(v[:3]))]
            if close:
                hint = "\n    Did you mean: %s ?" % ", ".join(sorted(close)[:4])
            raise ConfigError("Unknown key %r in %s.\n    Valid keys: %s%s"
                              % (key, where, list(valid), hint))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return _is_number(v) and math.isfinite(float(v))


def _check_sweep(entry: Dict[str, Any], field_name: str, where: str) -> List[Any]:
    values = as_sweep_list(entry[field_name])
    if len(values) == 0:
        raise ConfigError("%s.%s is an empty list; a sweep must contain at least one value."
                          % (where, field_name))
    return values


def _validate_sweep_values(field_name: str, values: Sequence[Any], where: str, method: str,
                           model: Optional[str], warnings_: List[str]) -> None:
    tag = "%s.%s" % (where, field_name)
    is_mpc_method = method in METHOD_DECLARATIONS and METHOD_DECLARATIONS[method].is_mpc
    for v in values:
        if field_name == "t0":
            if not _finite(v):
                raise ConfigError("%s: t0 value %r is not a finite number." % (tag, v))
            if not 0.0 < float(v) <= 1.0:
                raise ConfigError(
                    "%s: t0=%r must lie in (0, 1]. t0 is the canonical corruption strength, "
                    "not a model-native time; t0=0 leaves nothing to generate." % (tag, v))
        elif field_name == "beta":
            try:
                resolve_beta(v)
            except ValueError as exc:
                raise ConfigError(
                    "%s: %s beta is the exponent of the time schedule "
                    "s_k = s0 + (1 - s0)(k/N)^beta, so it must be a finite positive number. "
                    "beta = 1 is the uniform (legacy) grid." % (tag, exc))
            if float(v) != DEFAULT_BETA and float(v) > 8.0:
                warnings_.append(
                    "%s: beta=%g concentrates almost every step near the noisy start; at "
                    "large N the first intervals become numerically tiny." % (tag, float(v)))
        elif field_name == "steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: steps must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "solver":
            if v is not None and v not in VALID_SOLVERS:
                raise ConfigError("%s: unknown solver %r; valid: %s"
                                  % (tag, v, list(VALID_SOLVERS)))
            if model and v is not None:
                caps = MODEL_CAPABILITIES[model]
                if v not in caps.supported_solvers:
                    raise ConfigError(
                        "%s: %s does not support solver %r. %s is a MeanFlow model whose "
                        "dynamics are finite-interval transitions, not an instantaneous "
                        "velocity field, so ODE solvers do not apply."
                        % (tag, model.upper(), v, model.upper()))
        elif field_name == "K":
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError("%s: K must be an integer, got %r." % (tag, v))
            if v < 1:
                raise ConfigError("%s: K must be >= 1, got %d." % (tag, v))
            if v > 10:
                warnings_.append("%s: K=%d means %d nested differentiable model evaluations "
                                 "per control iteration; memory scales with K." % (tag, v, v - 1))
        elif field_name == "num_mpc_steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: num_mpc_steps must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "lam":
            if not _finite(v) or float(v) < 0.0:
                raise ConfigError("%s: lam must be finite and non-negative, got %r." % (tag, v))
        elif field_name == "n_ctrl":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: n_ctrl must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "lr":
            if not _finite(v) or float(v) <= 0.0:
                raise ConfigError("%s: lr must be a finite positive number, got %r." % (tag, v))
        elif field_name == "optimizer":
            if v not in VALID_OPTIMIZERS:
                raise ConfigError("%s: unknown optimizer %r; implemented in BOTH backends: %s"
                                  % (tag, v, list(VALID_OPTIMIZERS)))
            if method == "dflow" and v not in VALID_DFLOW_OPTIMIZERS:
                raise ConfigError(
                    "%s: D-Flow supports only %s in this iteration. The published method uses "
                    "LBFGS with line search; that is deliberately NOT implemented here, so "
                    "asking for %r would silently give you something else."
                    % (tag, list(VALID_DFLOW_OPTIMIZERS), v))
            if method == "rhso" and v not in VALID_RHSO_OPTIMIZERS:
                raise ConfigError(
                    "%s: RHSO supports only %s in this iteration, got %r. Its optimisation "
                    "problem is rebuilt from scratch at every outer stage, so any other "
                    "optimiser needs its own documented reset semantics first."
                    % (tag, list(VALID_RHSO_OPTIMIZERS), v))
        elif field_name == "warm_start":
            if not isinstance(v, bool):
                raise ConfigError("%s: warm_start must be a boolean, got %r." % (tag, v))
        elif field_name == "grad_clip":
            if v is not None and (not _finite(v) or float(v) <= 0.0):
                raise ConfigError("%s: grad_clip must be None or positive, got %r." % (tag, v))
        elif field_name == "phi_normalization":
            if v not in VALID_PHI_NORMALIZATIONS:
                raise ConfigError("%s: unknown phi_normalization %r; valid: %s"
                                  % (tag, v, list(VALID_PHI_NORMALIZATIONS)))
            if is_mpc_method and v != "half_sum_squared":
                warnings_.append("%s = %r changes the numerical scale of Phi, so Table E2's "
                                 "lambda values no longer mean what they meant in the paper. "
                                 "Retune lambda." % (tag, v))
            if method in ("pnp", "dflow", "rhso") and v != PER_MEASUREMENT_NORMALIZATION:
                warnings_.append(
                    "%s = %r departs from the per-measurement fidelity these methods default "
                    "to, so gamma0 / lr have to be retuned: %r changes the objective's scale "
                    "by orders of magnitude on some tasks." % (tag, v, v))
        elif field_name == "control_cost_normalization":
            if v not in VALID_CONTROL_COST_NORMALIZATIONS:
                raise ConfigError("%s: unknown control_cost_normalization %r; valid: %s"
                                  % (tag, v, list(VALID_CONTROL_COST_NORMALIZATIONS)))
            if v != "sum_squared":
                warnings_.append("%s = %r rescales the control cost relative to Phi; lambda "
                                 "needs retuning." % (tag, v))
        elif field_name == "delta_t_lambda_scaling":
            if v not in VALID_DELTA_T_LAMBDA_SCALINGS:
                raise ConfigError("%s: unknown delta_t_lambda_scaling %r; valid: %s"
                                  % (tag, v, list(VALID_DELTA_T_LAMBDA_SCALINGS)))
        elif field_name == "num_pnp_steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: num_pnp_steps must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "gamma0":
            if not _finite(v) or float(v) <= 0.0:
                raise ConfigError("%s: gamma0 must be a finite positive number, got %r."
                                  % (tag, v))
        elif field_name == "alpha":
            if not _finite(v) or float(v) <= 0.0:
                raise ConfigError("%s: alpha must be a finite positive number, got %r." % (tag, v))
            if float(v) > 1.0:
                warnings_.append(
                    "%s: alpha=%g lies outside the (0, 1] range PnP-Flow uses for its step-size "
                    "schedule gamma_k = gamma0 (1 - s_k)^alpha. That is allowed here, but it "
                    "damps the data term very aggressively near s = 1." % (tag, float(v)))
        elif field_name == "noise_samples":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: noise_samples must be an integer >= 1, got %r." % (tag, v))
            if v > 16:
                warnings_.append("%s: noise_samples=%d multiplies the denoiser cost of every "
                                 "PnP iteration by %d." % (tag, v, v))
        elif field_name == "num_opt_steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: num_opt_steps must be an integer >= 1, got %r." % (tag, v))
        elif field_name == "num_rhso_steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: num_rhso_steps must be an integer >= 1, got %r."
                                  % (tag, v))
        elif field_name == "mu":
            # _finite() rejects booleans, non-numbers, NaN and +/-inf in one place.
            if not _finite(v) or float(v) < 0.0:
                raise ConfigError(
                    "%s: mu must be a finite, non-negative number, got %r.\n"
                    "    mu weights RHSO's state-anchor penalty "
                    "R = sum_b ||q_b - x_k,b||^2 / (2 d_b); mu = 0 (the default) disables "
                    "it and reproduces the plain terminal-fidelity objective." % (tag, v))
        elif field_name == "mu_schedule":
            from .rhso import VALID_MU_SCHEDULES
            if not isinstance(v, str) or v not in VALID_MU_SCHEDULES:
                raise ConfigError(
                    "%s: unknown mu_schedule %r; valid: %s.\n"
                    "    Stage k of RHSO uses mu_k = mu * m(s_k): 'constant' (the default) "
                    "m = 1; 'increasing' m(t) = (t^2 + (1-t)^2) / (1-t)^2; 'decreasing' "
                    "its reciprocal." % (tag, v, list(VALID_MU_SCHEDULES)))
        elif field_name == "rhso_terminal_mode":
            from .rhso import VALID_TERMINAL_MODES, resolve_terminal_mode
            if v not in VALID_TERMINAL_MODES:
                raise ConfigError(
                    "%s: unknown rhso_terminal_mode %r; valid: %s.\n"
                    "    'direct' plans with ONE terminal prediction from the current "
                    "state; 'suffix' is the legacy standard-flow planner that integrates "
                    "the remaining trajectory; 'auto' (the default) picks whichever the "
                    "model's family already used."
                    % (tag, v, list(VALID_TERMINAL_MODES)))
            if model:
                try:
                    resolve_terminal_mode(v, MODEL_CAPABILITIES[model].dynamics_family)
                except ValueError as exc:
                    raise ConfigError("%s: %s" % (tag, exc))
        elif field_name in ("rhso_stage_diagnostics", "rhso_consistency_diagnostics",
                            "rhso_jacobian_diagnostics",
                            "rhso_gradient_authority_diagnostics"):
            if not isinstance(v, bool):
                raise ConfigError("%s: %s must be a boolean, got %r."
                                  % (tag, field_name, v))
        elif field_name in ("rhso_jacobian_probes", "rhso_jacobian_power_iters"):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: %s must be an integer >= 1, got %r."
                                  % (tag, field_name, v))
            if v > 64:
                warnings_.append(
                    "%s = %d: every probe is another JVP (and every power iteration a "
                    "JVP and a VJP) through the terminal planner. These are DIAGNOSTIC "
                    "evaluations -- they are excluded from the reported runtime and from "
                    "the algorithmic counters, but they still cost wall-clock time."
                    % (tag, v))
        elif field_name == "rhso_jacobian_seed":
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError("%s: rhso_jacobian_seed must be an integer, got %r."
                                  % (tag, v))
        elif field_name == "num_flower_steps":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: num_flower_steps must be an integer >= 1, got %r."
                                  % (tag, v))
        elif field_name == "flower_gamma":
            if isinstance(v, bool) or v not in (0, 1) or not isinstance(v, int):
                raise ConfigError(
                    "%s: flower_gamma must be 0 (the paper's reported setting: the "
                    "refinement returns the posterior mean) or 1 (adds a sample with the "
                    "posterior covariance), got %r." % (tag, v))
        elif field_name == "flower_cg_max_iters":
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ConfigError("%s: flower_cg_max_iters must be an integer >= 1, got %r."
                                  % (tag, v))
        elif field_name == "flower_cg_tol":
            if not _finite(v) or not 0.0 < float(v) < 1.0:
                raise ConfigError("%s: flower_cg_tol must be a number in (0, 1), got %r."
                                  % (tag, v))
        else:                                                       # pragma: no cover
            raise ConfigError("Sweepable field %r has no validation rule." % field_name)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Hard-fail on impossible configurations; return the list of soft warnings."""
    warnings_: List[str] = []
    if not isinstance(config, dict):
        raise ConfigError("The configuration must be a mapping.")
    _reject_unknown(config, TOP_LEVEL_KEYS, "config", REMOVED_KEYS)
    for section in ("runtime", "data", "experiments"):
        if section not in config:
            raise ConfigError("The configuration is missing the required %r section." % section)

    # ------------------------------------------------------------------ runtime
    rt = _require_mapping(config["runtime"], "runtime")
    _reject_unknown(rt, RUNTIME_KEYS, "runtime")
    if rt.get("accelerator", "auto") not in VALID_ACCELERATORS:
        raise ConfigError("runtime.accelerator must be one of %s." % (VALID_ACCELERATORS,))
    if not isinstance(rt.get("seed", 42), int) or isinstance(rt.get("seed", 42), bool):
        raise ConfigError("runtime.seed must be an integer.")
    if int(rt.get("max_atomic_jobs", 256)) < 1:
        raise ConfigError("runtime.max_atomic_jobs must be >= 1.")
    if rt.get("release_model_after_use", "auto") not in ("auto", True, False):
        raise ConfigError("runtime.release_model_after_use must be 'auto', true or false.")
    if not isinstance(rt.get("gpu_memory_profiling", True), bool):
        raise ConfigError("runtime.gpu_memory_profiling must be a boolean.")
    if not isinstance(rt.get("timing_detail", False), bool):
        raise ConfigError("runtime.timing_detail must be a boolean.")
    interval = rt.get("nvml_sample_interval", 0.02)
    if not _finite(interval) or not 0.001 <= float(interval) <= 1.0:
        raise ConfigError("runtime.nvml_sample_interval must lie in [0.001, 1.0] seconds "
                          "(it is the NVML sampling period used when a framework allocator "
                          "high-water mark is unavailable).")
    rep = rt.get("replicate", 0)
    if isinstance(rep, bool) or not isinstance(rep, int) or rep < 0:
        raise ConfigError("runtime.replicate must be a non-negative integer (0 = the default "
                          "single trial).")

    # ------------------------------------------------------------------ data
    data = _require_mapping(config["data"], "data")
    _reject_unknown(data, DATA_KEYS, "data",
                    {"num_images": "Removed. Each experiment declares its own num_images."})
    if data.get("source", "hf_imagenet_val") not in VALID_DATA_SOURCES:
        raise ConfigError("data.source must be one of %s. Synthetic images are refused: PSNR "
                          "and LPIPS against procedural textures say nothing about an ImageNet "
                          "prior." % (VALID_DATA_SOURCES,))
    if data.get("source") == "local_folder" and not data.get("local_folder"):
        raise ConfigError("data.source='local_folder' requires data.local_folder.")
    if int(data.get("image_size", 256)) != 256:
        raise ConfigError("data.image_size must be 256: every supported checkpoint is 256x256.")

    # ------------------------------------------------------------------ models (technical only)
    models_cfg = _require_mapping(config.get("models"), "models")
    for name, block in models_cfg.items():
        if name not in MODEL_REGISTRY_DEFAULTS:
            raise ConfigError("Unknown model %r in models:; known: %s" % (name, list(MODEL_NAMES)))
        block = _require_mapping(block, "models.%s" % name)
        for key in block:
            if key in ("methods",) + SWEEPABLE_FIELDS:
                raise ConfigError(
                    "models.%s.%s is an EXPERIMENT setting, not a technical one. Put it under "
                    "experiments.<name>.models.%s." % (name, key, name))
            if key not in MODEL_REGISTRY_DEFAULTS[name]:
                raise ConfigError("Unknown technical key %r for model %r.\n    Valid keys: %s"
                                  % (key, name, sorted(MODEL_REGISTRY_DEFAULTS[name])))
    jit_cfg = resolve_model_registry(config, "jit")
    if jit_cfg["variant"] not in jit_cfg["recommended_cfg"]:
        raise ConfigError("Unknown JiT variant %r; known: %s"
                          % (jit_cfg["variant"], sorted(jit_cfg["recommended_cfg"])))
    if jit_cfg["checkpoint_backend"] not in ("hf_mirror", "original_local"):
        raise ConfigError("jit.checkpoint_backend must be 'hf_mirror' or 'original_local'.")
    if jit_cfg["checkpoint_backend"] == "original_local" and not jit_cfg.get("local_checkpoint"):
        raise ConfigError("jit.checkpoint_backend='original_local' requires jit.local_checkpoint.")
    if not (0.0 < float(jit_cfg["t_eps"]) < 1.0):
        raise ConfigError("jit.t_eps must lie in (0, 1).")
    sit_cfg = resolve_model_registry(config, "sit")
    if (sit_cfg["guidance"] or {}).get("mode") not in ("official_first3", "all_channels"):
        raise ConfigError("sit.guidance.mode must be 'official_first3' or 'all_channels'.")
    for name in MODEL_NAMES:
        reg = resolve_model_registry(config, name)
        if int(reg["batch_size"]) < 1:
            raise ConfigError("models.%s.batch_size must be >= 1." % name)
        if reg.get("vae_encode_mode") is not None and \
                reg["vae_encode_mode"] not in ("mean", "sample"):
            raise ConfigError("models.%s.vae_encode_mode must be 'mean' or 'sample'." % name)

    # ------------------------------------------------------------------ defaults
    defaults = _require_mapping(config.get("defaults"), "defaults")
    _reject_unknown(defaults, DEFAULTS_KEYS, "defaults")
    for key in defaults:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(defaults, key, "defaults"), "defaults",
                                   "mpc_rhc", None, warnings_)
    if "record_loss_history" in defaults and not isinstance(defaults["record_loss_history"], bool):
        raise ConfigError("defaults.record_loss_history must be a boolean.")

    # ------------------------------------------------------------------ metrics
    metrics = _require_mapping(config.get("metrics"), "metrics")
    _reject_unknown(metrics, ("lpips", "lpips_net"), "metrics")
    if not metrics.get("lpips", True):
        warnings_.append("metrics.lpips is disabled, but LPIPS is a headline metric of this "
                         "benchmark; the results table will contain empty LPIPS columns.")

    # ------------------------------------------------------------------ experiments
    experiments = _require_mapping(config["experiments"], "experiments")
    if not experiments:
        raise ConfigError("experiments: is empty -- there is nothing to run.")
    any_enabled = False
    for exp_name, block in experiments.items():
        block = _require_mapping(block, "experiments.%s" % exp_name)
        where = "experiments.%s" % exp_name
        _reject_unknown(block, EXPERIMENT_KEYS, where, REMOVED_KEYS)
        if not block.get("enabled", False):
            continue
        any_enabled = True
        _validate_enabled_experiment(config, exp_name, block, where, warnings_)

    if not any_enabled:
        raise ConfigError("No experiment is enabled. Set `enabled: true` on at least one entry "
                          "under experiments:.")
    return warnings_


def _validate_enabled_experiment(config, exp_name, block, where, warnings_) -> None:
    problem = block.get("problem")
    if problem is None:
        raise ConfigError("%s is enabled but has no 'problem'." % where)
    if problem not in PROBLEM_DECLARATIONS:
        raise ConfigError("%s: unknown problem %r; known: %s"
                          % (where, problem, list(PROBLEM_NAMES)))
    decl = PROBLEM_DECLARATIONS[problem]

    n = block.get("num_images")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ConfigError("%s.num_images must be a positive integer, got %r." % (where, n))
    if n == 1:
        warnings_.append("%s runs on a single image; comparisons will be anecdotal." % where)

    group = block.get("measurement_noise_group")
    if group is not None:
        if not isinstance(group, str) or not group.strip():
            raise ConfigError(
                "%s.measurement_noise_group must be a non-empty string naming the pairing "
                "group, or be omitted entirely.\n    It makes the standard-normal "
                "measurement draw depend on (group, problem, structural degradation "
                "parameters, image) but NOT on sigma, so a sweep over sigma reuses one "
                "epsilon per image. Omitting it reproduces this repository's original "
                "seeding exactly." % where)
        if problem == "random_inpaint":
            warnings_.append(
                "%s pairs the measurement NOISE across sigma, but the random inpainting "
                "MASK is still seeded from the full parameter key (sigma included), which "
                "is this repository's existing behaviour and is deliberately left "
                "unchanged. A sigma sweep on this task therefore re-draws the mask, so "
                "(y - Ax)/sigma is NOT identical across the sweep. Pairing is exact for "
                "denoising, deblurring, super-resolution and box inpainting, whose "
                "operators do not depend on sigma." % where)

    params = _require_mapping(block.get("degradation"), "%s.degradation" % where)
    allowed = decl.required_params + decl.optional_params
    _reject_unknown(params, allowed, "%s.degradation" % where)
    merged = dict(decl.default_params)
    merged.update(params)
    for key in decl.required_params:
        if key not in merged:
            raise ConfigError("%s.degradation is missing required %r." % (where, key))
    _validate_problem_params(problem, merged, "%s.degradation" % where, warnings_)

    exp_defaults = _require_mapping(block.get("defaults"), "%s.defaults" % where)
    _reject_unknown(exp_defaults, DEFAULTS_KEYS, "%s.defaults" % where)
    for key in exp_defaults:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(exp_defaults, key, "%s.defaults" % where),
                                   "%s.defaults" % where, "mpc_rhc", None, warnings_)

    models_block = _require_mapping(block.get("models"), "%s.models" % where)
    if not models_block:
        raise ConfigError("%s is enabled but no model participates. Add at least one entry "
                          "under its 'models:' block." % where)
    for model_name, entry in models_block.items():
        if model_name not in MODEL_REGISTRY_DEFAULTS:
            raise ConfigError("%s.models: unknown model %r; known: %s"
                              % (where, model_name, list(MODEL_NAMES)))
        _validate_model_entry(config, problem, decl, merged, model_name,
                              _require_mapping(entry, "%s.models.%s" % (where, model_name)),
                              "%s.models.%s" % (where, model_name), warnings_)


def _validate_problem_params(problem: str, p: Dict[str, Any], where: str,
                             warnings_: List[str]) -> None:
    sigma = p.get("sigma", 0.0)
    if not _finite(sigma) or float(sigma) < 0.0:
        raise ConfigError("%s.sigma must be finite and non-negative, got %r." % (where, sigma))
    if float(sigma) == 0.0 and problem != "stroke_painting":
        warnings_.append("%s.sigma = 0 makes the measurement noiseless, which is not one of "
                         "the MPC-Flow paper's settings." % where)
    if problem == "super_resolution":
        factor = p.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
            raise ConfigError("%s.factor must be an integer >= 2, got %r." % (where, factor))
        if factor != 2:
            warnings_.append("%s.factor = %d departs from the paper's x2 strided-subsampling "
                             "operator." % (where, factor))
        if 256 % int(factor) != 0:
            raise ConfigError("%s.factor must divide 256." % where)
    if problem == "deblur":
        if not _finite(p.get("blur_sigma")) or float(p["blur_sigma"]) <= 0.0:
            raise ConfigError("%s.blur_sigma must be a positive number." % where)
        ksize = p.get("kernel_size", 7)
        if isinstance(ksize, bool) or not isinstance(ksize, int) or ksize < 1 or ksize % 2 == 0:
            raise ConfigError("%s.kernel_size must be a positive ODD integer so the kernel is "
                              "centred, got %r." % (where, ksize))
        if ksize > 129:
            raise ConfigError("%s.kernel_size=%d is impractically large." % (where, ksize))
        if p.get("padding", "reflect") != "reflect":
            raise ConfigError("%s.padding: only 'reflect' padding is implemented (it is the one "
                              "the three backends compute identically by index gather)." % where)
        if ksize < 2 * int(math.ceil(3.0 * float(p["blur_sigma"]))) + 1:
            warnings_.append("%s: kernel_size=%d truncates a sigma_b=%.3g Gaussian below ~3 "
                             "sigma. The paper does not specify the kernel support, so this is "
                             "a choice made here." % (where, ksize, float(p["blur_sigma"])))
    if problem == "random_inpaint":
        frac = p.get("missing_fraction")
        if not _finite(frac) or not 0.0 < float(frac) < 1.0:
            raise ConfigError("%s.missing_fraction must lie strictly in (0, 1), got %r."
                              % (where, frac))
    if problem == "box_inpaint":
        box = p.get("box")
        if isinstance(box, bool) or not isinstance(box, int) or not 0 < box < 256:
            raise ConfigError("%s.box must be an integer in (0, 256), got %r." % (where, box))
    if problem == "stroke_painting":
        preset = p.get("preset", "medium")
        if preset not in STROKE_PRESETS:
            raise ConfigError("%s.preset must be one of %s, got %r."
                              % (where, list(STROKE_PRESETS), preset))
        if not _finite(p.get("blur_sigma", 0.8)) or float(p.get("blur_sigma", 0.8)) <= 0:
            raise ConfigError("%s.blur_sigma must be a positive number." % where)
        if float(p.get("sigma", 0.0)) > 0.0:
            warnings_.append("%s.sigma > 0 adds Gaussian noise to the stroke measurement; the "
                             "default for this task is a noiseless y = A_G(x*)." % where)


def _validate_model_entry(config, problem, decl, problem_params, model_name, entry, where,
                          warnings_) -> None:
    _reject_unknown(entry, MODEL_ENTRY_KEYS, where)
    caps = MODEL_CAPABILITIES[model_name]
    reg = resolve_model_registry(config, model_name)

    if "guidance" in entry:
        guidance = _require_mapping(entry["guidance"], "%s.guidance" % where)
        warnings_.append("%s overrides guidance with %s; the registry default (%s) is the value "
                         "published for this checkpoint, and guidance intervals are expressed "
                         "in MODEL-NATIVE time. 'interval' is one interval, never a sweep."
                         % (where, guidance, reg["guidance"]))
        if guidance.get("interval") is not None:
            iv = guidance["interval"]
            if not isinstance(iv, (list, tuple)) or len(iv) != 2:
                raise ConfigError("%s.guidance.interval must be a 2-element [lo, hi]." % where)
            lo, hi = float(iv[0]), float(iv[1])
            if not 0.0 <= lo < hi <= 1.0:
                raise ConfigError("%s.guidance.interval must satisfy 0 <= lo < hi <= 1." % where)
        if guidance.get("scale") is not None and not _finite(guidance["scale"]):
            raise ConfigError("%s.guidance.scale must be a finite number or null." % where)
    if "batch_size" in entry:
        bs = entry["batch_size"]
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
            raise ConfigError("%s.batch_size must be a positive integer." % where)
        if caps.fixed_batch_shape:
            warnings_.append("%s overrides batch_size for a fixed-compiled-batch model; the "
                             "change costs exactly one recompilation." % where)
    if "record_loss_history" in entry and not isinstance(entry["record_loss_history"], bool):
        raise ConfigError("%s.record_loss_history must be a boolean." % where)

    for key in entry:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(entry, key, where), where, "mpc_rhc",
                                   model_name, warnings_)

    methods = _require_mapping(entry.get("methods"), "%s.methods" % where)
    if not methods:
        raise ConfigError("%s has no 'methods'. Add at least one of %s."
                          % (where, list(METHOD_NAMES)))
    for method_name, method_entry in methods.items():
        if method_name not in METHOD_DECLARATIONS:
            raise ConfigError("%s.methods: unknown method %r; known: %s"
                              % (where, method_name, list(METHOD_NAMES)))
        if method_name not in caps.supported_methods:
            raise ConfigError("%s: %s does not support method %r; supported: %s"
                              % (where, model_name.upper(), method_name,
                                 list(caps.supported_methods)))
        _validate_method_entry(config, problem, decl, problem_params, model_name, caps,
                               method_name,
                               _require_mapping(method_entry,
                                                "%s.methods.%s" % (where, method_name)),
                               "%s.methods.%s" % (where, method_name), entry, warnings_)


def _validate_method_entry(config, problem, decl, problem_params, model_name, caps, method_name,
                           entry, where, inherited_model_entry, warnings_) -> None:
    _reject_unknown(entry, METHOD_ENTRY_KEYS, where)
    method_decl = METHOD_DECLARATIONS[method_name]
    allowed = set(method_decl.fields) | {"record_loss_history"}

    for key in entry:
        if key in SWEEPABLE_FIELDS and key not in allowed:
            if key == "K":
                raise ConfigError(
                    "%s: 'K' is meaningless for %s and is rejected.\n"
                    "    %s has no planning discretisation to choose: SDEdit simply integrates, "
                    "and MPC-delta_t optimises a single control over one interval of length "
                    "delta.\n    Use method 'mpc_rhc' if you want a planning depth K."
                    % (where, method_name, method_decl.title))
            if key == "solver" and method_name == "pnp":
                raise ConfigError(
                    "%s: 'solver' does not apply to PnP-Flow. Its prior step is a SINGLE "
                    "denoiser evaluation D_s(q) = q + (1 - s) v(q, s), not an ODE solve; there "
                    "is no trajectory to integrate." % where)
            if key in ("mu", "mu_schedule"):
                raise ConfigError(
                    "%s: %r belongs to RHSO's state-anchor regularisation and does not "
                    "apply to %s.\n    It penalises how far the OPTIMISED STATE has moved "
                    "from the state the trajectory actually reached, which only exists in a "
                    "method that optimises a state in place.\n    MPC's control penalty is "
                    "a different quantity with a different name ('lam')."
                    % (where, key, method_decl.title))
            if key.startswith("rhso_"):
                raise ConfigError(
                    "%s: %r configures RHSO's terminal planner / theory diagnostics and "
                    "does not apply to %s.\n    Only a method that optimises a state "
                    "against its own terminal prediction has a terminal planner to choose "
                    "or per-stage diagnostics to record." % (where, key, method_decl.title))
            if key in ("lam", "control_cost_normalization") and method_name == "rhso":
                raise ConfigError(
                    "%s: %r is meaningless for RHSO and is rejected.\n"
                    "    RHSO optimises the generative STATE itself; it has no control "
                    "variable, so there is no control cost for lambda to trade the fidelity "
                    "off against.\n    Use 'mpc_rhc' or 'mpc_delta_t' if you want explicit "
                    "control optimisation." % (where, key))
            raise ConfigError(
                "%s: %r does not apply to method %r.\n    Fields for %s: %s"
                % (where, key, method_name, method_decl.title, sorted(allowed)))
    if "K" in inherited_model_entry and not method_decl.uses_K:
        warnings_.append("%s inherits a model-level 'K', which %s ignores."
                         % (where, method_name))

    for key in entry:
        if key in SWEEPABLE_FIELDS:
            _validate_sweep_values(key, _check_sweep(entry, key, where), where, method_name,
                                   model_name, warnings_)

    if method_name == "flower":
        if problem not in FLOWER_PROBLEMS:
            raise ConfigError(
                "%s: Flower is defined here for the linear operators %s only, not %r. Its "
                "refinement is a Gaussian proximal step that needs a linear H and a noisy "
                "measurement (stroke_painting is excluded)."
                % (where, list(FLOWER_PROBLEMS), problem))
        if not float(problem_params.get("sigma", 0.0) or 0.0) > 0.0:
            raise ConfigError(
                "%s: Flower weighs the data term by 1/sigma^2 with the TRUE noise level, so "
                "it needs sigma > 0; %s has sigma=%g."
                % (where, problem, float(problem_params.get("sigma", 0.0) or 0.0)))

    # `gaussian_likelihood` divides by sigma^2; a noiseless measurement makes it undefined.
    # Substituting a tiny epsilon would silently inflate the objective by ~1e16, so it is
    # rejected instead.
    phi_values = as_sweep_list(entry.get(
        "phi_normalization", inherited_model_entry.get("phi_normalization", None)))
    if "gaussian_likelihood" in [v for v in phi_values if v is not None]:
        sigma = float(problem_params.get("sigma", 0.0) or 0.0)
        if sigma <= 0.0:
            raise ConfigError(
                "%s: phi_normalization='gaussian_likelihood' needs a positive measurement "
                "noise level, but %s has sigma=%g. Its 1/(2 sigma^2) factor is undefined "
                "here, and quietly replacing sigma with a tiny epsilon would scale the "
                "objective by an arbitrary enormous constant.\n"
                "    Use 'half_mean_squared_per_measurement' (the PnP / D-Flow default) or "
                "'half_sum_squared', and let gamma0 / lr / lam carry the calibration."
                % (where, problem, sigma))

    t0_values = [float(v) for v in as_sweep_list(
        entry.get("t0", inherited_model_entry.get("t0", 0.8)))]
    if any(t < 1.0 for t in t0_values) and not caps.supports_encoding:
        raise ConfigError("%s: t0 < 1 needs the measurement-derived guide encoded into %s's "
                          "native state space, which this adapter does not support."
                          % (where, model_name.upper()))

    # The t0 / num_mpc_steps confound (see README).
    if method_decl.is_mpc and len(set(t0_values)) > 1:
        n_steps = as_sweep_list(entry.get("num_mpc_steps",
                                          inherited_model_entry.get("num_mpc_steps", 4)))
        if len(set(n_steps)) == 1:
            warnings_.append(
                "%s sweeps t0 over %s at a FIXED num_mpc_steps=%s, so delta = t0/N varies "
                "across the sweep (%s). Equal step counts do NOT mean equal execution "
                "resolution."
                % (where, sorted(set(t0_values)), n_steps[0],
                   ", ".join("t0=%.2f->delta=%.4g" % (t, t / float(n_steps[0]))
                             for t in sorted(set(t0_values)))))

    # beta on a single-interval trajectory is a no-op: warn rather than fail, because a
    # sweep that includes such a point is usually deliberate (it is its own control).
    beta_values = [float(v) for v in as_sweep_list(
        entry.get("beta", inherited_model_entry.get("beta", DEFAULT_BETA)))]
    if any(b != DEFAULT_BETA for b in beta_values):
        interval_field = {"sdedit": "steps", "dflow": "steps", "rhso": "num_rhso_steps",
                          "mpc_rhc": "num_mpc_steps", "mpc_delta_t": "num_mpc_steps",
                          "pnp": "num_pnp_steps", "flower": "num_flower_steps"}[method_name]
        intervals = as_sweep_list(entry.get(
            interval_field, inherited_model_entry.get(interval_field, None)))
        if method_name != "pnp" and intervals and all(
                isinstance(n, int) and not isinstance(n, bool) and n <= 1 for n in intervals):
            warnings_.append(
                "%s sweeps beta over %s, but %s = 1 leaves a single interval [s0, 1], where "
                "beta has NO mathematical effect. Those jobs will differ only by their "
                "recorded beta." % (where, sorted(set(beta_values)), interval_field))

    # D-Flow at t0 < 1 is an extension, not the published setup.  Say so once per entry.
    if method_name == "dflow" and any(t < 1.0 for t in t0_values):
        warnings_.append(
            "%s optimises an INTERMEDIATE flow state: D-Flow's published formulation "
            "optimises the source point (t0 = 1 here). At t0 < 1 the optimised variable q "
            "is the shared measurement-informed z_t0, which is an intentional research "
            "extension of this repository -- report it as such." % where)


# =====================================================================================
# Resolved atomic specs
# =====================================================================================
@dataclass(frozen=True)
class JobSpec:
    """One ATOMIC job: everything the runner needs, decided here and never re-derived."""
    # -- identity ---------------------------------------------------------------------
    job_id: str
    experiment: str
    problem: str
    problem_params: Dict[str, Any]
    problem_key: str                     # -> the shared InverseProblem instance
    num_images: int
    image_ids: Tuple[str, ...]

    # -- model ------------------------------------------------------------------------
    model: str
    dynamics_family: str                 # STANDARD_FLOW | MEANFLOW
    framework: str                       # "jax" | "torch"
    state_space: str                     # "latent" | "pixel"
    batch_size: int
    fixed_batch_shape: bool
    guidance: Dict[str, Any]

    # -- method -----------------------------------------------------------------------
    method: str                          # sdedit | mpc_rhc | mpc_delta_t | pnp | dflow | rhso
    K: Optional[int]                     # mpc_rhc only

    # -- initialisation / time --------------------------------------------------------
    t0: float
    beta: float                          # power-law time exponent; 1.0 = the uniform grid
    canonical_start_time: float          # s_start = 1 - t0
    native_start_time: float
    native_end_time: float
    native_time_mapping: str
    initialization_guide_mode: str
    initialization_kind: str             # "pure_prior_noise" | "measurement_informed"

    # -- SDEdit (also the D-Flow trajectory discretisation) ---------------------------
    steps: Optional[int]
    solver: Optional[str]

    # -- MPC --------------------------------------------------------------------------
    num_mpc_steps: Optional[int]
    # NOMINAL uniform spacing t0/num_mpc_steps, and None whenever beta != 1: a non-uniform
    # trajectory has no single physical step size.  Never use it as a per-step dt -- the real
    # values are s_{k+1} - s_k (see delta_min / delta_max below and src/schedule.py).
    delta: Optional[float]
    lam: Optional[float]
    n_ctrl: Optional[int]
    lr: Optional[float]
    phi_normalization: Optional[str]
    control_cost_normalization: Optional[str]
    delta_t_lambda_scaling: Optional[str]
    optimizer: Optional[str]
    warm_start: Optional[bool]
    grad_clip: Optional[float]
    record_loss_history: bool

    # -- reproducibility --------------------------------------------------------------
    seed: int
    replicate: int
    seed_recipes: Dict[str, str]
    hyperparameter_sources: Dict[str, str]

    # -- per-image cost estimates -----------------------------------------------------
    expected_control_iterations: int
    expected_model_evals: int
    expected_backprops: int

    # -- PnP-Flow ---------------------------------------------------------------------
    num_pnp_steps: Optional[int] = None
    gamma0: Optional[float] = None
    alpha: Optional[float] = None
    noise_samples: Optional[int] = None

    # -- D-Flow / RHSO ----------------------------------------------------------------
    num_opt_steps: Optional[int] = None

    # -- RHSO ---------------------------------------------------------------------------
    num_rhso_steps: Optional[int] = None
    # State-anchor regularisation weight. 0.0 = off = the objective RHSO had before this
    # field existed; None for every method that is not RHSO.
    mu: Optional[float] = None
    # RESOLVED terminal planner: "direct" | "suffix", never "auto". The resolution against
    # the model's dynamics family happens once, here, so nothing downstream re-derives it.
    rhso_terminal_mode: Optional[str] = None
    # What that planner IS, named honestly per family:
    #   "learned_finite_interval_map"      pMF / iMF: T_theta(q; s_k -> 1)
    #   "direct_clean_endpoint_prediction" JiT: x_hat_1(q, s_k)
    #   "suffix_integration"               the legacy integrated remaining suffix
    rhso_terminal_planner: Optional[str] = None
    # What ||p_k - r_k|| would MEAN for this job: a MeanFlow semigroup defect and a direct
    # predictor's endpoint shift are different claims about different objects.
    rhso_consistency_kind: Optional[str] = None
    # Theory-validation diagnostics. All default to off; they are EXTRA measurements and
    # are never counted as algorithmic compute.
    rhso_stage_diagnostics: Optional[bool] = None
    rhso_consistency_diagnostics: Optional[bool] = None
    rhso_jacobian_diagnostics: Optional[bool] = None
    rhso_jacobian_probes: Optional[int] = None
    rhso_jacobian_power_iters: Optional[int] = None
    rhso_jacobian_seed: Optional[int] = None
    # TASK-ALIGNED endpoint authority (see src/rhso_diagnostics.probe_gradient_authority).
    # A separate measurement from the random-direction probes above, and like them an
    # EXTRA one: it never enters the objective, the gradients or the executed trajectory.
    rhso_gradient_authority_diagnostics: Optional[bool] = None
    # Time dependence of the state-anchor weight ("constant" | "increasing" |
    # "decreasing"); None for every method that is not RHSO.
    mu_schedule: Optional[str] = None

    # -- resolved schedule metadata (derived; see src/schedule.py) ----------------------
    # delta_nominal_uniform is (1 - s0)/N for this method's own trajectory and is reported
    # for EVERY method; delta_min / delta_max are the true extreme step sizes.
    delta_nominal_uniform: Optional[float] = None
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None

    # -- further cost estimates (new methods) -----------------------------------------
    expected_objective_evals: int = 0
    expected_data_gradients: int = 0
    expected_denoiser_samples: int = 0

    # -- measurement realisation -------------------------------------------------------
    # OPT-IN common-random-number pairing group, or None (the default) for the seeding this
    # repository has always used.  Recorded on every row so a paired sweep is auditable.
    measurement_noise_group: Optional[str] = None

    # -- Flower -------------------------------------------------------------------------
    # None for every other method, so no existing job's resolved spec changes meaning.
    num_flower_steps: Optional[int] = None
    flower_gamma: Optional[int] = None
    flower_cg_max_iters: Optional[int] = None
    flower_cg_tol: Optional[float] = None

    @property
    def is_mpc(self) -> bool:
        return METHOD_DECLARATIONS[self.method].is_mpc

    @property
    def num_chunks(self) -> int:
        """Batched passes covering this job's images (one network call serves a batch)."""
        return max(1, -(-int(self.num_images) // max(1, int(self.batch_size))))

    @property
    def method_title(self) -> str:
        if self.method == "mpc_rhc":
            return "MPC-RHC (K=%d)" % (self.K or 1)
        if self.method == "mpc_delta_t":
            return "MPC-delta_t"
        if self.method == "pnp":
            return "PnP-Flow"
        if self.method == "dflow":
            return "D-Flow"
        if self.method == "rhso":
            return "RHSO"
        if self.method == "flower":
            return "Flower"
        return "SDEdit"

    @property
    def reconstruction_steps(self) -> int:
        """Outer trajectory / correction intervals, whatever the method calls them.

        SDEdit and D-Flow  : integration (or transition) intervals from s_start to 1.
        MPC                : receding-horizon replans.
        PnP                : correction cycles AFTER the initial prior projection.
        """
        if self.method == "sdedit":
            return int(self.steps)
        if self.method == "dflow":
            return int(self.steps)
        if self.method == "pnp":
            return int(self.num_pnp_steps)
        if self.method == "rhso":
            return int(self.num_rhso_steps)
        if self.method == "flower":
            return int(self.num_flower_steps)
        return int(self.num_mpc_steps)

    @property
    def leaf_dir(self) -> str:
        """Readable, collision-free directory name for one atomic job."""
        parts = ["t0=%.3f" % self.t0]
        if self.method == "sdedit":
            parts += ["steps=%d" % self.steps]
            if self.solver:
                parts.append(self.solver)
        elif self.method == "pnp":
            parts += ["N=%d" % self.num_pnp_steps, "g0=%g" % self.gamma0,
                      "a=%g" % self.alpha, "M=%d" % self.noise_samples]
            if self.phi_normalization != PER_MEASUREMENT_NORMALIZATION:
                parts.append("phi=%s" % self.phi_normalization)
        elif self.method == "dflow":
            parts += ["steps=%d" % self.steps]
            if self.solver:
                parts.append(self.solver)
            parts += ["opt=%d" % self.num_opt_steps, "lr=%g" % self.lr, self.optimizer]
            if self.phi_normalization != PER_MEASUREMENT_NORMALIZATION:
                parts.append("phi=%s" % self.phi_normalization)
        elif self.method == "rhso":
            parts += ["N=%d" % self.num_rhso_steps]
            if self.solver:
                parts.append(self.solver)
            parts += ["opt=%d" % self.num_opt_steps, "lr=%g" % self.lr, self.optimizer]
            # Omitted at the default, exactly like beta: mu=0 keeps the directory names a
            # pre-mu run would have produced, and job_id disambiguates regardless.
            if self.mu:
                parts.append("mu=%g" % self.mu)
            # Named only when it is NOT the constant default, so pre-existing paths stay.
            if self.mu_schedule and self.mu_schedule != RHSO_DEFAULTS["mu_schedule"]:
                parts.append("musched=%s" % self.mu_schedule)
            # Likewise for the terminal planner: named only when it is NOT the one this
            # family already used, so pre-existing paths are unchanged. A pMF job is
            # `direct` because it always was; a JiT job saying `direct` is the new planner
            # and says so in its path.
            if (self.rhso_terminal_mode
                    and self.rhso_terminal_mode != _legacy_terminal_mode(
                        self.dynamics_family)):
                parts.append("terminal=%s" % self.rhso_terminal_mode)
            flags = "".join(letter for letter, on in
                            (("s", self.rhso_stage_diagnostics),
                             ("c", self.rhso_consistency_diagnostics),
                             ("j", self.rhso_jacobian_diagnostics),
                             ("a", self.rhso_gradient_authority_diagnostics)) if on)
            if flags:
                parts.append("diag=%s" % flags)
            if self.phi_normalization != PER_MEASUREMENT_NORMALIZATION:
                parts.append("phi=%s" % self.phi_normalization)
        elif self.method == "flower":
            parts += ["N=%d" % self.num_flower_steps, "gamma=%d" % self.flower_gamma]
            if (self.flower_cg_max_iters != FLOWER_DEFAULTS["flower_cg_max_iters"]
                    or self.flower_cg_tol != FLOWER_DEFAULTS["flower_cg_tol"]):
                parts.append("cg=%d_%g" % (self.flower_cg_max_iters, self.flower_cg_tol))
        else:
            parts += ["N=%d" % self.num_mpc_steps]
            if self.K is not None:
                parts.append("K=%d" % self.K)
            parts += ["lam=%g" % self.lam, "nctrl=%d" % self.n_ctrl, "lr=%g" % self.lr,
                      self.optimizer]
            if self.warm_start:
                parts.append("warm")
            if self.grad_clip:
                parts.append("clip=%g" % self.grad_clip)
            if self.phi_normalization != "half_sum_squared":
                parts.append("phi=%s" % self.phi_normalization)
            if self.control_cost_normalization != "sum_squared":
                parts.append("cc=%s" % self.control_cost_normalization)
            if self.delta_t_lambda_scaling not in (None, "none"):
                parts.append("lamscale=%s" % self.delta_t_lambda_scaling)
        # beta changes the executed trajectory, so it belongs in the directory name; it is
        # omitted at the default so existing paths are unchanged.
        if float(self.beta) != DEFAULT_BETA:
            parts.append("beta=%g" % float(self.beta))
        if self.replicate:
            parts.append("rep=%d" % self.replicate)
        return "__".join(parts) + "__" + self.job_id[:8]

    @property
    def label(self) -> str:
        if self.method == "sdedit":
            return "%s | %s/%s | SDEdit | t0=%.2f steps=%d %s%s" % (
                self.experiment, self.model, self.problem, self.t0, self.steps,
                self.solver or "", self._beta_suffix)
        if self.method == "pnp":
            return ("%s | %s/%s | PnP-Flow | t0=%.2f N=%d gamma0=%g alpha=%g M=%d%s"
                    % (self.experiment, self.model, self.problem, self.t0, self.num_pnp_steps,
                       self.gamma0, self.alpha, self.noise_samples, self._beta_suffix))
        if self.method == "dflow":
            return ("%s | %s/%s | D-Flow | t0=%.2f steps=%d%s opt=%d lr=%g%s"
                    % (self.experiment, self.model, self.problem, self.t0, self.steps,
                       (" %s" % self.solver) if self.solver else "", self.num_opt_steps,
                       self.lr, self._beta_suffix))
        if self.method == "rhso":
            return ("%s | %s/%s | RHSO[%s] | t0=%.2f N=%d%s opt=%d lr=%g%s%s%s"
                    % (self.experiment, self.model, self.problem,
                       self.rhso_terminal_mode or "auto", self.t0,
                       self.num_rhso_steps, (" %s" % self.solver) if self.solver else "",
                       self.num_opt_steps, self.lr,
                       (" mu=%g" % self.mu) if self.mu else "", self._beta_suffix,
                       self._diagnostic_suffix
                       + ((" mu_schedule=%s" % self.mu_schedule)
                          if self.mu_schedule
                          and self.mu_schedule != RHSO_DEFAULTS["mu_schedule"] else "")))
        if self.method == "flower":
            return ("%s | %s/%s | Flower | t0=%.2f N=%d gamma=%d%s"
                    % (self.experiment, self.model, self.problem, self.t0,
                       self.num_flower_steps, self.flower_gamma, self._beta_suffix))
        return ("%s | %s/%s | %s | t0=%.2f N=%d %s=%.4g lam=%g nctrl=%d lr=%g%s"
                % (self.experiment, self.model, self.problem, self.method_title, self.t0,
                   self.num_mpc_steps,
                   "delta" if self.delta is not None else "dt_max",
                   self.delta if self.delta is not None else (self.delta_max or 0.0),
                   self.lam, self.n_ctrl, self.lr, self._beta_suffix))

    @property
    def _diagnostic_suffix(self) -> str:
        """' +diag(...)' naming the EXTRA measurements this job takes, or ''.

        Printed separately from the method's own settings because none of it changes the
        reconstruction -- it only adds measurements about it.
        """
        on = [name for name, flag in
              (("stage", self.rhso_stage_diagnostics),
               ("consistency", self.rhso_consistency_diagnostics),
               ("jacobian", self.rhso_jacobian_diagnostics),
               ("gradient_authority", self.rhso_gradient_authority_diagnostics)) if flag]
        return (" +diag(%s)" % ",".join(on)) if on else ""

    @property
    def _beta_suffix(self) -> str:
        """' beta=...' only when the schedule is NOT the legacy uniform one."""
        return "" if float(self.beta) == DEFAULT_BETA else " beta=%g" % float(self.beta)

    def figure_title(self) -> str:
        """Compact multi-line label for comparison grids (section 40 of the brief)."""
        lines = [MODEL_REGISTRY_DEFAULTS[self.model]["display_name"]]
        if self.method == "sdedit":
            lines.append("SDEdit")
            lines.append("t0=%.2g n=%d" % (self.t0, self.steps))
            if self.solver:
                lines.append(self.solver)
        elif self.method == "mpc_rhc":
            lines.append("MPC-RHC")
            lines.append("t0=%.2g K=%d" % (self.t0, self.K or 1))
            lines.append("N=%d lam=%g" % (self.num_mpc_steps, self.lam))
        elif self.method == "pnp":
            lines.append("PnP-Flow")
            lines.append("t0=%.2g N=%d" % (self.t0, self.num_pnp_steps))
            lines.append("g0=%g a=%g M=%d" % (self.gamma0, self.alpha, self.noise_samples))
        elif self.method == "dflow":
            lines.append("D-Flow")
            lines.append("t0=%.2g n=%d" % (self.t0, self.steps))
            lines.append("opt=%d lr=%g" % (self.num_opt_steps, self.lr))
        elif self.method == "rhso":
            lines.append("RHSO")
            lines.append("t0=%.2g N=%d" % (self.t0, self.num_rhso_steps))
            lines.append("opt=%d lr=%g" % (self.num_opt_steps, self.lr))
            # A mu sweep is the point of this field, so panels must never merge over it.
            if self.mu:
                lines.append("mu=%g" % self.mu)
            if self.mu_schedule and self.mu_schedule != RHSO_DEFAULTS["mu_schedule"]:
                lines.append("mu %s" % self.mu_schedule)
        elif self.method == "flower":
            lines.append("Flower")
            lines.append("t0=%.2g N=%d" % (self.t0, self.num_flower_steps))
            lines.append("gamma=%d" % self.flower_gamma)
        else:
            lines.append("MPC-dt")
            lines.append("t0=%.2g N=%d" % (self.t0, self.num_mpc_steps))
            lines.append("lam=%g" % self.lam)
        # Never merge beta-distinct configurations into one panel label.
        if float(self.beta) != DEFAULT_BETA:
            lines.append("beta=%g" % float(self.beta))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemRequest:
    """One shared inverse-problem specification, realised exactly once."""
    key: str
    problem: str
    params: Dict[str, Any]
    num_images: int
    seed: int
    guide_mode: str
    experiments: Tuple[str, ...]
    # Optional common-random-number pairing group.  Defaulted so a caller that predates the
    # field -- including a test building a request by hand -- keeps the original seeding.
    measurement_noise_group: Optional[str] = None


@dataclass
class ResourceRequirements:
    models: Tuple[str, ...]
    frameworks: Tuple[str, ...]
    repositories: Dict[str, Dict[str, Any]]
    packages: Tuple[str, ...]
    accelerators_supported: Dict[str, Tuple[str, ...]]
    data_source: str
    source_pool_size: int
    problems: Tuple[str, ...]
    needs_lpips: bool


@dataclass
class RunPlan:
    run_id: str
    created: str
    seed: int
    replicate: int
    output_dir: str
    cache_root: str
    resources: ResourceRequirements
    problems: Tuple[ProblemRequest, ...]
    specs: Tuple[JobSpec, ...]
    warnings: Tuple[str, ...]
    workload: Dict[str, Any]
    raw_config: Dict[str, Any]

    def specs_for_model(self, model: str) -> List[JobSpec]:
        return [s for s in self.specs if s.model == model]

    def scheduled_models(self) -> List[str]:
        """Model-major PHYSICAL order, deliberately distinct from experiment order."""
        return [m for m in MODEL_NAMES if any(s.model == m for s in self.specs)]

    def experiment_names(self) -> List[str]:
        """Logical order: the order the user wrote the experiments in."""
        return [name for name in self.raw_config["experiments"]
                if any(s.experiment == name for s in self.specs)]

    def problem_request(self, key: str) -> ProblemRequest:
        for p in self.problems:
            if p.key == key:
                return p
        raise KeyError("No problem request %r in the plan." % key)

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "created": self.created, "seed": self.seed,
                "replicate": self.replicate, "output_dir": self.output_dir,
                "cache_root": self.cache_root, "resources": asdict(self.resources),
                "problems": [asdict(p) for p in self.problems],
                "jobs": [s.to_dict() for s in self.specs],
                "warnings": list(self.warnings), "workload": self.workload}


# =====================================================================================
# Inheritance and sweep expansion
# =====================================================================================
BUILTIN_DEFAULTS: Dict[str, Any] = {
    "t0": 0.8,
    "beta": DEFAULT_BETA,                        # 1.0 -> the legacy uniform time grid
    "steps": 50,
    "solver": None,                              # -> the model's default_solver
    "num_mpc_steps": 4,
    "phi_normalization": "half_sum_squared",     # MPC only; PnP / D-Flow default below
    "control_cost_normalization": "sum_squared",
    "delta_t_lambda_scaling": "none",
    "optimizer": "adam",
    "warm_start": False,
    "grad_clip": None,
    "record_loss_history": True,
    # PnP-Flow / D-Flow
    "num_pnp_steps": PNP_DEFAULTS["num_pnp_steps"],
    "gamma0": PNP_DEFAULTS["gamma0"],
    "alpha": PNP_DEFAULTS["alpha"],
    "noise_samples": PNP_DEFAULTS["noise_samples"],
    "num_opt_steps": DFLOW_DEFAULTS["num_opt_steps"],
    "pnp_phi_normalization": PER_MEASUREMENT_NORMALIZATION,
    "dflow_steps": DFLOW_DEFAULTS["steps"],
    "dflow_lr": DFLOW_DEFAULTS["lr"],
    "dflow_optimizer": DFLOW_DEFAULTS["optimizer"],
    # RHSO
    "num_rhso_steps": RHSO_DEFAULTS["num_rhso_steps"],
    "rhso_num_opt_steps": RHSO_DEFAULTS["num_opt_steps"],
    "rhso_lr": RHSO_DEFAULTS["lr"],
    "rhso_optimizer": RHSO_DEFAULTS["optimizer"],
    "mu": RHSO_DEFAULTS["mu"],
    "mu_schedule": RHSO_DEFAULTS["mu_schedule"],
    "rhso_terminal_mode": RHSO_DEFAULTS["rhso_terminal_mode"],
    "rhso_stage_diagnostics": RHSO_DEFAULTS["rhso_stage_diagnostics"],
    "rhso_consistency_diagnostics": RHSO_DEFAULTS["rhso_consistency_diagnostics"],
    "rhso_jacobian_diagnostics": RHSO_DEFAULTS["rhso_jacobian_diagnostics"],
    "rhso_gradient_authority_diagnostics":
        RHSO_DEFAULTS["rhso_gradient_authority_diagnostics"],
    "rhso_jacobian_probes": DEFAULT_JACOBIAN_PROBES,
    "rhso_jacobian_power_iters": DEFAULT_JACOBIAN_POWER_ITERATIONS,
    "rhso_jacobian_seed": DEFAULT_JACOBIAN_SEED,
    # Flower
    "num_flower_steps": FLOWER_DEFAULTS["num_flower_steps"],
    "flower_gamma": FLOWER_DEFAULTS["flower_gamma"],
    "flower_cg_max_iters": FLOWER_DEFAULTS["flower_cg_max_iters"],
    "flower_cg_tol": FLOWER_DEFAULTS["flower_cg_tol"],
}

# Every field a JobSpec can carry from the sweep grid.  Fields a method does not use are
# resolved to None with source "not_applicable", so a result row never implies a value that
# played no part in the run.
SWEEP_ORDER: Tuple[str, ...] = (
    "t0", "beta", "steps", "solver", "K", "num_mpc_steps", "lam", "n_ctrl", "lr",
    "optimizer", "warm_start", "grad_clip", "phi_normalization",
    "control_cost_normalization", "delta_t_lambda_scaling",
    "num_pnp_steps", "gamma0", "alpha", "noise_samples", "num_opt_steps",
    "num_rhso_steps", "mu",
    "rhso_terminal_mode", "rhso_stage_diagnostics", "rhso_consistency_diagnostics",
    "rhso_jacobian_diagnostics", "rhso_jacobian_probes", "rhso_jacobian_power_iters",
    "rhso_jacobian_seed",
    # Appended, never inserted: a new axis with a single value leaves the Cartesian
    # product's ORDER -- and therefore the resolved job list -- exactly as it was.
    "rhso_gradient_authority_diagnostics",
    # Flower: appended for the same reason.
    "num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol",
    # RHSO anchor schedule: appended for the same reason.
    "mu_schedule",
)


def _inherit(field_name: str, layers: Sequence[Tuple[str, Dict[str, Any]]],
             fallback: Any, fallback_source: str):
    """Walk the precedence chain lowest -> highest; return (value, source)."""
    value, source = fallback, fallback_source
    for label, block in layers:
        if block and field_name in block:
            value, source = block[field_name], label
    return value, source


def _estimate_cost(method: str, values: Dict[str, Any], dynamics_family: str,
                   euler_final_step_for_heun: bool = False) -> Dict[str, int]:
    """Per-chunk compute estimate.  Measured counts are recorded alongside these at run time.

    SDEdit:   standard flow -> stage evaluations per step (euler 1, heun 2, rk4 4);
              MeanFlow      -> one learned transition per interval.
    MPC-RHC:  1 hoisted eval per replan + n_ctrl*(K-1) differentiable evals
              + MeanFlow only: 1 extra execution transition T(x; s -> s+delta).
    MPC-dt:   1 hoisted eval + n_ctrl look-ahead evals per replan, except the FINAL replan
              where s+delta = 1 makes the projection factor exactly zero.
    PnP:      1 initial prior projection + N*M denoiser evaluations; N data gradients.
              A denoiser evaluation is ONE network call for both families (a single
              velocity for standard flows, a single transition for MeanFlows) -- PnP never
              integrates an ODE.
    D-Flow:   num_opt_steps objective evaluations plus ONE final terminal evaluation with
              the optimised q, each costing a whole trajectory; the backward pass traverses
              the same model calls again, which is what `backprops` counts.
    Flower:   N destinations, one network call each, and ZERO backprops.  Its H / H^T
              applications depend on the operator and on how many CG iterations a
              deblurring solve takes, so no data-gradient estimate is made: the measured
              `data_gradient_evaluations` and `linear_solve_iterations` are the record.
    RHSO:     delegated to `rhso.rhso_cost_estimate`, which lives next to the loop it
              describes so the two cannot drift apart.  MeanFlow: N*M planning evaluations
              and N*(M+1) in total.  Standard flow: the solver's stage counts over each
              SHRINKING remaining suffix, plus one executed interval per stage -- never a
              crude N*M estimate.
    """
    def solver_evals(steps: int) -> int:
        if dynamics_family != STANDARD_FLOW:
            return int(steps)              # one learned transition per interval
        n = int(steps) * SOLVER_STAGE_EVALUATIONS.get(values.get("solver") or "euler", 1)
        if values.get("solver") == "heun" and euler_final_step_for_heun:
            n -= 1                         # the official JiT sampler's last step is Euler
        return n

    empty = {"control_iterations": 0, "model_evals": 0, "backprops": 0,
             "objective_evals": 0, "data_gradients": 0, "denoiser_samples": 0}

    if method == "sdedit":
        return dict(empty, model_evals=solver_evals(values["steps"]))

    if method == "pnp":
        n = int(values["num_pnp_steps"])
        m = int(values["noise_samples"])
        return dict(empty, model_evals=1 + n * m, objective_evals=n, data_gradients=n,
                    denoiser_samples=1 + n * m)

    if method == "flower":
        n = int(values["num_flower_steps"])
        return dict(empty, model_evals=n, denoiser_samples=n)

    if method == "rhso":
        from .rhso import rhso_cost_estimate       # local: rhso imports config at module scope
        est = rhso_cost_estimate(values, dynamics_family, euler_final_step_for_heun)
        return dict(empty, control_iterations=est["optimizer_iterations"],
                    model_evals=est["model_evals"], backprops=est["backprops"],
                    objective_evals=est["objective_evals"],
                    data_gradients=est["objective_evals"])

    if method == "dflow":
        per_objective = solver_evals(values["steps"])
        iterations = int(values["num_opt_steps"])
        return dict(empty,
                    control_iterations=iterations,
                    # +1 terminal evaluation with the FINAL optimised q
                    model_evals=per_objective * (iterations + 1),
                    backprops=per_objective * iterations,
                    objective_evals=iterations + 1,
                    data_gradients=iterations)

    num_steps = int(values["num_mpc_steps"])
    n_ctrl = int(values.get("n_ctrl") or 0)
    if method == "mpc_rhc":
        k = int(values.get("K") or 1)
        per_replan_planning = n_ctrl * max(0, k - 1)
        per_replan_total = 1 + per_replan_planning + (1 if dynamics_family == MEANFLOW else 0)
        return dict(empty, control_iterations=n_ctrl * num_steps,
                    model_evals=per_replan_total * num_steps,
                    backprops=per_replan_planning * num_steps,
                    objective_evals=n_ctrl * num_steps)
    planning_steps = max(0, num_steps - 1)
    return dict(empty, control_iterations=n_ctrl * num_steps,
                model_evals=num_steps + n_ctrl * planning_steps,
                backprops=n_ctrl * planning_steps,
                objective_evals=n_ctrl * num_steps)


def resolve_run_plan(config: Dict[str, Any], warnings_: Sequence[str] = (),
                     run_id: Optional[str] = None) -> RunPlan:
    """CONFIG -> fully resolved atomic JobSpec objects.  Nothing expensive happens here."""
    warnings_ = list(warnings_)
    rt = config["runtime"]
    seed = int(rt.get("seed", 42))
    replicate = int(rt.get("replicate", 0))
    run_id = run_id or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(rt.get("output_root", "outputs"), run_id)
    cache_root = rt.get("cache_root", "cache")
    global_defaults = config.get("defaults") or {}

    specs: List[JobSpec] = []
    problem_entries: Dict[str, Dict[str, Any]] = {}
    pool = 0
    seen_hp_warnings = set()

    for exp_name, block in config["experiments"].items():
        if not block.get("enabled"):
            continue
        problem = block["problem"]
        decl = PROBLEM_DECLARATIONS[problem]
        num_images = int(block["num_images"])
        pool = max(pool, num_images)

        problem_params = dict(decl.default_params)
        problem_params.update(block.get("degradation") or {})
        params_key = canonical_params_key(problem_params)
        noise_group = block.get("measurement_noise_group") or None
        # The problem identity depends on (problem, parameters, seed) ONLY -- never on the
        # experiment name, the model or the method, so two experiments asking for the same
        # specification provably share one measurement.
        # The pairing group takes part ONLY when it is set, so a configuration that never
        # mentions it keeps the problem keys -- and with them the shared measurements and
        # the output paths -- it had before the field existed.  Two experiments that differ
        # only in their group are two different measurement realisations and must not share
        # one InverseProblem.
        pkey = "prob_" + (stable_hash(problem, params_key, seed, "group", noise_group)
                          if noise_group else stable_hash(problem, params_key, seed))
        entry = problem_entries.setdefault(pkey, {
            "key": pkey, "problem": problem, "params": problem_params, "num_images": num_images,
            "seed": seed, "guide_mode": decl.guide_mode, "experiments": [],
            "measurement_noise_group": noise_group})
        entry["experiments"].append(exp_name)
        # A larger request supersedes a smaller one: same seed, same per-image draws, so the
        # smaller set is a prefix of the larger.
        entry["num_images"] = max(entry["num_images"], num_images)

        exp_defaults = block.get("defaults") or {}

        for model_name, model_entry in (block.get("models") or {}).items():
            reg = resolve_model_registry(config, model_name)
            caps = MODEL_CAPABILITIES[model_name]
            model_entry = model_entry or {}

            guidance = deep_merge(dict(reg["guidance"]), model_entry.get("guidance"))
            if model_name == "jit" and guidance.get("scale") is None:
                guidance = dict(guidance, scale=reg["recommended_cfg"].get(reg["variant"]),
                                scale_source="model_size_recommendation")
            batch_size = int(model_entry.get("batch_size", reg["batch_size"]))

            for method_name, method_entry in (model_entry.get("methods") or {}).items():
                method_entry = method_entry or {}
                mdecl = METHOD_DECLARATIONS[method_name]
                layers = [("defaults", global_defaults), ("experiment", exp_defaults),
                          ("model", model_entry), ("method", method_entry)]

                # Start from "this field played no part in this method", then switch on the
                # ones the method actually uses.  A field that stays not_applicable is None
                # in the resolved spec, the result row and the job fingerprint.
                axes: Dict[str, List[Any]] = {name: [None] for name in SWEEP_ORDER}
                axis_sources: Dict[str, str] = {name: "not_applicable" for name in SWEEP_ORDER}

                def axis(name: str, fallback: Any, fallback_source: str = "builtin") -> None:
                    value, source = _inherit(name, layers, fallback, fallback_source)
                    axes[name] = as_sweep_list(value)
                    axis_sources[name] = source

                axis("t0", BUILTIN_DEFAULTS["t0"])
                # Shared with t0: every method discretises time, so every method has a beta.
                axis("beta", BUILTIN_DEFAULTS["beta"])
                if method_name == "sdedit":
                    axis("steps", BUILTIN_DEFAULTS["steps"])
                    axis("solver", reg.get("default_solver"), "model_registry")
                elif method_name == "pnp":
                    for name in ("num_pnp_steps", "gamma0", "alpha", "noise_samples"):
                        axis(name, BUILTIN_DEFAULTS[name], UNTUNED)
                    axis("phi_normalization", BUILTIN_DEFAULTS["pnp_phi_normalization"])
                elif method_name == "rhso":
                    axis("num_rhso_steps", BUILTIN_DEFAULTS["num_rhso_steps"], UNTUNED)
                    axis("num_opt_steps", BUILTIN_DEFAULTS["rhso_num_opt_steps"], UNTUNED)
                    axis("lr", BUILTIN_DEFAULTS["rhso_lr"], UNTUNED)
                    # `builtin`, not UNTUNED: 0.0 is not a guess at a good weight, it is the
                    # feature being off. A row whose source reads `builtin` had no penalty.
                    axis("mu", BUILTIN_DEFAULTS["mu"])
                    # `builtin`: "constant" is the time-independent weight, not a guess.
                    axis("mu_schedule", BUILTIN_DEFAULTS["mu_schedule"])
                    axis("optimizer", BUILTIN_DEFAULTS["rhso_optimizer"])
                    # `builtin` again, not UNTUNED: "auto" reproduces the family's existing
                    # planner and every diagnostic starts off.
                    for name in ("rhso_terminal_mode", "rhso_stage_diagnostics",
                                 "rhso_consistency_diagnostics",
                                 "rhso_jacobian_diagnostics", "rhso_jacobian_probes",
                                 "rhso_jacobian_power_iters", "rhso_jacobian_seed",
                                 "rhso_gradient_authority_diagnostics"):
                        axis(name, BUILTIN_DEFAULTS[name])
                    # MeanFlow adapters have default_solver None, so a MeanFlow RHSO job
                    # resolves to no solver and the validator rejects any explicit one.
                    axis("solver", reg.get("default_solver"), "model_registry")
                    axis("phi_normalization", BUILTIN_DEFAULTS["pnp_phi_normalization"])
                elif method_name == "dflow":
                    axis("steps", BUILTIN_DEFAULTS["dflow_steps"], UNTUNED)
                    # MeanFlow adapters have default_solver None, so this resolves to None
                    # and the validator rejects any explicit solver for them.
                    axis("solver", reg.get("default_solver"), "model_registry")
                    axis("num_opt_steps", BUILTIN_DEFAULTS["num_opt_steps"], UNTUNED)
                    axis("lr", BUILTIN_DEFAULTS["dflow_lr"], UNTUNED)
                    axis("optimizer", BUILTIN_DEFAULTS["dflow_optimizer"])
                    axis("phi_normalization", BUILTIN_DEFAULTS["pnp_phi_normalization"])
                elif method_name == "flower":
                    axis("num_flower_steps", BUILTIN_DEFAULTS["num_flower_steps"], UNTUNED)
                    # `builtin`: gamma = 0 and the CG limits are the paper's settings
                    for name in ("flower_gamma", "flower_cg_max_iters", "flower_cg_tol"):
                        axis(name, BUILTIN_DEFAULTS[name])
                else:
                    axis("num_mpc_steps", BUILTIN_DEFAULTS["num_mpc_steps"])
                    for name in ("optimizer", "warm_start", "grad_clip", "phi_normalization",
                                 "control_cost_normalization"):
                        axis(name, BUILTIN_DEFAULTS[name])
                    if mdecl.uses_K:
                        axis("K", 1)
                    if method_name == "mpc_delta_t":
                        axis("delta_t_lambda_scaling",
                             BUILTIN_DEFAULTS["delta_t_lambda_scaling"])
                    # lam / n_ctrl / lr have no builtin fallback: their default depends on
                    # (problem, method, K) and is resolved per combination below.
                    for name in ("lam", "n_ctrl", "lr"):
                        value, source = _inherit(name, layers, "__ABSENT__", "table")
                        axes[name] = (["__ABSENT__"] if value == "__ABSENT__"
                                      else as_sweep_list(value))
                        axis_sources[name] = source

                record_loss, _ = _inherit("record_loss_history", layers,
                                          BUILTIN_DEFAULTS["record_loss_history"], "builtin")

                grids = []
                for name in SWEEP_ORDER:
                    seen_vals, uniq = set(), []
                    for v in axes[name]:
                        if repr(v) not in seen_vals:
                            seen_vals.add(repr(v))
                            uniq.append(v)
                    grids.append(uniq)

                for combo in itertools.product(*grids):
                    values = dict(zip(SWEEP_ORDER, combo))
                    t0 = float(values["t0"])
                    beta = resolve_beta(values["beta"])
                    values["beta"] = beta
                    s_start = canonical_start_time(t0)
                    sources = dict(axis_sources)

                    if mdecl.is_mpc:
                        n_steps = int(values["num_mpc_steps"])
                        # NOMINAL only: with beta != 1 there is no single step size, so the
                        # scalar is dropped rather than left to be misread as one.
                        delta = (t0 / n_steps) if beta == DEFAULT_BETA else None
                        fallback, label, hp_warn = resolve_mpc_hyperparameters(
                            problem, method_name, values["K"])
                        for w in hp_warn:
                            tag = (problem, method_name, values["K"], w)
                            if tag not in seen_hp_warnings:
                                seen_hp_warnings.add(tag)
                                warnings_.append("experiments.%s.models.%s.methods.%s: %s"
                                                 % (exp_name, model_name, method_name, w))
                        hp = {}
                        for name in ("lam", "n_ctrl", "lr"):
                            if values[name] == "__ABSENT__":
                                hp[name] = fallback[name]
                                sources[name] = label
                            else:
                                hp[name] = values[name]
                    else:
                        n_steps = None
                        delta = None
                        # D-Flow and RHSO use lr; SDEdit and PnP do not.
                        hp = {"lam": None, "n_ctrl": None, "lr": values["lr"]}
                    values["lam"], values["n_ctrl"], values["lr"] = \
                        hp["lam"], hp["n_ctrl"], hp["lr"]

                    # The ACTUAL resolved schedule of this job's own trajectory: the real
                    # per-step sizes, for metadata and for the structural checks.  PnP's N
                    # corrections sit inside a grid of N + 1 intervals.
                    trajectory_intervals = {
                        "sdedit": values["steps"], "dflow": values["steps"],
                        "mpc_rhc": n_steps, "mpc_delta_t": n_steps,
                        "rhso": values["num_rhso_steps"],
                        "pnp": (int(values["num_pnp_steps"]) + 1)
                        if values["num_pnp_steps"] is not None else None,
                        "flower": values["num_flower_steps"],
                    }[method_name]
                    if trajectory_intervals:
                        dts = grid_intervals(canonical_time_grid(
                            s_start, int(trajectory_intervals), beta))
                        delta_nominal = nominal_uniform_delta(s_start,
                                                              int(trajectory_intervals))
                        delta_min, delta_max = min(dts), max(dts)
                    else:
                        delta_nominal = delta_min = delta_max = None

                    # RHSO's terminal planner is resolved ONCE, against the model's family:
                    # "auto" becomes the concrete planner this job runs, and an impossible
                    # combination (a MeanFlow asked to integrate a suffix) fails here rather
                    # than deep inside a reconstruction.
                    if method_name == "rhso":
                        from .rhso import resolve_terminal_mode
                        from .rhso_diagnostics import consistency_kind
                        try:
                            values["rhso_terminal_mode"] = resolve_terminal_mode(
                                values["rhso_terminal_mode"], reg["dynamics_family"])
                        except ValueError as exc:
                            raise ConfigError(
                                "experiments.%s.models.%s.methods.rhso: %s"
                                % (exp_name, model_name, exc))
                        terminal_planner = rhso_terminal_planner(
                            values["rhso_terminal_mode"], model_name)
                        consistency = consistency_kind(reg["dynamics_family"],
                                                       values["rhso_terminal_mode"])
                        # A consistency measurement compares two stage-terminal predictions,
                        # so it implies the stage bookkeeping (mirrors
                        # rhso_diagnostics.settings_from_spec).
                        if values["rhso_consistency_diagnostics"]:
                            values["rhso_stage_diagnostics"] = True
                        # Same rule, same reason: the authority numbers are recorded on the
                        # per-(image, stage) rows that the stage bookkeeping creates, so
                        # asking for them alone is not an error and must not silently
                        # measure nothing.
                        if values["rhso_gradient_authority_diagnostics"]:
                            values["rhso_stage_diagnostics"] = True
                    else:
                        terminal_planner = consistency = None

                    cost = _estimate_cost(method_name, values, reg["dynamics_family"],
                                          bool(reg.get("euler_final_step_for_heun", False)))
                    # Every field that can change what runs takes part in the identity, so a
                    # different gamma0 / alpha / steps can never reuse another job's result.
                    hash_parts = [run_id, exp_name, problem, params_key, model_name,
                                         method_name, values["K"], t0, values["steps"],
                                         values["solver"], n_steps, hp["lam"], hp["n_ctrl"],
                                         hp["lr"], values["optimizer"], values["warm_start"],
                                         values["grad_clip"], values["phi_normalization"],
                                         values["control_cost_normalization"],
                                         values["delta_t_lambda_scaling"],
                                         values["num_pnp_steps"], values["gamma0"],
                                         values["alpha"], values["noise_samples"],
                                         values["num_opt_steps"],
                                         # beta and the RHSO horizon change what runs, so two
                                         # jobs differing only in them can never collide.
                                         values["num_rhso_steps"], values["mu"], beta,
                                         num_images, replicate]
                    # Appended ONLY when a job uses one of the new RHSO features, so a
                    # configuration that does not touch them keeps the job ids -- and
                    # therefore the output directories and the resume behaviour -- it had
                    # before these fields existed.
                    identity_extras = []
                    if method_name == "rhso":
                        if values["rhso_terminal_mode"] != _legacy_terminal_mode(
                                reg["dynamics_family"]):
                            identity_extras.append(("terminal_mode",
                                                    values["rhso_terminal_mode"]))
                        for name in ("rhso_stage_diagnostics",
                                     "rhso_consistency_diagnostics",
                                     "rhso_jacobian_diagnostics",
                                     "rhso_gradient_authority_diagnostics"):
                            if values[name]:
                                identity_extras.append((name, True))
                        if values["rhso_jacobian_diagnostics"]:
                            identity_extras.append(
                                ("jacobian", values["rhso_jacobian_probes"],
                                 values["rhso_jacobian_power_iters"],
                                 values["rhso_jacobian_seed"]))
                        # Only a NON-constant schedule enters the identity, so every job
                        # written before the field existed keeps its id and directory.
                        if values["mu_schedule"] not in (None, RHSO_DEFAULTS["mu_schedule"]):
                            identity_extras.append(("mu_schedule", values["mu_schedule"]))
                    # Flower's own fields, appended only for Flower jobs: every other job
                    # keeps the id it had before the method existed.
                    if method_name == "flower":
                        identity_extras.append(("flower", int(values["num_flower_steps"]),
                                                int(values["flower_gamma"]),
                                                int(values["flower_cg_max_iters"]),
                                                float(values["flower_cg_tol"])))
                    if identity_extras:
                        hash_parts.append(tuple(identity_extras))
                    # Appended ONLY when the experiment opts in, so an unpaired
                    # configuration keeps every job id it had before this field existed.
                    # sigma is already inside params_key above, so two noise levels remain
                    # separate jobs whether or not they share an epsilon.
                    if noise_group:
                        hash_parts.append(("measurement_noise_group", noise_group))
                    job_id = stable_hash(*hash_parts, size=6)

                    specs.append(JobSpec(
                        job_id=job_id, experiment=exp_name, problem=problem,
                        problem_params=dict(problem_params), problem_key=pkey,
                        num_images=num_images, image_ids=(),
                        model=model_name, dynamics_family=reg["dynamics_family"],
                        framework=reg["framework"], state_space=reg["state_space"],
                        batch_size=batch_size, fixed_batch_shape=caps.fixed_batch_shape,
                        guidance=dict(guidance),
                        method=method_name,
                        K=(int(values["K"]) if values["K"] is not None else None),
                        t0=t0, beta=beta, canonical_start_time=s_start,
                        native_start_time=native_time(s_start, reg["native_time_mapping"]),
                        native_end_time=native_time(1.0, reg["native_time_mapping"]),
                        native_time_mapping=reg["native_time_mapping"],
                        initialization_guide_mode=decl.guide_mode,
                        initialization_kind=("pure_prior_noise" if t0 >= 1.0
                                             else "measurement_informed"),
                        steps=(int(values["steps"]) if values["steps"] is not None else None),
                        solver=values["solver"],
                        num_mpc_steps=n_steps, delta=delta,
                        lam=(None if hp["lam"] is None else float(hp["lam"])),
                        n_ctrl=(None if hp["n_ctrl"] is None else int(hp["n_ctrl"])),
                        lr=(None if hp["lr"] is None else float(hp["lr"])),
                        phi_normalization=values["phi_normalization"],
                        control_cost_normalization=values["control_cost_normalization"],
                        delta_t_lambda_scaling=values["delta_t_lambda_scaling"],
                        optimizer=values["optimizer"],
                        warm_start=(None if values["warm_start"] is None
                                    else bool(values["warm_start"])),
                        grad_clip=(None if values["grad_clip"] is None
                                   else float(values["grad_clip"])),
                        record_loss_history=bool(record_loss),
                        seed=seed, replicate=replicate,
                        # Flower's noise recipe is recorded on Flower jobs only, so every
                        # other job's resolved spec (and its resume match) is unchanged.
                        seed_recipes=(dict(SEED_RECIPES, flower=_flower_seed_recipe())
                                      if method_name == "flower" else dict(SEED_RECIPES)),
                        hyperparameter_sources=sources,
                        expected_control_iterations=cost["control_iterations"],
                        expected_model_evals=cost["model_evals"],
                        expected_backprops=cost["backprops"],
                        num_pnp_steps=(int(values["num_pnp_steps"])
                                       if values["num_pnp_steps"] is not None else None),
                        gamma0=(float(values["gamma0"])
                                if values["gamma0"] is not None else None),
                        alpha=(float(values["alpha"]) if values["alpha"] is not None else None),
                        noise_samples=(int(values["noise_samples"])
                                       if values["noise_samples"] is not None else None),
                        num_opt_steps=(int(values["num_opt_steps"])
                                       if values["num_opt_steps"] is not None else None),
                        num_rhso_steps=(int(values["num_rhso_steps"])
                                        if values["num_rhso_steps"] is not None else None),
                        mu=(float(values["mu"]) if values["mu"] is not None else None),
                        rhso_terminal_mode=values["rhso_terminal_mode"],
                        rhso_terminal_planner=terminal_planner,
                        rhso_consistency_kind=consistency,
                        rhso_stage_diagnostics=(
                            None if values["rhso_stage_diagnostics"] is None
                            else bool(values["rhso_stage_diagnostics"])),
                        rhso_consistency_diagnostics=(
                            None if values["rhso_consistency_diagnostics"] is None
                            else bool(values["rhso_consistency_diagnostics"])),
                        rhso_jacobian_diagnostics=(
                            None if values["rhso_jacobian_diagnostics"] is None
                            else bool(values["rhso_jacobian_diagnostics"])),
                        rhso_jacobian_probes=(
                            None if values["rhso_jacobian_probes"] is None
                            else int(values["rhso_jacobian_probes"])),
                        rhso_jacobian_power_iters=(
                            None if values["rhso_jacobian_power_iters"] is None
                            else int(values["rhso_jacobian_power_iters"])),
                        rhso_jacobian_seed=(
                            None if values["rhso_jacobian_seed"] is None
                            else int(values["rhso_jacobian_seed"])),
                        rhso_gradient_authority_diagnostics=(
                            None if values["rhso_gradient_authority_diagnostics"] is None
                            else bool(values["rhso_gradient_authority_diagnostics"])),
                        measurement_noise_group=noise_group,
                        mu_schedule=(str(values["mu_schedule"])
                                     if values["mu_schedule"] is not None else None),
                        num_flower_steps=(int(values["num_flower_steps"])
                                          if values["num_flower_steps"] is not None else None),
                        flower_gamma=(int(values["flower_gamma"])
                                      if values["flower_gamma"] is not None else None),
                        flower_cg_max_iters=(int(values["flower_cg_max_iters"])
                                             if values["flower_cg_max_iters"] is not None
                                             else None),
                        flower_cg_tol=(float(values["flower_cg_tol"])
                                       if values["flower_cg_tol"] is not None else None),
                        delta_nominal_uniform=delta_nominal,
                        delta_min=delta_min, delta_max=delta_max,
                        expected_objective_evals=cost["objective_evals"],
                        expected_data_gradients=cost["data_gradients"],
                        expected_denoiser_samples=cost["denoiser_samples"]))

    # ---------------------------------------------------------------- resources
    required = [m for m in MODEL_NAMES if any(s.model == m for s in specs)]
    frameworks = sorted({resolve_model_registry(config, m)["framework"] for m in required})
    repositories = {m: {"url": resolve_model_registry(config, m)["repo_url"],
                        "rev": resolve_model_registry(config, m).get("repo_rev"),
                        "dirname": resolve_model_registry(config, m)["repo_dirname"]}
                    for m in required}
    packages = sorted({p for m in required
                       for p in resolve_model_registry(config, m)["packages"]})
    accel_support = {m: tuple(resolve_model_registry(config, m)["accelerators"])
                     for m in required}

    resources = ResourceRequirements(
        models=tuple(required), frameworks=tuple(frameworks), repositories=repositories,
        packages=tuple(packages), accelerators_supported=accel_support,
        data_source=config["data"].get("source", "hf_imagenet_val"),
        source_pool_size=pool, problems=tuple(dict.fromkeys(s.problem for s in specs)),
        needs_lpips=bool((config.get("metrics") or {}).get("lpips", True)))

    # ---------------------------------------------------------------- workload
    def bucket(keyfn):
        out: Dict[Any, Dict[str, int]] = {}
        for s in specs:
            e = out.setdefault(keyfn(s), {"jobs": 0, "reconstructions": 0,
                                          "control_iterations": 0, "model_evals": 0,
                                          "backprops": 0, "objective_evals": 0,
                                          "data_gradients": 0})
            e["jobs"] += 1
            e["reconstructions"] += s.num_images
            e["control_iterations"] += s.expected_control_iterations * s.num_chunks
            e["model_evals"] += s.expected_model_evals * s.num_chunks
            e["backprops"] += s.expected_backprops * s.num_chunks
            e["objective_evals"] += s.expected_objective_evals * s.num_chunks
            e["data_gradients"] += s.expected_data_gradients * s.num_chunks
        return out

    workload = {
        "enabled_experiments": len({s.experiment for s in specs}),
        "atomic_jobs": len(specs),
        "image_reconstructions": sum(s.num_images for s in specs),
        "control_iterations": sum(s.expected_control_iterations * s.num_chunks for s in specs),
        "model_evaluations": sum(s.expected_model_evals * s.num_chunks for s in specs),
        "backprops_through_model": sum(s.expected_backprops * s.num_chunks for s in specs),
        "objective_evaluations": sum(s.expected_objective_evals * s.num_chunks for s in specs),
        "data_gradient_evaluations": sum(s.expected_data_gradients * s.num_chunks
                                         for s in specs),
        "denoiser_samples": sum(s.expected_denoiser_samples * s.num_chunks for s in specs),
        "image_model_evaluations": sum(s.expected_model_evals * s.num_images for s in specs),
        "per_model": bucket(lambda s: s.model),
        "per_method": bucket(lambda s: s.method),
        "per_experiment": bucket(lambda s: s.experiment),
        "per_experiment_model_method": bucket(
            lambda s: (s.experiment, s.model, s.method)),
    }

    # ---------------------------------------------------------------- plan-level warnings
    limit = int(rt.get("max_atomic_jobs", 256))
    if len(specs) > limit:
        raise ConfigError(
            "The resolved Cartesian product contains %d atomic jobs, above "
            "runtime.max_atomic_jobs = %d.\n    Shorten the sweep lists, or raise the limit "
            "deliberately (--max-jobs on the command line) if you really mean it."
            % (len(specs), limit))
    if len(specs) > max(32, limit // 4):
        warnings_.append("Large sweep: %d atomic jobs / %s image reconstructions."
                         % (len(specs), "{:,}".format(workload["image_reconstructions"])))
    if len(frameworks) > 1:
        warnings_.append("This run mixes JAX and PyTorch in one process. Models are loaded "
                         "sequentially and released between families, but JAX does not return "
                         "device memory eagerly; prefer runtime.release_model_after_use: true.")
    # Paired-comparison completeness: warn when SDEdit has no partner at some t0.
    # Sorted, because set iteration order varies with PYTHONHASHSEED and the warning list
    # is written verbatim into run_metadata.json: an unordered loop makes two identical
    # runs produce two different metadata files.
    for (exp, model) in sorted({(s.experiment, s.model) for s in specs}):
        sel = [s for s in specs if s.experiment == exp and s.model == model]
        methods = {s.method for s in sel}
        compared = methods & set(COMPARED_METHODS)
        if compared and BASELINE_METHOD not in methods:
            warnings_.append(
                "%s / %s runs %s without a paired SDEdit baseline; the headline comparison of "
                "this repository is every strategy against SDEdit at the SAME t0."
                % (exp, model.upper(), ", ".join(sorted(compared))))
        elif compared:
            other_t0 = {s.t0 for s in sel if s.method in COMPARED_METHODS}
            sde_t0 = {s.t0 for s in sel if s.method == BASELINE_METHOD}
            missing = sorted(other_t0 - sde_t0)
            if missing:
                warnings_.append(
                    "%s / %s has %s jobs at t0=%s with no SDEdit job at the same t0; those "
                    "runs have no paired baseline."
                    % (exp, model.upper(), ", ".join(sorted(compared)),
                       ", ".join("%.2f" % t for t in missing)))
    ids = [s.job_id for s in specs]
    if len(set(ids)) != len(ids):                                       # pragma: no cover
        raise ConfigError("Internal error: duplicate job ids were generated.")

    problem_requests = tuple(
        ProblemRequest(key=e["key"], problem=e["problem"], params=e["params"],
                       num_images=e["num_images"], seed=e["seed"], guide_mode=e["guide_mode"],
                       experiments=tuple(dict.fromkeys(e["experiments"])),
                       measurement_noise_group=e.get("measurement_noise_group"))
        for e in problem_entries.values())

    return RunPlan(run_id=run_id, created=datetime.datetime.now().isoformat(), seed=seed,
                   replicate=replicate, output_dir=output_dir, cache_root=cache_root,
                   resources=resources, problems=problem_requests, specs=tuple(specs),
                   warnings=tuple(warnings_), workload=workload,
                   raw_config=copy.deepcopy(config))


# =====================================================================================
# Loading and printing
# =====================================================================================
def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ConfigError("%s does not contain a YAML mapping." % path)
    return config


def check_accelerator_compatibility(plan: RunPlan, accel: Dict[str, Any]) -> List[str]:
    """Fail early for impossible model/accelerator combinations; warn about slow ones."""
    kind = accel["kind"]
    problems, notes = [], []
    for model, supported in plan.resources.accelerators_supported.items():
        if kind not in supported:
            problems.append("%s cannot run on %s (supported: %s)"
                            % (model.upper(), kind, list(supported)))
    if problems:
        raise ConfigError(
            "Unsupported accelerator for this plan:\n    " + "\n    ".join(problems)
            + "\nChange runtime.accelerator, or stop using those models.")
    if kind == "cpu":
        notes.append("Running on CPU. Viable for structural validation only: MPC and D-Flow "
                     "need many forward AND backward passes through 256x256 models.")
        notes.append("GPU peak-memory columns will be empty on CPU; the memory comparison "
                     "needs a CUDA device.")
    if kind == "tpu" and "torch" in plan.resources.frameworks:
        notes.append("A TPU was selected but this plan needs PyTorch models (SiT/JiT), which "
                     "are configured for CUDA/CPU here. They will fall back to CPU.")
    if kind == "gpu":
        vram = accel.get("gpu_memory_mb") or 0
        if vram and vram < 24_000 and len(plan.resources.models) > 1:
            notes.append("GPU has %.1f GB: models will be loaded strictly one at a time."
                         % (vram / 1024.0))
        if vram and vram < 16_000:
            notes.append("MPC-delta_t, RHC K>1 and D-Flow backpropagate through the generative "
                         "model. If you hit OOM, reduce models.<m>.batch_size to 1.")
        if "jit" in plan.resources.models and not accel.get("bf16_likely", False):
            notes.append("This GPU probably lacks BF16 (compute capability < 8.0). JiT will use "
                         "FP32 compute, which is correct but slower; FP16 is deliberately "
                         "avoided because it destabilises pixel-space generation.")
    max_K = max([s.K or 1 for s in plan.specs] or [1])
    if max_K > 3 and kind != "cpu":
        notes.append("The plan contains RHC with K up to %d: memory for the control graph "
                     "scales with K-1 nested differentiable model evaluations." % max_K)
    dflow = [s for s in plan.specs if s.method == "dflow"]
    if dflow:
        deepest = max(s.expected_model_evals // max(1, (s.num_opt_steps or 1) + 1)
                      for s in dflow)
        if deepest > 1:
            notes.append("D-Flow keeps its whole trajectory in the autograd graph: the deepest "
                         "job in this plan chains %d model evaluations per objective, and "
                         "activation memory scales with that depth. There is no gradient "
                         "checkpointing in this implementation." % deepest)
    # The suffix warning describes the LEGACY standard-flow planner only.  It is resolved
    # from `rhso_terminal_mode`, not from the dynamics family: a JiT job with
    # `rhso_terminal_mode: direct` performs ONE endpoint prediction per inner objective and
    # never integrates -- let alone differentiates -- the remaining suffix, so claiming it
    # does would be simply false.  `auto` has already been resolved to `direct`/`suffix` by
    # the planner (see resolve_run_plan); the fallback keeps a stand-in spec honest.
    rhso = [s for s in plan.specs if s.method == "rhso"]

    def _mode(s) -> str:
        mode = getattr(s, "rhso_terminal_mode", None)
        if mode in (None, "auto"):
            return _legacy_terminal_mode(s.dynamics_family)
        return str(mode)

    suffix_jobs = [s for s in rhso
                   if s.dynamics_family == STANDARD_FLOW and _mode(s) == "suffix"]
    if suffix_jobs:
        deepest = max((s.num_rhso_steps or 1) for s in suffix_jobs)
        notes.append("Standard-flow RHSO in `suffix` terminal mode differentiates the whole "
                     "REMAINING suffix at every outer stage: its first stage chains up to %d "
                     "intervals in the autograd graph (cost then falls stage by stage). "
                     "MeanFlow RHSO plans with ONE learned transition regardless of the "
                     "horizon." % deepest)
    direct_flow_jobs = [s for s in rhso
                        if s.dynamics_family == STANDARD_FLOW and _mode(s) == "direct"]
    if direct_flow_jobs:
        notes.append("Standard-flow RHSO in `direct` terminal mode plans with ONE direct "
                     "clean-endpoint prediction x_hat_1(q, s_k) per inner objective: the "
                     "remaining suffix is neither integrated nor differentiated, so the "
                     "autograd graph does not deepen with the horizon.")
    if any(s.method == "pnp" and (s.noise_samples or 1) > 1 for s in plan.specs):
        notes.append("PnP with noise_samples > 1 evaluates the denoiser M times per "
                     "correction; the M realisations run sequentially here, so runtime scales "
                     "with M while peak memory does not.")
    return notes


def print_run_plan(plan: RunPlan, accel: Optional[Dict[str, Any]] = None,
                   notes: Sequence[str] = (), detail: bool = True) -> None:
    """The dry-run report (brief section 28)."""
    w = plan.workload
    print(RULE_ := "=" * 94)
    print("RESOLVED RUN PLAN   |   %s" % plan.run_id)
    print(RULE_)
    if accel:
        print("Accelerator          : %s%s"
              % (accel["kind"], "  (%s)" % accel["gpu_name"] if accel.get("gpu_name") else ""))
    print("Required models      : %s" % (", ".join(
        "%s [%s/%s/%s]" % (m.upper(), MODEL_REGISTRY_DEFAULTS[m]["framework"],
                           MODEL_REGISTRY_DEFAULTS[m]["state_space"],
                           MODEL_REGISTRY_DEFAULTS[m]["dynamics_family"])
        for m in plan.resources.models) or "(none)"))
    print("Frameworks           : %s" % (", ".join(plan.resources.frameworks) or "(none)"))
    print("Data                 : source=%s, shared pool=%d image(s)"
          % (plan.resources.data_source, plan.resources.source_pool_size))
    print("Inverse problems     : %s" % ", ".join(plan.resources.problems))
    print("Global seed          : %d   (replicate %d)" % (plan.seed, plan.replicate))
    print("Output directory     : %s" % plan.output_dir)

    print("-" * 94)
    print("SHARED PROBLEM INSTANCES   built once, reused by every model and method")
    for p in plan.problems:
        print("  %-17s %-52s images=%d" % (
            p.problem, ", ".join("%s=%s" % (k, p.params[k]) for k in sorted(p.params)),
            p.num_images))
        print("  %-17s guide g(y) = %s   used by: %s"
              % ("", p.guide_mode, ", ".join(p.experiments)))

    # ---------------------------------------------------------------- job counts
    print("-" * 94)
    print("ATOMIC JOBS")
    for exp in plan.experiment_names():
        exp_specs = [s for s in plan.specs if s.experiment == exp]
        print("\n%s   (%s, %d image%s)"
              % (exp, exp_specs[0].problem, exp_specs[0].num_images,
                 "s" if exp_specs[0].num_images != 1 else ""))
        for model in MODEL_NAMES:
            model_specs = [s for s in exp_specs if s.model == model]
            if not model_specs:
                continue
            print("  %s" % MODEL_REGISTRY_DEFAULTS[model]["display_name"])
            for method in METHOD_NAMES:
                sel = [s for s in model_specs if s.method == method]
                if not sel:
                    continue
                title = METHOD_DECLARATIONS[method].title
                print("    %s %s %d jobs"
                      % (title, "." * max(2, 20 - len(title)), len(sel)))
                if detail:
                    print("      t0 %s | beta %s" % (_axis(sel, "t0", "%.2f"),
                                                     _axis(sel, "beta", "%g")))
                    if method == "sdedit":
                        print("      steps %s | solver %s"
                              % (_axis(sel, "steps", "%s"), _axis(sel, "solver")))
                    elif method == "pnp":
                        print("      N %s | gamma0 %s | alpha %s | M %s"
                              % (_axis(sel, "num_pnp_steps", "%s"), _axis(sel, "gamma0", "%g"),
                                 _axis(sel, "alpha", "%g"), _axis(sel, "noise_samples", "%s")))
                        print("      phi %s | source %s"
                              % (_axis(sel, "phi_normalization"),
                                 ", ".join(sorted({s.hyperparameter_sources.get("gamma0", "?")
                                                   for s in sel}))))
                    elif method == "dflow":
                        print("      trajectory steps %s | solver %s | opt steps %s | lr %s"
                              % (_axis(sel, "steps", "%s"), _axis(sel, "solver"),
                                 _axis(sel, "num_opt_steps", "%s"), _axis(sel, "lr", "%g")))
                        print("      phi %s | source %s"
                              % (_axis(sel, "phi_normalization"),
                                 ", ".join(sorted({s.hyperparameter_sources.get("lr", "?")
                                                   for s in sel}))))
                    elif method == "rhso":
                        print("      outer stages %s | solver %s | opt steps %s | lr %s"
                              % (_axis(sel, "num_rhso_steps", "%s"), _axis(sel, "solver"),
                                 _axis(sel, "num_opt_steps", "%s"), _axis(sel, "lr", "%g")))
                        print("      mu %s   (0 = no state-anchor penalty)"
                              % _axis(sel, "mu", "%g"))
                        if any(s.mu_schedule not in (None, RHSO_DEFAULTS["mu_schedule"])
                               for s in sel):
                            print("      mu schedule %s   (stage k: mu_k = mu * m(s_k))"
                                  % _axis(sel, "mu_schedule"))
                        print("      terminal planner %s -> %s"
                              % (_axis(sel, "rhso_terminal_mode"),
                                 _axis(sel, "rhso_terminal_planner")))
                        diag = sorted({(job._diagnostic_suffix.strip() or "none")
                                       for job in sel})
                        print("      theory diagnostics %s   (EXTRA measurements; excluded "
                              "from the runtime and the algorithmic counters)"
                              % ", ".join(diag))
                        if any(s.measurement_noise_group for s in sel):
                            print("      measurement noise group %s   (one epsilon per "
                                  "image across sigma)"
                                  % _axis(sel, "measurement_noise_group"))
                        if any(s.rhso_jacobian_diagnostics for s in sel):
                            print("      jacobian probes %s | power iters %s | seed %s"
                                  % (_axis(sel, "rhso_jacobian_probes", "%s"),
                                     _axis(sel, "rhso_jacobian_power_iters", "%s"),
                                     _axis(sel, "rhso_jacobian_seed", "%s")))
                        print("      (N, M) pairs %s"
                              % ", ".join(sorted({"(%d, %d)" % (s.num_rhso_steps,
                                                                s.num_opt_steps)
                                                  for s in sel})))
                        print("      executed dt %s | phi %s | source %s"
                              % (_dt_axis(sel), _axis(sel, "phi_normalization"),
                                 ", ".join(sorted({s.hyperparameter_sources.get("lr", "?")
                                                   for s in sel}))))
                    elif method == "flower":
                        print("      N %s | gamma %s | CG max iters %s | CG tol %s"
                              % (_axis(sel, "num_flower_steps", "%s"),
                                 _axis(sel, "flower_gamma", "%s"),
                                 _axis(sel, "flower_cg_max_iters", "%s"),
                                 _axis(sel, "flower_cg_tol", "%g")))
                        print("      executed dt %s | source %s"
                              % (_dt_axis(sel), ", ".join(sorted(
                                  {s.hyperparameter_sources.get("num_flower_steps", "?")
                                   for s in sel}))))
                    else:
                        if method == "mpc_rhc":
                            print("      K %s" % _axis(sel, "K", "%s"))
                        print("      N %s | delta %s | executed dt %s"
                              % (_axis(sel, "num_mpc_steps", "%s"),
                                 _axis(sel, "delta", "%.4g"), _dt_axis(sel)))
                        print("      lam %s | n_ctrl %s | lr %s | source %s"
                              % (_axis(sel, "lam", "%g"), _axis(sel, "n_ctrl", "%s"),
                                 _axis(sel, "lr", "%g"),
                                 ", ".join(sorted({s.hyperparameter_sources.get("lam", "?")
                                                   for s in sel}))))

    print("-" * 94)
    print("TOTAL ATOMIC JOBS: %d" % w["atomic_jobs"])
    print("  image reconstructions       : %s" % "{:,}".format(w["image_reconstructions"]))
    print("  control iterations          : %s" % "{:,}".format(w["control_iterations"]))
    print("  generative-model evaluations: %s   (batched calls)"
          % "{:,}".format(w["model_evaluations"]))
    print("  ... as per-image work       : %s" % "{:,}".format(w["image_model_evaluations"]))
    print("  objective evaluations       : %s   (Phi / fidelity evaluations)"
          % "{:,}".format(w["objective_evaluations"]))
    print("  data-fidelity gradients     : %s   (PnP steps and D-Flow objectives)"
          % "{:,}".format(w["data_gradient_evaluations"]))
    print("  backprops through the model : %s   (SDEdit, RHC K=1 and PnP contribute ZERO)"
          % "{:,}".format(w["backprops_through_model"]))
    print("  jobs by model               : %s"
          % ", ".join("%s=%d" % (m.upper(), e["jobs"]) for m, e in w["per_model"].items()))
    print("  jobs by method              : %s"
          % ", ".join("%s=%d" % (m, e["jobs"]) for m, e in w["per_method"].items()))

    print("-" * 94)
    all_notes = list(plan.warnings) + list(notes)
    if all_notes:
        print("WARNINGS (%d)" % len(all_notes))
        for n in all_notes:
            print("  ! " + n)
    else:
        print("No warnings.")
    print("-" * 94)
    print("REMINDERS")
    print("  * delta = t0 / num_mpc_steps ONLY when beta = 1. Equal num_mpc_steps at different")
    print("    t0 does NOT mean equal execution resolution; the dt range above is what each")
    print("    job actually executes, and `delta` is null wherever beta != 1.")
    print("  * beta is the exponent of s_k = s0 + (1-s0)(k/N)^beta and applies to EVERY")
    print("    method. beta < 1 refines near clean space, beta = 1 is the legacy uniform grid")
    print("    (bitwise), beta > 1 refines near the noisy start. It is not PnP's alpha, and")
    print("    for MPC-RHC it shapes only the OUTER replanning grid, never the K-step plan.")
    print("  * MPC-Flow Table E2 hyperparameters were tuned for a CelebA 128x128 pixel-space")
    print("    U-Net. Here they are STARTING VALUES, never claims of optimality for JiT/pMF on")
    print("    ImageNet-256; every row records where its lam/n_ctrl/lr came from.")
    print("  * PnP's gamma0/alpha and D-Flow's lr/num_opt_steps have NO published value for")
    print("    these checkpoints either: their provenance reads %r until you sweep them." % UNTUNED)
    print("  * 'steps' means different compute per family: a D-Flow standard-flow step costs")
    print("    solver stage evaluations, a MeanFlow step costs ONE learned transition.")
    print("  * RHSO has no published hyperparameters at all: num_rhso_steps, num_opt_steps")
    print("    and lr are this repository's starting values, recorded as %r." % UNTUNED)
    print("  * RHSO's mu weights a STATE-anchor penalty (how far the optimised state moved")
    print("    from the state the trajectory reached); it is NOT MPC's control lambda, and")
    print("    mu = 0 is the plain terminal-fidelity objective.")
    print("  * RHSO's terminal planner: pMF plans with the LEARNED FINITE-INTERVAL MAP")
    print("    T_theta(q; s_k -> 1); JiT-direct plans with the DIRECT ENDPOINT PREDICTION")
    print("    x_hat_1(q, s_k). Both cost one terminal-planner evaluation per inner")
    print("    objective, but they are not the same learned object and are never reported")
    print("    as one. `suffix` is the legacy JiT planner that integrates the remainder.")
    print("  * The gradient-aligned authority diagnostic reports")
    print("    ||J^T g_end|| / ||g_end|| with g_end the FIDELITY-ONLY endpoint gradient;")
    print("    the state-anchor mu penalty is deliberately excluded, and the ratio is NaN")
    print("    (never a division by an invented epsilon) when the endpoint gradient")
    print("    vanishes. It is a measurement, not part of the objective.")
    print("  * Theory diagnostics are EXTRA measurements: their model evaluations, JVPs and")
    print("    VJPs are counted in the diagnostic_* columns and their time is subtracted")
    print("    from the reported runtime, so `runtime` stays algorithm runtime.")
    print("  * All methods share one epsilon per (model, image, replicate), so SDEdit, MPC,")
    print("    PnP, D-Flow and RHSO start from a bit-identical z_t0 at equal t0.")
    print(RULE_)


def _dt_axis(specs, limit=6):
    """The ACTUAL executed step-size range of each distinct schedule in `specs`.

    Printed as min-max pairs rather than two independent lists: with a beta sweep the
    minima and the maxima do not correspond position by position, and two separate lists
    would invite exactly the wrong reading.
    """
    seen, out = set(), []
    for s in specs:
        if s.delta_min is None:
            continue
        pair = "%.4g-%.4g" % (s.delta_min, s.delta_max)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    if not out:
        return "-"
    return ", ".join(out[:limit]) + (" ... (+%d)" % (len(out) - limit)
                                     if len(out) > limit else "")


def _axis(specs, attr, fmt="%s", limit=8):
    seen, out = set(), []
    for s in specs:
        v = getattr(s, attr)
        if repr(v) not in seen:
            seen.add(repr(v))
            out.append(v)
    shown = [(fmt % v) if v is not None else "-" for v in out[:limit]]
    return ", ".join(shown) + (" ... (+%d)" % (len(out) - limit) if len(out) > limit else "")
