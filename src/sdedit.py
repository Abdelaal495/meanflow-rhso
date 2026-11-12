"""Ordinary SDEdit reconstruction -- the paired baseline.

This module contains reconstruction ONLY.  Degradation lives in `problems.py`, and the
corrupted initial state is built by `models.base.build_initial_state`, exactly as it is for
both MPC methods:

        z_t0 = (1 - t0) * g(y) + t0 * eps

SDEdit then runs ordinary generation from s_start = 1 - t0 to the data endpoint s = 1.  The
measurement enters ONLY through the guide; there is no data-fidelity term and no control.
That is the point: it is what MPC's trajectory control is being compared against.

Standard flows (JiT, SiT)
    fixed-step integration of the instantaneous velocity field, with the Euler / Heun / RK4
    solvers from the SDEdit notebook.  Heun's final step falls back to Euler when the
    adapter advertises that policy (JiT's official sampler does).  Each step advances by
    the ACTUAL local dt_k = s_{k+1} - s_k of the schedule, so a non-uniform beta needs no
    change to the solver arithmetic.

MeanFlows (pMF, iMF)
    successive learned interval transitions along the canonical grid.  pMF is NOT treated as
    an ordinary velocity model; `steps` is the number of MeanFlow intervals and is always
    recorded explicitly in the results.

The grid itself comes from `schedule.canonical_time_grid`, the one place the universal
power-law exponent `beta` is applied (beta = 1 is the legacy uniform grid).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SOLVER_STAGE_EVALUATIONS
from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .schedule import canonical_time_grid, spec_beta
from .utils import MEANFLOW, STANDARD_FLOW


@dataclass
class ReconstructionStats:
    """Measured compute for one reconstruction (per chunk), never an estimate.

    The first block is the original set and keeps its meaning exactly.  The second was
    added with PnP-Flow and D-Flow, which spend their compute on things MPC does not have:

        data_gradient_evals   gradients of the DATA-FIDELITY term with respect to the
                              current state.  For a latent model this differentiates the
                              VAE decoder and the measurement operator -- it is NOT a
                              generative-trajectory backprop and is never counted as one.
        objective_evals       evaluations of the fidelity scalar itself.
        optimizer_iterations  Adam (or other optimiser) updates.  One iteration is NOT one
                              network evaluation: a D-Flow trajectory contains several.
        denoiser_samples      LOGICAL prior/denoiser applications, including PnP's initial
                              projection and every one of its M noise realisations, even if
                              an implementation later vectorises them into one network call.

    The third block was added with RHSO's optional state-anchor regularisation.  It splits
    the SAME scalar `loss_history` already records into its two parts, on the same
    per-image scale, so that for every recorded iteration

        loss_history[i] == fidelity_history[i] + mu * state_penalty_history[i]

    `state_penalty_history` is the UNWEIGHTED displacement R, not mu * R: mu is a resolved
    job field, so keeping R raw means a mu sweep can be re-weighted after the fact.  Both
    lists stay empty for every method that has no such split (and whenever
    `record_loss_history` is off), so nothing that reads `loss_history` changes.
    """
    control_iterations: int = 0
    model_evals_planning: int = 0        # v_theta / T_theta calls inside a control loop
    model_evals_total: int = 0           # every dynamics evaluation
    network_forwards: int = 0            # actual forward passes (CFG counted separately)
    backprops_through_model: int = 0
    loss_history: List[float] = field(default_factory=list)
    seconds: float = 0.0
    finite: bool = True

    # -- added with PnP-Flow / D-Flow ------------------------------------------------
    data_gradient_evals: int = 0
    objective_evals: int = 0
    optimizer_iterations: int = 0
    denoiser_samples: int = 0

    # -- added with RHSO's state-anchor regularisation --------------------------------
    fidelity_history: List[float] = field(default_factory=list)
    state_penalty_history: List[float] = field(default_factory=list)

    # -- added with the RHSO theory-validation diagnostics ----------------------------
    # STRICTLY SEPARATE from every counter above.  Stage-end evaluations, endpoint
    # consistency predictions, JVPs, VJPs and randomised spectral probes are EXTRA
    # measurements about the method, not compute the method spends, so counting them in
    # `model_evals_total` / `network_forwards` / `seconds` would silently inflate the
    # numbers a paper reports as the algorithm's cost.  `seconds` therefore excludes
    # `diagnostic_seconds` (see rhso._finalise), and with diagnostics off every field below
    # stays at zero / empty and nothing above changes.
    diagnostic_model_evals: int = 0
    diagnostic_network_forwards: int = 0
    diagnostic_jvps: int = 0
    diagnostic_vjps: int = 0
    diagnostic_seconds: float = 0.0
    # One entry per (REAL image, RHSO stage); see src/rhso_diagnostics.py.
    stage_records: List[Dict[str, Any]] = field(default_factory=list)
    # What ||p_k - r_k|| means for this run, and what the terminal planner is.  Set by the
    # RHSO loops so no downstream table has to guess the semantics from the model name.
    consistency_kind: Optional[str] = None
    terminal_planner_kind: Optional[str] = None
    diagnostic_settings: Optional[Dict[str, Any]] = None

    # -- added with Flower ---------------------------------------------------------------
    # Inner iterations of a linear solve (Flower's conjugate gradient for deblurring).  Each
    # one applies H and H^T, which are ALSO counted in `data_gradient_evals`; this field
    # records the solver's own iteration count.  0 for every other method.
    linear_solve_iterations: int = 0

    def merge(self, other: "ReconstructionStats") -> None:
        self.control_iterations += other.control_iterations
        self.model_evals_planning += other.model_evals_planning
        self.model_evals_total += other.model_evals_total
        self.network_forwards += other.network_forwards
        self.backprops_through_model += other.backprops_through_model
        self.loss_history.extend(other.loss_history)
        self.seconds += other.seconds
        self.finite = self.finite and other.finite
        self.data_gradient_evals += other.data_gradient_evals
        self.objective_evals += other.objective_evals
        self.optimizer_iterations += other.optimizer_iterations
        self.denoiser_samples += other.denoiser_samples
        self.fidelity_history.extend(other.fidelity_history)
        self.state_penalty_history.extend(other.state_penalty_history)
        self.diagnostic_model_evals += other.diagnostic_model_evals
        self.diagnostic_network_forwards += other.diagnostic_network_forwards
        self.diagnostic_jvps += other.diagnostic_jvps
        self.diagnostic_vjps += other.diagnostic_vjps
        self.diagnostic_seconds += other.diagnostic_seconds
        # Chunks are disjoint sets of images, so concatenating keeps one row per
        # (image, stage) across the whole job.
        self.stage_records.extend(other.stage_records)
        self.consistency_kind = self.consistency_kind or other.consistency_kind
        self.terminal_planner_kind = self.terminal_planner_kind or other.terminal_planner_kind
        self.diagnostic_settings = self.diagnostic_settings or other.diagnostic_settings
        self.linear_solve_iterations += other.linear_solve_iterations


# `canonical_time_grid` used to be defined HERE, which made an algorithm module the owner
# of the time discretisation every other method also depends on.  It now lives in the
# neutral `schedule.py` and carries the universal `beta` exponent; the name is re-exported
# so that `from src.sdedit import canonical_time_grid` keeps working unchanged.
__all__ = ["ReconstructionStats", "canonical_time_grid", "sdedit_flow", "sdedit_meanflow",
           "sdedit_reconstruct"]


def _is_finite(adapter: ModelAdapter, x) -> bool:
    if adapter.spec.framework == "torch":
        import torch
        return bool(torch.isfinite(x).all())
    import jax.numpy as jnp
    return bool(jnp.isfinite(x).all())


# =====================================================================================
# Standard flow: fixed-step ODE integration
# =====================================================================================
def sdedit_flow(adapter: StandardFlowAdapter, cond: Conditioning, x0, spec) -> Tuple[Any,
                                                                                     ReconstructionStats]:
    """Euler / Heun / RK4 integration from s_start to s = 1 (SDEdit notebook, section 14)."""
    import torch
    solver = spec.solver or "euler"
    if solver not in SOLVER_STAGE_EVALUATIONS:
        raise ValueError("Unknown solver %r" % solver)
    grid = canonical_time_grid(spec.canonical_start_time, int(spec.steps), spec_beta(spec))
    # The final-step policy is a MODEL property, read from the adapter -- not a name check.
    final_euler = bool(adapter.spec.euler_final_step_for_heun)

    stats = ReconstructionStats()
    started = time.perf_counter()
    # Sampling is pure inference: no graph is needed, unlike MPC.
    with torch.no_grad():
        x = x0.detach().clone()
        for k in range(int(spec.steps)):
            s, s_next = grid[k], grid[k + 1]
            dt = s_next - s
            adapter.reset_counters()

            v1 = adapter.velocity(x, s, cond)
            step_solver = solver
            if solver == "heun" and final_euler and k == int(spec.steps) - 1:
                step_solver = "euler"

            if step_solver == "euler":
                x_next = x + dt * v1
                evals = 1
            elif step_solver == "heun":
                v2 = adapter.velocity(x + dt * v1, s_next, cond)
                x_next = x + 0.5 * dt * (v1 + v2)
                evals = 2
            else:                                                        # rk4
                s_mid = 0.5 * (s + s_next)
                v2 = adapter.velocity(x + 0.5 * dt * v1, s_mid, cond)
                v3 = adapter.velocity(x + 0.5 * dt * v2, s_mid, cond)
                v4 = adapter.velocity(x + dt * v3, s_next, cond)
                x_next = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
                evals = 4

            stats.model_evals_total += evals
            stats.network_forwards += adapter.forward_counter
            if not _is_finite(adapter, x_next):
                raise FloatingPointError(
                    "%s produced a non-finite state at step %d/%d (s=%.6f -> %.6f) using %s."
                    % (adapter.spec.name, k, int(spec.steps) - 1, s, s_next, step_solver))
            x = x_next

    adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


# =====================================================================================
# MeanFlow: successive learned interval transitions
# =====================================================================================
def sdedit_meanflow(adapter: MeanFlowAdapter, cond: Conditioning, x0,
                    spec) -> Tuple[Any, ReconstructionStats]:
    """Successive transitions T(x; s_k -> s_{k+1}) along the canonical grid.

    The endpoints follow the universal beta schedule; the transitions themselves are the
    model's own learned finite-interval maps and are never approximated by a velocity.

    This is the notebooks' ordinary MeanFlow sampling, unchanged.  With `steps: 1` it is the
    single-interval transition straight to the data endpoint, which is what pMF is designed
    for; larger values subdivide the same interval.
    """
    grid = canonical_time_grid(spec.canonical_start_time, int(spec.steps), spec_beta(spec))
    stats = ReconstructionStats()
    started = time.perf_counter()
    x = x0

    for k in range(int(spec.steps)):
        adapter.reset_counters()
        x = adapter.transition(x, grid[k], grid[k + 1], cond)
        stats.model_evals_total += 1
        stats.network_forwards += adapter.forward_counter
        if not _is_finite(adapter, x):
            raise FloatingPointError(
                "%s produced a non-finite state at MeanFlow transition %d/%d (s=%.4f -> %.4f)."
                % (adapter.spec.name, k, int(spec.steps) - 1, grid[k], grid[k + 1]))

    x = adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


def sdedit_reconstruct(adapter: ModelAdapter, cond: Conditioning, x0, problem,
                       spec) -> Tuple[Any, ReconstructionStats]:
    """Family dispatch.  `problem` is accepted for interface symmetry with the MPC solvers
    and is deliberately unused: SDEdit never sees the measurement."""
    if spec.dynamics_family == STANDARD_FLOW:
        return sdedit_flow(adapter, cond, x0, spec)
    if spec.dynamics_family == MEANFLOW:
        return sdedit_meanflow(adapter, cond, x0, spec)
    raise ValueError("No SDEdit strategy for dynamics family %r" % spec.dynamics_family)



