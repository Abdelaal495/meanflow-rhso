"""Flower: a flow-matching solver for linear inverse problems, with no network backprop.

Based on Pourya, El Rawas and Unser, "Flower: A Flow-Matching Solver for Inverse Problems"
(ICLR 2026, arXiv 2509.26287).  Every iteration is

    1. destination   x_hat = x_t + (1 - t) v(x_t, t)                  (one network call)
    2. refinement    x_tilde = mu_t + gamma * kappa_t,   kappa_t ~ N(0, Sigma_t)
    3. progression   x_{t'} = (1 - t') eps + t' x_tilde               (fresh eps)

with the proximal (Gaussian-posterior) refinement

    Sigma_t = (nu_t^-2 I + sigma^-2 H^T H)^-1
    mu_t    = Sigma_t (nu_t^-2 x_hat + sigma^-2 H^T y),    nu_t = (1 - t) / sqrt(t^2 + (1 - t)^2)

and, like PnP-Flow, it NEVER backpropagates through the generative model.

What is published, and what this file adapts
--------------------------------------------
PUBLISHED (close to the paper):
  * the three steps above, on the canonical clock (t = 0 noise, t = 1 data);
  * the nu_t schedule;
  * gamma = 0, as in all of the paper's reported results.  gamma = 1 (`flower_gamma: 1`)
    adds an exact sample kappa_t with covariance Sigma_t;
  * sigma is the TRUE measurement noise level (`problem.sigma`, on the same [-1, 1] scale
    as y) -- an oracle value, exactly as in the paper;
  * conjugate gradient for a non-diagonal H^T H, at the paper's settings (at most 50
    iterations, relative residual 1e-5).

ADAPTED for this repository (documented, never presented as the paper's algorithm):
  * INITIALISATION.  Flower starts from pure noise.  Here it starts from the SAME shared
    z_t0 as every other strategy -- the benchmark's fairness invariant -- and its time grid
    is `canonical_time_grid(s0, N, beta)`, so at t0 = 1 (s0 = 0) it is the paper's start.
  * MEANFLOW (research extension).  A MeanFlow model has no instantaneous velocity, so the
    destination becomes the learned transition x_hat = T_theta(x_t; t -> 1).  For a
    flow-matching model x_hat is the conditional-mean estimate E[X_1 | X_t = x_t]; that
    interpretation does NOT automatically transfer to a learned MeanFlow transition.
  * PIXEL PRIORS ONLY.  The refinement acts on the image.  A latent prior would need an
    encoder inside the proximal step or an inner optimisation -- a different method -- so
    latent adapters are rejected, and the SiT / iMF capability entries do not list Flower.
  * LINEAR OPERATORS ONLY: denoising, deblurring, 2x strided super-resolution and the two
    inpainting tasks.  `stroke_painting` is rejected: it is noiseless by default, which
    makes sigma^-2 undefined, and it is not one of the tasks Flower is evaluated on here.

How the refinement is solved
----------------------------
The proximal step is solved in the adapter's own state space, with
H(x) = problem.apply(adapter.to_pixels(x, differentiable=True)).  For a pixel adapter
`to_pixels` is a pure layout permutation (the identity for pMF, BCHW -> BHWC for JiT), so
this is the same problem, solved exactly, with no inverse layout map needed.  H^T comes from
a vector-Jacobian product of that LINEAR map, so it is exact rather than hand-written.

The equation is multiplied through by nu_t^2 before it is solved,

    (I + rho H^T H) x_tilde = x_hat + rho H^T y + gamma (nu eps_1 + (nu^2 / sigma) H^T eps_2),
    rho = nu_t^2 / sigma^2,

which has the same solution but never forms nu_t^-2 (huge as t -> 1).  The gamma term has
covariance nu^2 (I + rho H^T H), so after the solve it contributes exactly Sigma_t: one
linear solve gives mu_t + gamma kappa_t.

    denoising, inpainting, strided SR   H^T H = diag(d), d = H^T(H(1)): closed form
    deblurring (separable, reflect)     not diagonal: per-image conjugate gradient

CG is warm-started at the destination x_hat (the solution as rho -> 0), and its step sizes
and stopping test are PER IMAGE, so an image's result does not depend on which other images
share its batch.  The cap `flower_cg_max_iters` defaults to the paper's 50; at the
benchmark's sigma = 0.05 the early blur solves (condition number ~ 1 + rho, rho = 400 at
t = 0) need roughly 60-80 iterations to reach 1e-5, which is why the deblurring screen also
tries 100.

Compute semantics
-----------------
N network evaluations (one destination per iteration), 0 backprops through the model.
Every application of H or H^T is counted in `data_gradient_evals` (H^T y and d are formed
once per reconstruction; CG costs one H and one H^T per iteration plus its initial
residual), and the CG iterations in `linear_solve_iterations`.
"""

from __future__ import annotations

import math
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from .models.base import Conditioning, MeanFlowAdapter, ModelAdapter, StandardFlowAdapter
from .schedule import canonical_time_grid, spec_beta
from .sdedit import ReconstructionStats, _is_finite
from .utils import MEANFLOW, STANDARD_FLOW, gaussian_noise

# H^T H is diagonal for these, so the refinement is elementwise.
DIAGONAL_PROBLEMS: Tuple[str, ...] = ("denoising", "random_inpaint", "box_inpaint",
                                      "super_resolution")
# H^T H is a symmetric blur-of-a-blur: conjugate gradient.
CG_PROBLEMS: Tuple[str, ...] = ("deblur",)
SUPPORTED_PROBLEMS: Tuple[str, ...] = DIAGONAL_PROBLEMS + CG_PROBLEMS
VALID_GAMMAS: Tuple[int, ...] = (0, 1)

# Recorded in the seed recipes of Flower jobs only (see config.resolve_run_plan).
FLOWER_SEED_RECIPE = (
    "seed(global_seed, 'flower', stream, model, image_id, replicate, iteration) with stream "
    "in {refresh, posterior_state, posterior_measurement}; independent of "
    "num_flower_steps, flower_gamma and the CG settings")


# =====================================================================================
# Schedule and deterministic noise
# =====================================================================================
def flower_time_grid(s_start: float, num_steps: int, beta: float = 1.0) -> List[float]:
    """The N + 1 times t_0 = s_start < ... < t_N = 1.  Iteration k refines at t_k."""
    n = int(num_steps)
    if n < 1:
        raise ValueError("num_flower_steps must be >= 1, got %r" % (num_steps,))
    return canonical_time_grid(s_start, n, beta)


def flower_nu(t: float) -> float:
    """nu_t = (1 - t) / sqrt(t^2 + (1 - t)^2): 1 at t = 0, 1/sqrt(2) at 1/2, -> 0 at 1."""
    t = float(t)
    return (1.0 - t) / math.sqrt(t * t + (1.0 - t) * (1.0 - t))


def flower_noise_parts(model: str, image_id: Any, replicate: int, iteration: int,
                       stream: str = "refresh") -> Tuple[Any, ...]:
    """Identity of ONE Flower noise draw.

    Fresh per iteration, deterministic across runs, drawn per IMAGE ID (so batching never
    changes an image's noise), and deliberately independent of num_flower_steps, gamma and
    the CG settings: iteration k draws the same noise whatever N is.  Three streams:

        refresh                 eps in the time progression x_{t'} = (1-t') eps + t' x_tilde
        posterior_state         eps_1 of the gamma = 1 sample (state shape)
        posterior_measurement   eps_2 of the gamma = 1 sample (measurement shape)

    It lives here rather than in `utils.py` so that the shared seeding module, and with it
    every existing job's seed recipes, is untouched.
    """
    return ("flower", str(stream), model, str(image_id), "rep%d" % int(replicate),
            "k%d" % int(iteration))


def _native_noise(adapter: ModelAdapter, spec, image_ids, iteration: int, stream: str):
    noise = np.stack([gaussian_noise(adapter.spec.native_shape,
                                     *flower_noise_parts(adapter.spec.name, image_id,
                                                         spec.replicate, iteration, stream))
                      for image_id in image_ids], axis=0)
    return adapter.to_native_noise(noise)


def _measurement_noise(ops, problem, spec, model: str, image_ids, iteration: int):
    shape = tuple(problem.measurement.shape[1:])
    noise = np.stack([gaussian_noise(shape, *flower_noise_parts(
        model, image_id, spec.replicate, iteration, "posterior_measurement"))
        for image_id in image_ids], axis=0)
    return ops.const(noise)


# =====================================================================================
# Supported configurations
# =====================================================================================
def check_flower_supported(adapter: ModelAdapter, problem, spec) -> None:
    """Fail loudly, before any compute, on anything Flower is not defined for here."""
    space = getattr(adapter.spec, "state_space", None)
    if space != "pixel" or getattr(spec, "state_space", "pixel") != "pixel":
        raise ValueError(
            "Flower supports pixel-space priors only (pMF, JiT); %s has state_space=%r. Its "
            "refinement is a proximal step on the IMAGE, and a latent version would need an "
            "encoder or an inner optimisation -- a different method, not implemented."
            % (adapter.spec.name, space if space != "pixel" else spec.state_space))
    if problem.name not in SUPPORTED_PROBLEMS:
        raise ValueError(
            "Flower supports the linear operators %s only; got %r. Its refinement needs a "
            "linear H with a Gaussian likelihood (stroke_painting is noiseless by default "
            "and is excluded, as is any nonlinear operator)."
            % (list(SUPPORTED_PROBLEMS), problem.name))
    if not float(problem.sigma) > 0.0:
        raise ValueError(
            "Flower needs a positive measurement noise level: its refinement weights the "
            "data term by 1/sigma^2, and %s has sigma=%g." % (problem.name,
                                                              float(problem.sigma)))
    gamma = getattr(spec, "flower_gamma", 0)
    if gamma not in VALID_GAMMAS:
        raise ValueError("flower_gamma must be 0 or 1, got %r" % (gamma,))


# =====================================================================================
# Framework primitives for the linear algebra
# =====================================================================================
class _Ops:
    """H, H^T and the per-image reductions, counting every H / H^T application."""

    def __init__(self, adapter: ModelAdapter, problem):
        self.adapter = adapter
        self.problem = problem
        self.B = adapter.backend()
        self.h_calls = 0
        self.ht_calls = 0

    @property
    def operator_applications(self) -> int:
        return self.h_calls + self.ht_calls

    def _forward(self, x):
        return self.problem.apply(self.adapter.to_pixels(x, differentiable=True), self.B)

    def H(self, x):
        self.h_calls += 1
        return self._forward(x)

    def y(self):
        return self.problem.measurement_tensor(self.B)

    def const(self, arr):
        return self.B.const(arr)


class _TorchOps(_Ops):
    def __init__(self, adapter, problem):
        super().__init__(adapter, problem)
        import torch
        self.torch = torch

    def H(self, x):
        with self.torch.no_grad():
            return super().H(x)

    def Ht(self, v, like):
        """Exact adjoint: the vjp of the linear map at any point (zeros)."""
        self.ht_calls += 1
        torch = self.torch
        z = torch.zeros_like(like).requires_grad_(True)
        with torch.enable_grad():
            out = self._forward(z)
            (g,) = torch.autograd.grad(out, z, grad_outputs=v.to(out.dtype))
        return g.detach()

    def bdot(self, a, b):
        n = int(a.shape[0])
        return (a * b).reshape(n, -1).sum(dim=1).reshape((n,) + (1,) * (a.dim() - 1))

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def where(self, cond, a, b):
        torch = self.torch
        a = a if torch.is_tensor(a) else torch.full_like(b, float(a))
        b = b if torch.is_tensor(b) else torch.full_like(a, float(b))
        return torch.where(cond, a, b)

    def any(self, cond) -> bool:
        return bool(cond.any())

    def max(self, x) -> float:
        return float(x.max())

    def ones_like(self, x):
        return self.torch.ones_like(x)

    def zeros_like(self, x):
        return self.torch.zeros_like(x)


class _JaxOps(_Ops):
    def __init__(self, adapter, problem):
        super().__init__(adapter, problem)
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = jax, jnp

    def Ht(self, v, like):
        """Exact adjoint: the vjp of the linear map at any point (zeros)."""
        self.ht_calls += 1
        _, pullback = self.jax.vjp(self._forward, self.jnp.zeros_like(like))
        return pullback(v.astype(like.dtype))[0]

    def bdot(self, a, b):
        n = int(a.shape[0])
        return self.jnp.sum((a * b).reshape(n, -1), axis=1).reshape((n,) + (1,) * (a.ndim - 1))

    def sqrt(self, x):
        return self.jnp.sqrt(x)

    def where(self, cond, a, b):
        return self.jnp.where(cond, a, b)

    def any(self, cond) -> bool:
        return bool(self.jnp.any(cond))

    def max(self, x) -> float:
        return float(self.jnp.max(x))

    def ones_like(self, x):
        return self.jnp.ones_like(x)

    def zeros_like(self, x):
        return self.jnp.zeros_like(x)


def make_ops(adapter: ModelAdapter, problem) -> _Ops:
    if adapter.spec.framework == "torch":
        return _TorchOps(adapter, problem)
    if adapter.spec.framework == "jax":
        return _JaxOps(adapter, problem)
    raise ValueError("No Flower linear algebra for framework %r" % adapter.spec.framework)


# =====================================================================================
# The refinement
# =====================================================================================
def conjugate_gradient(ops: _Ops, apply_A, b, x_init, max_iters: int,
                       tol: float) -> Tuple[Any, int, float]:
    """Solve A x = b for a symmetric positive-definite A, independently per image.

    Each image has its own step sizes and its own stopping test ||r|| <= tol ||b||; an image
    that has converged is frozen (zero step) while the others continue.  Returns
    (x, iterations run, worst relative residual).
    """
    x = x_init
    r = b - apply_A(x)
    p = r
    rr = ops.bdot(r, r)
    b_norm = ops.sqrt(ops.bdot(b, b))
    threshold = float(tol) * b_norm
    active = ops.sqrt(rr) > threshold
    iters = 0
    while iters < int(max_iters) and ops.any(active):
        Ap = apply_A(p)
        pAp = ops.bdot(p, Ap)
        alpha = ops.where(active, rr / ops.where(active, pAp, 1.0), 0.0)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = ops.bdot(r, r)
        beta = ops.where(active, rr_new / ops.where(active, rr, 1.0), 0.0)
        p = r + beta * p
        rr = ops.where(active, rr_new, rr)
        active = active & (ops.sqrt(rr) > threshold)
        iters += 1
    worst = ops.max(ops.sqrt(rr) / ops.where(b_norm > 0, b_norm, 1.0))
    return x, iters, worst


class FlowerRefinement:
    """The proximal step for ONE problem instance, with H^T y (and diag H^T H) cached."""

    def __init__(self, ops: _Ops, problem, like, cg_max_iters: int = 50,
                 cg_tol: float = 1e-5, solver: Optional[str] = None):
        self.ops = ops
        self.sigma = float(problem.sigma)
        self.solver = solver or ("closed_form" if problem.name in DIAGONAL_PROBLEMS else "cg")
        if self.solver == "closed_form" and problem.name not in DIAGONAL_PROBLEMS:
            raise ValueError("%s has a non-diagonal H^T H: no closed form" % problem.name)
        self.cg_max_iters = int(cg_max_iters)
        self.cg_tol = float(cg_tol)
        self.Hty = ops.Ht(ops.y(), like)
        self.diag = (ops.Ht(ops.H(ops.ones_like(like)), like)
                     if self.solver == "closed_form" else None)
        self.last_iterations = 0
        self.last_residual = 0.0

    def __call__(self, x_hat, t: float, gamma: int = 0, eps_state=None, eps_meas=None):
        """x_tilde = mu_t + gamma * kappa_t, from the destination x_hat at time t < 1."""
        ops = self.ops
        nu = flower_nu(t)
        rho = (nu * nu) / (self.sigma * self.sigma)
        rhs = x_hat + rho * self.Hty
        if gamma:
            # covariance nu^2 (I + rho H^T H): after the solve, exactly Sigma_t
            rhs = rhs + nu * eps_state + (nu * nu / self.sigma) * ops.Ht(eps_meas, x_hat)
        if self.solver == "closed_form":
            self.last_iterations, self.last_residual = 0, 0.0
            return rhs / (1.0 + rho * self.diag)

        def apply_A(v):
            return v + rho * ops.Ht(ops.H(v), v)

        # warm start at x_hat: with max_iters = 0 the refinement returns x_hat itself
        x, iters, residual = conjugate_gradient(ops, apply_A, rhs, x_hat,
                                                self.cg_max_iters, self.cg_tol)
        self.last_iterations, self.last_residual = iters, residual
        return x


# =====================================================================================
# The loops
# =====================================================================================
def _run(adapter: ModelAdapter, cond: Conditioning, x0, problem, spec, destination,
         stop_gradient) -> Tuple[Any, ReconstructionStats]:
    check_flower_supported(adapter, problem, spec)
    s0 = float(spec.canonical_start_time)
    N = int(spec.num_flower_steps)
    grid = flower_time_grid(s0, N, spec_beta(spec))
    gamma = int(getattr(spec, "flower_gamma", 0) or 0)
    image_ids = list(problem.image_ids)
    model = adapter.spec.name

    stats = ReconstructionStats()
    started = time.perf_counter()
    ops = make_ops(adapter, problem)
    refine = FlowerRefinement(ops, problem, x0,
                              cg_max_iters=int(getattr(spec, "flower_cg_max_iters", 50) or 50),
                              cg_tol=float(getattr(spec, "flower_cg_tol", 1e-5) or 1e-5))

    x = x0
    for k in range(N):
        t, t_next = float(grid[k]), float(grid[k + 1])

        # -- 1. destination (no gradient) ---------------------------------------------
        adapter.reset_counters()
        x_hat = stop_gradient(destination(x, t))
        stats.model_evals_total += 1
        stats.denoiser_samples += 1
        stats.network_forwards += adapter.forward_counter
        if not _is_finite(adapter, x_hat):
            raise FloatingPointError("%s produced a non-finite Flower destination at "
                                     "iteration %d/%d (t=%.6f)." % (model, k + 1, N, t))

        # -- 2. refinement ------------------------------------------------------------
        eps_state = eps_meas = None
        if gamma:
            eps_state = _native_noise(adapter, spec, image_ids, k, "posterior_state")
            eps_meas = _measurement_noise(ops, problem, spec, model, image_ids, k)
        x_tilde = stop_gradient(refine(x_hat, t, gamma, eps_state, eps_meas))
        stats.linear_solve_iterations += refine.last_iterations

        # -- 3. time progression (t_N = 1 exactly: the output is x_tilde) -------------
        if k == N - 1:
            x = x_tilde
        else:
            eps = _native_noise(adapter, spec, image_ids, k, "refresh")
            x = stop_gradient(adapter.corrupt(x_tilde, 1.0 - t_next, eps))
        if not _is_finite(adapter, x):
            raise FloatingPointError("%s produced a non-finite state at Flower iteration "
                                     "%d/%d (t=%.6f)." % (model, k + 1, N, t))

    stats.data_gradient_evals += ops.operator_applications
    x = adapter.block(x)
    stats.seconds = time.perf_counter() - started
    stats.finite = _is_finite(adapter, x)
    return x, stats


def flow_flower(adapter: StandardFlowAdapter, cond: Conditioning, x0, problem,
                spec) -> Tuple[Any, ReconstructionStats]:
    """Flower for an instantaneous-velocity pixel model (JiT): x_hat = x + (1 - t) v."""
    import torch

    def destination(x, t):
        with torch.no_grad():
            return x + (1.0 - t) * adapter.velocity(x, t, cond)

    return _run(adapter, cond, x0, problem, spec, destination, lambda v: v.detach())


def meanflow_flower(adapter: MeanFlowAdapter, cond: Conditioning, x0, problem,
                    spec) -> Tuple[Any, ReconstructionStats]:
    """Flower with the learned transition as the destination: x_hat = T(x; t -> 1).

    RESEARCH EXTENSION (see the module docstring).  Never differentiates the model.
    """
    import jax

    def destination(x, t):
        return adapter.transition(x, t, 1.0, cond)

    return _run(adapter, cond, x0, problem, spec, destination, jax.lax.stop_gradient)


def flower_reconstruct(adapter: ModelAdapter, cond: Conditioning, x0, problem,
                       spec) -> Tuple[Any, ReconstructionStats]:
    """Family dispatch, for callers that hold an adapter but not a registry."""
    if spec.dynamics_family == STANDARD_FLOW:
        return flow_flower(adapter, cond, x0, problem, spec)
    if spec.dynamics_family == MEANFLOW:
        return meanflow_flower(adapter, cond, x0, problem, spec)
    raise ValueError("No Flower strategy for dynamics family %r" % spec.dynamics_family)
