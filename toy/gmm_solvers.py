"""The six solvers of the image benchmark, on the toy prior interfaces (rebuttal Task 4).

Each function is a line-for-line port of its src/ counterpart onto two interfaces:

    flow-matching prior   .velocity(x, t), .endpoint(x, t) = x + (1 - t) v, .step(x, t, s)
    MeanFlow prior        .transition(x, t, s)

(toy/gmm_oracle.py gives the exact ones; learned models are wrapped the same way.)  The
canonical clock is used throughout: t = 0 noise, t = 1 data, t0 = 1 (start from noise),
beta = 1 (uniform grids).  Every row of a batch is an independent problem with its own
hyperparameters -- the image code sums the fidelity over the batch, so per-row gradients
never mix, and the optimiser here is a per-row Adam identical to torch.optim.Adam.

    solver        source                       fidelity (as in the image code)
    rhso          src/rhso.py                  per-measurement  (1 / 2m) ||Hx - y||^2
    dflow         src/dflow.py   (steps = 1)   per-measurement
    pnp           src/pnp.py                   per-measurement
    mpc_delta_t   src/mpc.py                   half sum         (1 / 2) ||Hx - y||^2
    mpc_rhc       src/mpc.py                   half sum
    flower        src/flower.py  (gamma = 0)   exact proximal step, dense solve (d = 16)

Cost accounting mirrors ReconstructionStats: `evals` = model evaluations (a MeanFlow
transition is ONE evaluation whatever it costs inside; see gmm_oracle.py), `backprops` =
evaluations the backward pass traverses, `data_grads` = data-fidelity gradients.

Every solver takes an optional `trace` list (default None: nothing recorded, results
unchanged).  When given, the solver appends the executed state after every outer step,
starting with x0, for trajectory figures.  RHSO, MPC-delta_t and MPC-RHC record x_k on their
outer grids; PnP-Flow its denoised iterates; Flower its refined estimate mu at every step
(not the re-noised state; the last mu is the output); D-Flow its optimised initial state q,
then the output G(q).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch


# =====================================================================================
# Batches of problems
# =====================================================================================
@dataclass
class Rows:
    """R independent inverse problems, padded to a common measurement count MMAX.

    H (R, MMAX, d) with zero rows beyond each problem's m; y (R, MMAX); mask (R, MMAX) marks
    real measurements; m (R,); sigma (R,); x0 (R, d) the shared initial noise; type_idx,
    prob_idx, seed_idx (R,) identify the row for the solvers' own noise (see NoiseSource)."""
    H: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    m: torch.Tensor
    sigma: torch.Tensor
    x0: torch.Tensor
    type_idx: torch.Tensor
    prob_idx: torch.Tensor
    seed_idx: torch.Tensor
    noise: "NoiseSource"

    def __len__(self):
        return self.x0.shape[0]

    def select(self, idx: torch.Tensor) -> "Rows":
        return Rows(*(getattr(self, f)[idx] for f in
                      ("H", "y", "mask", "m", "sigma", "x0", "type_idx", "prob_idx",
                       "seed_idx")), self.noise)

    @staticmethod
    def cat(parts: List["Rows"]) -> "Rows":
        return Rows(*(torch.cat([getattr(p, f) for p in parts]) for f in
                      ("H", "y", "mask", "m", "sigma", "x0", "type_idx", "prob_idx",
                       "seed_idx")), parts[0].noise)

    def repeat(self, times: int) -> "Rows":
        return Rows.cat([self] * times)


class NoiseSource:
    """Solver-internal noise (PnP reprojection, Flower refresh), deterministic per
    (problem set, problem type, problem, seed, stream, iteration) and independent of every
    hyperparameter, of which rows share a batch and of how many iterations a run has --
    iteration k draws the same noise whatever N is, as flower_noise_parts / pnp_reprojection
    _parts guarantee in the image code."""

    STREAMS = {"pnp": 1, "flower": 2}

    def __init__(self, set_seed: int, n_prob: int, n_seed: int, d: int, device, dtype):
        self.set_seed, self.n_prob, self.n_seed, self.d = set_seed, n_prob, n_seed, d
        self.device, self.dtype = device, dtype

    def draw(self, rows: Rows, stream: str, iteration: int) -> torch.Tensor:
        out = torch.empty_like(rows.x0)
        for t in torch.unique(rows.type_idx).tolist():
            g = torch.Generator(device=self.device)
            g.manual_seed((((self.set_seed * 1_000_003 + self.STREAMS[stream]) * 1009 + t)
                           * 100_003 + int(iteration)) % (2 ** 62))
            full = torch.randn((self.n_prob, self.n_seed, self.d), generator=g,
                               device=self.device, dtype=self.dtype)
            sel = rows.type_idx == t
            out[sel] = full[rows.prob_idx[sel], rows.seed_idx[sel]]
        return out


def residual(rows: Rows, x: torch.Tensor) -> torch.Tensor:
    return (torch.einsum("rmd,rd->rm", rows.H, x) - rows.y) * rows.mask


def phi_per_measurement(rows: Rows, x: torch.Tensor) -> torch.Tensor:
    """(1 / 2 m) ||Hx - y||^2 per row (src/problems.py, half_mean_squared_per_measurement)."""
    r = residual(rows, x)
    return 0.5 * (r * r).sum(1) / rows.m


def phi_half_sum(rows: Rows, x: torch.Tensor) -> torch.Tensor:
    """(1 / 2) ||Hx - y||^2 per row (half_sum_squared, MPC's default)."""
    r = residual(rows, x)
    return 0.5 * (r * r).sum(1)


def grad_phi_per_measurement(rows: Rows, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rmd,rm->rd", rows.H, residual(rows, x)) / rows.m[:, None]


class RowAdam:
    """torch.optim.Adam (betas 0.9 / 0.999, eps 1e-8, bias correction), elementwise, with a
    learning rate per row.  `lr` must broadcast against the parameter."""

    def __init__(self, param: torch.Tensor, lr: torch.Tensor, betas=(0.9, 0.999), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = torch.zeros_like(param)
        self.v = torch.zeros_like(param)
        self.t = 0

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        bc1 = 1 - self.b1 ** self.t
        bc2_sqrt = math.sqrt(1 - self.b2 ** self.t)
        denom = self.v.sqrt() / bc2_sqrt + self.eps
        return param - (self.lr / bc1) * self.m / denom


@dataclass
class Cost:
    """Per-problem cost of one solve (identical for every row of a run)."""
    evals: int = 0
    backprops: int = 0
    data_grads: int = 0
    linear_solves: int = 0


def _grid(n: int) -> List[float]:
    """canonical_time_grid(0, n, beta = 1): k / n, exact at both ends."""
    g = [k / n for k in range(n + 1)]
    g[-1] = 1.0
    return g


def _col(v: torch.Tensor) -> torch.Tensor:
    return v[:, None]


def _record(trace: Optional[list], x: torch.Tensor) -> None:
    if trace is not None:
        trace.append(x.detach().clone())


# =====================================================================================
# RHSO (src/rhso.py)
# =====================================================================================
def rhso(prior, rows: Rows, N: int, M: int, lr: torch.Tensor, mu: torch.Tensor,
         terminal: str = "direct", solver: str = "heun", trace: Optional[list] = None):
    """At s_k: M Adam steps on the state q against Phi(planner(q)) + mu R(q, x_k), with a
    fresh anchor and fresh Adam moments, then execute ONE interval from the final q.

    planner: MeanFlow T(q; s_k -> 1); flow matching 'direct' = q + (1 - s_k) v(q, s_k), or
    'suffix' = a differentiable Heun solve over the remaining outer grid [s_k, ..., 1].
    Execution: MeanFlow T(q; s_k -> s_{k+1}); flow matching one Heun step."""
    grid = _grid(N)
    d = rows.x0.shape[1]
    cost = Cost()
    x = rows.x0.clone()
    _record(trace, x)
    meanflow = prior.family == "meanflow"
    for k in range(N):
        s, s_next = grid[k], grid[k + 1]
        if meanflow:
            plan, per_obj = (lambda q, _s=s: prior.transition(q, _s, 1.0)), 1
        elif terminal == "direct":
            plan, per_obj = (lambda q, _s=s: prior.endpoint(q, _s)), 1
        else:
            plan = lambda q, _g=grid[k:]: prior.integrate(q, _g, solver)
            per_obj = prior.step_evals(solver) * (N - k)
        anchor = x.detach()
        q = anchor.clone()
        opt = RowAdam(q, _col(lr))
        for _ in range(M):
            q = q.detach().requires_grad_(True)
            fidelity = phi_per_measurement(rows, plan(q))
            penalty = 0.5 * ((q - anchor) ** 2).sum(1) / d          # R(q, x_k), 1/(2d)
            loss = (fidelity + mu * penalty).sum()
            (grad,) = torch.autograd.grad(loss, q)
            q = opt.step(q.detach(), grad)
            cost.evals += per_obj
            cost.backprops += per_obj
            cost.data_grads += 1
        with torch.no_grad():
            if meanflow:
                x = prior.transition(q, s, s_next)
                cost.evals += 1
            else:
                x = prior.step(q, s, s_next, solver)
                cost.evals += prior.step_evals(solver)
        _record(trace, x)
    return x.detach(), cost


# =====================================================================================
# D-Flow, one step (src/dflow.py)
# =====================================================================================
def dflow(prior, rows: Rows, num_opt_steps: int, lr: torch.Tensor, solver: str = "euler",
          trace: Optional[list] = None):
    """Adam on the initial state q through G(q; 0 -> 1): MeanFlow T(q; 0 -> 1); flow
    matching one Euler step q + v(q, 0) (the image code's steps = 1, solver = euler).  One
    more trajectory after the last update gives the reconstruction."""
    meanflow = prior.family == "meanflow"
    G = ((lambda q: prior.transition(q, 0.0, 1.0)) if meanflow
         else (lambda q: prior.step(q, 0.0, 1.0, solver)))
    per_obj = 1 if meanflow else prior.step_evals(solver)
    cost = Cost()
    q = rows.x0.clone()
    opt = RowAdam(q, _col(lr))
    for _ in range(int(num_opt_steps)):
        q = q.detach().requires_grad_(True)
        loss = phi_per_measurement(rows, G(q)).sum()
        (grad,) = torch.autograd.grad(loss, q)
        q = opt.step(q.detach(), grad)
        cost.evals += per_obj
        cost.backprops += per_obj
        cost.data_grads += 1
    with torch.no_grad():
        x = G(q)
    cost.evals += per_obj
    _record(trace, q)
    _record(trace, x)
    return x.detach(), cost


# =====================================================================================
# PnP-Flow (src/pnp.py), one noise realisation
# =====================================================================================
def pnp(prior, rows: Rows, num_pnp_steps: int, gamma0: torch.Tensor, alpha: torch.Tensor,
        trace: Optional[list] = None):
    """Initial projection of x0 at s0 = 0, then for s_k = k / (N + 1), k = 1..N: a gradient
    step of size gamma0 (1 - s_k)^alpha on the fidelity, reprojection s_k z + (1 - s_k) eps,
    and the denoiser (MeanFlow T(q; s_k -> 1); flow matching q + (1 - s_k) v(q, s_k)).
    Never differentiates the prior."""
    N = int(num_pnp_steps)
    meanflow = prior.family == "meanflow"
    denoise = ((lambda q, s: prior.transition(q, s, 1.0)) if meanflow
               else (lambda q, s: prior.endpoint(q, s)))
    cost = Cost()
    with torch.no_grad():
        _record(trace, rows.x0)
        x = denoise(rows.x0, 0.0)
        cost.evals += 1
        _record(trace, x)
        for k in range(1, N + 1):
            s = k / float(N + 1)
            gamma = gamma0 * (1.0 - s) ** alpha
            z = x - _col(gamma) * grad_phi_per_measurement(rows, x)
            cost.data_grads += 1
            eps = rows.noise.draw(rows, "pnp", k)
            x = denoise(s * z + (1.0 - s) * eps, s)
            cost.evals += 1
            _record(trace, x)
    return x, cost


# =====================================================================================
# MPC-delta_t and MPC-RHC (src/mpc.py)
# =====================================================================================
def mpc_delta_t(prior, rows: Rows, num_mpc_steps: int, n_ctrl: int, lam: torch.Tensor,
                lr: torch.Tensor, trace: Optional[list] = None):
    """At s: u minimises ||u||^2 + lam Phi(V(x_next(u))), x_next = nominal + dt u (MeanFlow,
    nominal = T(x; s -> s+dt)) or x + dt (v(x, s) + u) (flow matching); the value-to-go is
    Phi(T(x_next; s+dt -> 1)) or Phi(x_next + (1 - s - dt) v(x_next, s + dt)), and simply
    Phi(x_next) on the last step.  delta_t_lambda_scaling = none, as configured."""
    grid = _grid(int(num_mpc_steps))
    meanflow = prior.family == "meanflow"
    cost = Cost()
    x = rows.x0.clone()
    _record(trace, x)
    for k in range(int(num_mpc_steps)):
        s, s_next = grid[k], grid[k + 1]
        dt = s_next - s
        with torch.no_grad():
            if meanflow:
                base = prior.transition(x, s, s_next)
            else:
                base = x + dt * prior.velocity(x, s)
        cost.evals += 1
        terminal = s_next >= 1.0 - 1e-9
        u = torch.zeros_like(x)
        opt = RowAdam(u, _col(lr))
        for _ in range(int(n_ctrl)):
            u = u.detach().requires_grad_(True)
            x_next = base + dt * u
            if terminal:
                x_term = x_next
            elif meanflow:
                x_term = prior.transition(x_next, s_next, 1.0)
            else:
                x_term = prior.endpoint(x_next, s_next)
            loss = ((u * u).sum(1) + lam * phi_half_sum(rows, x_term)).sum()
            (grad,) = torch.autograd.grad(loss, u)
            u = opt.step(u.detach(), grad)
            used = 0 if terminal else 1
            cost.evals += used
            cost.backprops += used
            cost.data_grads += 1
        x = (base + dt * u).detach()
        _record(trace, x)
    return x, cost


def mpc_rhc(prior, rows: Rows, num_mpc_steps: int, n_ctrl: int, K: int, lam: torch.Tensor,
            lr: torch.Tensor, trace: Optional[list] = None):
    """Receding-horizon control with K planning intervals of h = (1 - s) / K over the
    remaining horizon; cost h sum ||u_k||^2 + lam Phi(x_K); execute the first control over
    the outer interval.  Flow matching: Euler steps x + h (v + u_k) (K = 1 needs no model in
    the graph).  MeanFlow: x_{k+1} = T(x_k; s_k -> s_{k+1}) + h u_k, the first transition
    computed once outside the graph; execution T(x; s -> s+dt) + dt u_0."""
    grid = _grid(int(num_mpc_steps))
    meanflow = prior.family == "meanflow"
    K = int(K)
    cost = Cost()
    x = rows.x0.clone()
    _record(trace, x)
    for step in range(int(num_mpc_steps)):
        s, s_next = grid[step], grid[step + 1]
        rem = 1.0 - s
        h = rem / K
        plan = [s + k * h for k in range(K + 1)]
        plan[-1] = 1.0
        with torch.no_grad():
            if meanflow:
                first = prior.transition(x, plan[0], plan[1])       # = T(x; s -> 1) if K = 1
            else:
                v0 = prior.velocity(x, s)
        cost.evals += 1
        U = torch.zeros((K,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        opt = RowAdam(U, lr[None, :, None])
        for _ in range(int(n_ctrl)):
            U = U.detach().requires_grad_(True)
            if meanflow:
                xk = first + (rem if K == 1 else h) * U[0]
                for k in range(1, K):
                    xk = prior.transition(xk, plan[k], plan[k + 1]) + h * U[k]
            else:
                xk = x + h * (v0 + U[0])
                for k in range(1, K):
                    xk = xk + h * (prior.velocity(xk, s + k * h) + U[k])
            control = h * (U * U).sum(2).sum(0)                       # = rem ||u||^2 at K=1
            loss = (control + lam * phi_half_sum(rows, xk)).sum()
            (grad,) = torch.autograd.grad(loss, U)
            U = opt.step(U.detach(), grad)
            cost.evals += K - 1
            cost.backprops += K - 1
            cost.data_grads += 1
        with torch.no_grad():
            if meanflow:
                x = prior.transition(x, s, s_next) + (s_next - s) * U[0]
                cost.evals += 1
            else:
                x = x + (s_next - s) * (v0 + U[0])
        _record(trace, x)
    return x.detach(), cost


# =====================================================================================
# Flower (src/flower.py), gamma = 0, exact refinement
# =====================================================================================
def flower_nu(t: float) -> float:
    return (1.0 - t) / math.sqrt(t * t + (1.0 - t) * (1.0 - t))


def flower(prior, rows: Rows, num_flower_steps: int, trace: Optional[list] = None):
    """For t_k = k / N: destination x_hat (MeanFlow T(x; t -> 1); flow matching
    x + (1 - t) v), refinement mu = (nu^-2 I + sigma^-2 H^T H)^-1 (nu^-2 x_hat +
    sigma^-2 H^T y) solved exactly (dense, float64), then x = (1 - t') eps + t' mu with
    fresh eps; the output is the last mu.  N evaluations, no backprop."""
    N = int(num_flower_steps)
    grid = _grid(N)
    meanflow = prior.family == "meanflow"
    cost = Cost()
    Hd = rows.H.double()
    HtH = torch.einsum("rmd,rme->rde", Hd, Hd)
    Hty = torch.einsum("rmd,rm->rd", Hd, rows.y.double())
    inv_s2 = (1.0 / rows.sigma.double() ** 2)[:, None]
    eye = torch.eye(rows.x0.shape[1], dtype=torch.float64, device=rows.x0.device)
    x = rows.x0.clone()
    _record(trace, x)
    with torch.no_grad():
        for k in range(N):
            t, t_next = grid[k], grid[k + 1]
            x_hat = prior.transition(x, t, 1.0) if meanflow else prior.endpoint(x, t)
            cost.evals += 1
            a = flower_nu(t) ** -2
            A = a * eye + inv_s2[:, :, None] * HtH
            b = a * x_hat.double() + inv_s2 * Hty
            mu = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1).to(x.dtype)
            cost.linear_solves += 1
            cost.data_grads += 1
            if k == N - 1:
                x = mu
            else:
                x = (1.0 - t_next) * rows.noise.draw(rows, "flower", k) + t_next * mu
            _record(trace, mu)
    return x, cost


# =====================================================================================
# Registry: structural knobs (one run each) and continuous knobs (one row block each)
# =====================================================================================
@dataclass
class SolverSpec:
    name: str
    fn: Callable
    structural: Dict[str, list]
    continuous: Dict[str, list] = field(default_factory=dict)
    fixed: Dict[str, object] = field(default_factory=dict)
    families: tuple = ("flow_matching", "meanflow")
    label: str = ""


# Image-grid gamma0 values were chosen for 256 x 256 RGB images under the per-measurement
# fidelity, whose gradient scales as 1 / m.  The toy grid keeps the per-step coefficient on
# H^T (Hx - y) the same: gamma0_toy = gamma0_image * m_toy / m_image, m_image = 3 * 256^2.
M_IMAGE = 3 * 256 * 256
PNP_GAMMA0_IMAGE = [25000.0, 100000.0, 200000.0, 400000.0, 800000.0, 1200000.0, 5000000.0]


def solver_specs() -> List[SolverSpec]:
    """The pre-registered pMF grids of configs/experiments_hpo_grid.yaml (RHSO at
    N * M = 160 as in the frozen benchmark; PnP's gamma0 rescaled as above)."""
    rhso_cont = dict(lr=[0.005, 0.01, 0.02, 0.03, 0.05, 0.1], mu=[0.0, 0.2, 1.0])
    return [
        SolverSpec("rhso", rhso, dict(NM=[(4, 40), (8, 20)]), rhso_cont,
                   fixed=dict(terminal="direct"), label="RHSO"),
        SolverSpec("rhso_suffix", rhso, dict(NM=[(4, 40), (8, 20)]), rhso_cont,
                   fixed=dict(terminal="suffix"), families=("flow_matching",),
                   label="RHSO (suffix)"),
        SolverSpec("dflow", dflow, dict(num_opt_steps=[160, 320, 640]),
                   dict(lr=[0.003, 0.01, 0.03, 0.1, 0.3]), label="D-Flow (1 step)"),
        SolverSpec("pnp", pnp, dict(num_pnp_steps=[10, 20, 50, 100, 200]),
                   dict(gamma0_image=PNP_GAMMA0_IMAGE, alpha=[0.1, 0.25, 0.5, 0.75, 1.0]),
                   label="PnP-Flow"),
        SolverSpec("mpc_delta_t", mpc_delta_t, dict(num_mpc_steps=[4, 8], n_ctrl=[20, 40]),
                   dict(lam=[10.0, 30.0, 180.0, 360.0, 540.0, 1000.0],
                        lr=[0.05, 0.1, 0.15, 0.3]), label="MPC-$\\Delta t$"),
        SolverSpec("mpc_rhc", mpc_rhc, dict(K=[1, 2, 3]),
                   dict(lam=[5.0, 15.0, 30.0, 60.0, 180.0, 240.0, 720.0],
                        lr=[0.02, 0.05, 0.1, 0.2]),
                   fixed=dict(num_mpc_steps=4, n_ctrl=40), label="MPC-RHC"),
        SolverSpec("flower", flower, dict(num_flower_steps=[50, 100, 200, 500]),
                   label="Flower"),
    ]


def call(spec: SolverSpec, prior, rows: Rows, structural: Dict, continuous: Dict[str, torch.Tensor],
         trace: Optional[list] = None):
    """Run one solver with one structural setting; continuous knobs are per-row tensors.
    `trace`: see the module docstring."""
    kw = dict(spec.fixed)
    if trace is not None:
        kw["trace"] = trace
    for key, value in structural.items():
        if key == "NM":
            kw["N"], kw["M"] = value
        else:
            kw[key] = value
    for key, value in continuous.items():
        if key == "gamma0_image":
            kw["gamma0"] = value * rows.m / M_IMAGE
        else:
            kw[key] = value
    return spec.fn(prior, rows, **kw)
