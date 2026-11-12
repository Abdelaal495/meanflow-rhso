#!/usr/bin/env python
"""Gaussian-mixture study with known posteriors (rebuttal Task 4).

A flow-matching model and a MeanFlow model share ONE MLP (width 512, depth 4) and ONE
training recipe on the same 16-D Gaussian mixture; every solver is later run on both and
scored against the EXACT posterior, which is again a Gaussian mixture.

    python toy/gmm_study.py stage1               # train both priors (cached), sanity checks
    python toy/gmm_study.py stage1 --retrain     # ignore the cached models
    python toy/gmm_study.py sweep                # MeanFlow stability sweep (10k steps each)
    python toy/gmm_study.py steptime             # time per training step, both models

Every command takes --device (default: cuda when available, else cpu).

Stage 1 (this file so far): the mixture, its closed-form posterior under y = Hx + sigma eps,
the exact velocity field of the mixture and its fine ODE solve, the two trained priors and
their sanity checks.  The solvers follow in stage 2.

The mixture
-----------
d = 16, 8 components with equal weights, means of norm 2 in random directions, and
anisotropic covariances: each a random rotation of diag(s_j^2) with s_j = 0.5 j^(-a),
a = log(25) / log(16), so s runs from 0.5 down to 0.02 and some directions carry almost no
signal.  One fixed seed.  The mixture is then centred and scaled by one scalar so that the
average variance per coordinate is 1 (the scalar keeps every ratio above).

Time convention (as in toy/common.py): t = 0 noise, t = 1 data, x_t = (1 - t) x0 + t x1.

The exact velocity
------------------
Given component i, x_t is Gaussian with mean t m_i and covariance S_i(t) = t^2 C_i + (1-t)^2 I,
and (x1, x0, x_t) are jointly Gaussian, so

    v(x, t) = E[x1 - x0 | x_t = x] = sum_i r_i(x, t) [ m_i + (t C_i - (1 - t) I) S_i(t)^-1 (x - t m_i) ]

with r_i the posterior responsibilities.  C_i and I share C_i's eigenvectors, so every term
is diagonal in that basis and the formula is finite on all of [0, 1].
"""
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MeanFlowMLP, VelocityMLP, freeze, train_flow_matching, train_meanflow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "gmm_study"
OUT = ROOT / "outputs" / "gmm_study"

DIM, COMPONENTS = 16, 8
MIXTURE_SEED = 20261111
MEAN_NORM, S_MAX, S_MIN = 2.0, 0.5, 0.02
TRAIN_POOL = 1_000_000

# one recipe for both priors (common.train_* defaults: Adam 3e-4, 500 warm-up steps then
# cosine, grad clip 1, EMA 0.999)
WIDTH, DEPTH = 512, 4
TRAIN_SEED, TRAIN_STEPS, TRAIN_BATCH = 0, 40_000, 1024


@dataclass(frozen=True)
class Recipe:
    """The knobs the stability sweep varies.  `lr` and `s_min` (the data) are shared by both
    priors -- a flow-matching model is only ever compared with a MeanFlow model trained on the
    same data with the same learning rate, steps and batch; `long_share` (share of s != t
    pairs) and `adaptive_p` (adaptive-weight exponent) exist only in the MeanFlow objective."""
    lr: float = 3e-4
    s_min: float = S_MIN
    long_share: float = 0.5
    adaptive_p: float = 0.5

    def tag(self, kind: str) -> str:
        """Suffix naming the non-default knobs that apply to `kind` ('' for the defaults, so
        models cached before the knobs existed still load)."""
        default, parts = Recipe(), []
        if self.lr != default.lr:
            parts.append("lr%g" % self.lr)
        if self.s_min != default.s_min:
            parts.append("smin%g" % self.s_min)
        if kind == "meanflow":
            if self.long_share != default.long_share:
                parts.append("long%g" % self.long_share)
            if self.adaptive_p != default.adaptive_p:
                parts.append("p%g" % self.adaptive_p)
        return "".join("_" + p for p in parts)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================================
# Gaussian mixtures
# =====================================================================================
@dataclass
class GaussianMixture:
    """weights (K,), means (K, d), covs (K, d, d); float64 throughout."""
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    log_evidence: Optional[float] = None          # set on a posterior: log p(y)
    _eig: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)
    _chol: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def k(self) -> int:
        return len(self.weights)

    @property
    def d(self) -> int:
        return self.means.shape[1]

    def eig(self):
        """(eigenvalues (K, d), eigenvectors (K, d, d)) of every component covariance."""
        if self._eig is None:
            lam, vec = np.linalg.eigh(self.covs)
            self._eig = (np.clip(lam, 0.0, None), vec)
        return self._eig

    def chol(self):
        if self._chol is None:
            jitter = 1e-12 * np.eye(self.d)
            self._chol = np.linalg.cholesky(self.covs + jitter)
        return self._chol

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        comp = rng.choice(self.k, size=n, p=self.weights)
        z = rng.standard_normal((n, self.d))
        x = self.means[comp] + np.einsum("nij,nj->ni", self.chol()[comp], z)
        return x, comp

    def component_log_prob(self, x: np.ndarray) -> np.ndarray:
        """log w_i + log N(x; m_i, C_i), shape (n, K)."""
        L = self.chol()
        out = np.empty((len(x), self.k))
        for i in range(self.k):
            diff = x - self.means[i]
            sol = np.linalg.solve(L[i], diff.T).T
            logdet = 2.0 * np.log(np.diag(L[i])).sum()
            out[:, i] = (np.log(self.weights[i]) - 0.5 * (sol ** 2).sum(1)
                         - 0.5 * logdet - 0.5 * self.d * math.log(2 * math.pi))
        return out

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        return _logsumexp(self.component_log_prob(x), axis=1)

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        lp = self.component_log_prob(x)
        return np.exp(lp - _logsumexp(lp, axis=1)[:, None])

    def mean(self) -> np.ndarray:
        return self.weights @ self.means

    def covariance(self) -> np.ndarray:
        mu = self.mean()
        second = np.einsum("k,kij->ij", self.weights, self.covs
                           + np.einsum("ki,kj->kij", self.means, self.means))
        return second - np.outer(mu, mu)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal matrix (QR with the sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))[None, :]


def component_scales(d: int = DIM, s_min: float = S_MIN) -> np.ndarray:
    """s_j = S_MAX * j^(-a), j = 1..d, with s_d = s_min: a power law from 0.5 to s_min
    (0.02 by default)."""
    a = math.log(S_MAX / s_min) / math.log(d)
    return S_MAX * np.arange(1, d + 1, dtype=np.float64) ** (-a)


def make_mixture(seed: int = MIXTURE_SEED, d: int = DIM, k: int = COMPONENTS,
                 s_min: float = S_MIN) -> GaussianMixture:
    """`s_min` changes only the scales; means and rotations come from the same seed."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((k, d))
    means = MEAN_NORM * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    s = component_scales(d, s_min)
    covs = np.stack([R @ np.diag(s ** 2) @ R.T for R in (random_rotation(d, rng)
                                                         for _ in range(k))])
    raw = GaussianMixture(np.full(k, 1.0 / k), means, covs)
    # centre, then ONE scalar so the average variance per coordinate is 1
    c = 1.0 / math.sqrt(np.trace(raw.covariance()) / d)
    return GaussianMixture(raw.weights, (means - raw.mean()) * c, covs * c * c)


# =====================================================================================
# Operators and the closed-form posterior
# =====================================================================================
OPERATOR_SEED = 20261112


def make_operator(name: str, d: int = DIM, seed: int = OPERATOR_SEED) -> np.ndarray:
    """H as a dense (m, d) matrix: denoising I, masking 50% of the coordinates (fixed),
    projection onto 8 fixed random Gaussian rows (entries N(0, 1/d))."""
    rng = np.random.default_rng([seed, {"denoising": 0, "masking": 1, "projection": 2}[name]])
    if name == "denoising":
        return np.eye(d)
    if name == "masking":
        keep = np.sort(rng.permutation(d)[: d // 2])
        return np.eye(d)[keep]
    if name == "projection":
        return rng.standard_normal((d // 2, d)) / math.sqrt(d)
    raise ValueError(name)


def posterior(prior: GaussianMixture, H: np.ndarray, sigma: float,
              y: np.ndarray) -> GaussianMixture:
    """p(x | y) for y = Hx + sigma eps, exactly: a Gaussian mixture.

    Per component (standard linear-Gaussian update):
        S_i  = H C_i H^T + sigma^2 I,      K_i = C_i H^T S_i^-1
        m_i' = m_i + K_i (y - H m_i),      C_i' = C_i - K_i H C_i
        w_i' proportional to w_i N(y; H m_i, S_i)
    """
    m = H.shape[0]
    log_w = np.empty(prior.k)
    means = np.empty_like(prior.means)
    covs = np.empty_like(prior.covs)
    for i in range(prior.k):
        C = prior.covs[i]
        S = H @ C @ H.T + sigma ** 2 * np.eye(m)
        L = np.linalg.cholesky(S)
        r = y - H @ prior.means[i]
        alpha = np.linalg.solve(L, r)
        log_w[i] = (math.log(prior.weights[i]) - 0.5 * alpha @ alpha
                    - np.log(np.diag(L)).sum() - 0.5 * m * math.log(2 * math.pi))
        K = np.linalg.solve(S, H @ C).T                  # C H^T S^-1  (C, S symmetric)
        means[i] = prior.means[i] + K @ r
        Ci = C - K @ H @ C
        covs[i] = 0.5 * (Ci + Ci.T)
    log_evidence = float(_logsumexp(log_w, axis=0))
    return GaussianMixture(np.exp(log_w - log_evidence), means, covs, log_evidence)


# =====================================================================================
# The exact flow of the mixture
# =====================================================================================
def exact_velocity(mix: GaussianMixture, x: np.ndarray, t: float) -> np.ndarray:
    """v(x, t) = E[x1 - x0 | x_t = x] for x1 ~ mix, x0 ~ N(0, I); float64, any t in [0, 1]."""
    lam, vec = mix.eig()
    t = float(t)
    n, d = x.shape
    logits = np.empty((n, mix.k))
    terms = np.empty((mix.k, n, d))
    for i in range(mix.k):
        z = (x - t * mix.means[i]) @ vec[i]              # coordinates in C_i's eigenbasis
        denom = t * t * lam[i] + (1.0 - t) ** 2           # eigenvalues of S_i(t)
        logits[:, i] = (math.log(mix.weights[i]) - 0.5 * (z * z / denom).sum(1)
                        - 0.5 * np.log(denom).sum())
        gain = (t * lam[i] - (1.0 - t)) / denom
        terms[i] = mix.means[i] + (z * gain) @ vec[i].T
    r = np.exp(logits - _logsumexp(logits, axis=1)[:, None])
    return np.einsum("ni,nid->nd", r, terms.transpose(1, 0, 2))


def exact_flow(mix: GaussianMixture, x0: np.ndarray, steps: int = 1000, t0: float = 0.0,
               t1: float = 1.0) -> np.ndarray:
    """Fine RK4 solve of dx/dt = v(x, t) from t0 to t1 (float64)."""
    x = np.asarray(x0, np.float64).copy()
    h = (t1 - t0) / steps
    for k in range(steps):
        t = t0 + k * h
        k1 = exact_velocity(mix, x, t)
        k2 = exact_velocity(mix, x + 0.5 * h * k1, t + 0.5 * h)
        k3 = exact_velocity(mix, x + 0.5 * h * k2, t + 0.5 * h)
        k4 = exact_velocity(mix, x + h * k3, t + h)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def exact_velocity_rows(mix: GaussianMixture, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """exact_velocity with one time per row, t of shape (n,)."""
    lam, vec = mix.eig()
    t = np.asarray(t, np.float64)[:, None]
    n, d = x.shape
    logits = np.empty((n, mix.k))
    terms = np.empty((n, mix.k, d))
    for i in range(mix.k):
        z = (x - t * mix.means[i]) @ vec[i]
        denom = t * t * lam[i][None, :] + (1.0 - t) ** 2
        logits[:, i] = (math.log(mix.weights[i]) - 0.5 * (z * z / denom).sum(1)
                        - 0.5 * np.log(denom).sum(1))
        terms[:, i] = mix.means[i] + (z * (t * lam[i][None, :] - (1.0 - t)) / denom) @ vec[i].T
    r = np.exp(logits - _logsumexp(logits, axis=1)[:, None])
    return np.einsum("ni,nid->nd", r, terms)


def exact_transition(mix: GaussianMixture, x: np.ndarray, t: np.ndarray, s: np.ndarray,
                     steps: int = 400) -> np.ndarray:
    """Phi_{t -> s}(x) of the exact flow, one (t, s) per row: RK4 in tau in [0, 1] on
    time t + tau (s - t), so every row gets `steps` steps whatever its interval."""
    x = np.asarray(x, np.float64).copy()
    t, s = np.asarray(t, np.float64), np.asarray(s, np.float64)
    span, h = (s - t)[:, None], 1.0 / steps

    def f(y, tau):
        return span * exact_velocity_rows(mix, y, t + tau * (s - t))

    for k in range(steps):
        tau = k * h
        k1 = f(x, tau)
        k2 = f(x + 0.5 * h * k1, tau + 0.5 * h)
        k3 = f(x + 0.5 * h * k2, tau + 0.5 * h)
        k4 = f(x + h * k3, tau + h)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def exact_average_velocity(mix: GaussianMixture, z: np.ndarray, t: np.ndarray, s: np.ndarray,
                           steps: int = 400) -> np.ndarray:
    """u*(z, t, s) = (Phi_{t -> s}(z) - z) / (s - t); the exact velocity where s = t."""
    t, s = np.asarray(t, np.float64), np.asarray(s, np.float64)
    out = exact_velocity_rows(mix, z, t)
    long = s != t
    if long.any():
        out[long] = ((exact_transition(mix, z[long], t[long], s[long], steps) - z[long])
                     / (s[long] - t[long])[:, None])
    return out


def meanflow_error_floor(mix: GaussianMixture, z: np.ndarray, v: np.ndarray, t: np.ndarray,
                         s: np.ndarray, steps: int = 400, eps: float = 1e-5) -> np.ndarray:
    """The raw MeanFlow error of the EXACT average velocity u*, per training pair.

    The trainer's target is v + (s - t) du/dt with du/dt taken along the conditional path
    (tangent v = x1 - x0, not the exact velocity v*).  Since u* obeys the MeanFlow identity
    along the exact flow, u* = v* + (s - t) [J u* v* + d_t u*], the difference is
        u* - target = -(I + (s - t) J u*) (v - v*) = -dPhi_{t -> s}/dz (v - v*(z, t)),
    so the error is |dPhi_{t -> s}/dz (v - v*)|^2 (just |v - v*|^2 where s = t).  No model
    can do better in expectation: u* is E[target | z, t, s].  The Jacobian-vector product is
    a central difference of the RK4 map along v - v*."""
    t, s = np.asarray(t, np.float64), np.asarray(s, np.float64)
    w = v - exact_velocity_rows(mix, z, t)
    out = (w ** 2).sum(1)
    long = s != t
    m = int(long.sum())
    if m:
        zl, wl = z[long], w[long]
        both = exact_transition(mix, np.concatenate([zl + eps * wl, zl - eps * wl]),
                                np.tile(t[long], 2), np.tile(s[long], 2), steps)
        out[long] = (((both[:m] - both[m:]) / (2 * eps)) ** 2).sum(1)
    return out


# =====================================================================================
# The two priors
# =====================================================================================
def _cache_path(kind: str, steps: int, batch: int, recipe: Recipe = Recipe()) -> Path:
    return CACHE / ("%s_w%d_d%d_s%d_b%d_seed%d%s.pt"
                    % (kind, WIDTH, DEPTH, steps, batch, TRAIN_SEED, recipe.tag(kind)))


def _device_label(device: str) -> str:
    if str(device).startswith("cuda"):
        return "GPU (%s)" % torch.cuda.get_device_name(torch.device(device))
    return "CPU (%d threads)" % torch.get_num_threads()


def train_prior(kind: str, mix: GaussianMixture, steps: int, batch: int, recipe: Recipe,
                device: str, verbose: bool = True, log_every: int = 1000):
    """Train one prior from scratch on `device`; returns (EMA model, info dict)."""
    cls = VelocityMLP if kind == "flow_matching" else MeanFlowMLP
    torch.manual_seed(TRAIN_SEED)
    model = cls(dim=mix.d, width=WIDTH, depth=DEPTH).to(device)

    def data_fn(rng):
        return mix.sample(TRAIN_POOL, rng)[0].astype(np.float32)

    log: List[dict] = []
    if verbose:
        print("training %s: %d steps, batch %d, width %d, depth %d, %s, %s ..."
              % (kind, steps, batch, WIDTH, DEPTH, recipe, _device_label(device)))
    started = time.perf_counter()
    if kind == "flow_matching":
        trained = train_flow_matching(model, data_fn, steps, batch, TRAIN_SEED, lr=recipe.lr,
                                      verbose=verbose, log_every=log_every, log=log)
    else:
        trained = train_meanflow(model, data_fn, steps, batch, TRAIN_SEED, lr=recipe.lr,
                                 long_share=recipe.long_share, adaptive_p=recipe.adaptive_p,
                                 verbose=verbose, log_every=log_every, log=log)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    info = dict(kind=kind, steps=steps, batch=batch, width=WIDTH, depth=DEPTH, seed=TRAIN_SEED,
                seconds=seconds, device=_device_label(device), threads=torch.get_num_threads(),
                log=log, recipe=dict(lr=recipe.lr, s_min=recipe.s_min,
                                     long_share=recipe.long_share, adaptive_p=recipe.adaptive_p,
                                     schedule="500 warm-up + cosine", clip=1.0, ema=0.999))
    return trained, info


def load_or_train(kind: str, mix: GaussianMixture, retrain: bool = False,
                  steps: int = TRAIN_STEPS, batch: int = TRAIN_BATCH, verbose: bool = True,
                  recipe: Recipe = Recipe(), device: str = "cpu"):
    """kind = 'flow_matching' | 'meanflow'.  Returns (frozen model on `device`, info dict)."""
    path = _cache_path(kind, steps, batch, recipe)
    if path.exists() and not retrain:
        cls = VelocityMLP if kind == "flow_matching" else MeanFlowMLP
        model = cls(dim=mix.d, width=WIDTH, depth=DEPTH)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"])
        if verbose:
            print("loaded %s (trained in %.0f s)" % (path, state["info"]["seconds"]))
        return freeze(model.to(device)), state["info"]
    trained, info = train_prior(kind, mix, steps, batch, recipe, device, verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in trained.state_dict().items()}, "info": info}, path)
    if verbose:
        print("saved %s (%.0f s)" % (path, info["seconds"]))
    return freeze(trained), info


@torch.no_grad()
def fm_velocity(model, x: torch.Tensor, t: float) -> torch.Tensor:
    return model(x, torch.full((x.shape[0],), float(t), device=x.device))


@torch.no_grad()
def mf_transition(model, x: torch.Tensor, t: float, s: float) -> torch.Tensor:
    tt = torch.full((x.shape[0],), float(t), device=x.device)
    ss = torch.full((x.shape[0],), float(s), device=x.device)
    return x + (float(s) - float(t)) * model(x, tt, ss)


@torch.no_grad()
def fm_sample(model, x0: torch.Tensor, steps: int = 100, t0: float = 0.0) -> torch.Tensor:
    """Heun integration of the learned velocity from t0 to 1."""
    x, h = x0.clone(), (1.0 - t0) / steps
    for k in range(steps):
        t = t0 + k * h
        a = fm_velocity(model, x, t)
        b = fm_velocity(model, x + h * a, t + h)
        x = x + 0.5 * h * (a + b)
    return x


@torch.no_grad()
def mf_sample(model, x0: torch.Tensor, steps: int = 1) -> torch.Tensor:
    x = x0.clone()
    for k in range(steps):
        x = mf_transition(model, x, k / steps, (k + 1) / steps)
    return x


# =====================================================================================
# Sanity checks
# =====================================================================================
JUMP_TIMES = (0.25, 0.5, 0.75)        # T(x; t -> 1) is also checked from these times
PASS_ONE_STEP_SW2 = 0.05              # MeanFlow T(x; 0 -> 1): sliced W2 at most this
PASS_VAR_LO = (0.7, 1.5)              # ... and low-signal variance ratio in this range
PASS_EIGHT_STEP_FACTOR = 1.5          # MeanFlow 8 steps: SW2 within this factor of FM Heun-100


def sliced_w2(a: np.ndarray, b: np.ndarray, n_proj: int = 512, seed: int = 0) -> float:
    """Sliced Wasserstein-2 between two equal-size samples (exact 1-D W2 per direction)."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((a.shape[1], n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    pa, pb = np.sort(a @ theta, axis=0), np.sort(b @ theta, axis=0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def low_signal_variance_ratio(mix: GaussianMixture, x: np.ndarray) -> Tuple[float, float]:
    """Within-component variance of x along each component's 8 largest / 8 smallest
    eigendirections, relative to the true eigenvalue (geometric mean; 1 = exact).

    Samples are assigned to their most likely component, so the same statistic computed on
    true samples is the reference (assignment adds a small bias)."""
    lam, vec = mix.eig()
    comp = mix.responsibilities(x).argmax(1)
    hi, lo = [], []
    half = mix.d // 2
    for i in range(mix.k):
        xi = x[comp == i]
        if len(xi) < 50:
            continue
        var = ((xi - xi.mean(0)) @ vec[i]).var(0)
        ratio = var / lam[i]
        order = np.argsort(lam[i])                       # ascending
        lo.append(np.log(ratio[order[:half]]))
        hi.append(np.log(ratio[order[half:]]))
    return float(np.exp(np.mean(hi))), float(np.exp(np.mean(lo)))


def _recipe_from(args) -> Recipe:
    return Recipe(lr=args.lr, s_min=args.s_min, long_share=args.long_share,
                  adaptive_p=args.adaptive_p)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy().astype(np.float64)


def stage1(args) -> Dict:
    torch.set_num_threads(args.threads or torch.get_num_threads())
    device, recipe = args.device, _recipe_from(args)
    mix = make_mixture(s_min=recipe.s_min)
    lam = mix.eig()[0]
    print("mixture: d=%d, K=%d, mean norms %.3f-%.3f, component std %.4f-%.4f, average "
          "variance %.4f" % (mix.d, mix.k, np.linalg.norm(mix.means, axis=1).min(),
                             np.linalg.norm(mix.means, axis=1).max(), math.sqrt(lam.min()),
                             math.sqrt(lam.max()), np.trace(mix.covariance()) / mix.d))
    fm, fm_info = load_or_train("flow_matching", mix, args.retrain, args.steps, args.batch,
                                recipe=recipe, device=device)
    mf, mf_info = load_or_train("meanflow", mix, args.retrain, args.steps, args.batch,
                                recipe=recipe, device=device)

    n = args.samples
    rng = np.random.default_rng(1)
    truth, _ = mix.sample(n, rng)
    truth2, _ = mix.sample(n, rng)
    x0 = rng.standard_normal((n, mix.d))

    def dev(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.astype(np.float32)).to(device)

    # the exact flow in segments, keeping the states at JUMP_TIMES (same grid as one solve)
    started = time.perf_counter()
    knots = (0.0,) + JUMP_TIMES + (1.0,)
    states, x = {0.0: x0}, x0
    for a, b in zip(knots[:-1], knots[1:]):
        x = exact_flow(mix, x, steps=int(round(args.ode_steps * (b - a))), t0=a, t1=b)
        states[b] = x
    exact = states[1.0]
    exact_seconds = time.perf_counter() - started

    gen = {
        "flow-matching ODE (Heun, 100 steps)": _to_np(fm_sample(fm, dev(x0), 100)),
        "MeanFlow, 1 step T(x; 0->1)": _to_np(mf_sample(mf, dev(x0), 1)),
        "MeanFlow, 2 steps": _to_np(mf_sample(mf, dev(x0), 2)),
        "MeanFlow, 8 steps": _to_np(mf_sample(mf, dev(x0), 8)),
        "exact flow (RK4, %d steps)" % args.ode_steps: exact,
    }
    for t in JUMP_TIMES:
        # x_t is the exact flow's state at t, so the reference is the same trajectory's end
        gen["MeanFlow, 1 step T(x; %g->1)" % t] = _to_np(mf_transition(mf, dev(states[t]), t, 1.0))
        heun = int(round(100 * (1 - t)))
        gen["flow-matching Heun %d steps, %g->1" % (heun, t)] = _to_np(
            fm_sample(fm, dev(states[t]), heun, t0=t))
    floor = sliced_w2(truth, truth2)
    ll_true = float(mix.log_prob(truth2).mean())
    ref_hi, ref_lo = low_signal_variance_ratio(mix, truth2)
    rows = []
    for name, x in gen.items():
        hi, lo = low_signal_variance_ratio(mix, x)
        diff = x - exact
        rows.append(dict(
            sampler=name, sliced_w2=sliced_w2(x, truth), mean_log_density=float(mix.log_prob(x).mean()),
            var_ratio_high_signal=hi, var_ratio_low_signal=lo,
            rmse_vs_exact_flow=float(np.sqrt((diff ** 2).mean())),
            median_rel_err_vs_exact_flow=float(np.median(np.linalg.norm(diff, axis=1)
                                                         / np.linalg.norm(exact, axis=1)))))
    by_name = {r["sampler"]: r for r in rows}
    one = by_name["MeanFlow, 1 step T(x; 0->1)"]
    eight = by_name["MeanFlow, 8 steps"]["sliced_w2"]
    heun = by_name["flow-matching ODE (Heun, 100 steps)"]["sliced_w2"]
    criteria = [
        ("MeanFlow 1-step SW2 <= %g" % PASS_ONE_STEP_SW2, one["sliced_w2"],
         one["sliced_w2"] <= PASS_ONE_STEP_SW2),
        ("MeanFlow 1-step var lo in [%g, %g]" % PASS_VAR_LO, one["var_ratio_low_signal"],
         PASS_VAR_LO[0] <= one["var_ratio_low_signal"] <= PASS_VAR_LO[1]),
        ("MeanFlow 8-step SW2 <= %g x FM Heun-100 (%.4f)" % (PASS_EIGHT_STEP_FACTOR, heun),
         eight, eight <= PASS_EIGHT_STEP_FACTOR * heun),
    ]
    result = dict(
        samples=n, sliced_w2_floor=floor, mean_log_density_true=ll_true,
        var_ratio_reference=dict(high_signal=ref_hi, low_signal=ref_lo),
        exact_flow_seconds=exact_seconds, rows=rows,
        pass_criteria=[dict(name=c, value=v, passed=bool(ok)) for c, v, ok in criteria],
        passed=all(ok for _, _, ok in criteria),
        training={k: {kk: vv for kk, vv in v.items() if kk != "log"}
                  for k, v in (("flow_matching", fm_info), ("meanflow", mf_info))})

    print("\nsanity checks on %d samples (same noise for every sampler)" % n)
    print("  sliced W2 floor (two independent true samples): %.4f" % floor)
    print("  mean log density of true samples: %.3f;  within-component variance ratio of "
          "true samples: high-signal %.3f, low-signal %.3f" % (ll_true, ref_hi, ref_lo))
    print("  rows 'from t': x_t is the exact flow's state at t; every RMSE / rel err is against "
          "the same trajectory's endpoint")
    print("  %-38s %9s %9s %9s %9s %11s %9s" % ("sampler", "SW2", "log p", "var hi", "var lo",
                                              "RMSE exact", "rel err"))
    for r in rows:
        print("  %-38s %9.4f %9.3f %9.3f %9.3f %11.4f %9.4f"
              % (r["sampler"], r["sliced_w2"], r["mean_log_density"], r["var_ratio_high_signal"],
                 r["var_ratio_low_signal"], r["rmse_vs_exact_flow"],
                 r["median_rel_err_vs_exact_flow"]))
    print("  pass criteria:")
    for c, v, ok in criteria:
        print("    [%s] %-52s %.4f" % ("PASS" if ok else "FAIL", c, v))
    for kind, info in (("flow matching", fm_info), ("MeanFlow", mf_info)):
        print("  training %-14s %6.0f s  (%d steps, batch %d, %s)"
              % (kind, info["seconds"], info["steps"], info["batch"],
                 info.get("device", "CPU (%d threads)" % info["threads"])))

    out = OUT / ("stage1" + recipe.tag("meanflow"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "sanity.json").write_text(json.dumps(result, indent=2))
    _plot_training(fm_info, mf_info, out / "training_curves.png")
    print("wrote %s and training_curves.png" % (out / "sanity.json"))
    return result


def _plot_training(fm_info, mf_info, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, info, title in ((axes[0], fm_info, "flow matching (mse)"),
                            (axes[1], mf_info, "MeanFlow (raw mse to JVP target)")):
        log = info.get("log") or []
        ax.plot([e["step"] for e in log], [e["raw_mse"] for e in log])
        ax.set_yscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =====================================================================================
# MeanFlow stability sweep
# =====================================================================================
SWEEP_GRID = dict(long_share=(0.25, 0.5), adaptive_p=(0.5, 1.0), lr=(3e-4, 1e-4),
                  s_min=(0.02, 0.05))
SWEEP_STEPS, SWEEP_LOG_EVERY, SWEEP_SMOOTH = 10_000, 100, 5
BLOWUP_FACTOR = 10.0


def _smooth(values: List[float], width: int = SWEEP_SMOOTH) -> np.ndarray:
    """Running median over `width` logged batches (one batch's raw error is noisy)."""
    v = np.asarray(values, np.float64)
    return np.array([np.median(v[max(0, i - width + 1): i + 1]) for i in range(len(v))])


def replay_meanflow_batches(mix: GaussianMixture, steps: int, batch: int, long_share: float,
                            device: str, at_steps, width: int = WIDTH, depth: int = DEPTH,
                            seed: int = TRAIN_SEED) -> Dict[int, Tuple[np.ndarray, ...]]:
    """The (x1, x0, t, s) that common.train_meanflow draws at `at_steps`, as float64 numpy.

    Replays its random stream call for call: the same pool, the same seed, the throwaway
    EMA initialisation it performs on the CPU generator, then per step randint, randn, rand,
    rand, rand on `device`.  The batches do not depend on the learning rate or the adaptive
    exponent.  tests/test_gmm_study.py checks the replay against the trainer itself."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = torch.from_numpy(mix.sample(TRAIN_POOL, rng)[0].astype(np.float32)).to(device)
    MeanFlowMLP(dim=mix.d, width=width, depth=depth)    # the trainer's EMA copy draws its init
    want, out = set(at_steps), {}
    for step in range(1, max(want) + 1):
        x1 = data[torch.randint(0, len(data), (batch,), device=device)]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, device=device)
        ramp = min(1.0, step / (steps / 3))
        long = torch.rand(batch, device=device) < long_share * ramp
        s = torch.where(long, t + (1 - t) * torch.rand(batch, device=device) * ramp, t)
        if step in want:
            out[step] = tuple(a.cpu().numpy().astype(np.float64) for a in (x1, x0, t, s))
    return out


def summarize_curve(log: List[dict]) -> Dict:
    """Peak and final raw MeanFlow error, and whether the curve blows up: the smoothed error
    rises above BLOWUP_FACTOR x the lowest smoothed value reached before it."""
    steps = [e["step"] for e in log]
    sm = _smooth([e["raw_mse"] for e in log])
    running_min = np.minimum.accumulate(sm)
    rise = sm / running_min
    peak = int(np.argmax(sm))
    worst = int(np.argmax(rise))
    return dict(peak_raw=float(sm[peak]), peak_step=steps[peak],
                final_raw=float(np.mean([e["raw_mse"] for e in log[-SWEEP_SMOOTH:]])),
                min_raw=float(sm.min()), max_rise=float(rise[worst]), max_rise_step=steps[worst],
                blows_up=bool(rise.max() > BLOWUP_FACTOR))


def sweep(args) -> List[Dict]:
    """MeanFlow only, SWEEP_STEPS steps per setting (each run has its own full schedule:
    warm-up, cosine to zero and the interval curriculum over the first third), raw error
    logged every SWEEP_LOG_EVERY steps.  One JSON per setting under outputs/gmm_study/sweep/
    (an existing file is reused, so an interrupted sweep resumes)."""
    import itertools
    torch.set_num_threads(args.threads or torch.get_num_threads())
    out = OUT / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows, mixtures = [], {}
    keys = list(SWEEP_GRID)
    for values in itertools.product(*(SWEEP_GRID[k] for k in keys)):
        recipe = Recipe(**dict(zip(keys, values)))
        path = out / ("meanflow_s%d_b%d%s.json" % (args.steps, args.batch, recipe.tag("meanflow")
                                                   or "_default"))
        if path.exists():
            record = json.loads(path.read_text())
            print("reused %s" % path.name)
        else:
            if recipe.s_min not in mixtures:
                mixtures[recipe.s_min] = make_mixture(s_min=recipe.s_min)
            _, info = train_prior("meanflow", mixtures[recipe.s_min], args.steps, args.batch,
                                  recipe, args.device, verbose=False, log_every=SWEEP_LOG_EVERY)
            record = dict(recipe=info["recipe"], steps=args.steps, batch=args.batch,
                          seconds=info["seconds"], device=info["device"], log=info["log"])
            path.write_text(json.dumps(record, indent=1))
        summary = summarize_curve(record["log"])
        log = record["log"]                 # time per step between logs, so the one-off
        a, b = log[min(1, len(log) - 1)], log[-1]          # data-pool sampling is excluded
        row = dict(long_share=recipe.long_share, adaptive_p=recipe.adaptive_p, lr=recipe.lr,
                   s_min=recipe.s_min, seconds=record["seconds"],
                   ms_per_step=(1000.0 * (b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                                if b["step"] > a["step"] else float("nan")),
                   device=record["device"], **summary)
        rows.append(row)
        print("  long %-4g p %-3g lr %-6g s_min %-4g  peak %10.1f @%5d  final %8.1f  "
              "max rise x%-7.1f %s  (%.0f s, %.1f ms/step)"
              % (row["long_share"], row["adaptive_p"], row["lr"], row["s_min"], row["peak_raw"],
                 row["peak_step"], row["final_raw"], row["max_rise"],
                 "BLOWS UP" if row["blows_up"] else "stable  ", row["seconds"], row["ms_per_step"]))
    _write_sweep_table(rows, out)
    _plot_sweep(out, args.steps, args.batch)
    return rows


def _write_sweep_table(rows: List[Dict], out: Path) -> None:
    import csv
    cols = ["long_share", "adaptive_p", "lr", "s_min", "peak_raw", "peak_step", "final_raw",
            "min_raw", "max_rise", "max_rise_step", "blows_up", "seconds", "ms_per_step", "device"]
    with open(out / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print("wrote %s (blow-up: smoothed raw error > %g x the lowest value before it)"
          % (out / "table.csv", BLOWUP_FACTOR))


def _plot_sweep(out: Path, steps: int, batch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for i, long_share in enumerate(SWEEP_GRID["long_share"]):
        for j, p in enumerate(SWEEP_GRID["adaptive_p"]):
            ax = axes[i, j]
            for lr in SWEEP_GRID["lr"]:
                for s_min in SWEEP_GRID["s_min"]:
                    r = Recipe(lr=lr, s_min=s_min, long_share=long_share, adaptive_p=p)
                    path = out / ("meanflow_s%d_b%d%s.json" % (steps, batch,
                                                               r.tag("meanflow") or "_default"))
                    if not path.exists():
                        continue
                    log = json.loads(path.read_text())["log"]
                    ax.plot([e["step"] for e in log], _smooth([e["raw_mse"] for e in log]),
                            label="lr %g, s_min %g" % (lr, s_min),
                            ls="-" if s_min == SWEEP_GRID["s_min"][0] else "--")
            ax.set_yscale("log")
            ax.set_title("long share %g, adaptive p %g" % (long_share, p), fontsize=9)
            ax.axvline(steps / 3, color="0.6", lw=0.8)       # curriculum fully on
            if i == 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("raw MeanFlow error (running median)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)


def _sweep_path(out: Path, steps: int, batch: int, recipe: Recipe) -> Path:
    return out / ("meanflow_s%d_b%d%s.json" % (steps, batch, recipe.tag("meanflow") or "_default"))


def floor(args) -> Dict:
    """The raw-error floor (meanflow_error_floor) on the sweep's own batches at
    --long-share, next to the sweep's curves for every lr / s_min at that share and
    --adaptive-p.  'final' is the mean over the last SWEEP_SMOOTH logged batches, as in the
    sweep table."""
    out = OUT / "sweep"
    final_steps = [args.steps - SWEEP_LOG_EVERY * k for k in range(SWEEP_SMOOTH)][::-1]
    curve_steps = sorted(set([1] + list(range(500, args.steps + 1, 500)) + final_steps))
    result = dict(long_share=args.long_share, adaptive_p=args.adaptive_p, steps=args.steps,
                  batch=args.batch, rk4_steps=args.rk4_steps, curves={})
    started = time.perf_counter()
    for s_min in SWEEP_GRID["s_min"]:
        mix = make_mixture(s_min=s_min)
        batches = replay_meanflow_batches(mix, args.steps, args.batch, args.long_share,
                                          args.device, curve_steps)
        curve = []
        for step in curve_steps:
            x1, x0, t, s = batches[step]
            z = (1 - t)[:, None] * x0 + t[:, None] * x1
            err = meanflow_error_floor(mix, z, x1 - x0, t, s, steps=args.rk4_steps)
            long = s != t
            curve.append(dict(step=step, floor=float(err.mean()),
                              floor_long=float(err[long].mean()) if long.any() else None,
                              floor_short=float(err[~long].mean()), long_pairs=int(long.sum())))
        result["curves"][str(s_min)] = curve
        print("  s_min %g: floor on %d batches (%.0f s so far)"
              % (s_min, len(curve), time.perf_counter() - started))

    rows = []
    print("\n  long share %g, adaptive p %g: raw MeanFlow error of the exact average velocity "
          "on the same batches" % (args.long_share, args.adaptive_p))
    print("  %-6s %-8s %12s %12s %8s %14s %14s" % ("s_min", "lr", "sweep final", "floor final",
                                                 "ratio", "sweep peak", "floor max"))
    for s_min in SWEEP_GRID["s_min"]:
        curve = result["curves"][str(s_min)]
        fl_final = float(np.mean([c["floor"] for c in curve if c["step"] in final_steps]))
        fl_max = max(curve, key=lambda c: c["floor"])
        for lr in SWEEP_GRID["lr"]:
            recipe = Recipe(lr=lr, s_min=s_min, long_share=args.long_share,
                            adaptive_p=args.adaptive_p)
            path = _sweep_path(out, args.steps, args.batch, recipe)
            summary = summarize_curve(json.loads(path.read_text())["log"])
            rows.append(dict(s_min=s_min, lr=lr, sweep_final=summary["final_raw"],
                             floor_final=fl_final, sweep_peak=summary["peak_raw"],
                             sweep_peak_step=summary["peak_step"], floor_max=fl_max["floor"],
                             floor_max_step=fl_max["step"]))
            print("  %-6g %-8g %12.1f %12.1f %8.2f %8.1f @%5d %8.1f @%5d"
                  % (s_min, lr, summary["final_raw"], fl_final, summary["final_raw"] / fl_final,
                     summary["peak_raw"], summary["peak_step"], fl_max["floor"], fl_max["step"]))
    result["rows"] = rows
    name = "floor_long%g_p%g" % (args.long_share, args.adaptive_p)
    (out / (name + ".json")).write_text(json.dumps(result, indent=1))
    _plot_floor(result, out, name + ".png")
    print("wrote %s (.json, .png)" % (out / name))
    return result


def _plot_floor(result: Dict, out: Path, filename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, s_min in zip(axes, SWEEP_GRID["s_min"]):
        for lr in SWEEP_GRID["lr"]:
            recipe = Recipe(lr=lr, s_min=s_min, long_share=result["long_share"],
                            adaptive_p=result["adaptive_p"])
            log = json.loads(_sweep_path(out, result["steps"], result["batch"], recipe)
                             .read_text())["log"]
            ax.plot([e["step"] for e in log], _smooth([e["raw_mse"] for e in log]),
                    label="trained, lr %g" % lr)
        curve = result["curves"][str(s_min)]
        ax.plot([c["step"] for c in curve], [c["floor"] for c in curve], "k.-",
                label="exact average velocity (floor)")
        ax.set_yscale("log")
        ax.set_title("long share %g, p %g, s_min %g" % (result["long_share"],
                                                        result["adaptive_p"], s_min), fontsize=9)
        ax.set_xlabel("step")
        ax.axvline(result["steps"] / 3, color="0.6", lw=0.8)
    axes[0].set_ylabel("raw MeanFlow error")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=120)
    plt.close(fig)


def steptime(args) -> Dict:
    """Seconds per training step of both priors on --device, measured after a warm-up."""
    torch.set_num_threads(args.threads or torch.get_num_threads())
    mix = make_mixture()
    result = {}
    for kind in ("flow_matching", "meanflow"):
        _, info = train_prior(kind, mix, args.steps, args.batch, Recipe(), args.device,
                              verbose=False, log_every=args.steps // 4)
        log = info["log"]                                  # steps 1, n/4, n/2, 3n/4, n
        a, b = log[1], log[-1]
        ms = 1000.0 * (b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
        result[kind] = ms
        print("  %-14s %.2f ms/step  (batch %d, steps %d-%d, %s)"
              % (kind, ms, args.batch, a["step"], b["step"], info["device"]))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, steps):
        q.add_argument("--device", default=default_device(),
                       help="cuda when available, else cpu (default: %(default)s)")
        q.add_argument("--steps", type=int, default=steps)
        q.add_argument("--batch", type=int, default=TRAIN_BATCH)
        q.add_argument("--threads", type=int, default=0)

    q = sub.add_parser("stage1", help="train both priors (cached) and run the sanity checks")
    common(q, TRAIN_STEPS)
    q.add_argument("--retrain", action="store_true")
    q.add_argument("--samples", type=int, default=20_000)
    q.add_argument("--ode-steps", type=int, default=1000)
    q.add_argument("--lr", type=float, default=Recipe.lr, help="both priors")
    q.add_argument("--s-min", type=float, default=Recipe.s_min,
                   help="thinnest component std before scaling (both priors)")
    q.add_argument("--long-share", type=float, default=Recipe.long_share, help="MeanFlow only")
    q.add_argument("--adaptive-p", type=float, default=Recipe.adaptive_p, help="MeanFlow only")
    q.set_defaults(run=stage1)
    q = sub.add_parser("sweep", help="MeanFlow stability sweep over %s" % ", ".join(SWEEP_GRID))
    common(q, SWEEP_STEPS)
    q.set_defaults(run=sweep)
    q = sub.add_parser("floor", help="raw-error floor of the exact average velocity on the "
                       "sweep's batches")
    common(q, SWEEP_STEPS)
    q.add_argument("--long-share", type=float, default=0.25)
    q.add_argument("--adaptive-p", type=float, default=1.0)
    q.add_argument("--rk4-steps", type=int, default=400)
    q.set_defaults(run=floor)
    q = sub.add_parser("steptime", help="time per training step of both priors")
    common(q, 400)
    q.set_defaults(run=steptime)
    a = p.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
