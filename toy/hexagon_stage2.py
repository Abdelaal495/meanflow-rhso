#!/usr/bin/env python
"""Hexagon study, stage 2 (rebuttal Task 4): the six solvers on the learned hexagon priors.

    python -u toy/hexagon_stage2.py                 # everything, on the GPU, with a log
    python -u toy/hexagon_stage2.py --smoke         # tiny seed counts, short widening
    python -u toy/hexagon_stage2.py --tasks corner off_edge --priors imeanflow
    python -u toy/hexagon_stage2.py --rebuild-from-log outputs/hexagon_study/stage2/log_run1.txt \
        --tasks corner off_edge x_coordinate_s0.1

Every finished (task, prior, solver) is saved to stage2/results/ as it completes and skipped
on the next run (--fresh recomputes); the CSVs and tables are assembled from all saved
results.  --rebuild-from-log restores finished solvers from a run's log.txt (selected
setting, final grids and statuses, no retuning) and re-evaluates only their test sets; the
per-setting tuning scores are not in the log, so tuning.csv lacks those solvers.

Rows are run in chunks of at most ROW_EVAL_BUDGET / depth rows per call (and at most
MAX_ROWS_PER_CALL), depth = the model evaluations one backward pass holds (RHSO with the Heun
suffix: 2N).  A row-evaluation takes about 5.4 KB, so a call peaks near 1.4 GB, within the
4 GB GPU; the peak memory of every setting is logged, and a setting that runs over a minute
logs its progress.  The per-row computation
and noise do not depend on the chunking, but float32 GPU matrix products round differently
for different batch sizes (outputs differ by about 1e-6 to 1e-4 between chunk sizes).

Priors (low-frequency time features, toy/hexagon_study.py --features low; same MLP and
recipe): flow matching; iMF (the primary MeanFlow prior); MeanFlow p = 0.5 (a second one).
Solvers (toy/gmm_solvers.py, ports of src/ with the image code's cost accounting): RHSO
(N * M = 160), D-Flow (one step), MPC-RHC, MPC-Delta t, PnP-Flow, Flower; for flow matching
also RHSO with the integrated Heun suffix, as in the image code.  `evals` counts model
evaluations (one MeanFlow transition = one evaluation), `backprops` the evaluations the
backward pass traverses.

One objective for every solver.  Every row's fidelity is 1/2 ||Hx - y||^2: the solvers
written with the per-measurement form (1 / 2m) ||Hx - y||^2 (RHSO, D-Flow, PnP) get the
normaliser m = 1 (Rows.m), the MPC solvers use the half sum, and Flower's proximal step
uses a weight w on it (w = 1 / sigma^2).  So in the control task every solver sees
Phi(x) = 1/2 ||x - c||^2; in the inverse problem every solver sees 1/2 (x_1 - y)^2, and
Flower uses the known noise level (w = 1 / sigma^2, not tuned), as in the image code.

Control task: H = I, y = c.  Targets:
    corner     c = the lower-right corner (1, -sqrt 3); p* = c
    off_edge   c = (0, -sqrt 3 - OFF_EDGE), OFF_EDGE below the midpoint of the lower edge;
               p* = (0, -sqrt 3), the hexagon point nearest c
Tuning on 64 noise seeds, selection by the mean distance to p*; test on 512 fresh seeds.
Metrics: distance to c, to the hexagon and to p* (mean and standard error).

Inverse problem: y = x*_1 + sigma eps (the x-coordinate), sigma in {0.1, 0.3}, x* a data
sample.  The exact posterior along the hexagon: with x = b(s) + j, b(s) the boundary point
at arc length s ~ U[0, 12) and j ~ N(0, 0.02^2 I), p(s | y) is proportional to
N(y; b_1(s), sigma^2 + 0.02^2), and x | s, y is Gaussian (x_1 shrunk toward y, x_2 = b_2(s)
+ jitter); the mean and the samples come from a midpoint rule in s with QUAD_CELLS cells.
Tuning on 64 problems x 16 seeds, selection by the mean over problems of the sliced W2
between the 16 outputs and 16 exact posterior samples; test on 512 problems x 16 seeds.
Metrics: MSE to x* (||x - x*||^2 / 2), MSE to the posterior mean, sliced W2 to 16 exact
posterior samples per problem (the 'posterior samples' row is its floor), distance to the
hexagon.

Grids.  Each solver starts from the pre-registered pMF grid (gmm_solvers.solver_specs),
with MPC-Delta t's lambda extended down to 1 and 3, and Flower's control weight on
{1, 10, 100, 1e3, 1e4}.  After tuning, every knob whose winner sits on an edge of its grid
is extended past that edge (EXTEND below) and the solver is retuned (scored settings are
cached), until every winner is interior, sits at a domain bound (mu = 0, K = 1, one step,
N = 1 or 160 for RHSO's N * M = 160), or a knob was extended MAX_EXTENSIONS times
(MAX_BUDGET_EXTENSIONS for step counts, which raise the cost and may have no interior
optimum: more steps rarely hurt).
grids.csv reports each knob's final grid, winner and status.

Outputs in outputs/hexagon_study/stage2/: control.csv, inverse.csv, grids.csv, tuning.csv,
log.txt, trajectories_corner.png, trajectories_off_edge.png (16 test seeds per panel; PnP
and Flower show their denoised estimates only).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_solvers as S                                    # noqa: E402
import hexagon_study as H                                  # noqa: E402

OUT = H.OUT / "stage2"
FEATURES = "low"
PRIORS = {"flow_matching": "flow matching", "imeanflow": "iMF", "meanflow_p0.5": "MeanFlow p=0.5"}
SOLVERS = ("rhso", "rhso_suffix", "dflow", "mpc_rhc", "mpc_delta_t", "pnp", "flower")
FIGURE_SOLVERS = ("rhso", "dflow", "mpc_rhc", "mpc_delta_t", "pnp", "flower")
DENOISED_TRACE = ("pnp", "flower")           # their traces after x0 are denoised estimates

APOTHEM = math.sqrt(3.0)
OFF_EDGE = 0.5
TARGETS = {"corner": (np.array([1.0, -APOTHEM]), np.array([1.0, -APOTHEM])),
           "off_edge": (np.array([0.0, -APOTHEM - OFF_EDGE]), np.array([0.0, -APOTHEM]))}
SIGMAS = (0.1, 0.3)
FLOWER_CONTROL_WEIGHT = [1.0, 10.0, 100.0, 1000.0, 10000.0]
MPC_DT_EXTRA_LAMBDA = [1.0, 3.0]

TUNE_SEED, TEST_SEED = 20261120, 20261121
CONTROL_TUNE, CONTROL_TEST = 64, 512
INV_TUNE, INV_TEST = (64, 16), (512, 16)
QUAD_CELLS = 24_000
TRACE_SEEDS = 16
ROW_EVAL_BUDGET = 262_144     # rows x model evaluations held for backprop, per call (~1.4 GB)
MAX_ROWS_PER_CALL = 32_768
MAX_EXTENSIONS = 6            # learning rates, weights, mu, lambda, gamma0, alpha, K, (N, M)
MAX_BUDGET_EXTENSIONS = 2     # step counts (x4 past the pre-registered grid at most)
RHSO_BUDGET = 160


# =====================================================================================
# The learned priors on the solver interfaces
# =====================================================================================
class FlowPrior:
    family = "flow_matching"

    def __init__(self, model):
        self.model = model

    def velocity(self, x, t):
        return self.model(x, torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype))

    def endpoint(self, x, t):
        return x + (1.0 - float(t)) * self.velocity(x, t)

    def step(self, x, t, s, solver: str = "heun"):
        dt = float(s) - float(t)
        v1 = self.velocity(x, t)
        if solver == "euler":
            return x + dt * v1
        return x + 0.5 * dt * (v1 + self.velocity(x + dt * v1, s))

    @staticmethod
    def step_evals(solver: str = "heun") -> int:
        return 1 if solver == "euler" else 2

    def integrate(self, x, grid, solver: str = "heun"):
        for a, b in zip(grid[:-1], grid[1:]):
            x = self.step(x, a, b, solver)
        return x


class MeanFlowPrior:
    family = "meanflow"

    def __init__(self, model):
        self.model = model

    def transition(self, x, t, s):
        t, s = float(t), float(s)
        if s == t:
            return x
        full = lambda v: torch.full((x.shape[0],), v, device=x.device, dtype=x.dtype)
        return x + (s - t) * self.model(x, full(t), full(s))


def load_priors(device, names) -> Dict[str, object]:
    out = {}
    for name in names:
        kind, p = H.MODELS[name]
        model, _ = H.load_or_train(kind, H.TRAIN_STEPS, H.TRAIN_BATCH, device, p=p,
                                   features=FEATURES)
        out[name] = FlowPrior(model) if kind == "flow_matching" else MeanFlowPrior(model)
    return out


# =====================================================================================
# Tasks
# =====================================================================================
def make_rows(Hm: np.ndarray, Y: np.ndarray, sigma: float, X0: np.ndarray, set_seed: int,
              device) -> S.Rows:
    """Rows for P problems x S seeds (problem-major) sharing the operator Hm (m, 2); Y (P, m);
    X0 (P, S, 2).  Rows.m = 1: every per-measurement fidelity is the half sum (see above)."""
    P, Sd, d = X0.shape
    m = Hm.shape[0]
    Pi, Si = np.meshgrid(np.arange(P), np.arange(Sd), indexing="ij")
    Pi, Si = Pi.ravel(), Si.ravel()
    R = len(Pi)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    li = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
    return S.Rows(H=f(np.broadcast_to(Hm, (R, m, d)).copy()), y=f(Y[Pi]),
                  mask=f(np.ones((R, m))), m=f(np.ones(R)), sigma=f(np.full(R, sigma)),
                  x0=f(X0[Pi, Si]), type_idx=li(np.zeros(R)), prob_idx=li(Pi), seed_idx=li(Si),
                  noise=S.NoiseSource(set_seed, P, Sd, d, device, torch.float32))


class ControlTask:
    kind = "control"

    def __init__(self, name: str, n_seed: int, set_seed: int, device):
        self.name = name
        self.c, self.p_star = TARGETS[name]
        rng = np.random.default_rng([set_seed, 1 if name == "corner" else 2])
        self.X0 = rng.standard_normal((1, n_seed, 2))
        self.rows = make_rows(np.eye(2), self.c[None, :], 1.0, self.X0, set_seed, device)

    def score(self, x: np.ndarray) -> float:
        """Selection score of one setting's outputs (rows in order): mean distance to p*."""
        return float(np.linalg.norm(x - self.p_star, axis=1).mean())

    def metrics(self, x: np.ndarray) -> Dict:
        dc = np.linalg.norm(x - self.c, axis=1)
        dh = H.distance_to_hexagon(x)
        dp = np.linalg.norm(x - self.p_star, axis=1)
        se = lambda a: float(a.std(ddof=1) / math.sqrt(len(a)))
        return dict(dist_c=float(dc.mean()), dist_c_se=se(dc), dist_hex=float(dh.mean()),
                    dist_hex_se=se(dh), dist_pstar=float(dp.mean()), dist_pstar_se=se(dp),
                    dist_pstar_median=float(np.median(dp)), finite=bool(np.isfinite(x).all()))


class Posterior:
    """The exact posterior of x given y = x_1 + sigma eps under the hexagon data model."""

    def __init__(self, y: float, sigma: float):
        ds = H.PERIMETER / QUAD_CELLS
        self.s = (np.arange(QUAD_CELLS) + 0.5) * ds
        self.ds, self.sigma = ds, sigma
        b = H.boundary_point(self.s)
        var = sigma ** 2 + H.JITTER ** 2
        logw = -0.5 * (y - b[:, 0]) ** 2 / var
        w = np.exp(logw - logw.max())
        self.w = w / w.sum()
        self.k = H.JITTER ** 2 / var                          # shrinkage of x_1 toward y
        self.y = y
        self.cond_mean = np.stack([b[:, 0] + self.k * (y - b[:, 0]), b[:, 1]], 1)
        self.std = np.array([H.JITTER * math.sqrt(1 - self.k), H.JITTER])

    def mean(self) -> np.ndarray:
        return self.w @ self.cond_mean

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        i = rng.choice(len(self.w), size=n, p=self.w)
        s = self.s[i] + (rng.uniform(size=n) - 0.5) * self.ds
        b = H.boundary_point(s)
        mean = np.stack([b[:, 0] + self.k * (self.y - b[:, 0]), b[:, 1]], 1)
        return mean + self.std * rng.standard_normal((n, 2))


class InverseTask:
    kind = "inverse"

    def __init__(self, sigma: float, n_prob: int, n_seed: int, set_seed: int, device):
        self.name, self.sigma = "x_coordinate_s%g" % sigma, sigma
        rng = np.random.default_rng([set_seed, int(round(1000 * sigma))])
        self.x_star = H.sample(n_prob, rng)
        self.y = self.x_star[:, :1] + sigma * rng.standard_normal((n_prob, 1))
        self.X0 = rng.standard_normal((n_prob, n_seed, 2))
        self.n_prob, self.n_seed = n_prob, n_seed
        self.rows = make_rows(np.array([[1.0, 0.0]]), self.y, sigma, self.X0, set_seed, device)
        self.post = [Posterior(float(v), sigma) for v in self.y[:, 0]]
        prng = np.random.default_rng([set_seed, 7, int(round(1000 * sigma))])
        self.ref = [p.sample(n_seed, prng) for p in self.post]           # scored against
        self.ref_b = np.stack([p.sample(n_seed, prng) for p in self.post])  # the floor row

    def score(self, x: np.ndarray) -> float:
        """Selection score of one setting's outputs (rows problem-major): mean over problems
        of the sliced W2 between the n_seed outputs and n_seed exact posterior samples."""
        X = x.reshape(self.n_prob, self.n_seed, 2)
        return float(np.mean([H.sliced_w2(X[p], self.ref[p], n_proj=256, seed=p)
                              for p in range(self.n_prob)]))

    def metrics(self, X: np.ndarray) -> Dict:
        """X: (P, S, 2)."""
        P = X.shape[0]
        per_prob = ((X - self.x_star[:, None]) ** 2).mean(2).mean(1)
        pm = np.stack([p.mean() for p in self.post])[:, None]
        sw = [H.sliced_w2(X[p], self.ref[p], n_proj=256, seed=p) for p in range(P)]
        dh = H.distance_to_hexagon(X.reshape(-1, 2))
        return dict(mse_x=float(per_prob.mean()), mse_x_se=float(per_prob.std(ddof=1) / math.sqrt(P)),
                    mse_post_mean=float(((X - pm) ** 2).mean()), sw2=float(np.mean(sw)),
                    dist_hex=float(dh.mean()), finite=bool(np.isfinite(X).all()))


# =====================================================================================
# Solvers and their grids
# =====================================================================================
def _flower_weighted(prior, rows: S.Rows, num_flower_steps: int, weight: torch.Tensor,
                     trace=None):
    """Flower with a weight w on 1/2 ||Hx - y||^2 (w = 1 / sigma^2), the control task's knob."""
    return S.flower(prior, dataclasses.replace(rows, sigma=weight.rsqrt()), num_flower_steps,
                    trace=trace)


def specs_for(task) -> List[S.SolverSpec]:
    specs = {s.name: s for s in S.solver_specs()}
    m = specs["mpc_delta_t"]
    specs["mpc_delta_t"] = dataclasses.replace(
        m, continuous=dict(m.continuous, lam=sorted(MPC_DT_EXTRA_LAMBDA + m.continuous["lam"])))
    if task.kind == "control":
        f = specs["flower"]
        specs["flower"] = S.SolverSpec("flower", _flower_weighted, f.structural,
                                       dict(weight=FLOWER_CONTROL_WEIGHT), label=f.label)
    return [specs[n] for n in SOLVERS]


RHSO_N = [n for n in range(1, RHSO_BUDGET + 1) if RHSO_BUDGET % n == 0]
LOG3 = ("lr", "lam", "gamma0_image", "weight")
INT2 = ("num_opt_steps", "num_pnp_steps", "num_flower_steps", "num_mpc_steps", "n_ctrl")


def _sig(v: float) -> float:
    return float("%.4g" % v)


def extend(knob: str, grid: list, side: str) -> Optional[object]:
    """The next value past the `side` ('min' or 'max') edge of a knob's grid, or None at a
    domain bound.  Log knobs x3 (alpha x2), mu x3 up (0 is its bound), step counts x2,
    K +1 (bound 1), RHSO's (N, M) along the divisors of N * M = 160."""
    lo, hi = grid[0], grid[-1]
    up = side == "max"
    if knob == "NM":
        n = hi[0] if up else lo[0]
        nxt = [k for k in RHSO_N if (k > n if up else k < n)]
        if not nxt:
            return None
        k = nxt[0] if up else nxt[-1]
        return (k, RHSO_BUDGET // k)
    if knob in LOG3:
        return _sig(hi * 3) if up else _sig(lo / 3)
    if knob == "alpha":
        return _sig(hi * 2) if up else _sig(lo / 2)
    if knob == "mu":
        return _sig(hi * 3) if up else (None if lo == 0 else _sig(lo / 3))
    if knob in INT2:
        return int(hi * 2) if up else (lo // 2 if lo // 2 >= 1 else None)
    if knob == "K":
        return hi + 1 if up else (lo - 1 if lo > 1 else None)
    raise KeyError(knob)


def _key(structural: Dict, continuous: Dict) -> str:
    return json.dumps({**{k: (list(v) if isinstance(v, tuple) else v) for k, v in structural.items()},
                       **continuous}, sort_keys=True)


def graph_depth(spec, structural: Dict) -> int:
    """Model evaluations one backward pass holds per row (1 for the no-grad solvers)."""
    if spec.name == "rhso" and spec.fixed.get("terminal") == "suffix":
        return 2 * int(structural["NM"][0])           # Heun: 2 evaluations per interval
    if spec.name == "rhso_suffix":
        return 2 * int(structural["NM"][0])
    if spec.name == "mpc_rhc":
        return max(1, int(structural["K"]) - 1)
    return 1


def rows_per_call(spec, structural: Dict) -> int:
    return max(32, min(MAX_ROWS_PER_CALL, ROW_EVAL_BUDGET // graph_depth(spec, structural)))


def run_point(spec, prior, rows: S.Rows, structural: Dict, cont_points: List[Dict], trace=None):
    """Continuous points of one structural setting, one row block each, run in chunks of
    rows_per_call rows.  Returns (outputs (len(cont_points) * len(rows), 2), cost, peak MB)."""
    B = len(rows)
    big = rows.repeat(len(cont_points))
    cont = {k: torch.as_tensor(np.repeat([p[k] for p in cont_points], B), dtype=torch.float32,
                               device=rows.x0.device) for k in cont_points[0]}
    chunk = rows_per_call(spec, structural)
    cuda = rows.x0.is_cuda
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    outs, cost, steps = [], None, []
    n_calls = -(-len(big) // chunk)
    last = time.perf_counter()
    for i, a in enumerate(range(0, len(big), chunk)):
        idx = torch.arange(a, min(a + chunk, len(big)), device=rows.x0.device)
        tr = [] if trace is not None else None
        x, cost = S.call(spec, prior, big.select(idx), structural, {k: v[idx] for k, v in cont.items()},
                         trace=tr)
        outs.append(x.detach().double().cpu().numpy())
        if trace is not None:
            steps.append(tr)
        if PROGRESS_LOG is not None and time.perf_counter() - last > 60 and i + 1 < n_calls:
            last = time.perf_counter()
            PROGRESS_LOG("      %-12s %s: call %d of %d done (%d rows/call, peak %.0f MB so far)" % (
                spec.name, _key(structural, {}), i + 1, n_calls, chunk,
                torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0))
    if trace is not None:                                  # stitch the chunks' traces
        for k in range(len(steps[0])):
            trace.append(torch.cat([s[k] for s in steps]))
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0
    run_point.last_peak_mb = peak
    return np.concatenate(outs), cost


run_point.last_peak_mb = 0.0
PROGRESS_LOG = None              # set by run(): a progress line at least once a minute


def tune(spec, prior, task, smoke: bool, log) -> Dict:
    """Tune on the task's rows, widening every edge-hitting knob (see the module docstring)."""
    rows = task.rows
    B = len(rows)
    grids = {k: [tuple(v) if isinstance(v, list) else v for v in g]
             for k, g in list(spec.structural.items()) + list(spec.continuous.items())}
    structural_keys = list(spec.structural)
    extensions = {k: 0 for k in grids}
    cap = lambda k: 2 if smoke else (MAX_BUDGET_EXTENSIONS if k in INT2 else MAX_EXTENSIONS)
    cache: Dict[str, Dict] = {}
    rnd = 0
    while True:
        rnd += 1
        skeys = structural_keys
        ckeys = [k for k in grids if k not in structural_keys]
        new = 0
        for svals in itertools.product(*[grids[k] for k in skeys]):
            st = dict(zip(skeys, svals))
            cps = [dict(zip(ckeys, c)) for c in itertools.product(*[grids[k] for k in ckeys])] or [dict()]
            todo = [cp for cp in cps if _key(st, cp) not in cache]
            if not todo:
                continue
            started = time.perf_counter()
            x, cost = run_point(spec, prior, rows, st, todo)
            for ci, cp in enumerate(todo):
                xb = x[ci * B:(ci + 1) * B]
                finite = bool(np.isfinite(xb).all())
                cache[_key(st, cp)] = dict(structural={k: (list(v) if isinstance(v, tuple) else v)
                                                       for k, v in st.items()},
                                           continuous=cp, score=task.score(xb) if finite else math.inf,
                                           finite=finite, evals=cost.evals, backprops=cost.backprops)
            new += len(todo)
            log("    %-12s round %d %-34s %4d new settings x %5d rows  %5.1f s  (%d rows/call, "
                "peak %.0f MB)" % (spec.name, rnd, _key(st, {}), len(todo), B,
                                   time.perf_counter() - started, rows_per_call(spec, st),
                                   run_point.last_peak_mb))
        current = [cache[_key(dict(zip(skeys, s)), dict(zip(ckeys, c)))]
                   for s in itertools.product(*[grids[k] for k in skeys])
                   for c in (itertools.product(*[grids[k] for k in ckeys]) if ckeys else [()])]
        best = min(current, key=lambda r: r["score"])
        winner = {**{k: (tuple(v) if isinstance(v, list) else v) for k, v in best["structural"].items()},
                  **best["continuous"]}
        grown = False
        status = {}
        for k, g in grids.items():
            if len(g) < 2:
                status[k] = "single value"
                continue
            side = "min" if winner[k] == g[0] else ("max" if winner[k] == g[-1] else None)
            if side is None:
                status[k] = "interior"
                continue
            nxt = extend(k, g, side)
            if nxt is None:
                status[k] = "domain bound (%s)" % side
            elif extensions[k] >= cap(k):
                status[k] = "extension cap (%s)" % side
            else:
                g = sorted(g + [nxt], key=lambda v: v[0] if isinstance(v, tuple) else v)
                grids[k] = g
                extensions[k] += 1
                grown = True
                status[k] = "extended (%s)" % side
        if not grown:
            break
        log("    %-12s widened: %s" % (spec.name, ", ".join(
            "%s -> %s" % (k, json.dumps([list(v) if isinstance(v, tuple) else v for v in grids[k]]))
            for k, s in status.items() if s.startswith("extended"))))
    return dict(best=best, winner=winner, grids=grids, status=status, records=list(cache.values()))


def run_selected(spec, prior, rows: S.Rows, best: Dict, trace=None):
    st = {k: (tuple(v) if isinstance(v, list) else v) for k, v in best["structural"].items()}
    return run_point(spec, prior, rows, st, [best["continuous"]], trace=trace)


# =====================================================================================
# The study
# =====================================================================================
CONTROL_COLS = ["task", "prior", "solver", "selected", "not_interior", "evals", "backprops",
                "data_grads", "linear_solves", "dist_c", "dist_c_se", "dist_hex", "dist_hex_se",
                "dist_pstar", "dist_pstar_se", "dist_pstar_median", "tuning_dist_pstar"]
INVERSE_COLS = ["task", "sigma", "prior", "solver", "selected", "not_interior", "evals",
                "backprops", "data_grads", "linear_solves", "mse_x", "mse_x_se",
                "mse_post_mean", "sw2", "dist_hex", "tuning_sw2"]
GRID_COLS = ["task", "prior", "solver", "knob", "grid", "winner", "status"]


def _selected_str(best: Dict) -> str:
    return json.dumps(dict(**best["structural"], **best["continuous"]))


def _not_interior(status: Dict) -> str:
    return " ".join("%s:%s" % (k, s) for k, s in status.items() if s not in ("interior", "single value"))


def _ckpt_path(results: Path, task: str, prior: str, solver: str) -> Path:
    return results / ("%s__%s__%s.json" % (task, prior, solver))


def _save(path: Path, obj: Dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _status_for(spec, grids: Dict, winner: Dict, extensions: Dict, smoke: bool = False) -> Dict:
    """The status tune() assigns at its last round, from the final grids and extension counts."""
    status = {}
    for k, g in grids.items():
        if len(g) < 2:
            status[k] = "single value"
            continue
        side = "min" if winner[k] == g[0] else ("max" if winner[k] == g[-1] else None)
        if side is None:
            status[k] = "interior"
        elif extend(k, g, side) is None:
            status[k] = "domain bound (%s)" % side
        else:
            status[k] = "extension cap (%s)" % side
    return status


_WIDENED = re.compile(r"(\w+) -> (.+?)(?=, \w+ -> |$)")


def parse_log(path: Path) -> Dict:
    """From a run's log.txt: per (task, prior, solver) the selected setting, the widened grids
    (knob -> final grid, number of extensions) and the logged test metrics."""
    out, task, prior = {}, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line[10:] if len(line) > 10 else ""
        if body.startswith("task "):
            task = body[5:].strip()
        elif body.startswith("  prior "):
            prior = body[8:].strip()
        elif "widened:" in body:
            solver = body.split()[0]
            entry = out.setdefault((task, prior, solver), dict(widened={}, extensions={}))
            for k, g in _WIDENED.findall(body.split("widened: ", 1)[1]):
                entry["widened"][k] = json.loads(g)
                entry["extensions"][k] = entry["extensions"].get(k, 0) + 1
        elif body.lstrip().startswith("-> "):
            rest = body.lstrip()[3:]
            solver = rest.split()[0]
            sel = rest.split("selected ", 1)[1]
            selected = json.JSONDecoder().raw_decode(sel)[0]
            nums = dict(re.findall(r"(dist to p\*|to c|to hexagon|MSE to x\*|SW2) ([0-9.]+)", sel))
            entry = out.setdefault((task, prior, solver), dict(widened={}, extensions={}))
            entry["selected"], entry["logged"] = selected, {k: float(v) for k, v in nums.items()}
    return {k: v for k, v in out.items() if "selected" in v}


def _tuned_from_log(spec, entry: Dict) -> Dict:
    grids = {k: [tuple(v) if isinstance(v, list) else v for v in g]
             for k, g in list(spec.structural.items()) + list(spec.continuous.items())}
    for k, g in entry["widened"].items():
        grids[k] = [tuple(v) if isinstance(v, list) else v for v in g]
    sel = entry["selected"]
    structural = {k: sel[k] for k in spec.structural}
    continuous = {k: sel[k] for k in sel if k not in spec.structural}
    winner = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sel.items()}
    extensions = {k: entry["extensions"].get(k, 0) for k in grids}
    return dict(best=dict(structural=structural, continuous=continuous, score=float("nan")),
                winner=winner, grids=grids, status=_status_for(spec, grids, winner, extensions),
                records=[], from_log=True)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "smoke" if args.smoke else OUT
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")
    started = time.perf_counter()

    def log(msg):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    global PROGRESS_LOG
    PROGRESS_LOG = log
    H.require_cuda(args.device)
    device = args.device
    priors = load_priors(device, args.priors)
    from_log = parse_log(Path(args.rebuild_from_log)) if args.rebuild_from_log else None
    ct, cx = (8, 16) if args.smoke else (CONTROL_TUNE, CONTROL_TEST)
    it, ix = ((4, 4), (8, 4)) if args.smoke else (INV_TUNE, INV_TEST)
    tasks = []
    for name in args.tasks:
        if name in TARGETS:
            tasks.append((ControlTask(name, ct, TUNE_SEED, device),
                          ControlTask(name, cx, TEST_SEED, device)))
        else:
            sigma = float(name.split("_s")[-1])
            tasks.append((InverseTask(sigma, *it, TUNE_SEED, device),
                          InverseTask(sigma, *ix, TEST_SEED, device)))
    log("=" * 60)
    log("hexagon stage 2 on %s: priors %s, tasks %s%s%s; row-eval budget %d per call" % (
        device, ", ".join(priors), ", ".join(t.name for t, _ in tasks), " (smoke)" if args.smoke else "",
        " (rebuilt from %s)" % args.rebuild_from_log if from_log else "", ROW_EVAL_BUDGET))

    for tune_task, test_task in tasks:
        log("task %s" % test_task.name)
        ref_path = _ckpt_path(results, test_task.name, "reference", "reference")
        ref_rows = []
        if test_task.kind == "inverse":
            pm = np.stack([p.mean() for p in test_task.post])
            for label, X in (("posterior mean", np.repeat(pm[:, None], test_task.n_seed, 1)),
                             ("posterior samples", test_task.ref_b)):
                m = test_task.metrics(X)
                ref_rows.append(dict(task=test_task.name, sigma=test_task.sigma, prior="exact",
                                     solver=label, **{k: v for k, v in m.items() if k != "finite"}))
        for pname, prior in priors.items():
            log("  prior %s" % pname)
            if test_task.kind == "control":                  # the uncontrolled sampler
                with torch.no_grad():
                    if prior.family == "meanflow":
                        x = prior.transition(test_task.rows.x0, 0.0, 1.0)
                    else:
                        x = prior.integrate(test_task.rows.x0, S._grid(100), "heun")
                m = test_task.metrics(x.double().cpu().numpy())
                ref_rows.append(dict(task=test_task.name, prior=PRIORS[pname],
                                     solver="no control (%s)" % ("1 step" if prior.family == "meanflow" else "Heun 100"),
                                     **{k: v for k, v in m.items() if k != "finite"}))
            for spec in specs_for(test_task):
                if prior.family not in spec.families:
                    continue
                ckpt = _ckpt_path(results, test_task.name, pname, spec.name)
                if ckpt.exists() and not args.fresh:
                    log("    %-12s saved result found, skipped" % spec.name)
                    continue
                t0 = time.perf_counter()
                if from_log is not None:
                    entry = from_log.get((test_task.name, pname, spec.name))
                    if entry is None:
                        log("    %-12s not finished in the log, left for a full run" % spec.name)
                        continue
                    tuned = _tuned_from_log(spec, entry)
                else:
                    entry = None
                    tuned = tune(spec, prior, tune_task, args.smoke, log)
                best = tuned["best"]
                record = dict(task=test_task.name, prior=pname, solver=spec.name, kind=test_task.kind,
                              from_log=bool(tuned.get("from_log")),
                              tuning=[dict(task=test_task.name, prior=pname, solver=spec.name,
                                           setting=_selected_str(r), score=r["score"], finite=r["finite"],
                                           evals=r["evals"], backprops=r["backprops"])
                                      for r in tuned["records"]],
                              grids=[dict(task=test_task.name, prior=pname, solver=spec.name, knob=k,
                                          grid=json.dumps([list(v) if isinstance(v, tuple) else v for v in g]),
                                          winner=json.dumps(list(tuned["winner"][k]) if isinstance(tuned["winner"][k], tuple)
                                                            else tuned["winner"][k]),
                                          status=tuned["status"][k]) for k, g in tuned["grids"].items()])
                if not tuned.get("from_log") and not math.isfinite(best["score"]):
                    log("    %s: no finite setting" % spec.name)
                    record["row"] = None
                    _save(ckpt, record)
                    continue
                x, cost = run_selected(spec, prior, test_task.rows, best)
                flags = _not_interior(tuned["status"])
                common = dict(task=test_task.name, prior=PRIORS[pname], solver=spec.label,
                              selected=_selected_str(best), not_interior=flags,
                              evals=cost.evals, backprops=cost.backprops,
                              data_grads=cost.data_grads, linear_solves=cost.linear_solves)
                score = None if tuned.get("from_log") else best["score"]
                if test_task.kind == "control":
                    m = test_task.metrics(x)
                    record["row"] = dict(common, tuning_dist_pstar=score,
                                         **{k: v for k, v in m.items() if k != "finite"})
                    shown = (m["dist_pstar"], m["dist_c"], m["dist_hex"])
                    if spec.name in FIGURE_SOLVERS:
                        sub = test_task.rows.select(torch.arange(min(TRACE_SEEDS, len(test_task.rows)),
                                                                 device=test_task.rows.x0.device))
                        tr = []
                        run_selected(spec, prior, sub, best, trace=tr)
                        record["trace"] = np.stack([t.double().cpu().numpy() for t in tr], 1).tolist()
                    msg = "dist to p* %.4f, to c %.4f, to hexagon %.4f" % shown
                    logged = ("dist to p*", "to c", "to hexagon")
                else:
                    X = x.reshape(test_task.n_prob, test_task.n_seed, 2)
                    m = test_task.metrics(X)
                    record["row"] = dict(common, sigma=test_task.sigma, tuning_sw2=score,
                                         **{k: v for k, v in m.items() if k != "finite"})
                    shown = (m["mse_x"], m["sw2"], m["dist_hex"])
                    msg = "MSE to x* %.4f, SW2 %.4f, to hexagon %.4f" % shown
                    logged = ("MSE to x*", "SW2", "to hexagon")
                if entry is not None:                      # agreement with the logged values
                    gaps = [abs(round(v, 4) - entry["logged"][k]) for v, k in zip(shown, logged)]
                    record["log_check_max_gap"] = max(gaps)
                    msg += "; logged run: max gap %.4f" % max(gaps)
                _save(ckpt, record)
                log("    -> %-12s selected %s%s; %s (evals %d, backprops %d, %.0f s)" % (
                    spec.name, _selected_str(best), " [NOT INTERIOR: %s]" % flags if flags else "",
                    msg, cost.evals, cost.backprops, time.perf_counter() - t0))
        _save(ref_path, dict(rows=ref_rows))

    assemble(out, log)
    log("done in %.0f s; wrote %s" % (time.perf_counter() - started, out))
    logf.close()


def assemble(out: Path, log=print) -> None:
    """CSVs, figures and tables from every saved result in out/results."""
    results = out / "results"
    order = list(TARGETS) + ["x_coordinate_s%g" % s for s in SIGMAS]
    solver_order = {s: i for i, s in enumerate(SOLVERS)}
    prior_order = {p: i for i, p in enumerate(PRIORS)}
    recs = [json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    refs = {r_task: rec for rec in recs if "rows" in rec for r_task in {row["task"] for row in rec["rows"]}}
    solved = sorted([r for r in recs if "rows" not in r],
                    key=lambda r: (order.index(r["task"]) if r["task"] in order else 99,
                                   prior_order.get(r["prior"], 9), solver_order.get(r["solver"], 99)))
    control_rows, inverse_rows, grid_rows, tuning_rows, traces = [], [], [], [], {}
    for task in order:
        ref = refs.get(task)
        for row in (ref["rows"] if ref else []):
            (control_rows if task in TARGETS else inverse_rows).append(row)
        for rec in [r for r in solved if r["task"] == task]:
            if rec.get("row"):
                (control_rows if rec["kind"] == "control" else inverse_rows).append(rec["row"])
            grid_rows += rec["grids"]
            tuning_rows += rec["tuning"]
            if rec.get("trace"):
                traces[(task, rec["prior"], rec["solver"])] = np.asarray(rec["trace"])
    # the reference rows interleave with the priors in run order: put each 'no control' row
    # before its prior's solvers
    if control_rows:
        control_rows.sort(key=lambda r: (order.index(r["task"]),
                                         list(PRIORS.values()).index(r["prior"]),
                                         0 if r["solver"].startswith("no control") else 1))
    for name, cols, rows in (("control.csv", CONTROL_COLS, control_rows),
                             ("inverse.csv", INVERSE_COLS, inverse_rows),
                             ("grids.csv", GRID_COLS, grid_rows),
                             ("tuning.csv", ["task", "prior", "solver", "setting", "score", "finite",
                                             "evals", "backprops"], tuning_rows)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    for target in TARGETS:
        if any(k[0] == target for k in traces):
            plot_trajectories(out, target, traces, [p for p in PRIORS if any(k[1] == p for k in traces)])
    gaps = [(r["task"], r["prior"], r["solver"], r["log_check_max_gap"]) for r in solved if "log_check_max_gap" in r]
    if gaps:
        print("\nrebuilt from the log: %d solvers; largest gap between the re-evaluated and the logged "
              "test metrics (4 decimals) %.4f" % (len(gaps), max(g[3] for g in gaps)))
        for g in gaps:
            if g[3] > 0:
                print("  gap %.4f  %s %s %s" % (g[3], g[0], g[1], g[2]))
    print_tables(control_rows, inverse_rows, grid_rows)


def print_tables(control_rows, inverse_rows, grid_rows):
    for task in TARGETS:
        rows = [r for r in control_rows if r["task"] == task]
        if not rows:
            continue
        c, p = TARGETS[task]
        print("\ncontrol, target %s: c = (%.3f, %.3f), p* = (%.3f, %.3f); mean (se)"
              % (task, c[0], c[1], p[0], p[1]))
        print("  %-15s %-22s %17s %17s %17s %6s %6s  %s" % ("prior", "solver", "dist to p*", "dist to c",
                                                          "dist to hexagon", "evals", "bprop", "selected"))
        for r in rows:
            print("  %-15s %-22s %8.4f (%6.4f) %8.4f (%6.4f) %8.4f (%6.4f) %6s %6s  %s%s" % (
                r["prior"], r["solver"], r["dist_pstar"], r["dist_pstar_se"], r["dist_c"], r["dist_c_se"],
                r["dist_hex"], r["dist_hex_se"], r.get("evals", ""), r.get("backprops", ""),
                r.get("selected", ""), "  NOT INTERIOR " + r["not_interior"] if r.get("not_interior") else ""))
    for sigma in SIGMAS:
        rows = [r for r in inverse_rows if r["sigma"] == sigma]
        if not rows:
            continue
        print("\ninverse problem, y = x_1 + %g eps; selection by sliced W2" % sigma)
        print("  %-15s %-22s %17s %9s %8s %9s %6s %6s  %s" % ("prior", "solver", "MSE to x* (se)", "MSE pm",
                                                             "SW2", "d hexagon", "evals", "bprop", "selected"))
        for r in rows:
            print("  %-15s %-22s %8.4f (%6.4f) %9.4f %8.4f %9.4f %6s %6s  %s%s" % (
                r["prior"], r["solver"], r["mse_x"], r["mse_x_se"], r["mse_post_mean"], r["sw2"],
                r["dist_hex"], r.get("evals", ""), r.get("backprops", ""), r.get("selected", ""),
                "  NOT INTERIOR " + r["not_interior"] if r.get("not_interior") else ""))
    bad = [g for g in grid_rows if g["status"] not in ("interior", "single value")]
    print("\nknobs whose winner is not interior: %d of %d" % (
        len(bad), sum(g["status"] != "single value" for g in grid_rows)))
    for g in bad:
        print("  %-18s %-14s %-12s %-15s winner %-10s %-24s grid %s" % (
            g["task"], g["prior"], g["solver"], g["knob"], g["winner"], g["status"], g["grid"]))


def plot_trajectories(out: Path, target: str, traces: Dict, prior_names: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c, p_star = TARGETS[target]
    labels = {s.name: s.label for s in S.solver_specs()}
    fig, axes = plt.subplots(len(prior_names), len(FIGURE_SOLVERS),
                             figsize=(2.6 * len(FIGURE_SOLVERS), 2.7 * len(prior_names)), squeeze=False)
    hexagon = np.vstack([H.VERTICES, H.VERTICES[:1]])
    for i, pname in enumerate(prior_names):
        for j, sname in enumerate(FIGURE_SOLVERS):
            ax = axes[i, j]
            ax.plot(hexagon[:, 0], hexagon[:, 1], color="0.7", lw=1.0, zorder=1)
            tr = traces.get((target, pname, sname))
            if tr is not None:
                for path in tr:
                    if sname in DENOISED_TRACE:          # the denoised estimates only
                        path = path[1:]
                    ax.plot(path[:, 0], path[:, 1], color="C0", lw=0.7, alpha=0.6, zorder=2)
                    ax.scatter(path[0, 0], path[0, 1], s=6, color="0.4", zorder=3)
                    ax.scatter(path[-1, 0], path[-1, 1], s=10, color="C3", zorder=4)
            ax.scatter([c[0]], [c[1]], marker="*", s=90, color="k", zorder=5)
            ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                title = labels[sname].replace("$\\Delta t$", "Δt")
                ax.set_title(title + (" (denoised)" if sname in DENOISED_TRACE else ""), fontsize=9)
            if j == 0:
                ax.set_ylabel(PRIORS[pname], fontsize=9)
    fig.suptitle("target %s (star): 16 test seeds, start grey, end red; PnP-Flow and Flower "
                 "show their denoised estimate at each step" % target, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / ("trajectories_%s.png" % target), dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--priors", nargs="+", default=list(PRIORS), choices=list(PRIORS))
    p.add_argument("--tasks", nargs="+", default=list(TARGETS) + ["x_coordinate_s%g" % s for s in SIGMAS])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--fresh", action="store_true", help="recompute solvers with a saved result")
    p.add_argument("--rebuild-from-log", default=None,
                   help="restore finished solvers from this log.txt (no retuning)")
    run(p.parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
