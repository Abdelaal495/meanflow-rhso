#!/usr/bin/env python
"""Hexagon study, stage 2 (rebuttal Task 4): the six solvers on the learned hexagon priors.

    python -u toy/hexagon_stage2.py                 # everything, on the GPU, with a log
    python -u toy/hexagon_stage2.py --smoke         # tiny grids and seed counts
    python -u toy/hexagon_stage2.py --tasks corner off_edge --priors imeanflow

Priors (low-frequency time features, toy/hexagon_study.py --features low; same MLP and
recipe): flow matching; iMF (the primary MeanFlow prior); MeanFlow p = 0.5 (a second one).
Solvers (toy/gmm_solvers.py, ports of src/ with the image code's cost accounting): RHSO
(N * M = 160), D-Flow (one step), MPC-RHC, MPC-Delta t, PnP-Flow, Flower, on the
pre-registered pMF grids (gmm_solvers.solver_specs); for flow matching also RHSO with the
integrated Heun suffix, as in the image code.  `evals` counts model evaluations (one
MeanFlow transition = one evaluation), `backprops` the evaluations the backward pass
traverses.

Control task: Phi(x) = 1/2 ||x - c||^2, i.e. H = I, y = c.  The solvers keep the image
code's fidelity normalisations: RHSO, D-Flow and PnP use the per-measurement form
(1 / 2m) ||x - c||^2 with m = 2; MPC uses the half sum 1/2 ||x - c||^2.  Flower has no
weight on Phi in the image code (sigma is the known noise level there), so in the control
task its sigma is tuned on {1, 0.3, 0.1, 0.03, 0.01}, a weight 1 / sigma^2 on Phi like MPC's
lambda.  Targets:
    corner     c = the lower-right corner (1, -sqrt 3); p* = c
    off_edge   c = (0, -sqrt 3 - OFF_EDGE), OFF_EDGE below the midpoint of the lower edge;
               p* = (0, -sqrt 3), the hexagon point nearest c
Tuning: every grid point on 64 noise seeds, selection by the mean distance to p*; test: 512
fresh seeds.  Metrics: distance to c, to the hexagon and to p* (mean and standard error).

Inverse problem: y = x*_1 + sigma eps (the x-coordinate), sigma in {0.1, 0.3}, x* a data
sample.  The exact posterior along the hexagon: with x = b(s) + j, b(s) the boundary point
at arc length s ~ U[0, 12) and j ~ N(0, 0.02^2 I), p(s | y) is proportional to
N(y; b_1(s), sigma^2 + 0.02^2), and x | s, y is Gaussian (x_1 shrunk toward y, x_2 = b_2(s)
+ jitter); the mean and the samples come from a midpoint rule in s with QUAD_CELLS cells.
Tuning: 64 problems x 4 seeds, selection by MSE to x*; test: 512 problems x 16 seeds.
Metrics: MSE to x* (||x - x*||^2 / 2), MSE to the posterior mean, sliced W2 between each
problem's 16 outputs and 16 exact posterior samples (the 'posterior samples' row is its
floor), distance to the hexagon.

Outputs in outputs/hexagon_study/stage2/: control.csv, inverse.csv, tuning.csv, log.txt,
trajectories_corner.png, trajectories_off_edge.png (16 test seeds per panel).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

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

APOTHEM = math.sqrt(3.0)
OFF_EDGE = 0.5
TARGETS = {"corner": (np.array([1.0, -APOTHEM]), np.array([1.0, -APOTHEM])),
           "off_edge": (np.array([0.0, -APOTHEM - OFF_EDGE]), np.array([0.0, -APOTHEM]))}
SIGMAS = (0.1, 0.3)
FLOWER_CONTROL_SIGMA = [1.0, 0.3, 0.1, 0.03, 0.01]

TUNE_SEED, TEST_SEED = 20261120, 20261121
CONTROL_TUNE, CONTROL_TEST = 64, 512
INV_TUNE, INV_TEST = (64, 4), (512, 16)
QUAD_CELLS = 24_000
TRACE_SEEDS = 16


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
    """Rows for P problems x S seeds sharing the operator Hm (m, 2); Y (P, m); X0 (P, S, 2)."""
    P, Sd, d = X0.shape
    m = Hm.shape[0]
    Pi, Si = np.meshgrid(np.arange(P), np.arange(Sd), indexing="ij")
    Pi, Si = Pi.ravel(), Si.ravel()
    R = len(Pi)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    li = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
    return S.Rows(H=f(np.broadcast_to(Hm, (R, m, d)).copy()), y=f(Y[Pi]),
                  mask=f(np.ones((R, m))), m=f(np.full(R, m)), sigma=f(np.full(R, sigma)),
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

    def score(self, x: np.ndarray, rows: S.Rows = None) -> np.ndarray:
        """Per-row selection score: the distance to p*."""
        return np.linalg.norm(x - self.p_star, axis=1)

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

    def __init__(self, sigma: float, n_prob: int, n_seed: int, set_seed: int, device,
                 posteriors: bool = True):
        self.name, self.sigma = "x_coordinate_s%g" % sigma, sigma
        rng = np.random.default_rng([set_seed, int(round(1000 * sigma))])
        self.x_star = H.sample(n_prob, rng)
        self.y = self.x_star[:, :1] + sigma * rng.standard_normal((n_prob, 1))
        self.X0 = rng.standard_normal((n_prob, n_seed, 2))
        self.n_prob, self.n_seed = n_prob, n_seed
        self.rows = make_rows(np.array([[1.0, 0.0]]), self.y, sigma, self.X0, set_seed, device)
        self.post = [Posterior(float(v), sigma) for v in self.y[:, 0]] if posteriors else None

    def score(self, x: np.ndarray, rows: S.Rows) -> np.ndarray:
        """Per-row selection score: the squared error to x*, per coordinate."""
        return ((x - self.x_star[rows.prob_idx.cpu().numpy()]) ** 2).mean(1)

    def metrics(self, X: np.ndarray, post_samples) -> Dict:
        """X: (P, S, 2)."""
        P = X.shape[0]
        per_prob = ((X - self.x_star[:, None]) ** 2).mean(2).mean(1)
        pm = np.stack([p.mean() for p in self.post])[:, None]
        sw = [H.sliced_w2(X[p], post_samples[p], n_proj=256, seed=p) for p in range(P)]
        dh = H.distance_to_hexagon(X.reshape(-1, 2))
        return dict(mse_x=float(per_prob.mean()), mse_x_se=float(per_prob.std(ddof=1) / math.sqrt(P)),
                    mse_post_mean=float(((X - pm) ** 2).mean()), sw2=float(np.mean(sw)),
                    dist_hex=float(dh.mean()), finite=bool(np.isfinite(X).all()))


# =====================================================================================
# Solvers
# =====================================================================================
def _flower_weighted(prior, rows: S.Rows, num_flower_steps: int, sigma: torch.Tensor,
                     trace=None):
    """Flower with a tuned sigma (the control task's weight 1 / sigma^2 on Phi)."""
    return S.flower(prior, dataclasses.replace(rows, sigma=sigma), num_flower_steps, trace=trace)


def specs_for(task) -> List[S.SolverSpec]:
    specs = {s.name: s for s in S.solver_specs()}
    if task.kind == "control":
        f = specs["flower"]
        specs["flower"] = S.SolverSpec("flower", _flower_weighted, f.structural,
                                       dict(sigma=FLOWER_CONTROL_SIGMA), label=f.label)
    return [specs[n] for n in SOLVERS]


def _points(grid: Dict[str, list], smoke: bool, take: int):
    keys = list(grid)
    values = [grid[k][:take] if smoke else grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [dict()]


def run_point(spec, prior, rows: S.Rows, structural: Dict, cont_points: List[Dict], trace=None):
    """All continuous points of one structural setting in one batch (one row block each)."""
    B = len(rows)
    big = rows.repeat(len(cont_points))
    cont = {k: torch.as_tensor(np.repeat([p[k] for p in cont_points], B), dtype=torch.float32,
                               device=rows.x0.device) for k in cont_points[0]}
    x, cost = S.call(spec, prior, big, structural, cont, trace=trace)
    return x.detach().double().cpu().numpy(), cost


def _jsonable(d: Dict) -> Dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def tune(spec, prior, task, smoke: bool, log) -> Dict:
    rows = task.rows
    B = len(rows)
    cont_points = _points(spec.continuous, smoke, 2)
    records = []
    for st in _points(spec.structural, smoke, 1):
        started = time.perf_counter()
        x, cost = run_point(spec, prior, rows, st, cont_points)
        for ci, cp in enumerate(cont_points):
            xb = x[ci * B:(ci + 1) * B]
            finite = bool(np.isfinite(xb).all())
            score = float(task.score(xb, rows).mean()) if finite else math.inf
            records.append(dict(structural=_jsonable(st), continuous=cp, score=score,
                                finite=finite, evals=cost.evals, backprops=cost.backprops))
        log("    %-12s %-30s %3d settings x %4d rows  %5.1f s" % (
            spec.name, json.dumps(_jsonable(st)), len(cont_points), B, time.perf_counter() - started))
    best = min(records, key=lambda r: r["score"])
    edges = []
    for k, grid in list(spec.structural.items()) + list(spec.continuous.items()):
        grid = grid[:1] if (smoke and k in spec.structural) else (grid[:2] if smoke else grid)
        if len(grid) < 2:
            continue
        v = best["structural"].get(k, best["continuous"].get(k))
        v = tuple(v) if isinstance(v, list) else v
        if v == grid[0] or v == grid[-1]:
            edges.append("%s=%s" % (k, "min" if v == grid[0] else "max"))
    return dict(best=best, edges=edges, records=records)


def run_selected(spec, prior, rows: S.Rows, best: Dict, trace=None):
    st = {k: (tuple(v) if isinstance(v, list) else v) for k, v in best["structural"].items()}
    return run_point(spec, prior, rows, st, [best["continuous"]], trace=trace)


# =====================================================================================
# The study
# =====================================================================================
CONTROL_COLS = ["task", "prior", "solver", "selected", "grid_edges", "evals", "backprops",
                "data_grads", "linear_solves", "dist_c", "dist_c_se", "dist_hex", "dist_hex_se",
                "dist_pstar", "dist_pstar_se", "dist_pstar_median", "tuning_dist_pstar"]
INVERSE_COLS = ["task", "sigma", "prior", "solver", "selected", "grid_edges", "evals",
                "backprops", "data_grads", "linear_solves", "mse_x", "mse_x_se",
                "mse_post_mean", "sw2", "dist_hex", "tuning_mse_x"]


def _selected_str(best: Dict) -> str:
    return json.dumps(dict(**best["structural"], **best["continuous"]))


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "smoke" if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w", encoding="utf-8")
    started = time.perf_counter()

    def log(msg):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    H.require_cuda(args.device)
    device = args.device
    priors = load_priors(device, args.priors)
    ct, cx = (8, 16) if args.smoke else (CONTROL_TUNE, CONTROL_TEST)
    it, ix = ((4, 2), (8, 4)) if args.smoke else (INV_TUNE, INV_TEST)
    tasks = []
    for name in args.tasks:
        if name in TARGETS:
            tasks.append((ControlTask(name, ct, TUNE_SEED, device),
                          ControlTask(name, cx, TEST_SEED, device)))
        else:
            sigma = float(name.split("_s")[-1])
            tasks.append((InverseTask(sigma, *it, TUNE_SEED, device, posteriors=False),
                          InverseTask(sigma, *ix, TEST_SEED, device)))
    log("hexagon stage 2 on %s: priors %s, tasks %s%s" % (
        device, ", ".join(priors), ", ".join(t.name for t, _ in tasks), " (smoke)" if args.smoke else ""))

    control_rows, inverse_rows, tuning_rows, traces = [], [], [], {}
    for tune_task, test_task in tasks:
        log("task %s" % test_task.name)
        if test_task.kind == "inverse":
            prng = np.random.default_rng([TEST_SEED, 7, int(round(1000 * test_task.sigma))])
            post_a = [p.sample(test_task.n_seed, prng) for p in test_task.post]
            post_b = np.stack([p.sample(test_task.n_seed, prng) for p in test_task.post])
            pm = np.stack([p.mean() for p in test_task.post])
            for label, X in (("posterior mean", np.repeat(pm[:, None], test_task.n_seed, 1)),
                             ("posterior samples", post_b)):
                m = test_task.metrics(X, post_a)
                inverse_rows.append(dict(task=test_task.name, sigma=test_task.sigma, prior="exact",
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
                control_rows.append(dict(task=test_task.name, prior=PRIORS[pname],
                                         solver="no control (%s)" % ("1 step" if prior.family == "meanflow" else "Heun 100"),
                                         **{k: v for k, v in m.items() if k != "finite"}))
            for spec in specs_for(test_task):
                if prior.family not in spec.families:
                    continue
                t0 = time.perf_counter()
                tuned = tune(spec, prior, tune_task, args.smoke, log)
                best = tuned["best"]
                for r in tuned["records"]:
                    tuning_rows.append(dict(task=test_task.name, prior=pname, solver=spec.name,
                                            setting=_selected_str(r), score=r["score"],
                                            finite=r["finite"], evals=r["evals"], backprops=r["backprops"]))
                if not math.isfinite(best["score"]):
                    log("    %s: no finite setting" % spec.name)
                    continue
                x, cost = run_selected(spec, prior, test_task.rows, best)
                common = dict(task=test_task.name, prior=PRIORS[pname], solver=spec.label,
                              selected=_selected_str(best), grid_edges=" ".join(tuned["edges"]),
                              evals=cost.evals, backprops=cost.backprops,
                              data_grads=cost.data_grads, linear_solves=cost.linear_solves)
                if test_task.kind == "control":
                    m = test_task.metrics(x)
                    control_rows.append(dict(common, tuning_dist_pstar=best["score"],
                                             **{k: v for k, v in m.items() if k != "finite"}))
                    if spec.name in FIGURE_SOLVERS:
                        sub = test_task.rows.select(torch.arange(min(TRACE_SEEDS, len(test_task.rows)),
                                                                 device=test_task.rows.x0.device))
                        tr = []
                        run_selected(spec, prior, sub, best, trace=tr)
                        traces[(test_task.name, pname, spec.name)] = np.stack(
                            [t.double().cpu().numpy() for t in tr], 1)
                    log("    -> %-12s selected %s%s; dist to p* %.4f, to c %.4f, to hexagon %.4f "
                        "(evals %d, backprops %d, %.0f s)" % (
                            spec.name, _selected_str(best), " [edge: %s]" % " ".join(tuned["edges"]) if tuned["edges"] else "",
                            m["dist_pstar"], m["dist_c"], m["dist_hex"], cost.evals, cost.backprops,
                            time.perf_counter() - t0))
                else:
                    X = x.reshape(test_task.n_prob, test_task.n_seed, 2)
                    m = test_task.metrics(X, post_a)
                    inverse_rows.append(dict(common, sigma=test_task.sigma, tuning_mse_x=best["score"],
                                             **{k: v for k, v in m.items() if k != "finite"}))
                    log("    -> %-12s selected %s%s; MSE to x* %.4f, SW2 %.4f, to hexagon %.4f "
                        "(evals %d, backprops %d, %.0f s)" % (
                            spec.name, _selected_str(best), " [edge: %s]" % " ".join(tuned["edges"]) if tuned["edges"] else "",
                            m["mse_x"], m["sw2"], m["dist_hex"], cost.evals, cost.backprops,
                            time.perf_counter() - t0))

    for name, cols, rows in (("control.csv", CONTROL_COLS, control_rows),
                             ("inverse.csv", INVERSE_COLS, inverse_rows),
                             ("tuning.csv", ["task", "prior", "solver", "setting", "score", "finite",
                                             "evals", "backprops"], tuning_rows)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    for target in TARGETS:
        if any(k[0] == target for k in traces):
            plot_trajectories(out, target, traces, list(priors))
    log("done in %.0f s; wrote %s" % (time.perf_counter() - started, out))
    print_tables(control_rows, inverse_rows)
    logf.close()


def print_tables(control_rows, inverse_rows):
    for task in TARGETS:
        rows = [r for r in control_rows if r["task"] == task]
        if not rows:
            continue
        c, p = TARGETS[task]
        print("\ncontrol, target %s: c = (%.3f, %.3f), p* = (%.3f, %.3f); 512 test seeds, mean (se)"
              % (task, c[0], c[1], p[0], p[1]))
        print("  %-15s %-22s %17s %17s %17s %6s %6s  %s" % ("prior", "solver", "dist to p*", "dist to c",
                                                          "dist to hexagon", "evals", "bprop", "selected"))
        for r in rows:
            print("  %-15s %-22s %8.4f (%6.4f) %8.4f (%6.4f) %8.4f (%6.4f) %6s %6s  %s%s" % (
                r["prior"], r["solver"], r["dist_pstar"], r["dist_pstar_se"], r["dist_c"], r["dist_c_se"],
                r["dist_hex"], r["dist_hex_se"], r.get("evals", ""), r.get("backprops", ""),
                r.get("selected", ""), "  EDGE " + r["grid_edges"] if r.get("grid_edges") else ""))
    for sigma in SIGMAS:
        rows = [r for r in inverse_rows if r["sigma"] == sigma]
        if not rows:
            continue
        print("\ninverse problem, y = x_1 + %g eps; 512 problems x 16 seeds" % sigma)
        print("  %-15s %-22s %17s %9s %8s %9s %6s %6s  %s" % ("prior", "solver", "MSE to x* (se)", "MSE pm",
                                                             "SW2", "d hexagon", "evals", "bprop", "selected"))
        for r in rows:
            print("  %-15s %-22s %8.4f (%6.4f) %9.4f %8.4f %9.4f %6s %6s  %s%s" % (
                r["prior"], r["solver"], r["mse_x"], r["mse_x_se"], r["mse_post_mean"], r["sw2"],
                r["dist_hex"], r.get("evals", ""), r.get("backprops", ""), r.get("selected", ""),
                "  EDGE " + r["grid_edges"] if r.get("grid_edges") else ""))


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
                    ax.plot(path[:, 0], path[:, 1], color="C0", lw=0.7, alpha=0.6, zorder=2)
                    ax.scatter(path[0, 0], path[0, 1], s=6, color="0.4", zorder=3)
                    ax.scatter(path[-1, 0], path[-1, 1], s=10, color="C3", zorder=4)
            ax.scatter([c[0]], [c[1]], marker="*", s=90, color="k", zorder=5)
            ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(labels[sname].replace("$\\Delta t$", "Δt"), fontsize=9)
            if j == 0:
                ax.set_ylabel(PRIORS[pname], fontsize=9)
    fig.suptitle("target %s (star): 16 test seeds, start grey, end red" % target, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / ("trajectories_%s.png" % target), dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--priors", nargs="+", default=list(PRIORS), choices=list(PRIORS))
    p.add_argument("--tasks", nargs="+", default=list(TARGETS) + ["x_coordinate_s%g" % s for s in SIGMAS])
    p.add_argument("--smoke", action="store_true")
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
