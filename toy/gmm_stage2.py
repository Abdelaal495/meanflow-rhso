#!/usr/bin/env python
"""Gaussian-mixture study, stage 2 (rebuttal Task 4): every solver on both prior families,
tuned and scored against the EXACT posterior.

    python toy/gmm_stage2.py --block oracle            # primary study: exact priors
    python toy/gmm_stage2.py --block oracle --smoke    # tiny problem sets and grids

Blocks
------
oracle   flow matching = the exact velocity of the mixture (direct endpoint x + (1 - t) v
         = E[x1 | x_t]; RHSO also with the integrated Heun suffix, as in the image code);
         MeanFlow = the exact finite-interval map Psi(x; t -> s), an RK4 solve with
         gmm_oracle.RK4_STEPS steps.  One evaluation of Psi counts as ONE model evaluation
         (as a learned transition does); its internal velocity evaluations are reported in
         `map_internal_velocity_evals`.
learned  the trained VelocityMLP / MeanFlowMLP (only if a learned MeanFlow passes the
         stage-1 criteria).

Problems (toy/gmm_study.make_operator): denoising sigma in {0.1, 0.2, 0.4}, masking 50%
sigma 0.05, random 8 x 16 projection sigma 0.05.  The same x* are used for every problem
type (paired); y and the initial noise x0 are drawn per (problem, seed) and shared by every
solver and setting (the pairing invariant of the image benchmark).

Protocol
--------
Tuning: 64 problems x 4 seeds, every point of the pre-registered pMF grid
(gmm_solvers.solver_specs), selection per (prior, solver, problem type) by the lowest mean
squared error to x*; settings with any non-finite output are excluded; a selection on a grid
edge is flagged.  Test: 512 different problems x 16 seeds at the selected settings.

Metrics (test set), per (prior, solver, problem type):
    mse_x            ||x_hat - x*||^2 / d                     (mean over seeds, then problems;
                                                               se over problems)
    mse_post_mean    ||x_hat - E[x | y]||^2 / d
    logp_mean/median log p(x_hat | y) under the exact posterior
    meas_rmse        sqrt(mean_j (H x_hat - y)_j^2)
    sw2              sliced W2 between the 16 seeds' outputs and 16 exact posterior samples,
                     per problem (the 'posterior samples' reference row is its floor)
    mse_hi / mse_lo  the error along the 8 largest / 8 smallest eigendirections of the data
                     covariance (per direction, mean); per-direction values in the .json
Reference rows: the exact posterior mean (the MMSE estimator) and exact posterior samples.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_study as G                                      # noqa: E402
import gmm_oracle as O                                     # noqa: E402
import gmm_solvers as S                                    # noqa: E402

PROBLEMS = [("denoising", 0.1), ("denoising", 0.2), ("denoising", 0.4),
            ("masking", 0.05), ("projection", 0.05)]
TUNE_SEED, TEST_SEED = 20261113, 20261114
N_TUNE, S_TUNE, N_TEST, S_TEST = 64, 4, 512, 16
GRAD_CHUNK, NOGRAD_CHUNK = 16384, 32768


def problem_key(name: str, sigma: float) -> str:
    return "%s_s%g" % (name, sigma)


# =====================================================================================
# Problem sets
# =====================================================================================
class ProblemSet:
    """x* (P, d) shared by the problem types; per type H, sigma, y (P, m), the exact
    posterior per problem; x0 (P, S, d); and the torch Rows of every (type, problem, seed)."""

    def __init__(self, mix, set_seed: int, n_prob: int, n_seed: int, device, dtype=torch.float32,
                 problems=PROBLEMS, posteriors: bool = True):
        self.mix, self.n_prob, self.n_seed, self.problems = mix, n_prob, n_seed, problems
        rng = np.random.default_rng(set_seed)
        self.x_star, _ = mix.sample(n_prob, rng)
        self.x0 = rng.standard_normal((n_prob, n_seed, mix.d))
        self.H, self.y, self.post = [], [], []
        d = mix.d
        mmax = max(G.make_operator(name).shape[0] for name, _ in problems)
        parts = []
        noise = S.NoiseSource(set_seed, n_prob, n_seed, d, device, dtype)
        for ti, (name, sigma) in enumerate(problems):
            H = G.make_operator(name)
            m = H.shape[0]
            y = self.x_star @ H.T + sigma * np.random.default_rng([set_seed, ti]).standard_normal((n_prob, m))
            self.H.append(H)
            self.y.append(y)
            if posteriors:
                self.post.append([G.posterior(mix, H, sigma, y[p]) for p in range(n_prob)])
            P, Sd = np.meshgrid(np.arange(n_prob), np.arange(n_seed), indexing="ij")
            P, Sd = P.ravel(), Sd.ravel()
            R = len(P)
            Hp = np.zeros((mmax, d)); Hp[:m] = H
            yp = np.zeros((n_prob, mmax)); yp[:, :m] = y
            mask = np.zeros(mmax); mask[:m] = 1
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
            ti_ = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
            parts.append(S.Rows(H=t(np.broadcast_to(Hp, (R, mmax, d)).copy()), y=t(yp[P]),
                                mask=t(np.broadcast_to(mask, (R, mmax)).copy()),
                                m=t(np.full(R, m)), sigma=t(np.full(R, sigma)),
                                x0=t(self.x0[P, Sd]), type_idx=ti_(np.full(R, ti)),
                                prob_idx=ti_(P), seed_idx=ti_(Sd), noise=noise))
        self.rows_by_type = parts

    def rows(self, types: List[int]) -> S.Rows:
        return S.Rows.cat([self.rows_by_type[t] for t in types])


# =====================================================================================
# Priors
# =====================================================================================
def oracle_priors(mix, device) -> Dict[str, object]:
    tm = O.TorchMixture(mix, device=device, dtype=torch.float32)
    return {"flow_matching": O.FlowOracle(tm), "meanflow": O.MeanFlowOracle(tm, O.RK4_STEPS)}


def internal_evals(prior) -> int:
    return getattr(prior, "internal_velocity_evals", 0) if prior.family == "meanflow" else 0


# =====================================================================================
# Running a solver over many rows
# =====================================================================================
NO_GRAD_SOLVERS = ("pnp", "flower")


def run_chunked(spec: S.SolverSpec, prior, rows: S.Rows, structural: Dict,
                continuous: Dict[str, torch.Tensor], chunk: int):
    outs, cost = [], None
    for a in range(0, len(rows), chunk):
        idx = torch.arange(a, min(a + chunk, len(rows)), device=rows.x0.device)
        part = rows.select(idx)
        cont = {k: v[idx] for k, v in continuous.items()}
        x, c = S.call(spec, prior, part, structural, cont)
        outs.append(x.detach())
        cost = c
    return torch.cat(outs), cost


def _structural_points(spec: S.SolverSpec, smoke: bool):
    keys = list(spec.structural)
    values = [spec.structural[k][:1] if smoke else spec.structural[k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def _continuous_points(spec: S.SolverSpec, smoke: bool):
    keys = list(spec.continuous)
    values = [(spec.continuous[k][:2] if smoke else spec.continuous[k]) for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [dict()]


def _jsonable(d: Dict) -> Dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def tune(spec: S.SolverSpec, prior, pset: ProblemSet, smoke: bool, log) -> List[Dict]:
    """Every grid point on the tuning set; returns one record per (point, problem type)."""
    device = pset.rows_by_type[0].x0.device
    types = list(range(len(pset.problems)))
    base = pset.rows(types)
    B = len(base)
    x_star = torch.as_tensor(pset.x_star, dtype=base.x0.dtype, device=device)
    chunk = NOGRAD_CHUNK if spec.name in NO_GRAD_SOLVERS else GRAD_CHUNK
    records = []
    cont_points = _continuous_points(spec, smoke)
    for st in _structural_points(spec, smoke):
        started = time.perf_counter()
        rows = base.repeat(len(cont_points))
        cont = {k: torch.as_tensor(np.repeat([p[k] for p in cont_points], B),
                                   dtype=base.x0.dtype, device=device)
                for k in (cont_points[0] if cont_points[0] else {})}
        x, cost = run_chunked(spec, prior, rows, st, cont, chunk)
        err = ((x - x_star[rows.prob_idx]) ** 2).mean(1)
        finite = torch.isfinite(x).all(1)
        for ci, cp in enumerate(cont_points):
            block = slice(ci * B, (ci + 1) * B)
            for t in types:
                sel = base.type_idx == t
                e, f = err[block][sel], finite[block][sel]
                records.append(dict(type=t, structural=_jsonable(st), continuous=cp,
                                    mse=float(e.mean()) if bool(f.all()) else math.inf,
                                    finite=bool(f.all()), evals=cost.evals,
                                    backprops=cost.backprops))
        log("    %-12s %-26s %3d settings x %d rows  %.0f s" % (
            spec.name, json.dumps(_jsonable(st)), len(cont_points), B,
            time.perf_counter() - started))
    return records


def select(spec: S.SolverSpec, records: List[Dict], n_types: int, smoke: bool) -> List[Dict]:
    chosen = []
    for t in range(n_types):
        mine = [r for r in records if r["type"] == t]
        best = min(mine, key=lambda r: r["mse"])
        edges = []
        for k, grid in list(spec.structural.items()) + list(spec.continuous.items()):
            grid = grid[:1] if (smoke and k in spec.structural) else (grid[:2] if smoke else grid)
            if len(grid) < 2:
                continue
            v = best["structural"].get(k, best["continuous"].get(k))
            v = tuple(v) if isinstance(v, list) else v
            if v == grid[0] or v == grid[-1]:
                edges.append("%s=%s" % (k, "min" if v == grid[0] else "max"))
        chosen.append(dict(best, edges=edges))
    return chosen


def test(spec: S.SolverSpec, prior, pset: ProblemSet, chosen: List[Dict]) -> Tuple[Dict, Dict]:
    """Outputs (P, S, d) numpy per problem type at its selected setting, and its cost."""
    device = pset.rows_by_type[0].x0.device
    chunk = NOGRAD_CHUNK if spec.name in NO_GRAD_SOLVERS else GRAD_CHUNK
    groups: Dict[str, List[int]] = {}
    for t, c in enumerate(chosen):
        groups.setdefault(json.dumps(c["structural"], sort_keys=True), []).append(t)
    outputs, costs = {}, {}
    for key, types in groups.items():
        st = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(key).items()}
        rows = pset.rows(types)
        cont = {}
        for k in chosen[types[0]]["continuous"]:
            per_type = {t: chosen[t]["continuous"][k] for t in types}
            cont[k] = torch.as_tensor([per_type[int(t)] for t in rows.type_idx.tolist()],
                                      dtype=rows.x0.dtype, device=device)
        x, cost = run_chunked(spec, prior, rows, st, cont, chunk)
        x = x.double().cpu().numpy()
        for t in types:
            sel = (rows.type_idx == t).cpu().numpy()
            outputs[t] = x[sel].reshape(pset.n_prob, pset.n_seed, -1)
            costs[t] = cost
    return outputs, costs


# =====================================================================================
# Metrics against the exact posterior
# =====================================================================================
def data_eigvecs(mix) -> np.ndarray:
    lam, vec = np.linalg.eigh(mix.covariance())
    return vec[:, np.argsort(lam)[::-1]]                    # columns, largest variance first


def metrics(pset: ProblemSet, t: int, X: np.ndarray, U: np.ndarray, post_samples, n_proj=256) -> Dict:
    """X: (P, S, d) outputs for problem type t."""
    P, Sd, d = X.shape
    xs = pset.x_star[:, None, :]
    post = pset.post[t]
    pm = np.stack([p.mean() for p in post])[:, None, :]
    per_prob = ((X - xs) ** 2).mean(2).mean(1)
    H, y = pset.H[t], pset.y[t]
    resid = np.einsum("md,psd->psm", H, X) - y[:, None, :]
    logp = np.concatenate([post[p].log_prob(X[p]) for p in range(P)])
    finite = np.isfinite(X).all()
    proj = np.einsum("psd,de->pse", X - xs, U) ** 2           # per eigendirection
    per_dir = proj.reshape(-1, d).mean(0)
    sw = [G.sliced_w2(X[p], post_samples[p], n_proj=n_proj, seed=p) for p in range(P)]
    half = d // 2
    return dict(mse_x=float(per_prob.mean()), mse_x_se=float(per_prob.std(ddof=1) / math.sqrt(P)),
                mse_post_mean=float(((X - pm) ** 2).mean()),
                logp_mean=float(logp.mean()), logp_median=float(np.median(logp)),
                meas_rmse=float(np.sqrt((resid ** 2).mean(2)).mean()),
                sw2=float(np.mean(sw)), mse_hi=float(per_dir[:half].mean()),
                mse_lo=float(per_dir[half:].mean()), per_direction=[float(v) for v in per_dir],
                finite=bool(finite))


# =====================================================================================
# The study
# =====================================================================================
CSV_COLUMNS = ["block", "prior", "solver", "problem", "sigma", "selected", "grid_edges",
               "evals", "backprops", "data_grads", "linear_solves",
               "map_internal_velocity_evals", "mse_x", "mse_x_se", "mse_post_mean",
               "logp_mean", "logp_median", "meas_rmse", "sw2", "mse_hi", "mse_lo",
               "tuning_mse"]


def run(args):
    started = time.perf_counter()
    out = G.OUT if not args.smoke else G.OUT / "smoke"
    work = out / ("%s_work" % args.block)
    work.mkdir(parents=True, exist_ok=True)
    logf = open(work / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    mix = G.make_mixture()
    device = args.device
    n_tune, s_tune = (4, 2) if args.smoke else (N_TUNE, S_TUNE)
    n_test, s_test = (8, 2) if args.smoke else (N_TEST, S_TEST)
    log("block %s on %s; tuning %d x %d, test %d x %d" % (args.block, device, n_tune, s_tune,
                                                         n_test, s_test))
    tune_set = ProblemSet(mix, TUNE_SEED, n_tune, s_tune, device, posteriors=False)
    test_set = ProblemSet(mix, TEST_SEED, n_test, s_test, device)
    if args.block == "oracle":
        priors = oracle_priors(mix, device)
    else:
        import gmm_learned
        priors = gmm_learned.learned_priors(mix, device)
    if args.priors:
        priors = {k: v for k, v in priors.items() if k in args.priors}
    specs = [s for s in S.solver_specs() if not args.solvers or s.name in args.solvers]
    U = data_eigvecs(mix)
    prng = np.random.default_rng(TEST_SEED + 1)
    post_a = {t: [p.sample(s_test, prng)[0] for p in test_set.post[t]]
              for t in range(len(PROBLEMS))}
    post_b = {t: [p.sample(s_test, prng)[0] for p in test_set.post[t]]
              for t in range(len(PROBLEMS))}

    rows_out, details = [], {}
    for t, (name, sigma) in enumerate(PROBLEMS):          # reference rows
        pm = np.stack([p.mean() for p in test_set.post[t]])
        for label, X, ref in (("posterior mean", np.repeat(pm[:, None], s_test, 1), post_a[t]),
                              ("posterior samples", np.stack(post_b[t]), post_a[t])):
            m = metrics(test_set, t, X, U, ref)
            details["reference|%s|%s" % (label, problem_key(name, sigma))] = m
            rows_out.append(dict(block=args.block, prior="exact", solver=label, problem=name,
                                 sigma=sigma, selected="", grid_edges="", evals=0, backprops=0,
                                 data_grads=0, linear_solves=0, map_internal_velocity_evals=0,
                                 tuning_mse="", **{k: v for k, v in m.items()
                                                   if k not in ("per_direction", "finite")}))

    tuning_rows = []
    for prior_name, prior in priors.items():
        for spec in specs:
            if prior.family not in spec.families:
                continue
            tag = "%s__%s" % (prior_name, spec.name)
            cached = work / (tag + ".json")
            if cached.exists():
                saved = json.loads(cached.read_text())
                log("  %s: reused %s" % (tag, cached.name))
            else:
                log("  %s: tuning" % tag)
                t0 = time.perf_counter()
                records = tune(spec, prior, tune_set, args.smoke, log)
                chosen = select(spec, records, len(PROBLEMS), args.smoke)
                t1 = time.perf_counter()
                outputs, costs = test(spec, prior, test_set, chosen)
                t2 = time.perf_counter()
                result = {}
                for t, (name, sigma) in enumerate(PROBLEMS):
                    m = metrics(test_set, t, outputs[t], U, post_a[t])
                    c = costs[t]
                    result[str(t)] = dict(metrics=m, chosen=chosen[t],
                                          cost=dict(evals=c.evals, backprops=c.backprops,
                                                    data_grads=c.data_grads,
                                                    linear_solves=c.linear_solves))
                saved = dict(records=records, result=result, tune_seconds=t1 - t0,
                             test_seconds=t2 - t1)
                cached.write_text(json.dumps(saved))
                log("  %s: tuned in %.0f s, tested in %.0f s" % (tag, t1 - t0, t2 - t1))
            for t, (name, sigma) in enumerate(PROBLEMS):
                r = saved["result"][str(t)]
                m, ch, c = r["metrics"], r["chosen"], r["cost"]
                details["%s|%s|%s" % (prior_name, spec.name, problem_key(name, sigma))] = m
                rows_out.append(dict(
                    block=args.block, prior=prior_name, solver=spec.name, problem=name,
                    sigma=sigma, selected=json.dumps(dict(ch["structural"], **ch["continuous"])),
                    grid_edges=";".join(ch["edges"]), map_internal_velocity_evals=internal_evals(prior),
                    tuning_mse=ch["mse"], **c,
                    **{k: v for k, v in m.items() if k not in ("per_direction", "finite")}))
                log("    %-14s %-11s %-16s mse %.4f  post-mean %.4f  lo %.4f hi %.4f  "
                    "sw2 %.3f  eval/BP %d/%d  %s" % (
                        prior_name, spec.name, problem_key(name, sigma), m["mse_x"],
                        m["mse_post_mean"], m["mse_lo"], m["mse_hi"], m["sw2"], c["evals"],
                        c["backprops"], ("edges: " + ",".join(ch["edges"])) if ch["edges"] else ""))
            for rec in saved["records"]:
                name, sigma = PROBLEMS[rec["type"]]
                tuning_rows.append(dict(block=args.block, prior=prior_name, solver=spec.name,
                                        problem=name, sigma=sigma,
                                        setting=json.dumps(dict(rec["structural"], **rec["continuous"])),
                                        tuning_mse=rec["mse"], finite=rec["finite"],
                                        evals=rec["evals"], backprops=rec["backprops"]))

    write_outputs(args.block, out, rows_out, tuning_rows, details, priors)
    log("done in %.0f s" % (time.perf_counter() - started))


def write_outputs(block, out: Path, rows, tuning_rows, details, priors):
    note = ("One evaluation of the MeanFlow oracle map Psi(x; t -> s) is one RK4 solve with "
            "%d steps (%d exact-velocity evaluations); it is counted as ONE model evaluation, "
            "as a learned transition is." % (O.RK4_STEPS, 4 * O.RK4_STEPS)) if block == "oracle" else ""
    with open(out / ("%s_results.csv" % block), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    with open(out / ("%s_tuning.csv" % block), "w", newline="") as f:
        cols = ["block", "prior", "solver", "problem", "sigma", "setting", "tuning_mse",
                "finite", "evals", "backprops"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in tuning_rows:
            w.writerow(r)
    (out / ("%s_details.json" % block)).write_text(json.dumps(dict(note=note, metrics=details), indent=1))
    write_latex(block, out, rows, note)
    plot(block, out, rows, details)
    print("wrote %s_{results,tuning}.csv, %s_rows_*.tex, %s_*.png in %s" % (block, block, block, out))
    if note:
        print("note: " + note)


LATEX_METRICS = [("mse_x", "MSE to $x^\\star$"), ("mse_post_mean", "squared distance to the posterior mean"),
                 ("sw2", "sliced $W_2$ to exact posterior samples"),
                 ("mse_lo", "MSE along the 8 lowest-variance data directions"),
                 ("mse_hi", "MSE along the 8 highest-variance data directions")]


def write_latex(block, out: Path, rows, note):
    order = [problem_key(n, s) for n, s in PROBLEMS]
    labels = {s.name: s.label for s in S.solver_specs()}
    labels.update({"posterior mean": "posterior mean (MMSE)", "posterior samples": "posterior samples"})
    prior_label = {"flow_matching": "FM", "meanflow": "MF", "exact": "exact"}
    for key, caption in LATEX_METRICS:
        lines = ["%% %s, block %s. Columns: %s." % (caption, block, ", ".join(order)),
                 "% Rows: prior & solver & Eval./BP per problem (denoising sigma=0.2 setting) & values."]
        if note:
            lines.append("% " + note)
        groups: Dict[Tuple[str, str], Dict[str, Dict]] = {}
        for r in rows:
            groups.setdefault((r["prior"], r["solver"]), {})[problem_key(r["problem"], r["sigma"])] = r
        for (prior, solver), by in groups.items():
            ref = by.get(problem_key("denoising", 0.2), next(iter(by.values())))
            cost = "--" if prior == "exact" else "%d/%d" % (ref["evals"], ref["backprops"])
            vals = []
            for k in order:
                r = by.get(k)
                if r is None:
                    vals.append("--")
                elif key == "mse_x":
                    vals.append("%.4f $\\pm$ %.4f" % (r["mse_x"], r["mse_x_se"]))
                else:
                    vals.append("%.4f" % r[key])
            lines.append("%s & %s & %s & %s \\\\" % (prior_label.get(prior, prior),
                                                    labels.get(solver, solver), cost, " & ".join(vals)))
        (out / ("%s_rows_%s.tex" % (block, key))).write_text("\n".join(lines) + "\n")


def plot(block, out: Path, rows, details):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    priors = [p for p in dict.fromkeys(r["prior"] for r in rows) if p != "exact"]
    solvers = list(dict.fromkeys(r["solver"] for r in rows if r["prior"] != "exact"))
    labels = {s.name: s.label.replace("$\\Delta t$", "Δt") for s in S.solver_specs()}
    sigmas = [s for n, s in PROBLEMS if n == "denoising"]
    fig, axes = plt.subplots(1, len(priors), figsize=(5 * len(priors), 3.8), sharey=True,
                             squeeze=False)
    for ax, prior in zip(axes[0], priors):
        for solver in solvers:
            ys = [next((r["mse_x"] for r in rows if r["prior"] == prior and r["solver"] == solver
                        and r["problem"] == "denoising" and r["sigma"] == s), np.nan) for s in sigmas]
            if not all(np.isnan(ys)):
                ax.plot(sigmas, ys, "o-", label=labels.get(solver, solver))
        ys = [next(r["mse_x"] for r in rows if r["solver"] == "posterior mean"
                   and r["problem"] == "denoising" and r["sigma"] == s) for s in sigmas]
        ax.plot(sigmas, ys, "k--", label="posterior mean (MMSE)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(sigmas); ax.set_xticklabels(sigmas)
        ax.set_xlabel("denoising noise σ"); ax.set_title("%s prior (%s)" % (prior, block), fontsize=9)
    axes[0, 0].set_ylabel("MSE to x* (test set)")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / ("%s_error_vs_sigma.png" % block), dpi=130); plt.close(fig)

    shown = [problem_key("denoising", 0.2), problem_key("projection", 0.05)]
    fig, axes = plt.subplots(len(shown), len(priors), figsize=(5 * len(priors), 3.4 * len(shown)),
                             squeeze=False, sharex=True)
    for i, pk in enumerate(shown):
        for j, prior in enumerate(priors):
            ax = axes[i, j]
            for solver in solvers:
                m = details.get("%s|%s|%s" % (prior, solver, pk))
                if m:
                    ax.plot(range(1, 17), m["per_direction"], ".-", label=labels.get(solver, solver))
            m = details["reference|posterior mean|%s" % pk]
            ax.plot(range(1, 17), m["per_direction"], "k--", label="posterior mean (MMSE)")
            ax.set_yscale("log"); ax.set_title("%s, %s prior" % (pk, prior), fontsize=9)
            ax.axvline(8.5, color="0.7", lw=0.8)
            if i == len(shown) - 1:
                ax.set_xlabel("data-covariance eigendirection (1 = largest variance)")
            if j == 0:
                ax.set_ylabel("MSE along the direction")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / ("%s_per_direction.png" % block), dpi=130); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", choices=["oracle", "learned"], default="oracle")
    p.add_argument("--device", default=G.default_device())
    p.add_argument("--smoke", action="store_true", help="tiny problem sets and grids")
    p.add_argument("--priors", nargs="*", default=None)
    p.add_argument("--solvers", nargs="*", default=None)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
