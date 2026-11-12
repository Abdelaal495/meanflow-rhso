#!/usr/bin/env python
"""Hexagon study (rebuttal Task 4), modelled on the toy of MPC-Flow (Webber et al., 2026,
Appendix C).

    python -u toy/hexagon_study.py train --model flow_matching    # one model per run, cached
    python -u toy/hexagon_study.py train --model meanflow_p0
    python -u toy/hexagon_study.py train --model meanflow_p0.5
    python -u toy/hexagon_study.py train --model imeanflow         # iMF objective (v-loss)
    python -u toy/hexagon_study.py train --model flow_matching --features low   # low-frequency
    python -u toy/hexagon_study.py stage1 --features low           # time features (see below)
    python -u toy/hexagon_study.py stage1                          # sanity table

Training needs CUDA (`--device cuda`, the default) and stops at once when it is missing.  It
prints a progress line every 1,000 steps (step, loss, elapsed time) and writes the same lines
to outputs/hexagon_study/train_<model>.log.

Data: uniform on the boundary of a regular hexagon with side 2, centred at the origin, with
a flat lower edge (vertices 2 (cos k pi/3, sin k pi/3)), plus N(0, 0.02^2 I) jitter.

Priors: a flow-matching model and a MeanFlow model with the SAME MLP (toy/common.py, 2-D,
width 256, depth 4) and the SAME recipe (Adam 3e-4, 500 warm-up steps + cosine, grad clip 1,
EMA 0.999, steps, batch, seed); MeanFlow is trained with the MeanFlow objective.  Stage 1
sweeps the MeanFlow adaptive-weight exponent p in {0, 0.5} (p = 0 is plain squared error).

Sanity metrics (stage 1), each on 20,000 samples with the same noise for every sampler:
    sliced W2 to fresh data, against the two-sample floor;
    the distribution of the distance to the hexagon (mean, median, 90th / 99th percentile,
    share within 0.05 = 2.5 jitter std) of flow-matching Heun-100, MeanFlow 1, 2, 8 steps;
    the jumps T(x_t; t -> 1) from x_t = (1 - t) x0 + t x1 (x1 fresh data, so x_t has the
    true marginal) at t = 0.25, 0.5, 0.75, with flow-matching Heun from the same x_t, and
    the RMS gap between the two endpoints.

Time features (--features): `default` is the MLPs of toy/common.py (features of t, s and
s - t up to 2^7 pi rad per unit time); `low` is MeanFlowMLPLowFreq / VelocityMLPLowFreq
(raw t plus sin/cos up to 2 pi * 2 rad per unit time; MeanFlow conditioned on (t, s - t)).
Each feature set has its own cache files, logs (suffix _tflow) and stage-1 folder.

Time convention (toy/common.py): t = 0 noise, t = 1 data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MeanFlowMLP, MeanFlowMLPLowFreq, VelocityMLP,  # noqa: E402
                    VelocityMLPLowFreq, freeze, train_flow_matching, train_imeanflow,
                    train_meanflow)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "hexagon"
OUT = ROOT / "outputs" / "hexagon_study"

SIDE, JITTER = 2.0, 0.02
VERTICES = np.array([[SIDE * math.cos(k * math.pi / 3), SIDE * math.sin(k * math.pi / 3)]
                     for k in range(6)])
LOWER_RIGHT_CORNER = VERTICES[5]                 # (1, -sqrt 3)
PERIMETER = 6 * SIDE

WIDTH, DEPTH = 256, 4
TRAIN_SEED, TRAIN_STEPS, TRAIN_BATCH, TRAIN_POOL = 0, 30_000, 1024, 1_000_000
SWEEP_P = (0.0, 0.5)
MODELS = {"flow_matching": ("flow_matching", None),
          **{"meanflow_p%g" % p: ("meanflow", p) for p in SWEEP_P},
          "imeanflow": ("imeanflow", None)}
# the one-jump priors in the stage-1 table: model name -> label
MEANFLOW_LABELS = {**{"meanflow_p%g" % p: "MeanFlow p=%g" % p for p in SWEEP_P},
                   "imeanflow": "iMF"}
# pass criteria (those of the Gaussian-mixture study that apply in 2-D; its low-signal
# variance ratio has no counterpart here): one-step sliced W2 at most ONE_STEP_SW2_MAX, and
# 8-step sliced W2 within EIGHT_STEP_RATIO_MAX of flow-matching Heun-100
ONE_STEP_SW2_MAX, EIGHT_STEP_RATIO_MAX = 0.05, 1.5
# time-feature sets: name -> (flow-matching class, MeanFlow class, cache/log suffix,
#                             the one-jump priors of its stage-1 table)
FEATURES = {"default": (VelocityMLP, MeanFlowMLP, "", tuple(MEANFLOW_LABELS)),
            "low": (VelocityMLPLowFreq, MeanFlowMLPLowFreq, "_tflow",
                    ("meanflow_p0.5", "imeanflow"))}
PROGRESS_EVERY = 1000
JUMP_TIMES = (0.25, 0.5, 0.75)
NEAR = 0.05


def require_cuda(device: str) -> None:
    """Stop at once unless the device is CUDA and a forward + backward runs on it."""
    if not str(device).startswith("cuda"):
        raise SystemExit("error: this study runs on the GPU only; pass --device cuda (got %r)" % device)
    if not torch.cuda.is_available():
        raise SystemExit("error: CUDA is not available (torch %s, built for CUDA %s)"
                         % (torch.__version__, torch.version.cuda))
    layer = torch.nn.Linear(2, 8).to(device)
    layer(torch.randn(16, 2, device=device)).square().mean().backward()
    torch.cuda.synchronize()
    print("CUDA ok: %s, torch %s" % (torch.cuda.get_device_name(device), torch.__version__),
          flush=True)


class ProgressLog(list):
    """The trainers' log list.  Every entry at a multiple of PROGRESS_EVERY steps is also
    printed (flushed) and appended to a log file, so a run can be watched while it trains."""

    def __init__(self, name: str, steps: int, path: Path):
        super().__init__()
        self.name, self.steps, self.path = name, steps, path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def line(self, text: str) -> None:
        print(text, flush=True)
        with open(self.path, "a") as f:
            f.write(text + "\n")

    def append(self, entry: dict) -> None:
        super().append(entry)
        if entry["step"] % PROGRESS_EVERY == 0:
            self.line("%s  %s  step %5d/%d  loss %.5f  elapsed %6.1f s"
                      % (time.strftime("%H:%M:%S"), self.name, entry["step"], self.steps,
                         entry["loss"], entry["seconds"]))


# =====================================================================================
# The hexagon
# =====================================================================================
def boundary_point(s: np.ndarray) -> np.ndarray:
    """The boundary point at arc length s in [0, 12) from vertex 0, counter-clockwise."""
    s = np.mod(np.asarray(s, np.float64), PERIMETER)
    edge = np.minimum((s // SIDE).astype(int), 5)
    u = (s - edge * SIDE) / SIDE
    a, b = VERTICES[edge], VERTICES[(edge + 1) % 6]
    return a + u[..., None] * (b - a)


def sample(n: int, rng: np.random.Generator, jitter: float = JITTER) -> np.ndarray:
    """Uniform on the boundary (uniform arc length), plus N(0, jitter^2 I)."""
    return boundary_point(rng.uniform(0.0, PERIMETER, n)) + jitter * rng.standard_normal((n, 2))


def distance_to_hexagon(x: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the hexagon's boundary (exact, 6 segments)."""
    x = np.asarray(x, np.float64)
    best = np.full(x.shape[:-1], np.inf)
    for k in range(6):
        a, b = VERTICES[k], VERTICES[(k + 1) % 6]
        ab = b - a
        u = np.clip(((x - a) @ ab) / (ab @ ab), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(x - (a + u[..., None] * ab), axis=-1))
    return best


def sliced_w2(a: np.ndarray, b: np.ndarray, n_proj: int = 512, seed: int = 0) -> float:
    """Sliced Wasserstein-2 between two equal-size samples (exact 1-D W2 per direction)."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((a.shape[1], n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    pa, pb = np.sort(a @ theta, axis=0), np.sort(b @ theta, axis=0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


# =====================================================================================
# The priors
# =====================================================================================
def _cache_path(kind: str, steps: int, batch: int, p: float = None,
                features: str = "default") -> Path:
    tag = ("" if p is None else "_p%g" % p) + FEATURES[features][2]
    return CACHE / ("%s_w%d_d%d_s%d_b%d_seed%d%s.pt" % (kind, WIDTH, DEPTH, steps, batch,
                                                        TRAIN_SEED, tag))


def load_or_train(kind: str, steps: int, batch: int, device: str, p: float = None,
                  retrain: bool = False, verbose: bool = True, features: str = "default"):
    """kind 'flow_matching', 'meanflow' (adaptive exponent p) or 'imeanflow' (iMF v-loss).
    Returns (model, info).
    Training logs to outputs/hexagon_study/train_<model>.log (see ProgressLog)."""
    path = _cache_path(kind, steps, batch, p, features)
    fm_cls, mf_cls, suffix, _ = FEATURES[features]
    cls = fm_cls if kind == "flow_matching" else mf_cls
    if path.exists() and not retrain:
        model = cls(dim=2, width=WIDTH, depth=DEPTH)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"])
        if verbose:
            print("loaded %s (trained in %.0f s)" % (path.name, state["info"]["seconds"]))
        return freeze(model.to(device)), state["info"]
    torch.manual_seed(TRAIN_SEED)
    model = cls(dim=2, width=WIDTH, depth=DEPTH).to(device)
    data_fn = lambda rng: sample(TRAIN_POOL, rng).astype(np.float32)
    name = (kind if p is None else "%s_p%g" % (kind, p)) + suffix
    log = ProgressLog(name, steps, OUT / ("train_%s.log" % name))
    log.line("%s  training %s: %d steps, batch %d, width %d, depth %d, %s"
             % (time.strftime("%Y-%m-%d %H:%M:%S"), name, steps, batch, WIDTH, DEPTH, device))
    started = time.perf_counter()
    if kind == "flow_matching":
        trained = train_flow_matching(model, data_fn, steps, batch, TRAIN_SEED, verbose=False,
                                      log_every=1000, log=log)
    elif kind == "imeanflow":
        trained = train_imeanflow(model, data_fn, steps, batch, TRAIN_SEED, verbose=False,
                                  log_every=1000, log=log)
    else:
        trained = train_meanflow(model, data_fn, steps, batch, TRAIN_SEED, adaptive_p=p,
                                 verbose=False, log_every=1000, log=log)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    info = dict(kind=kind, steps=steps, batch=batch, width=WIDTH, depth=DEPTH, seed=TRAIN_SEED,
                adaptive_p=p, seconds=seconds, device=str(device), log=list(log),
                features=features, architecture=cls.__name__,
                recipe="Adam 3e-4, 500 warm-up + cosine, grad clip 1, EMA 0.999"
                + ("; iMF v-loss, logit-normal times, s != t on half the batch"
                   if kind == "imeanflow" else ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in trained.state_dict().items()}, "info": info}, path)
    log.line("%s  saved %s (%.0f s, %.1f ms/step)" % (time.strftime("%H:%M:%S"), path.name,
                                                      seconds, 1000 * seconds / steps))
    return freeze(trained), info


@torch.no_grad()
def fm_sample(model, x: torch.Tensor, steps: int = 100, t0: float = 0.0) -> torch.Tensor:
    """Heun integration of the learned velocity from t0 to 1."""
    h = (1.0 - t0) / steps
    for k in range(steps):
        t = t0 + k * h
        tt = torch.full((x.shape[0],), t, device=x.device)
        a = model(x, tt)
        b = model(x + h * a, tt + h)
        x = x + 0.5 * h * (a + b)
    return x


@torch.no_grad()
def mf_transition(model, x: torch.Tensor, t: float, s: float) -> torch.Tensor:
    tt = torch.full((x.shape[0],), float(t), device=x.device)
    ss = torch.full((x.shape[0],), float(s), device=x.device)
    return x + (float(s) - float(t)) * model(x, tt, ss)


@torch.no_grad()
def mf_sample(model, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
    for k in range(steps):
        x = mf_transition(model, x, k / steps, (k + 1) / steps)
    return x


# =====================================================================================
# Stage 1
# =====================================================================================
def describe(x: np.ndarray, truth: np.ndarray) -> Dict:
    dist = distance_to_hexagon(x)
    return dict(sliced_w2=sliced_w2(x, truth), dist_mean=float(dist.mean()),
                dist_median=float(np.median(dist)), dist_p90=float(np.percentile(dist, 90)),
                dist_p99=float(np.percentile(dist, 99)), near_share=float((dist <= NEAR).mean()))


def train(args) -> None:
    require_cuda(args.device)
    kind, p = MODELS[args.model]
    load_or_train(kind, args.steps, args.batch, args.device, p=p, retrain=args.retrain,
                  features=args.features)


def jvp_scale(model, device, n: int = 8192, seed: int = 3) -> Dict:
    """Median |(s - t) du/dt| and median |u| at random t < s on the true marginal, du/dt
    the JVP along the model's own velocity (u(z, t, t), 1, 0): the term both MeanFlow
    objectives put into the target (iMF with that tangent, MeanFlow with x1 - x0)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    s = t + (1 - t) * rng.uniform(0, 1, n)
    z = (1 - t)[:, None] * rng.standard_normal((n, 2)) + t[:, None] * sample(n, rng)
    f = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)
    Z, T, S = f(z), f(t), f(s)
    with torch.no_grad():
        v = model(Z, T, T)
        u, dudt = torch.func.jvp(model, (Z, T, S), (v, torch.ones_like(T), torch.zeros_like(S)))
    corr = ((S - T)[:, None] * dudt).norm(dim=1)
    return dict(median_abs_correction=float(corr.median()), median_abs_u=float(u.norm(dim=1).median()))


def stage1(args) -> Dict:
    device = args.device
    require_cuda(device)
    features = args.features
    fm, fm_info = load_or_train("flow_matching", args.steps, args.batch, device,
                                retrain=args.retrain, features=features)
    mfs = {}
    for name in FEATURES[features][3]:
        kind, p = MODELS[name]
        mfs[MEANFLOW_LABELS[name]] = load_or_train(kind, args.steps, args.batch, device, p=p,
                                                   retrain=args.retrain,
                                                   features=features) + (name,)

    n = args.samples
    rng = np.random.default_rng(1)
    truth, truth2 = sample(n, rng), sample(n, rng)
    x0 = rng.standard_normal((n, 2))
    x1 = sample(n, rng)                                  # for the marginal states x_t
    dev = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
    np64 = lambda t: t.cpu().numpy().astype(np.float64)

    rows = [dict(sampler="data (second sample)", prior="", **describe(truth2, truth))]
    rows.append(dict(sampler="Heun 100 steps", prior="flow matching",
                     **describe(np64(fm_sample(fm, dev(x0), 100)), truth)))
    for label, (mf, _, _) in mfs.items():
        for k in (1, 2, 8):
            rows.append(dict(sampler="%d step%s" % (k, "" if k == 1 else "s"),
                             prior=label,
                             **describe(np64(mf_sample(mf, dev(x0), k)), truth)))
    for t in JUMP_TIMES:
        xt = (1 - t) * x0 + t * x1
        heun = int(round(100 * (1 - t)))
        ref = np64(fm_sample(fm, dev(xt), heun, t0=t))
        rows.append(dict(sampler="Heun %d steps from t=%g" % (heun, t), prior="flow matching",
                         **describe(ref, truth)))
        for label, (mf, _, _) in mfs.items():
            out = np64(mf_transition(mf, dev(xt), t, 1.0))
            rows.append(dict(sampler="1 jump T(x; %g -> 1)" % t, prior=label,
                             gap_to_fm=float(np.sqrt(((out - ref) ** 2).sum(1).mean())),
                             **describe(out, truth)))
    floor = rows[0]["sliced_w2"]
    infos = dict(flow_matching=fm_info, **{name: info for _, info, name in mfs.values()})
    infos = {k + FEATURES[features][2]: v for k, v in infos.items()}
    fm_heun = rows[1]["sliced_w2"]
    criteria = []
    for label in mfs:
        sw = {r["sampler"]: r["sliced_w2"] for r in rows if r["prior"] == label}
        one, eight = sw["1 step"], sw["8 steps"]
        near1 = [r["near_share"] for r in rows if r["prior"] == label and r["sampler"] == "1 step"][0]
        criteria.append(dict(prior=label, features=features, one_step_sw2=one,
                             one_step_pass=one <= ONE_STEP_SW2_MAX,
                             eight_step_sw2=eight, eight_step_over_fm_heun=eight / fm_heun,
                             eight_step_pass=eight <= EIGHT_STEP_RATIO_MAX * fm_heun,
                             one_step_near_share=near1, **jvp_scale(mfs[label][0], device)))

    print("\nsanity checks on %d samples (same noise for every sampler); sliced W2 floor %.4f"
          % (n, floor))
    print("  %-15s %-25s %8s %8s %8s %8s %8s %8s %8s" % (
        "prior", "sampler", "SW2", "d mean", "d med", "d p90", "d p99", "<=0.05", "gap FM"))
    for r in rows:
        print("  %-15s %-25s %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f %8s" % (
            r["prior"], r["sampler"], r["sliced_w2"], r["dist_mean"], r["dist_median"],
            r["dist_p90"], r["dist_p99"], r["near_share"],
            "%.4f" % r["gap_to_fm"] if "gap_to_fm" in r else ""))
    print("\npass criteria (%s time features): one-step SW2 <= %.2f; 8-step SW2 <= %.1fx "
          "flow-matching Heun-100 (%.4f)" % (features, ONE_STEP_SW2_MAX, EIGHT_STEP_RATIO_MAX, fm_heun))
    for c in criteria:
        print("  %-15s one-step SW2 %.4f %s   8-step SW2 %.4f = %.1fx FM %s   overall %s"
              "   one-step <=0.05 %.3f   median |(s-t) du/dt| %.2f vs |u| %.2f"
              % (c["prior"], c["one_step_sw2"], "PASS" if c["one_step_pass"] else "FAIL",
                 c["eight_step_sw2"], c["eight_step_over_fm_heun"],
                 "PASS" if c["eight_step_pass"] else "FAIL",
                 "PASS" if c["one_step_pass"] and c["eight_step_pass"] else "FAIL",
                 c["one_step_near_share"], c["median_abs_correction"], c["median_abs_u"]))
    for name, info in infos.items():
        print("  training %-16s %5.0f s  (%d steps, batch %d, %.1f ms/step, %s)"
              % (name, info["seconds"], info["steps"], info["batch"],
                 1000 * info["seconds"] / info["steps"], info["device"]))

    out = OUT / ("stage1" if features == "default" else "stage1_%s" % features)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["prior", "sampler", "sliced_w2", "dist_mean", "dist_median", "dist_p90", "dist_p99",
            "near_share", "gap_to_fm"]
    with open(out / "sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with open(out / "criteria.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(criteria[0]))
        w.writeheader()
        w.writerows(criteria)
    (out / "sanity.json").write_text(json.dumps(dict(
        samples=n, sliced_w2_floor=floor, rows=rows, criteria=criteria,
        training={k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in infos.items()}),
        indent=2))
    _plot(out, fm, mfs, x0[:4000], truth[:4000], device, infos)
    print("wrote %s (sanity.csv, criteria.csv, sanity.json, samples.png, training_curves.png)" % out)
    return dict(rows=rows, floor=floor)


def _plot(out: Path, fm, mfs, x0, truth, device, infos) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
    panels = [("data", truth), ("FM Heun 100", fm_sample(fm, dev(x0), 100).cpu().numpy())]
    for label, (mf, _, _) in mfs.items():
        panels += [("%s, 1 step" % label, mf_sample(mf, dev(x0), 1).cpu().numpy()),
                   ("%s, 8 steps" % label, mf_sample(mf, dev(x0), 8).cpu().numpy())]
    cols = 4
    rows_ = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(3 * cols, 3.1 * rows_))
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    hexagon = np.vstack([VERTICES, VERTICES[:1]])
    for ax, (title, pts) in zip(axes, panels):
        ax.plot(hexagon[:, 0], hexagon[:, 1], color="0.6", lw=0.8)
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.4)
        ax.set_title(title, fontsize=9); ax.set_aspect("equal")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out / "samples.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, info in infos.items():
        log = info.get("log") or []
        ax.plot([e["step"] for e in log], [e["raw_mse"] for e in log], label=name)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("raw training error")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train one prior (cached unless --retrain)")
    t.add_argument("--model", required=True, choices=sorted(MODELS))
    q = sub.add_parser("stage1", help="the sanity table (trains any prior not yet cached)")
    q.add_argument("--samples", type=int, default=20_000)
    for c in (t, q):
        c.add_argument("--device", default="cuda")
        c.add_argument("--steps", type=int, default=TRAIN_STEPS)
        c.add_argument("--batch", type=int, default=TRAIN_BATCH)
        c.add_argument("--retrain", action="store_true")
        c.add_argument("--features", choices=sorted(FEATURES), default="default",
                       help="time features of both MLPs (see the module docstring)")
    t.set_defaults(run=train)
    q.set_defaults(run=stage1)
    a = p.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    # A CUDA error can leave the interpreter hanging at shutdown; print the error and leave
    # with os._exit so a failed run always ends.
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
