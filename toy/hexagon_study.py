#!/usr/bin/env python
"""Hexagon study (rebuttal Task 4), modelled on the toy of MPC-Flow (Webber et al., 2026,
Appendix C).

    python -u toy/hexagon_study.py train --model flow_matching    # one model per run, cached
    python -u toy/hexagon_study.py train --model meanflow_p0
    python -u toy/hexagon_study.py train --model meanflow_p0.5
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
from common import MeanFlowMLP, VelocityMLP, freeze, train_flow_matching, train_meanflow  # noqa: E402

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
          **{"meanflow_p%g" % p: ("meanflow", p) for p in SWEEP_P}}
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
def _cache_path(kind: str, steps: int, batch: int, p: float = None) -> Path:
    tag = "" if p is None else "_p%g" % p
    return CACHE / ("%s_w%d_d%d_s%d_b%d_seed%d%s.pt" % (kind, WIDTH, DEPTH, steps, batch,
                                                        TRAIN_SEED, tag))


def load_or_train(kind: str, steps: int, batch: int, device: str, p: float = None,
                  retrain: bool = False, verbose: bool = True):
    """kind 'flow_matching' or 'meanflow' (adaptive exponent p).  Returns (model, info).
    Training logs to outputs/hexagon_study/train_<model>.log (see ProgressLog)."""
    path = _cache_path(kind, steps, batch, p)
    cls = VelocityMLP if kind == "flow_matching" else MeanFlowMLP
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
    name = kind if p is None else "meanflow_p%g" % p
    log = ProgressLog(name, steps, OUT / ("train_%s.log" % name))
    log.line("%s  training %s: %d steps, batch %d, width %d, depth %d, %s"
             % (time.strftime("%Y-%m-%d %H:%M:%S"), name, steps, batch, WIDTH, DEPTH, device))
    started = time.perf_counter()
    if kind == "flow_matching":
        trained = train_flow_matching(model, data_fn, steps, batch, TRAIN_SEED, verbose=False,
                                      log_every=1000, log=log)
    else:
        trained = train_meanflow(model, data_fn, steps, batch, TRAIN_SEED, adaptive_p=p,
                                 verbose=False, log_every=1000, log=log)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    info = dict(kind=kind, steps=steps, batch=batch, width=WIDTH, depth=DEPTH, seed=TRAIN_SEED,
                adaptive_p=p, seconds=seconds, device=str(device), log=list(log),
                recipe="Adam 3e-4, 500 warm-up + cosine, grad clip 1, EMA 0.999")
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
    load_or_train(kind, args.steps, args.batch, args.device, p=p, retrain=args.retrain)


def stage1(args) -> Dict:
    device = args.device
    require_cuda(device)
    fm, fm_info = load_or_train("flow_matching", args.steps, args.batch, device,
                                retrain=args.retrain)
    mfs = {p: load_or_train("meanflow", args.steps, args.batch, device, p=p, retrain=args.retrain)
           for p in SWEEP_P}

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
    for p, (mf, _) in mfs.items():
        for k in (1, 2, 8):
            rows.append(dict(sampler="%d step%s" % (k, "" if k == 1 else "s"),
                             prior="MeanFlow p=%g" % p,
                             **describe(np64(mf_sample(mf, dev(x0), k)), truth)))
    for t in JUMP_TIMES:
        xt = (1 - t) * x0 + t * x1
        heun = int(round(100 * (1 - t)))
        ref = np64(fm_sample(fm, dev(xt), heun, t0=t))
        rows.append(dict(sampler="Heun %d steps from t=%g" % (heun, t), prior="flow matching",
                         **describe(ref, truth)))
        for p, (mf, _) in mfs.items():
            out = np64(mf_transition(mf, dev(xt), t, 1.0))
            rows.append(dict(sampler="1 jump T(x; %g -> 1)" % t, prior="MeanFlow p=%g" % p,
                             gap_to_fm=float(np.sqrt(((out - ref) ** 2).sum(1).mean())),
                             **describe(out, truth)))
    floor = rows[0]["sliced_w2"]
    infos = dict(flow_matching=fm_info, **{"meanflow_p%g" % p: info for p, (_, info) in mfs.items()})

    print("\nsanity checks on %d samples (same noise for every sampler); sliced W2 floor %.4f"
          % (n, floor))
    print("  %-15s %-25s %8s %8s %8s %8s %8s %8s %8s" % (
        "prior", "sampler", "SW2", "d mean", "d med", "d p90", "d p99", "<=0.05", "gap FM"))
    for r in rows:
        print("  %-15s %-25s %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f %8s" % (
            r["prior"], r["sampler"], r["sliced_w2"], r["dist_mean"], r["dist_median"],
            r["dist_p90"], r["dist_p99"], r["near_share"],
            "%.4f" % r["gap_to_fm"] if "gap_to_fm" in r else ""))
    for name, info in infos.items():
        print("  training %-16s %5.0f s  (%d steps, batch %d, %.1f ms/step, %s)"
              % (name, info["seconds"], info["steps"], info["batch"],
                 1000 * info["seconds"] / info["steps"], info["device"]))

    out = OUT / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["prior", "sampler", "sliced_w2", "dist_mean", "dist_median", "dist_p90", "dist_p99",
            "near_share", "gap_to_fm"]
    with open(out / "sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    (out / "sanity.json").write_text(json.dumps(dict(
        samples=n, sliced_w2_floor=floor, rows=rows,
        training={k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in infos.items()}),
        indent=2))
    _plot(out, fm, mfs, x0[:4000], truth[:4000], device, infos)
    print("wrote %s (sanity.csv, sanity.json, samples.png, training_curves.png)" % out)
    return dict(rows=rows, floor=floor)


def _plot(out: Path, fm, mfs, x0, truth, device, infos) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
    panels = [("data", truth), ("FM Heun 100", fm_sample(fm, dev(x0), 100).cpu().numpy())]
    for p, (mf, _) in mfs.items():
        panels += [("MF p=%g, 1 step" % p, mf_sample(mf, dev(x0), 1).cpu().numpy()),
                   ("MF p=%g, 8 steps" % p, mf_sample(mf, dev(x0), 8).cpu().numpy())]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.1))
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
