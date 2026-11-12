#!/usr/bin/env python
"""RHSO on a two-moons prior in 2-D (the toy example of the paper).

    python toy/rhso_two_moons.py                 # train on the CPU, then draw
    python toy/rhso_two_moons.py --retrain       # ignore the cached model
    python toy/rhso_two_moons.py --N 1 2 4 8 --M 40 --lr 0.05 [--budget 160]

A small MLP u(x, t, s) models the average velocity from time t to time s on the two-moons
distribution (t = 0 noise, t = 1 data), so one evaluation transports a state from t to s.
A target x* sits at the tip of one moon.  From one noise sample the left panel overlays the
uncontrolled flow-ODE trajectory and RHSO for several N: at each of the N stages the
current state is optimised so that the one-jump terminal prediction T(q; t_k -> 1) lands
on x*, one interval is executed, and the plan is discarded.  The right panel plots the
distance of each stage's terminal prediction from x* before and after that stage's
optimisation.  N = 1 is the D-Flow-like special case.

By default u is obtained by trajectory distillation from a flow-matching teacher;
`--trainer meanflow` uses the MeanFlow identity u = v + (s - t) du/dt instead, which is
less stable on this problem.  Everything runs on the CPU.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from common import MeanFlowMLP, TimeFeatures, VelocityMLP, train_meanflow  # noqa: F401

DEVICE = "cpu"
DATA_NOISE = 0.0          # jitter on the moons; 0 = pure curves, like MPC-Flow's hexagon
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "toy_two_moons_meanflow.pt"
TEXTWIDTH_IN = 5.5


# ===================================================================================== data
def two_moons(n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """sklearn.make_moons re-implemented (no dependency), then centred and scaled ~x1.6 so
    the data sits at the scale of N(0, I) noise."""
    n_up = n // 2
    n_lo = n - n_up
    th = rng.uniform(0, math.pi, n_up)
    upper = np.stack([np.cos(th), np.sin(th)], 1)
    th = rng.uniform(0, math.pi, n_lo)
    lower = np.stack([1.0 - np.cos(th), 1.0 - np.sin(th) - 0.5], 1)
    x = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    return ((x - np.array([0.5, 0.25])) * 1.6).astype(np.float32)


def moon_point(which: str, theta: float) -> np.ndarray:
    """A point ON a moon at angle theta, in the same centred/scaled coordinates."""
    if which == "upper":
        p = np.array([math.cos(theta), math.sin(theta)])
    else:
        p = np.array([1.0 - math.cos(theta), 1.0 - math.sin(theta) - 0.5])
    return ((p - np.array([0.5, 0.25])) * 1.6).astype(np.float32)


# ==================================================================================== model
# TimeFeatures, MeanFlowMLP, VelocityMLP and the MeanFlow trainer live in toy/common.py (shared
# with toy/gmm_study.py).  At dim = 2 they are bit-for-bit the classes this file defined.
def train(model: MeanFlowMLP, steps: int, batch: int, seed: int, long_share: float = 0.5,
          verbose: bool = True):
    """MeanFlow objective (Geng et al.) on two moons; see common.train_meanflow."""
    return train_meanflow(model, lambda rng: two_moons(200_000, DATA_NOISE, rng), steps, batch,
                          seed, long_share=long_share, verbose=verbose)


def integrate(v, x, t0: torch.Tensor, t1: torch.Tensor, steps: int = 16):
    """Heun integration of dx/dt = v(x, t) from per-sample times t0 to t1 (no gradient)."""
    h = (t1 - t0) / steps
    for k in range(steps):
        t = t0 + k * h
        a = v(x, t)
        b = v(x + h[:, None] * a, t + h)
        x = x + 0.5 * h[:, None] * (a + b)
    return x


def train_distilled(model: MeanFlowMLP, steps: int, batch: int, seed: int, verbose: bool = True,
                    distill_steps: int = 0):
    """Trajectory distillation: learn the AVERAGE velocity of a flow-matching teacher.

    1. v_theta(x, t) by flow matching -- stable, a few minutes on a CPU.
    2. u_phi(x_t, t, s) = (x_s - x_t) / (s - t) with x_s the teacher's ODE solution from x_t,
       and u_phi(x, t, t) = v_theta(x, t): plain regression on well-defined targets.

    This is the same object a MeanFlow learns through its identity u = v + (s - t) du/dt
    (see `train`), obtained here by a route that cannot diverge; in 2-D the identity-based
    trainer spikes to raw errors in the hundreds whenever long intervals enter, and its
    one-jump samples stay smeared.  For a figure whose point is the one-jump terminal
    prediction, the transition map must actually be right.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = torch.from_numpy(two_moons(200_000, DATA_NOISE, rng))
    v = VelocityMLP().to(DEVICE)
    opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        x1 = data[torch.randint(0, len(data), (batch,))]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch)
        z = (1 - t)[:, None] * x0 + t[:, None] * x1
        loss = ((v(z, t) - (x1 - x0)) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if verbose and (step % 2000 == 0 or step == 1):
            print("  [teacher] step %5d  flow-matching mse %.4f  (%.0fs)" % (step, loss.item(), time.perf_counter() - started))
    v.eval()
    for q in v.parameters():
        q.requires_grad_(False)

    steps = distill_steps or steps
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    ema = MeanFlowMLP().to(DEVICE); ema.load_state_dict(model.state_dict())
    for step in range(1, steps + 1):
        x1 = data[torch.randint(0, len(data), (batch,))]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch)
        long = torch.rand(batch) < 0.75
        s = torch.where(long, t + (1 - t) * torch.rand(batch), t)
        z = (1 - t)[:, None] * x0 + t[:, None] * x1
        with torch.no_grad():
            xs = integrate(v, z, t, s, 16)
            gap = (s - t)[:, None]
            target = torch.where(gap > 1e-6, (xs - z) / gap.clamp_min(1e-6), v(z, t))
        loss = ((model(z, t, s) - target) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if verbose and (step % 2000 == 0 or step == 1):
            print("  [distil]  step %5d  mse %.4f  (%.0fs)" % (step, loss.item(), time.perf_counter() - started))
    ema.teacher = v                      # kept for the uncontrolled ODE, so both use one field
    return ema


def load_or_train(retrain: bool, steps: int, batch: int, seed: int, long_share: float,
                  trainer: str = "distill", distill_steps: int = 0) -> MeanFlowMLP:
    cache = CACHE.with_name("toy_two_moons_%s_noise%g.pt" % (trainer, DATA_NOISE))
    model = MeanFlowMLP().to(DEVICE)
    if cache.exists() and not retrain:
        state = torch.load(cache, map_location=DEVICE)
        # the distilled model carries its teacher as a submodule, saved separately under "v"
        model.load_state_dict({k: v for k, v in state["u"].items() if not k.startswith("teacher.")})
        if state.get("v") is not None:
            model.teacher = VelocityMLP().to(DEVICE); model.teacher.load_state_dict(state["v"])
        print("loaded", cache)
    else:
        print("training (%s) on two moons: %d steps, batch %d, CPU ..." % (trainer, steps, batch))
        model = (train_distilled(model, steps, batch, seed, distill_steps=distill_steps)
                 if trainer == "distill" else train(model, steps, batch, seed, long_share))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"u": model.state_dict(),
                    "v": getattr(model, "teacher", None) and model.teacher.state_dict()}, cache)
        print("saved", cache)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ================================================================================ dynamics
def transition(model, x, t: float, s: float):
    """x at time t  ->  x at time s, in ONE learned jump."""
    tt = torch.full((x.shape[0],), t)
    ss = torch.full((x.shape[0],), s)
    return x + (s - t) * model(x, tt, ss)


def velocity(model, x, t):
    """v(x, t): the teacher's field if the model was distilled, else u(x, t, t)."""
    teacher = getattr(model, "teacher", None)
    tt = torch.full((x.shape[0],), t)
    return teacher(x, tt) if teacher is not None else model(x, tt, tt)


def ode_trajectory(model, x0, n_steps: int = 200):
    """The uncontrolled flow: Heun integration of the instantaneous velocity."""
    x = x0.clone()
    pts = [x.clone()]
    h = 1.0 / n_steps
    with torch.no_grad():
        for k in range(n_steps):
            t = k * h
            v1 = velocity(model, x, t)
            v2 = velocity(model, x + h * v1, t + h)
            x = x + 0.5 * h * (v1 + v2)
            pts.append(x.clone())
    return torch.cat(pts).numpy()


def rhso(model, x0, target, N: int, M: int, lr: float, mu: float = 0.0):
    """Receding-horizon state optimisation with a MeanFlow planner.

    Stage k (time t_k = k / N):  optimise q from x_k so that the one-jump terminal prediction
    T(q) = q + (1 - t_k) u(q, t_k, 1) lands on the target (Adam, M steps, fresh moments);
    then execute ONE interval, x_{k+1} = q + (t_{k+1} - t_k) u(q, t_k, t_{k+1}), and discard
    the plan.  mu > 0 adds the state-anchor penalty ||q - x_k||^2 (0 here, as in the paper's
    final benchmark).  Returns the executed states, the optimised states, and the terminal
    predictions before and after each stage's optimisation.
    """
    grid = [k / N for k in range(N + 1)]
    x = x0.clone()
    states, optimised, pred_before, pred_after = [x.clone()], [], [], []
    for k in range(N):
        tk, tn = grid[k], grid[k + 1]
        with torch.no_grad():
            pred_before.append(transition(model, x, tk, 1.0).clone())
        q = x.clone().requires_grad_(True)
        opt = torch.optim.Adam([q], lr=lr)
        for _ in range(M):
            loss = ((transition(model, q, tk, 1.0) - target) ** 2).sum()
            if mu > 0:
                loss = loss + mu * ((q - x) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        q = q.detach()
        with torch.no_grad():
            pred_after.append(transition(model, q, tk, 1.0).clone())
            x = transition(model, q, tk, tn)
        optimised.append(q.clone())
        states.append(x.clone())
    to_np = lambda lst: torch.cat(lst).numpy()
    return dict(states=to_np(states), optimised=to_np(optimised),
                pred_before=to_np(pred_before), pred_after=to_np(pred_after), grid=grid)


# ================================================================================== figure
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "axes.linewidth": 0.5,
    })
    return plt


def diagnostic(model, out: Path):
    """Samples from the trained prior over the data: the check that training worked."""
    plt = style()
    rng = np.random.default_rng(1)
    data = two_moons(3000, DATA_NOISE, rng)
    x0 = torch.randn(3000, 2)
    with torch.no_grad():
        one = transition(model, x0, 0.0, 1.0).numpy()
        x = x0.clone()
        for k in range(8):
            x = transition(model, x, k / 8, (k + 1) / 8)
        eight = x.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for ax, pts, title in zip(axes, (data, one, eight), ("data", "1-step samples", "8-step samples")):
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, color="#4c78a8")
        ax.set_title(title, fontsize=8); ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)
        ax.tick_params(labelsize=6)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    # a number: mean distance from each sample to its nearest data point
    d1 = np.sqrt(((one[:, None, :] - data[None, :, :]) ** 2).sum(-1)).min(1).mean()
    d8 = np.sqrt(((eight[:, None, :] - data[None, :, :]) ** 2).sum(-1)).min(1).mean()
    dd = np.sqrt(((data[:500, None, :] - data[None, 500:, :]) ** 2).sum(-1)).min(1).mean()
    print("mean nearest-data distance: 1-step %.3f   8-step %.3f   (data-to-data, sample spacing %.3f)" % (d1, d8, dd))


def pick_start(model, target, seeds, min_dist: float, min_len: float, x0_fixed=None):
    """A noise sample whose UNCONTROLLED path is long enough to see and ends at least
    min_dist from the target, so the controlled paths have somewhere visible to go.  The
    first seed that qualifies is used; recorded, hence reproducible.  --x0 overrides."""
    if x0_fixed is not None:
        return -1, torch.tensor([x0_fixed], dtype=torch.float32)
    for sd in seeds:
        x0 = torch.from_numpy(np.random.default_rng(sd).normal(0, 1, (1, 2)).astype(np.float32))
        path = ode_trajectory(model, x0, 100)
        length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        if np.linalg.norm(path[-1] - target.numpy()[0]) >= min_dist and length >= min_len:
            return sd, x0
    raise SystemExit("no seed in the list gives a long uncontrolled path ending far from the target")


def draw(model, args):
    plt = style()
    from matplotlib.lines import Line2D
    target = torch.from_numpy(moon_point(args.moon, args.theta))[None, :]
    seed, x0 = pick_start(model, target, list(range(40)), args.min_dist, args.min_len,
                          None if args.x0 == [] else args.x0)
    tgt = target.numpy()[0]
    print("start seed %d, x0 = %s, target = %s" % (seed, x0.numpy()[0].round(3), tgt.round(3)))

    free = ode_trajectory(model, x0, 200)
    def M_for(N):                      # --budget fixes N*M; otherwise every stage gets --M
        return max(1, args.budget // N) if args.budget else args.M
    runs = {N: rhso(model, x0, target, N, M_for(N), args.lr, args.mu) for N in args.N}
    for N, r in runs.items():
        print("  N=%d  M=%-3d final distance to target %.3f   (uncontrolled %.3f)"
              % (N, M_for(N), np.linalg.norm(r["states"][-1] - tgt), np.linalg.norm(free[-1] - tgt)))

    # the support drawn as two curves, the way MPC-Flow's eval.py draws the hexagon boundary
    th = np.linspace(0, math.pi, 400)
    curve_up = np.stack([moon_point("upper", a) for a in th]); curve_lo = np.stack([moon_point("lower", a) for a in th])
    colours = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7"][:len(args.N)]   # Okabe-Ito, as in MPC-Flow

    # Two panels: the picture, and the number behind it -- how far each stage's terminal
    # prediction is from the target, before and after that stage's optimisation.
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.58))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25, left=0.01, right=0.99,
                          top=0.97, bottom=0.16)
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])

    for c_ in (curve_up, curve_lo):
        ax.plot(c_[:, 0], c_[:, 1], color="#333333", lw=2.0, alpha=0.8, zorder=1, solid_capstyle="round")
    ax.plot(free[:, 0], free[:, 1], color="#222222", lw=1.1, ls="--", zorder=3)
    ax.plot(free[-1, 0], free[-1, 1], marker="o", ms=4.5, mfc="white", mec="#222222", mew=1.0, zorder=4)
    for (N, r), c in zip(runs.items(), colours):
        st, q = r["states"], r["optimised"]
        for k in range(N):                                   # the optimisation move, dotted
            ax.plot([st[k, 0], q[k, 0]], [st[k, 1], q[k, 1]], color=c, lw=0.8, ls=":", zorder=3)
            ax.plot([q[k, 0], st[k + 1, 0]], [q[k, 1], st[k + 1, 1]], color=c, lw=1.5, zorder=3)
        ax.plot(st[1:-1, 0], st[1:-1, 1], ls="none", marker="o", ms=2.6, color=c, zorder=4)
        ax.plot(st[-1, 0], st[-1, 1], ls="none", marker="o", ms=4.5, color=c, zorder=5)
    ax.plot(x0[0, 0], x0[0, 1], marker="o", ms=5.0, color="grey", zorder=6)
    ax.plot(tgt[0], tgt[1], marker="*", ms=13, color="red", mec="darkred", mew=0.8, zorder=7)
    ax.annotate("$x_0$", (x0[0, 0], x0[0, 1]), (-11, -10), textcoords="offset points", fontsize=7)
    ax.annotate("$x^\\star$", (tgt[0], tgt[1]), (7, -9), textcoords="offset points", fontsize=7,
                color="#c1272d")
    # limits from what is drawn, with a margin, so no path leaves the canvas
    pts = np.concatenate([curve_up, curve_lo, free, x0.numpy(), tgt[None]] +
                         [r["states"] for r in runs.values()] + [r["optimised"] for r in runs.values()])
    lo, hi = pts.min(0) - 0.2, pts.max(0) + 0.2
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")

    # right panel: distance of the terminal prediction to the target, per stage
    for (N, r), c in zip(runs.items(), colours):
        t_mid = np.array(r["grid"][:-1])
        d_before = np.linalg.norm(r["pred_before"] - tgt, axis=1)
        d_after = np.linalg.norm(r["pred_after"] - tgt, axis=1)
        for k in range(N):
            ax2.plot([t_mid[k], t_mid[k]], [d_before[k], d_after[k]], color=c, lw=0.8, ls=":")
        ax2.plot(t_mid, d_before, ls="none", marker="o", ms=3.0, mfc="white", mec=c, mew=0.9)
        ax2.plot(t_mid, d_after, ls="none", marker="o", ms=3.0, color=c)
        ax2.plot(np.append(t_mid, 1.0), np.append(d_after, np.linalg.norm(r["states"][-1] - tgt)),
                 color=c, lw=1.2, alpha=0.9)
    ax2.axhline(np.linalg.norm(free[-1] - tgt), color="#222222", lw=1.0, ls="--")
    ax2.set_yscale("log")
    ax2.set_xlabel("stage time $t_k$", fontsize=7, labelpad=2)
    ax2.set_ylabel("$\\|T(\\cdot; t_k \\to 1) - x^\\star\\|$", fontsize=7, labelpad=2)
    ax2.tick_params(labelsize=6, length=2, pad=1.5)
    ax2.set_xlim(-0.04, 1.04)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    handles = [Line2D([], [], color="#222222", ls="--", lw=1.1, marker="o", ms=3.5, mfc="white",
                      markevery=[1], label="uncontrolled flow (and its endpoint)")]
    handles += [Line2D([], [], color=c, lw=1.5, marker="o", ms=3, label="RHSO, $N=%d$" % N)
                for N, c in zip(args.N, colours)]
    handles += [Line2D([], [], color="#666666", ls=":", lw=0.9, label="optimise the state"),
                Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="#666666",
                       label="prediction before / after")]
    fig.legend(handles=handles, fontsize=6.2, loc="lower center", ncol=len(handles), frameon=False,
               handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    out = Path(args.out) / "toy_rhso_two_moons.pdf"
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print("wrote", out, "(+ .png)")
    return runs, free


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--data-noise", type=float, default=0.0,
                   help="Gaussian jitter added to the moons (0: pure curves, as MPC-Flow samples its hexagon boundary)")
    p.add_argument("--trainer", choices=("distill", "meanflow"), default="distill",
                   help="how u(x,t,s) is obtained: distilled from a flow-matching teacher (default), "
                        "or the MeanFlow identity with JVP")
    p.add_argument("--distill-steps", type=int, default=30000,
                   help="distillation steps (the teacher uses --steps); 0 = same as --steps")
    p.add_argument("--long-share", type=float, default=0.5,
                   help="share of training pairs with s != t once the curriculum has ramped")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--M", type=int, default=40, help="inner optimisation steps per stage")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--moon", choices=("upper", "lower"), default="upper")
    p.add_argument("--theta", type=float, default=0.0, help="angle on the moon; 0 is a tip")
    p.add_argument("--min-dist", type=float, default=1.5,
                   help="required distance between the uncontrolled endpoint and the target")
    p.add_argument("--min-len", type=float, default=1.2, help="required length of the uncontrolled path")
    p.add_argument("--x0", type=float, nargs="*", default=[-1.5, 1.5],
                   help="the noise start (fixed, like MPC-Flow's x0 = (-0.75, -0.5)); pass 'auto' logic by giving no value: --x0 with --min-dist selects a seed")
    p.add_argument("--budget", type=int, default=0,
                   help="total inner steps shared by the N stages (M = budget / N); 0 = use --M")
    p.add_argument("--out", default=str(ROOT / "figures"))
    args = p.parse_args()

    global DATA_NOISE
    DATA_NOISE = args.data_noise
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = load_or_train(args.retrain, args.steps, args.batch, args.seed, args.long_share, args.trainer,
                          args.distill_steps)
    diagnostic(model, Path(args.out) / "toy_two_moons_check.png")
    draw(model, args)


if __name__ == "__main__":
    main()
