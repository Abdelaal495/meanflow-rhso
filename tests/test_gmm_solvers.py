"""Gaussian-mixture study, stage 2 (rebuttal Task 4): the oracle priors and the toy solvers.

Checks, on the CPU in float64 (no trained model):

1. the torch exact velocity equals the numpy one of toy/gmm_study.py; the direct endpoint
   x + (1 - t) v is exactly E[x1 | x_t] (the closed-form posterior of x_t = t x1 + (1-t) x0);
   the oracle map with RK4_STEPS steps is within 1e-3 relative RMSE of the 1000-step map
   from t = 0, 0.25, 0.5, 0.75, and the 1000-step torch map equals the numpy exact flow;
   backprop through the map matches a finite difference;
2. the per-row Adam equals torch.optim.Adam run separately on each row with its own lr;
3. rows never interact: every solver gives a row the same output alone and in a batch
   (which is what lets one run carry many problems and many hyperparameter settings), and
   the solvers' own noise depends only on (problem, seed, iteration);
4. cost accounting: RHSO equals src/rhso.py's own cost model (MeanFlow; flow matching with
   the direct planner and with the Heun suffix), and every other solver equals the image
   code's per-iteration accounting;
5. Flower's refinement is the exact proximal step (denoising closed form), and one-step
   Euler D-Flow on the exact velocity is degenerate (x + v(x, 0) = E[x1] for every x).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))
sys.path.insert(0, str(ROOT))

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

import gmm_oracle as O    # noqa: E402
import gmm_solvers as S   # noqa: E402
import gmm_stage2 as T    # noqa: E402
import gmm_study as G     # noqa: E402

RESULTS = []
DT = torch.float64


def report(name, ok, detail=""):
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    RESULTS.append(bool(ok))


def check_oracle(mix, tm):
    print("\n1. the oracle priors")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 16)) * 1.3
    worst = max(np.abs(tm.velocity(torch.tensor(x), t).numpy() - G.exact_velocity(mix, x, t)).max()
                for t in (0.0, 0.2, 0.7, 0.99, 1.0))
    report("torch velocity = numpy exact velocity", worst < 1e-10, "max difference %.1e" % worst)

    fm = O.FlowOracle(tm)
    t = 0.6
    end = fm.endpoint(torch.tensor(x[:16]), t).numpy()
    ref = np.stack([G.posterior(mix, t * np.eye(16), 1 - t, xi).mean() for xi in x[:16]])
    err = np.abs(end - ref).max()
    report("x + (1 - t) v = E[x1 | x_t]", err < 1e-9, "max difference %.1e at t = %g" % (err, t))

    check = O.rk4_step_check(mix, candidates=(O.RK4_STEPS,), n=1024)
    worst = check["rows"][0]["worst"]
    report("RK4_STEPS = %d map within 1e-3 of 1000 steps" % O.RK4_STEPS,
           worst <= O.CHECK_TOLERANCE,
           "worst relative RMSE %.1e over t = 0, 0.25, 0.5, 0.75 -> 1" % worst)
    x0 = rng.standard_normal((64, 16))
    a = O.MeanFlowOracle(tm, 1000).transition(torch.tensor(x0), 0.0, 1.0).numpy()
    b = G.exact_flow(mix, x0, steps=1000)
    report("1000-step torch map = numpy exact flow", np.abs(a - b).max() < 1e-9,
           "max difference %.1e" % np.abs(a - b).max())

    mf = O.MeanFlowOracle(tm, O.RK4_STEPS)
    H = torch.tensor(G.make_operator("projection"))
    y = torch.tensor(rng.standard_normal(8))
    f = lambda q: 0.5 * ((mf.transition(q, 0.25, 1.0) @ H.T - y) ** 2).sum()
    q = torch.tensor(rng.standard_normal((1, 16)), requires_grad=True)
    (g,) = torch.autograd.grad(f(q), q)
    e = 1e-6
    fd = torch.stack([(f(q.detach() + e * torch.eye(16)[i]) - f(q.detach() - e * torch.eye(16)[i])) / (2 * e)
                      for i in range(16)])
    rel = (g[0] - fd).abs().max() / fd.abs().max()
    report("backprop through the map = finite difference", rel < 1e-6, "max rel. difference %.1e" % rel)


def check_adam():
    print("\n2. per-row Adam")
    rng = np.random.default_rng(1)
    A = torch.tensor(rng.standard_normal((3, 4, 4)))
    A = A @ A.transpose(1, 2) + torch.eye(4)
    b = torch.tensor(rng.standard_normal((3, 4)))
    lrs = torch.tensor([0.003, 0.05, 0.3], dtype=DT)
    loss = lambda x: (0.5 * torch.einsum("ri,rij,rj->r", x, A, x) - (b * x).sum(1)).sum()
    x = torch.zeros(3, 4, dtype=DT)
    opt = S.RowAdam(x, lrs[:, None])
    for _ in range(60):
        x = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(loss(x), x)
        x = opt.step(x.detach(), g)
    worst = 0.0
    for r in range(3):
        p = torch.zeros(1, 4, dtype=DT, requires_grad=True)
        ref = torch.optim.Adam([p], lr=float(lrs[r]))
        for _ in range(60):
            ref.zero_grad()
            (0.5 * p[0] @ A[r] @ p[0] - b[r] @ p[0]).backward()
            ref.step()
        worst = max(worst, float((p.detach()[0] - x[r]).abs().max()))
    report("RowAdam = torch.optim.Adam per row", worst < 1e-12, "60 steps, lr 0.003 / 0.05 / 0.3; max difference %.1e" % worst)


def small_set(mix):
    return T.ProblemSet(mix, 123, 3, 2, "cpu", dtype=DT, posteriors=False)


def check_rows(mix, priors):
    print("\n3. rows never interact")
    pset = small_set(mix)
    rows = pset.rows(list(range(len(T.PROBLEMS))))
    n = len(rows)
    lr = torch.full((n,), 0.05, dtype=DT)
    one = torch.ones(n, dtype=DT)
    runs = {
        "rhso": lambda p, r, k: S.rhso(p, r, 2, 3, lr[:k], 0.2 * one[:k]),
        "rhso_suffix": lambda p, r, k: S.rhso(p, r, 2, 3, lr[:k], 0.2 * one[:k], terminal="suffix"),
        "dflow": lambda p, r, k: S.dflow(p, r, 4, lr[:k]),
        "pnp": lambda p, r, k: S.pnp(p, r, 3, 2.0 * one[:k], 0.5 * one[:k]),
        "mpc_delta_t": lambda p, r, k: S.mpc_delta_t(p, r, 3, 4, 30.0 * one[:k], lr[:k]),
        "mpc_rhc": lambda p, r, k: S.mpc_rhc(p, r, 2, 4, 3, 30.0 * one[:k], lr[:k]),
        "flower": lambda p, r, k: S.flower(p, r, 4),
    }
    idx = torch.tensor([0, 7, 13, n - 1])
    worst = 0.0
    for name, fn in runs.items():
        for prior in priors.values():
            if name == "rhso_suffix" and prior.family == "meanflow":
                continue
            full, _ = fn(prior, rows, n)
            part, _ = fn(prior, rows.select(idx), len(idx))
            worst = max(worst, float((full[idx] - part).abs().max()))
    report("every solver: a row alone = the row in a batch", worst < 1e-12,
           "7 solvers x 2 priors, rows from all 5 problem types; max difference %.1e" % worst)
    sub = rows.select(idx)
    a = rows.noise.draw(rows, "flower", 5)[idx]
    b = rows.noise.draw(sub, "flower", 5)
    c = rows.noise.draw(rows, "pnp", 5)[idx]
    report("solver noise depends on (problem, seed, iteration) only",
           torch.equal(a, b) and not torch.equal(a, c), "and differs between streams")


def check_costs(mix, priors):
    print("\n4. cost accounting")
    from src.rhso import rhso_cost_estimate
    from src.utils import MEANFLOW, STANDARD_FLOW
    pset = small_set(mix)
    rows = pset.rows([0])
    n = len(rows)
    lr, one = torch.full((n,), 0.05, dtype=DT), torch.ones(n, dtype=DT)
    fm, mf = priors["flow_matching"], priors["meanflow"]
    ok, details = True, []
    for N, M in ((2, 3), (4, 2)):
        for prior, fam, mode in ((mf, MEANFLOW, "direct"), (fm, STANDARD_FLOW, "direct"),
                                 (fm, STANDARD_FLOW, "suffix")):
            _, c = S.rhso(prior, rows, N, M, lr, 0 * one, terminal=mode)
            ref = rhso_cost_estimate(dict(num_rhso_steps=N, num_opt_steps=M, solver="heun",
                                          rhso_terminal_mode=mode), fam)
            ok &= (c.evals, c.backprops) == (ref["model_evals"], ref["backprops"])
    details.append("RHSO = src.rhso.rhso_cost_estimate")
    expect = []
    N, C, K, steps = 3, 4, 3, 5
    for prior in (fm, mf):
        meanflow = prior.family == "meanflow"
        expect += [
            (S.dflow(prior, rows, steps, lr)[1], (steps + 1, steps)),
            (S.pnp(prior, rows, steps, one, one)[1], (steps + 1, 0)),
            (S.mpc_delta_t(prior, rows, N, C, one, lr)[1], (N + (N - 1) * C, (N - 1) * C)),
            (S.mpc_rhc(prior, rows, N, C, K, one, lr)[1],
             (N * (1 + C * (K - 1) + (1 if meanflow else 0)), N * C * (K - 1))),
            (S.mpc_rhc(prior, rows, N, C, 1, one, lr)[1], (N * (2 if meanflow else 1), 0)),
            (S.flower(prior, rows, steps)[1], (steps, 0)),
        ]
    ok2 = all((c.evals, c.backprops) == e for c, e in expect)
    report("RHSO equals the image code's cost model", ok, "MeanFlow, FM direct, FM Heun suffix; (N, M) = (2, 3), (4, 2)")
    report("D-Flow, PnP, MPC-dt, MPC-RHC, Flower counts", ok2,
           "evals / backprops as src/dflow, pnp, mpc, flower count them")


def check_flower_dflow(mix, priors):
    print("\n5. Flower's refinement and degenerate one-step D-Flow")
    pset = small_set(mix)
    rows = pset.rows([1])                                   # denoising, sigma 0.2
    fm = priors["flow_matching"]
    x, _ = S.flower(fm, rows, 1)                            # t = 0: nu = 1, x_hat = E[x1]
    xhat = fm.endpoint(rows.x0, 0.0)
    s2 = rows.sigma[:, None] ** 2
    ref = (xhat + rows.y / s2) / (1 + 1 / s2)
    err = float((x - ref).abs().max())
    report("Flower step = (x_hat / nu^2 + y / sigma^2) / (1 / nu^2 + 1 / sigma^2)", err < 1e-12,
           "denoising, max difference %.1e" % err)
    out, _ = S.dflow(fm, rows, 20, torch.full((len(rows),), 0.1, dtype=DT))
    gap = float((out - torch.tensor(mix.mean())).abs().max())
    report("one-step Euler D-Flow on the exact v returns E[x1]", gap < 1e-12,
           "x + v(x, 0) = E[x1] for every x, so its gradient is 0 (max gap %.1e)" % gap)


def main():
    mix = G.make_mixture()
    tm = O.TorchMixture(mix, "cpu", DT)
    priors = {"flow_matching": O.FlowOracle(tm), "meanflow": O.MeanFlowOracle(tm, O.RK4_STEPS)}
    check_oracle(mix, tm)
    check_adam()
    check_rows(mix, priors)
    check_costs(mix, priors)
    check_flower_dflow(mix, priors)
    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
