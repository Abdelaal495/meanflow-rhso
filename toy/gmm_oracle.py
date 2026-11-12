"""Oracle priors for the Gaussian-mixture study (rebuttal Task 4, primary study).

No training: both priors are the EXACT transport of the mixture in toy/gmm_study.py, in
torch, batched, differentiable and on any device.

    FlowOracle     the exact velocity v(x, t) = E[x1 - x0 | x_t = x]  (the flow-matching
                   interface); its direct endpoint estimate x + (1 - t) v(x, t) is exactly
                   E[x1 | x_t = x]; and a fixed-step solve over a grid (the integrated
                   suffix of the image code's flow RHSO).
    MeanFlowOracle the exact finite-interval map Psi(x; t -> s) of the same flow (the
                   MeanFlow interface), computed by a fixed-step RK4 solve of v with
                   `rk4_steps` steps over [t, s], whatever the interval.  Autograd
                   differentiates the discrete solve exactly.

Cost accounting follows the image code.  One call of `velocity` is one model evaluation of
the flow-matching prior; one call of `transition` is ONE evaluation of the MeanFlow prior,
although it runs 4 x rk4_steps velocity evaluations internally -- those are counted
separately (`internal_velocity_evals`) and reported, never added to the evaluation count.

Time convention: t = 0 noise, t = 1 data, x_t = (1 - t) x0 + t x1.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


class TorchMixture:
    """The mixture's means and per-component eigendecompositions as torch tensors."""

    def __init__(self, mix, device="cpu", dtype=torch.float64):
        lam, vec = mix.eig()
        self.device, self.dtype = torch.device(device), dtype
        self.k, self.d = mix.k, mix.d
        as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)
        self.log_w = as_t(np.log(mix.weights))                  # (K,)
        self.means = as_t(mix.means)                            # (K, d)
        self.lam = as_t(lam)                                    # (K, d)
        self.vec = as_t(vec)                                    # (K, d, d), columns = eigvecs
        self.mean_coords = torch.einsum("kd,kde->ke", self.means, self.vec)   # V_i^T m_i
        # all K eigenbases side by side, so every per-component product is ONE matmul:
        # to_coords (d, K*d): x -> [V_1^T x, ..., V_K^T x];  from_coords (K*d, d): back
        self.to_coords = self.vec.permute(1, 0, 2).reshape(self.d, self.k * self.d)
        self.from_coords = self.vec.permute(0, 2, 1).reshape(self.k * self.d, self.d)

    def velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """v(x, t) for a batch x (n, d) at one time t (see gmm_study.exact_velocity):
        sum_i r_i [m_i + V_i diag(g_i) V_i^T (x - t m_i)]."""
        t = float(t)
        n = x.shape[0]
        z = (x @ self.to_coords).view(n, self.k, self.d) - t * self.mean_coords   # (n, K, d)
        denom = t * t * self.lam + (1.0 - t) ** 2                                  # (K, d)
        logits = (self.log_w - 0.5 * (z * z / denom).sum(-1)
                  - 0.5 * torch.log(denom).sum(-1))                               # (n, K)
        r = torch.softmax(logits, dim=1)
        gain = (t * self.lam - (1.0 - t)) / denom                                  # (K, d)
        weighted = (r.unsqueeze(-1) * gain * z).reshape(n, self.k * self.d)
        return r @ self.means + weighted @ self.from_coords


class FlowOracle:
    """The flow-matching interface: v, the direct endpoint and fixed-step solves."""

    family = "flow_matching"

    def __init__(self, tm: TorchMixture):
        self.tm = tm

    def velocity(self, x, t):
        return self.tm.velocity(x, t)

    def endpoint(self, x, t):
        """x + (1 - t) v(x, t) = E[x1 | x_t = x], exactly.  One evaluation."""
        return x + (1.0 - float(t)) * self.tm.velocity(x, t)

    def step(self, x, t, s, solver: str = "heun"):
        """One solver step over [t, s] (the image code's flow_step, no final-step policy:
        the exact velocity is finite at t = 1).  Euler 1 evaluation, Heun 2."""
        dt = float(s) - float(t)
        v1 = self.tm.velocity(x, t)
        if solver == "euler":
            return x + dt * v1
        v2 = self.tm.velocity(x + dt * v1, s)
        return x + 0.5 * dt * (v1 + v2)

    @staticmethod
    def step_evals(solver: str = "heun") -> int:
        return 1 if solver == "euler" else 2

    def integrate(self, x, grid: Sequence[float], solver: str = "heun"):
        for a, b in zip(grid[:-1], grid[1:]):
            x = self.step(x, a, b, solver)
        return x


class MeanFlowOracle:
    """The MeanFlow interface: Psi(x; t -> s) by `rk4_steps` RK4 steps of the exact v."""

    family = "meanflow"

    def __init__(self, tm: TorchMixture, rk4_steps: int):
        self.tm, self.rk4_steps = tm, int(rk4_steps)

    @property
    def internal_velocity_evals(self) -> int:
        """Velocity evaluations inside ONE map evaluation (reported, not counted as evals)."""
        return 4 * self.rk4_steps

    def transition(self, x, t, s):
        t, s = float(t), float(s)
        if s == t:
            return x
        h = (s - t) / self.rk4_steps
        v = self.tm.velocity
        for k in range(self.rk4_steps):
            a = t + k * h
            k1 = v(x, a)
            k2 = v(x + 0.5 * h * k1, a + 0.5 * h)
            k3 = v(x + 0.5 * h * k2, a + 0.5 * h)
            k4 = v(x + h * k3, a + h)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


# =====================================================================================
# Choosing rk4_steps: the map against a 1000-step reference
# =====================================================================================
CHECK_STARTS = (0.0, 0.25, 0.5, 0.75)
CHECK_TOLERANCE = 1e-3


def rk4_step_check(mix, candidates=(1, 2, 4, 8, 16, 32, 64), n: int = 4096,
                   reference_steps: int = 1000, device="cpu", seed: int = 7):
    """Relative RMSE of Psi_n(x; t -> 1) against Psi_1000(x; t -> 1) for x the exact flow's
    state at t (from N(0, I) noise), t in CHECK_STARTS.  Relative RMSE =
    sqrt(mean |Psi_n - Psi_ref|^2) / sqrt(mean |Psi_ref|^2).  Returns
    {"rows": [...], "chosen": smallest n within CHECK_TOLERANCE at every start}."""
    tm = TorchMixture(mix, device=device)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(rng.standard_normal((n, mix.d)), dtype=tm.dtype, device=tm.device)
    starts = {}
    ref_map = MeanFlowOracle(tm, reference_steps)
    prev = 0.0
    with torch.no_grad():
        for t in CHECK_STARTS:                 # exact states at each start, on one trajectory
            if t > prev:
                x = MeanFlowOracle(tm, int(round(reference_steps * (t - prev)))).transition(x, prev, t)
            starts[t] = x
            prev = t
        refs = {t: ref_map.transition(starts[t], t, 1.0) for t in CHECK_STARTS}
        rows = []
        for steps in candidates:
            m = MeanFlowOracle(tm, steps)
            errs = {}
            for t in CHECK_STARTS:
                out = m.transition(starts[t], t, 1.0)
                errs[t] = float(torch.sqrt(((out - refs[t]) ** 2).mean())
                                / torch.sqrt((refs[t] ** 2).mean()))
            rows.append(dict(rk4_steps=steps, rel_rmse={str(t): e for t, e in errs.items()},
                             worst=max(errs.values())))
    passing = [r["rk4_steps"] for r in rows if r["worst"] <= CHECK_TOLERANCE]
    return dict(rows=rows, chosen=min(passing) if passing else None, tolerance=CHECK_TOLERANCE,
                reference_steps=reference_steps, samples=n, starts=list(CHECK_STARTS))


# Chosen by rk4_step_check (python toy/gmm_oracle.py): the smallest step count whose map is
# within 1e-3 relative RMSE of the 1000-step reference from t = 0, 0.25, 0.5 and 0.75 to 1.
RK4_STEPS = 16


def main():
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gmm_study as G
    result = rk4_step_check(G.make_mixture(), device=G.default_device())
    print("relative RMSE of Psi_n(x; t -> 1) vs %d RK4 steps (%d samples, x = exact state at t)"
          % (result["reference_steps"], result["samples"]))
    print("  %-9s" % "steps" + "".join("%12s" % ("t = %g" % t) for t in CHECK_STARTS) + "%12s" % "worst")
    for r in result["rows"]:
        print("  %-9d" % r["rk4_steps"] + "".join("%12.2e" % r["rel_rmse"][str(t)] for t in CHECK_STARTS)
              + "%12.2e" % r["worst"])
    print("smallest within %g at every start: %s (RK4_STEPS = %d)"
          % (result["tolerance"], result["chosen"], RK4_STEPS))
    G.OUT.mkdir(parents=True, exist_ok=True)
    with open(G.OUT / "oracle_rk4_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rk4_steps"] + ["rel_rmse_from_t%g" % t for t in CHECK_STARTS] + ["worst", "chosen"])
        for r in result["rows"]:
            w.writerow([r["rk4_steps"]] + ["%.3e" % r["rel_rmse"][str(t)] for t in CHECK_STARTS]
                       + ["%.3e" % r["worst"], int(r["rk4_steps"] == RK4_STEPS)])
    if result["chosen"] != RK4_STEPS:
        raise SystemExit("RK4_STEPS = %d disagrees with the check (%s)" % (RK4_STEPS, result["chosen"]))


if __name__ == "__main__":
    main()
