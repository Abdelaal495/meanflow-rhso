"""Gaussian-mixture study (rebuttal Task 4): the exact quantities every solver is scored on.

Checks, with no trained model involved:

1. the mixture has the stated structure (8 equal weights, d = 16, scales 0.5 -> 0.02 as a
   power law, centred, average variance 1) and its samples have the analytic moments;
2. the closed-form posterior matches IMPORTANCE SAMPLING of prior x likelihood -- log
   evidence, posterior mean, covariance and component probabilities, each within Monte
   Carlo error -- for denoising, masking and the random projection; the proposal is an
   EQUAL-weight mixture of inflated posterior components, so the closed-form weights are
   tested rather than assumed, and a deliberately wrong posterior (sigma doubled) fails;
3. the exact velocity equals (E[x1 | x_t] - x_t) / (1 - t) with E[x1 | x_t] the closed-form
   posterior of the observation x_t = t x1 + (1 - t) x0 (validated by 2), and has the
   known values at t = 0 (-x) and t = 1 (x);
4. the fine ODE solve of the exact velocity carries N(0, I) to the mixture (sliced W2 at the
   two-sample floor), and sliced W2 itself is right on a known Gaussian pair;
5. toy/common.py reproduces the pre-refactor two-moons trainers BIT FOR BIT (the MeanFlow
   trainer and the distillation trainer, against toy/rhso_two_moons.py at commit 0d1eaed);
6. the raw-error floor of the MeanFlow target: the per-row velocity equals the scalar one,
   the batch replay reproduces common.train_meanflow's own batches (a learning rate of 0
   keeps the model at its initialisation, so its logged raw error must be recomputed
   exactly), and the closed form |dPhi/dz (v - v*)|^2 equals the error of the exact average
   velocity against the target built by finite differences, as the trainer builds it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

import importlib
import math
import subprocess
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import gmm_study as G  # noqa: E402

RESULTS = []
PRE_REFACTOR_COMMIT = "0d1eaed"


def report(name, ok, detail=""):
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", name, detail))
    RESULTS.append(bool(ok))


def check_mixture(mix):
    print("\n1. the mixture")
    lam = mix.eig()[0]
    stds = np.sqrt(np.sort(lam, axis=1)[:, ::-1])
    s = G.component_scales()
    c = stds[0, 0] / s[0]
    report("structure", mix.k == 8 and mix.d == 16 and np.allclose(mix.weights, 1 / 8)
           and np.allclose(stds, c * s[None, :], rtol=1e-8)
           and abs(s[0] - 0.5) < 1e-12 and abs(s[-1] - 0.02) < 1e-12
           and np.allclose(np.diff(np.log(s)) / np.diff(np.log(np.arange(1, 17))),
                           np.log(s[-1] / s[0]) / np.log(16)),
           "8 x 1/8; per-component std %.3f..%.4f = %.3f x (0.5..0.02), an exact power law"
           % (stds.max(), stds.min(), c))
    report("centred, average variance 1",
           np.abs(mix.mean()).max() < 1e-12 and abs(np.trace(mix.covariance()) / 16 - 1) < 1e-12,
           "raw means have norm 2; after centring and scaling their norms are %.2f..%.2f"
           % (np.linalg.norm(mix.means, axis=1).min(), np.linalg.norm(mix.means, axis=1).max()))
    x, comp = mix.sample(400_000, np.random.default_rng(5))
    emp_cov = np.cov(x.T)
    report("samples have the analytic moments",
           np.abs(x.mean(0)).max() < 0.01 and np.abs(emp_cov - mix.covariance()).max() < 0.02
           and np.abs(np.bincount(comp) / len(comp) - 1 / 8).max() < 0.005,
           "max |mean| %.4f, max |cov - analytic| %.4f" % (np.abs(x.mean(0)).max(),
                                                           np.abs(emp_cov - mix.covariance()).max()))


def importance_check(mix, post_claimed, H, sigma, y, n, rng):
    """IS estimates of log Z, mean, covariance, component probabilities, with standard errors;
    returns the largest |estimate - claimed| / SE over all of them."""
    k = post_claimed.k
    # proposal: equal-weight mixture of the claimed posterior components, covariance x1.5
    # (std x1.22): heavier than the target in every direction, without the weight
    # degeneracy a wider proposal suffers in 16 dimensions
    prop = G.GaussianMixture(np.full(k, 1.0 / k), post_claimed.means,
                             1.5 * post_claimed.covs + 1e-8 * np.eye(mix.d))
    x, _ = prop.sample(n, rng)
    resid = y[None, :] - x @ H.T
    m = H.shape[0]
    log_lik = (-0.5 * (resid ** 2).sum(1) / sigma ** 2 - m * math.log(sigma)
               - 0.5 * m * math.log(2 * math.pi))
    log_w = mix.log_prob(x) + log_lik - prop.log_prob(x)
    shift = log_w.max()
    w = np.exp(log_w - shift)
    Z = w.mean()
    se_Z = w.std() / math.sqrt(n)
    log_Z = math.log(Z) + shift
    wn = w / w.sum()
    ess = 1.0 / (wn ** 2).sum()

    def est(f):                                         # self-normalised estimate and its SE
        mean = (wn[:, None] * f).sum(0)
        se = np.sqrt((wn[:, None] ** 2 * (f - mean) ** 2).sum(0))
        return mean, se

    z = [abs(log_Z - post_claimed.log_evidence) / (se_Z / Z)]
    mu, se = est(x)
    z += list(np.abs(mu - post_claimed.mean()) / se)
    centred = x - post_claimed.mean()
    second, se2 = est(np.einsum("ni,nj->nij", centred, centred).reshape(n, -1))
    z += list(np.abs(second - post_claimed.covariance().ravel()) / np.maximum(se2, 1e-300))
    resp = mix.responsibilities(x)                      # p(i | x) = p(i | x, y)
    r, se_r = est(resp)
    z += list(np.abs(r - post_claimed.weights) / np.maximum(se_r, 1e-12))
    return max(z), ess, log_Z


def check_posterior(mix):
    print("\n2. closed-form posterior vs importance sampling")
    rng = np.random.default_rng(11)
    for name, sigma in (("denoising", 0.2), ("masking", 0.05), ("projection", 0.05)):
        H = G.make_operator(name)
        x_star, _ = mix.sample(1, rng)
        y = H @ x_star[0] + sigma * rng.standard_normal(H.shape[0])
        post = G.posterior(mix, H, sigma, y)
        zmax, ess, log_Z = importance_check(mix, post, H, sigma, y, 300_000, rng)
        wrong = G.posterior(mix, H, 2 * sigma, y)
        wrong.log_evidence = post.log_evidence           # only the distribution is wrong
        zwrong, _, _ = importance_check(mix, wrong, H, sigma, y, 300_000, rng)
        report("%s (sigma=%g)" % (name, sigma), zmax < 5.0 and zwrong > 10.0,
               "max |IS - closed form| / SE = %.2f over log Z, mean, cov, weights (ESS %.0f); "
               "a sigma-doubled posterior scores %.0f" % (zmax, ess, zwrong))


def check_velocity(mix):
    print("\n3. the exact velocity")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 16)) * 1.3
    worst = 0.0
    for t in (0.1, 0.5, 0.9):
        v = G.exact_velocity(mix, x, t)
        ref = np.stack([(G.posterior(mix, t * np.eye(16), 1.0 - t, xi).mean() - xi) / (1.0 - t)
                        for xi in x])
        worst = max(worst, np.abs(v - ref).max() / np.abs(ref).max())
    report("v = (E[x1 | x_t] - x_t) / (1 - t)", worst < 1e-9,
           "max relative difference %.1e at t = 0.1, 0.5, 0.9" % worst)
    v0, v1 = G.exact_velocity(mix, x, 0.0), G.exact_velocity(mix, x, 1.0)
    report("endpoints: v(x, 0) = -x, v(x, 1) = x",
           np.abs(v0 + x).max() < 1e-12 and np.abs(v1 - x).max() < 1e-9 and np.isfinite(v1).all(),
           "finite on all of [0, 1]")


def check_flow(mix):
    print("\n4. the exact flow and sliced W2")
    rng = np.random.default_rng(9)
    mu = np.zeros(16); mu[0] = 2.0
    a = rng.standard_normal((20_000, 16)); b = rng.standard_normal((20_000, 16)) + mu
    sw = G.sliced_w2(a, b, n_proj=2000)
    report("sliced W2 on N(0, I) vs N(2 e1, I)", abs(sw - 2.0 / 4.0) < 0.02,
           "%.4f, exact |mu| / sqrt(d) = 0.5" % sw)
    n = 4000
    truth, _ = mix.sample(n, rng)
    truth2, _ = mix.sample(n, rng)
    x1 = G.exact_flow(mix, rng.standard_normal((n, 16)), steps=200)
    floor, sw = G.sliced_w2(truth, truth2), G.sliced_w2(x1, truth)
    report("exact flow carries N(0, I) to the mixture", sw < 1.3 * floor,
           "SW2 %.4f vs two-sample floor %.4f (n = %d)" % (sw, floor, n))


def check_refactor():
    print("\n5. toy/common.py reproduces the pre-refactor two-moons trainers")
    import torch
    with tempfile.TemporaryDirectory() as tmp:
        src = subprocess.run(["git", "show", "%s:toy/rhso_two_moons.py" % PRE_REFACTOR_COMMIT],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        (Path(tmp) / "pre_refactor_two_moons.py").write_text(src, encoding="utf-8")
        sys.path.insert(0, tmp)
        try:
            old = importlib.import_module("pre_refactor_two_moons")
            new = importlib.import_module("rhso_two_moons")

            def run(mod, fn):
                torch.manual_seed(0)
                return fn(mod, mod.MeanFlowMLP()).state_dict()

            for label, fn in (("MeanFlow trainer", lambda m, net: m.train(net, 60, 64, 3, verbose=False)),
                              ("distillation trainer", lambda m, net: m.train_distilled(
                                  net, 30, 64, 3, verbose=False, distill_steps=30))):
                a, b = run(old, fn), run(new, fn)
                report("%s bit for bit" % label,
                       list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a),
                       "%d tensors, same names (cached models still load)" % len(a))
        finally:
            sys.path.remove(tmp)


def check_floor(mix):
    print("\n6. the raw-error floor of the MeanFlow target")
    import torch
    from common import MeanFlowMLP, train_meanflow
    rng = np.random.default_rng(21)
    x = rng.standard_normal((64, 16)) * 1.3
    worst = max(np.abs(G.exact_velocity_rows(mix, x, np.full(64, t)) - G.exact_velocity(mix, x, t)).max()
                for t in (0.0, 0.3, 0.97, 1.0))
    report("per-row exact velocity = scalar one", worst < 1e-12, "max difference %.1e" % worst)

    torch.manual_seed(4)
    model = MeanFlowMLP(dim=16, width=32, depth=2)
    log = []
    train_meanflow(model, lambda r: mix.sample(G.TRAIN_POOL, r)[0].astype(np.float32), steps=9,
                   batch=64, seed=0, long_share=0.5, lr=0.0, verbose=False, log_every=1, log=log)
    batches = G.replay_meanflow_batches(mix, 9, 64, 0.5, "cpu", range(1, 10), width=32, depth=2)
    diffs, n_long = [], 0
    for entry in log:
        x1, x0, t, s = (torch.from_numpy(a.astype(np.float32)) for a in batches[entry["step"]])
        n_long += int((s != t).sum())
        z = (1 - t)[:, None] * x0 + t[:, None] * x1
        v = x1 - x0
        u, dudt = torch.func.jvp(lambda z_, t_, s_: model(z_, t_, s_), (z, t, s),
                                 (v, torch.ones_like(t), torch.zeros_like(s)))
        err = ((u - (v + (s - t)[:, None] * dudt)) ** 2).sum(1).mean().item()
        diffs.append(abs(err - entry["raw_mse"]) / entry["raw_mse"])
    report("batch replay = the trainer's batches", len(log) == 9 and max(diffs) < 1e-6 and n_long > 0,
           "9 steps, %d long pairs; logged raw error recomputed to %.1e" % (n_long, max(diffs)))

    n = 24
    x1, _ = mix.sample(n, rng)
    x0 = rng.standard_normal((n, 16))
    t = rng.uniform(0.1, 0.6, n)
    s = t + rng.uniform(0.2, 1.0, n) * (1 - t)
    z = (1 - t)[:, None] * x0 + t[:, None] * x1
    v = x1 - x0
    closed = G.meanflow_error_floor(mix, z, v, t, s)
    d = 1e-5                                    # the trainer's total derivative, tangent (v, 1, 0)
    dudt = (G.exact_average_velocity(mix, z + d * v, t + d, s)
            - G.exact_average_velocity(mix, z - d * v, t - d, s)) / (2 * d)
    direct = ((G.exact_average_velocity(mix, z, t, s) - (v + (s - t)[:, None] * dudt)) ** 2).sum(1)
    rel = np.abs(closed - direct).max() / direct.max()
    coarse = G.meanflow_error_floor(mix, z, v, t, s, steps=200)
    conv = np.abs(coarse - closed).max() / closed.max()
    report("floor = |dPhi/dz (v - v*)|^2", rel < 1e-6 and conv < 1e-4,
           "max rel. difference to the finite-difference target %.1e; RK4 200 vs 400 steps %.1e"
           % (rel, conv))


def main():
    mix = G.make_mixture()
    check_mixture(mix)
    check_posterior(mix)
    check_velocity(mix)
    check_flow(mix)
    check_refactor()
    check_floor(mix)
    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
