"""Execute Flower (src/flower.py) against synthetic adapters, and its configurations.

No checkpoint is involved.  The "models" are analytic:

    ToyMeanFlow   (JAX, BHWC, pixel)   T(x; a -> b) = x + (b - a) f(x),  f = 0.7 (target - x) + 0.2 sin 3x
    ToyJiT        (torch, BCHW, pixel) v(x, s) = f(x), so x + (1 - t) v == T(x; t -> 1) above

so both families produce the SAME destination, and the torch path (with its BCHW <-> BHWC
layout permutation inside H) can be checked against the JAX one.  A latent twin of the
MeanFlow toy checks the rejection.

Covered: nu_t; the closed-form refinement against conjugate gradient (denoising, both
inpaintings, strided SR) and against the plan's textbook formula; CG on the reflect-padded
blur against a dense float64 solve, and the exactness of the vjp adjoint; the gamma = 1
sample's covariance; Eval./BP counts (N and 0), the H/H^T and CG counters and the planner's
cost model; determinism, noise independent of num_flower_steps, batch independence; the
last iteration returning x_tilde; the rejections (latent adapter, stroke_painting, sigma = 0,
SiT/iMF, bad gamma); job ids of every committed configuration unchanged; resume of
pre-Flower artefacts; the generated screen / smoke configurations; hpo.py's flower knob,
`--lock` and `finalize`; and make_flower_configs.py freeze on a planted screen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib
import csv
import dataclasses
import hashlib
import importlib.util
import io
import math
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

from spec_support import make_spec
from src.models.base import AdapterSpec, Conditioning, MeanFlowAdapter, StandardFlowAdapter
from src.problems import InverseProblem
from src.utils import (FLOW_ASCENDING, MEANFLOW, MEANFLOW_DESCENDING, NUMPY_BACKEND,
                       STANDARD_FLOW, gaussian_kernel_1d)

REPO = Path(__file__).resolve().parents[1]
RES = 16
BATCH = 2
IDS = ("img0", "img1")


# =====================================================================================
# Synthetic adapters
# =====================================================================================
def _drift(x, target, sin):
    return 0.7 * (target - x) + 0.2 * sin(3.0 * x)


class ToyMeanFlow(MeanFlowAdapter):
    def __init__(self, target, state_space="pixel", name="pmf"):
        super().__init__(name, {})
        self.target = jnp.asarray(target)
        self.spec = AdapterSpec(
            name=name, display_name=name, dynamics_family=MEANFLOW, framework="jax",
            state_space=state_space, native_shape=(RES, RES, 3), layout="BHWC",
            pixel_resolution=RES, prediction_kind="mean_velocity",
            native_time_mapping=MEANFLOW_DESCENDING, batch_size=BATCH,
            fixed_batch_shape=True, num_classes=1000, null_label=1000, guidance={},
            checkpoint={})

    def to_native_noise(self, noise):
        return jnp.asarray(noise, jnp.float32)

    def encode_pixels(self, pixels):
        return jnp.asarray(pixels, jnp.float32)

    def to_pixels(self, state, differentiable=False):
        return state if differentiable else np.asarray(jax.device_get(state), np.float32)

    def _lerp(self, guide, noise, keep, add):
        return keep * guide + add * noise

    def transition(self, state, s_from, s_to, conditioning):
        self.count_forwards(1)
        n = state.shape[0]
        return state + (float(s_to) - float(s_from)) * _drift(state, self.target[:n], jnp.sin)


class ToyJiT(StandardFlowAdapter):
    """Torch, native BCHW: H = apply(permute(x)) exercises the layout path."""

    def __init__(self, target):
        import torch
        super().__init__("jit", {})
        self.torch = torch
        self.device, self.integration_dtype = torch.device("cpu"), torch.float32
        self.target = torch.from_numpy(np.ascontiguousarray(
            np.asarray(target, np.float32).transpose(0, 3, 1, 2)))
        self.spec = AdapterSpec(
            name="jit", display_name="jit", dynamics_family=STANDARD_FLOW, framework="torch",
            state_space="pixel", native_shape=(RES, RES, 3), layout="BCHW",
            pixel_resolution=RES, prediction_kind="clean",
            native_time_mapping=FLOW_ASCENDING, batch_size=BATCH, fixed_batch_shape=False,
            num_classes=1000, null_label=1000, guidance={}, checkpoint={})

    def to_native_noise(self, noise):
        return self.torch.from_numpy(np.ascontiguousarray(
            np.asarray(noise, np.float32).transpose(0, 3, 1, 2)))

    def encode_pixels(self, pixels):
        return self.to_native_noise(pixels)

    def to_pixels(self, state, differentiable=False):
        if differentiable:
            return state.permute(0, 2, 3, 1)
        return np.ascontiguousarray(state.detach().numpy().transpose(0, 2, 3, 1))

    def _lerp(self, guide, noise, keep, add):
        return keep * guide + add * noise

    def velocity(self, state, s, conditioning):
        self.count_forwards(1)
        return _drift(state, self.target[:state.shape[0]], self.torch.sin)


# =====================================================================================
# Problems built like src/problems.build_problem, at 16x16
# =====================================================================================
PARAMS = {"denoising": {}, "box_inpaint": {"box": 6}, "random_inpaint": {"missing_fraction": 0.7},
          "super_resolution": {"factor": 2},
          "deblur": {"blur_sigma": 1.0, "kernel_size": 7, "padding": "reflect"}}
SIGMA = {"denoising": 0.2, "box_inpaint": 0.05, "random_inpaint": 0.01,
         "super_resolution": 0.05, "deblur": 0.05}


def make_problem(name, sigma=None, n=BATCH):
    rng = np.random.default_rng(7)
    gt = (rng.standard_normal((n, RES, RES, 3)) * 0.3).astype(np.float32)
    mask = None
    if name == "box_inpaint":
        mask = np.ones((n, RES, RES, 1), np.float32)
        mask[:, 5:11, 5:11, :] = 0.0
    elif name == "random_inpaint":
        mask = (rng.random((n, RES, RES, 1)) >= 0.7).astype(np.float32)
    sigma = SIGMA.get(name, 0.05) if sigma is None else sigma
    p = InverseProblem(name=name, key="k_" + name, sigma=sigma, params=dict(PARAMS.get(name, {})),
                       ground_truth=gt, measurement=np.zeros((n, 1, 1, 3), np.float32),
                       display_measurement=gt, initialization_guide=gt, guide_mode="identity",
                       mask=mask, image_ids=IDS[:n] if n <= 2 else tuple("img%d" % i for i in range(n)))
    clean = np.asarray(p.apply(gt, NUMPY_BACKEND), np.float32)
    noise = rng.standard_normal(clean.shape).astype(np.float32) * sigma
    if mask is not None:
        noise = noise * mask
    p.measurement = (clean + noise).astype(np.float32)
    p._cache.clear()
    return p


def report(name, ok, detail):
    print("  [%s] %-44s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def flower_spec(model="pmf", **over):
    """A real resolved Flower spec; t0 = 1 (pure noise start), as in the screen."""
    over.setdefault("t0", 1.0)
    return make_spec("flower", model=model, **over)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =====================================================================================
def check_math(results):
    from src.flower import (FlowerRefinement, flower_nu, flower_time_grid, make_ops)
    print("\nnu_t, the time grid and the refinement")

    nus = [flower_nu(0.0), flower_nu(0.5), flower_nu(1.0 - 1e-6)]
    results.append(report("nu_t at 0, 0.5, 1-1e-6",
                          abs(nus[0] - 1.0) < 1e-12 and abs(nus[1] - 1 / math.sqrt(2)) < 1e-12
                          and 0 < nus[2] < 1.01e-6,
                          "%.6f, %.6f (1/sqrt 2), %.3e" % tuple(nus)))
    g = flower_time_grid(0.0, 50)
    gb = flower_time_grid(0.2, 8, 2.0)
    results.append(report("grid: t_0 = s0, t_N = 1 exactly, beta honoured",
                          len(g) == 51 and g[0] == 0.0 and g[-1] == 1.0 and gb[0] == 0.2
                          and gb[-1] == 1.0 and abs(gb[1] - (0.2 + 0.8 / 64)) < 1e-12
                          and all(flower_nu(a) > flower_nu(b) for a, b in zip(g[:-1], g[1:])),
                          "51 times on [0, 1]; beta=2 gives t_1 = %.4f; nu strictly decreasing"
                          % gb[1]))

    rng = np.random.default_rng(0)
    for name in ("denoising", "box_inpaint", "random_inpaint", "super_resolution"):
        problem = make_problem(name)
        adapter = ToyMeanFlow(problem.ground_truth)
        ops = make_ops(adapter, problem)
        x_hat = jnp.asarray(rng.standard_normal((BATCH, RES, RES, 3)).astype(np.float32))
        e1 = jnp.asarray(rng.standard_normal(x_hat.shape).astype(np.float32))
        e2 = jnp.asarray(rng.standard_normal(problem.measurement.shape).astype(np.float32))
        closed = FlowerRefinement(ops, problem, x_hat, solver="closed_form")
        cg = FlowerRefinement(ops, problem, x_hat, cg_max_iters=500, cg_tol=1e-7, solver="cg")
        worst = 0.0
        for t in (0.0, 0.3, 0.9, 0.995):
            for gamma in (0, 1):
                a = closed(x_hat, t, gamma, e1, e2)
                b = cg(x_hat, t, gamma, e1, e2)
                worst = max(worst, rel(a, b))
        # the plan's unscaled textbook formula, elementwise, in float64
        t = 0.3
        nu, s = flower_nu(t), problem.sigma
        hty = np.asarray(closed.Hty, np.float64)
        d = np.asarray(closed.diag, np.float64)
        textbook = ((nu ** -2) * np.asarray(x_hat, np.float64) + (s ** -2) * hty) / \
            (nu ** -2 + (s ** -2) * d)
        tb = rel(closed(x_hat, t), textbook)
        dvals = sorted(set(np.round(d.ravel(), 6)))
        results.append(report("closed form == CG: %s" % name, worst < 1e-5 and tb < 1e-6,
                              "max rel diff %.1e over t, gamma; vs textbook mu_t %.1e; "
                              "diag(H^T H) values %s" % (worst, tb, dvals)))


def dense_operator(problem):
    """H as a dense (m, n) float64 matrix, from the NumPy operator on basis images."""
    n = RES * RES * 3
    basis = np.eye(n, dtype=np.float32).reshape(n, RES, RES, 3)
    single = dataclasses.replace(problem, _cache={})
    cols = np.asarray(single.apply(basis, NUMPY_BACKEND), np.float64).reshape(n, -1)
    return cols.T


def check_cg(results):
    from src.flower import FlowerRefinement, conjugate_gradient, flower_nu, make_ops
    print("\nConjugate gradient on the blur")
    problem = make_problem("deblur")
    adapter = ToyMeanFlow(problem.ground_truth)
    ops = make_ops(adapter, problem)
    rng = np.random.default_rng(1)
    x_hat = jnp.asarray(rng.standard_normal((BATCH, RES, RES, 3)).astype(np.float32))

    H = dense_operator(problem)
    v = rng.standard_normal((BATCH,) + problem.measurement.shape[1:]).astype(np.float32)
    lhs = np.sum(np.asarray(ops.H(x_hat), np.float64) * v)
    rhs = np.sum(np.asarray(x_hat, np.float64) * np.asarray(ops.Ht(jnp.asarray(v), x_hat)))
    sym = float(np.abs(H - H.T).max())
    results.append(report("vjp adjoint is exact on the reflect blur",
                          abs(lhs - rhs) / abs(lhs) < 1e-5 and sym > 1e-3,
                          "<Hx, v> = %.6f, <x, H^T v> = %.6f; H itself is NOT symmetric "
                          "(max |H - H^T| = %.3f), so H^T is not H" % (lhs, rhs, sym)))

    def dense(t, b):
        nu, s = flower_nu(t), problem.sigma
        A = (nu ** -2) * np.eye(H.shape[1]) + (s ** -2) * H.T @ H
        y = problem.measurement[b].reshape(-1).astype(np.float64)
        return np.linalg.solve(A, (nu ** -2) * np.asarray(x_hat[b], np.float64).ravel()
                               + (s ** -2) * H.T @ y)

    # (a) given room, CG reaches the paper's tolerance and the textbook solution
    roomy = FlowerRefinement(ops, problem, x_hat, cg_max_iters=500)
    worst_res, worst_err, iters = 0.0, 0.0, []
    for t in (0.0, 0.5, 0.95):
        out = np.asarray(roomy(x_hat, t), np.float64)
        iters.append(roomy.last_iterations)
        worst_res = max(worst_res, roomy.last_residual)
        worst_err = max(worst_err, max(rel(out[b].ravel(), dense(t, b)) for b in range(BATCH)))
    results.append(report("CG converges on the blur (tol 1e-5)",
                          worst_res <= 1e-5 and worst_err < 1e-4,
                          "iterations %s at t = 0, 0.5, 0.95; worst relative residual %.1e; "
                          "vs dense float64 solve of the textbook system %.1e"
                          % (iters, worst_res, worst_err)))

    # (b) at the paper's cap of 50, the early, ill-conditioned solves (cond ~ 1 + rho, with
    # rho = nu^2 / sigma^2 = 400 at t = 0) stop at the cap: that is the paper's setting,
    # reported rather than hidden; the late solves converge well inside it
    refine = FlowerRefinement(ops, problem, x_hat)            # paper settings: 50, 1e-5
    capped = []
    for t in (0.0, 0.95):
        out = np.asarray(refine(x_hat, t), np.float64)
        start = max(rel(np.asarray(x_hat[b], np.float64).ravel(), dense(t, b)) for b in range(BATCH))
        err = max(rel(out[b].ravel(), dense(t, b)) for b in range(BATCH))
        capped.append((refine.last_iterations, refine.last_residual, start, err))
    results.append(report("paper cap: 50 iterations bind only early",
                          capped[0][0] == 50 and capped[0][1] < 1e-3
                          and capped[0][3] < 0.01 * capped[0][2]
                          and capped[1][0] < 50 and capped[1][1] <= 1e-5,
                          "t=0: %d iters, residual %.1e, error %.1e (warm start %.1e); "
                          "t=0.95: %d iters, residual %.1e"
                          % (capped[0][0], capped[0][1], capped[0][3], capped[0][2],
                             capped[1][0], capped[1][1])))

    # per-image CG: an image's solve does not depend on its batch partner
    one = make_problem("deblur").subset([0])
    ops1 = make_ops(ToyMeanFlow(one.ground_truth), one)
    alone = np.asarray(FlowerRefinement(ops1, one, x_hat[:1])(x_hat[:1], 0.0))
    paired = np.asarray(refine(x_hat, 0.0))[:1]
    results.append(report("CG is per image (batch independent)", rel(alone, paired) < 1e-6,
                          "image 0 alone vs in a batch of 2: rel diff %.1e" % rel(alone, paired)))

    # gamma = 1: the extra term is A^-1 (nu e1 + nu^2/sigma H^T e2); its covariance is Sigma_t
    p = make_problem("random_inpaint", n=2)
    ops_p = make_ops(ToyMeanFlow(p.ground_truth), p)
    ref_p = FlowerRefinement(ops_p, p, x_hat)
    t, nu, s = 0.4, flower_nu(0.4), p.sigma
    draws = []
    for i in range(40):
        r = np.random.default_rng(100 + i)
        e1 = jnp.asarray(r.standard_normal(x_hat.shape).astype(np.float32))
        e2 = jnp.asarray(r.standard_normal(p.measurement.shape).astype(np.float32))
        draws.append(np.asarray(ref_p(x_hat, t, 1, e1, e2) - ref_p(x_hat, t, 0)))
    draws = np.stack(draws)
    m = np.broadcast_to(p.mask, x_hat.shape).astype(bool)
    var_obs, var_mis = float(draws[:, m].var()), float(draws[:, ~m].var())
    want_obs, want_mis = 1.0 / (nu ** -2 + s ** -2), nu ** 2
    results.append(report("gamma = 1 sample has covariance Sigma_t",
                          abs(var_obs / want_obs - 1) < 0.05 and abs(var_mis / want_mis - 1) < 0.05,
                          "observed var %.3e (want %.3e), missing var %.4f (want %.4f)"
                          % (var_obs, want_obs, var_mis, want_mis)))


def run_flower(adapter, problem, spec):
    from src.flower import flower_reconstruct
    cond = Conditioning(labels=np.zeros((len(problem.image_ids),), np.int32), guidance={})
    eps = adapter.prior_sample(problem.image_ids)
    x0 = adapter.initial_state(None, spec.t0, eps)
    return x0, flower_reconstruct(adapter, cond, x0, problem, spec)


def check_end_to_end(results):
    from src.config import _estimate_cost
    from src.flower import (FlowerRefinement, _native_noise, flower_noise_parts, make_ops)
    print("\nEnd to end: MeanFlow (JAX) and standard flow (torch)")
    N = 6
    spec = flower_spec("pmf", num_flower_steps=N)
    results.append(report("planner: Eval. = N, BP = 0",
                          spec.expected_model_evals == N and spec.expected_backprops == 0
                          and spec.expected_denoiser_samples == N and spec.flower_gamma == 0
                          and spec.flower_cg_max_iters == 50 and spec.flower_cg_tol == 1e-5
                          and spec.hyperparameter_sources["num_flower_steps"] == "method"
                          and "flower" in spec.seed_recipes,
                          "N=%d -> %d evals, %d backprops; gamma %d, CG %d/%g"
                          % (N, spec.expected_model_evals, spec.expected_backprops,
                             spec.flower_gamma, spec.flower_cg_max_iters, spec.flower_cg_tol)))

    lines = []
    ok = True
    for name in ("denoising", "box_inpaint", "random_inpaint", "super_resolution", "deblur"):
        problem = make_problem(name)
        adapter = ToyMeanFlow(problem.ground_truth)
        x0, (state, stats) = run_flower(adapter, problem, spec)
        est = _estimate_cost("flower", {"num_flower_steps": N}, MEANFLOW)
        expect_ops = 3 if name != "deblur" else 1 + 2 * (N + stats.linear_solve_iterations)
        y = problem.measurement
        rmse0 = float(np.sqrt(np.mean((np.asarray(problem.apply(np.asarray(x0), NUMPY_BACKEND)) - y) ** 2)))
        rmse1 = float(np.sqrt(np.mean((np.asarray(problem.apply(np.asarray(state), NUMPY_BACKEND)) - y) ** 2)))
        good = (stats.model_evals_total == N == est["model_evals"]
                and stats.network_forwards == N and stats.denoiser_samples == N
                and stats.backprops_through_model == 0 == est["backprops"]
                and stats.data_gradient_evals == expect_ops
                and (stats.linear_solve_iterations > 0) == (name == "deblur")
                and stats.finite and rmse1 < rmse0)
        ok &= good
        lines.append("%s %d/%d/%d ops=%d cg=%d rmse %.3f->%.3f"
                     % (name[:6], stats.model_evals_total, stats.backprops_through_model,
                        stats.network_forwards, stats.data_gradient_evals,
                        stats.linear_solve_iterations, rmse0, rmse1))
    results.append(report("counters on all five tasks (pMF)", ok,
                          "evals/BP/forwards = N/0/N, H+H^T = 3 (diagonal) or "
                          "1 + 2(N + CG iters) (blur), measurement RMSE falls"))
    for line in lines:
        print("        %s" % line)

    problem = make_problem("deblur")
    adapter = ToyMeanFlow(problem.ground_truth)
    _, (first, _) = run_flower(adapter, problem, spec)
    _, (again, _) = run_flower(adapter, problem, spec)
    results.append(report("determinism", np.array_equal(np.asarray(first), np.asarray(again)),
                          "a repeated deblurring run is bitwise identical"))

    a = _native_noise(adapter, flower_spec("pmf", num_flower_steps=5), IDS, 3, "refresh")
    b = _native_noise(adapter, flower_spec("pmf", num_flower_steps=500), IDS, 3, "refresh")
    c = _native_noise(adapter, spec, IDS, 4, "refresh")
    d = _native_noise(adapter, spec, IDS, 3, "posterior_state")
    swapped = _native_noise(adapter, spec, IDS[::-1], 3, "refresh")
    results.append(report("noise: per (image, iteration), independent of N",
                          np.array_equal(np.asarray(a), np.asarray(b))
                          and not np.array_equal(np.asarray(a), np.asarray(c))
                          and not np.array_equal(np.asarray(a), np.asarray(d))
                          and np.array_equal(np.asarray(a)[0], np.asarray(swapped)[1])
                          and not np.array_equal(np.asarray(a)[0], np.asarray(a)[1])
                          and "500" not in repr(flower_noise_parts("pmf", "img0", 0, 3)),
                          "iteration 3 is the same at N=5 and N=500; differs by iteration, "
                          "stream and image; follows the image id, not the batch slot"))

    # batch independence end to end (deblur: per-image CG, per-image noise)
    single = problem.subset([1])
    _, (alone, _) = run_flower(ToyMeanFlow(single.ground_truth), single, spec)
    results.append(report("image 1 alone == image 1 in a batch of 2",
                          rel(alone, np.asarray(first)[1:]) < 1e-5,
                          "rel diff %.1e" % rel(alone, np.asarray(first)[1:])))

    # N = 1: the output is x_tilde = refine(T(x0; s0 -> 1)), no re-noising
    one = flower_spec("pmf", num_flower_steps=1)
    x0, (out1, st1) = run_flower(adapter, problem, one)
    ops = make_ops(adapter, problem)
    manual = FlowerRefinement(ops, problem, x0)(adapter.transition(x0, 0.0, 1.0, None), 0.0)
    results.append(report("last iteration returns x_tilde",
                          np.allclose(np.asarray(out1), np.asarray(manual), atol=1e-6)
                          and st1.model_evals_total == 1,
                          "N=1 output == refine(T(x0; 0 -> 1), t=0)"))

    g1 = flower_spec("pmf", num_flower_steps=N, flower_gamma=1)
    _, (s1, st_g1) = run_flower(adapter, problem, g1)
    _, (s1b, _) = run_flower(adapter, problem, g1)
    results.append(report("gamma = 1 runs, is deterministic, differs",
                          np.array_equal(np.asarray(s1), np.asarray(s1b))
                          and not np.allclose(np.asarray(s1), np.asarray(first))
                          and st_g1.backprops_through_model == 0
                          and st_g1.data_gradient_evals == 1 + 2 * (N + st_g1.linear_solve_iterations) + N,
                          "one extra H^T per iteration for the sample (%d operator applications)"
                          % st_g1.data_gradient_evals))

    # torch / JiT toy: same destination as the JAX toy, BCHW state, H through a permute
    try:
        import torch  # noqa: F401
    except ImportError:
        results.append(report("torch JiT path", False, "torch is not installed"))
        return
    jspec = flower_spec("jit", num_flower_steps=N)
    worst, counts = 0.0, []
    for name in ("denoising", "super_resolution", "deblur"):
        problem = make_problem(name)
        # named "jit" so the prior noise and Flower's noise (both seeded by model name) match
        _, (pm, pst) = run_flower(ToyMeanFlow(problem.ground_truth, name="jit"), problem, spec)
        _, (jt, jst) = run_flower(ToyJiT(problem.ground_truth), problem, jspec)
        jt_bhwc = jt.detach().numpy().transpose(0, 2, 3, 1)
        worst = max(worst, rel(jt_bhwc, np.asarray(pm)))
        counts.append((jst.model_evals_total, jst.backprops_through_model,
                       jst.data_gradient_evals == pst.data_gradient_evals))
    results.append(report("torch JiT path == JAX pMF path",
                          worst < 1e-4 and all(c == (N, 0, True) for c in counts)
                          and tuple(jt.shape) == (BATCH, 3, RES, RES),
                          "flow_flower (BCHW) vs meanflow_flower (BHWC) on denoising, SR, "
                          "deblur: max rel diff %.1e; counts %s" % (worst, counts)))


def check_rejections(results):
    from src.config import ConfigError
    from src.flower import flower_reconstruct
    from spec_support import build_plan
    print("\nRejections")
    spec = flower_spec("pmf", num_flower_steps=2)
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})

    def raises(fn, *needles):
        try:
            fn()
        except (ValueError, ConfigError) as exc:
            msg = str(exc)
            return all(n in msg for n in needles), msg.split("\n")[0][:90]
        return False, "no error"

    problem = make_problem("denoising")
    latent = ToyMeanFlow(problem.ground_truth, state_space="latent", name="imf")
    x0 = latent.prior_sample(IDS)
    ok, msg = raises(lambda: flower_reconstruct(latent, cond, x0, problem, spec), "pixel-space")
    results.append(report("latent adapter rejected", ok, msg))

    stroke = dataclasses.replace(problem, name="stroke_painting")
    pix = ToyMeanFlow(problem.ground_truth)
    ok, msg = raises(lambda: flower_reconstruct(pix, cond, pix.prior_sample(IDS), stroke, spec),
                     "stroke_painting")
    results.append(report("stroke_painting rejected", ok, msg))

    noiseless = make_problem("denoising", sigma=0.0)
    ok, msg = raises(lambda: flower_reconstruct(pix, cond, pix.prior_sample(IDS), noiseless, spec),
                     "sigma")
    results.append(report("sigma = 0 rejected", ok, msg))

    for model in ("sit", "imf"):
        ok, msg = raises(lambda: build_plan({"flower": {"num_flower_steps": 4}}, model=model),
                         "does not support method 'flower'")
        results.append(report("config: %s has no flower" % model.upper(), ok, msg))

    import copy
    import spec_support
    from src.config import validate_config
    cfg = copy.deepcopy(spec_support._BASE_CONFIG)
    cfg["experiments"]["s"] = {"enabled": True, "problem": "stroke_painting", "num_images": 2,
                               "models": {"pmf": {"methods": {"flower": {}}}}}
    ok, msg = raises(lambda: validate_config(cfg), "stroke_painting")
    results.append(report("config: stroke_painting rejected", ok, msg))
    ok1, msg1 = raises(lambda: build_plan({"flower": {"flower_gamma": 2}}), "flower_gamma")
    ok2, _ = raises(lambda: build_plan({"flower": {"num_pnp_steps": 4}}), "does not apply")
    ok3, _ = raises(lambda: build_plan({"pnp": {"num_flower_steps": 4}}), "does not apply")
    results.append(report("config: bad gamma / foreign fields", ok1 and ok2 and ok3, msg1))


# job ids, output directories and costs of every configuration committed before Flower
# (a digest over the 14,967 jobs of these files at commit 4859dfa, which this change matched)
PRE_FLOWER_CONFIGS = [
    "experiments.yaml", "experiments_8img_refinement_under10h.yaml",
    "experiments_fair_guidance_refinement.yaml", "experiments_final_frozen100.yaml",
    "experiments_focus_guidance.yaml", "experiments_full_hparam_sweep.yaml",
    "experiments_hpo_grid.yaml", "experiments_hpo_sweep_checkpoints.yaml",
    "experiments_imagenet100_final.yaml", "experiments_matched_screen.yaml",
    "experiments_matched_screen_smoke.yaml", "experiments_mega_tuning.yaml",
    "experiments_overnight_tuning.yaml", "experiments_theory_N_fixedB_final100.yaml",
    "experiments_theory_N_fixedM_final100.yaml", "experiments_theory_beta_schedule.yaml",
    "experiments_theory_gradient_authority.yaml", "experiments_theory_jacobian_8img.yaml",
    "experiments_theory_jit_suffix_final100.yaml", "experiments_theory_mu_denoising.yaml",
    "experiments_theory_noise_strength.yaml", "experiments_theory_resources_batch4.yaml",
    "experiments_theory_smoke.yaml", "experiments_theory_validation_final100.yaml"]
PRE_FLOWER_DIGEST = "b66357f391d49e2d55b8dd460cb563315c3b229197f608689a94c33923cc7545"


def job_digest():
    from src.config import load_config, resolve_run_plan, validate_config
    h, n = hashlib.sha256(), 0
    for name in PRE_FLOWER_CONFIGS:
        cfg = load_config(str(REPO / "configs" / name))
        plan = resolve_run_plan(cfg, validate_config(cfg), run_id="ids")
        for s in plan.specs:
            h.update(("%s|%s|%s|%s|%s\n" % (name, s.job_id, s.leaf_dir, s.expected_model_evals,
                                            s.expected_backprops)).encode())
            n += 1
    return h.hexdigest(), n


def check_identity(results):
    print("\nJob identity and resume")
    digest, n = job_digest()
    results.append(report("existing job ids / dirs / costs unchanged", digest == PRE_FLOWER_DIGEST,
                          "%d jobs in %d configs, digest %s" % (n, len(PRE_FLOWER_CONFIGS),
                                                                 digest[:16])))
    from src.utils import jsonable
    with contextlib.redirect_stdout(io.StringIO()):
        spec_run = importlib.util.spec_from_file_location("run_module", REPO / "run.py")
        run = importlib.util.module_from_spec(spec_run)
        spec_run.loader.exec_module(run)
    flower_names = ("num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol")
    pnp = jsonable(make_spec("pnp", num_pnp_steps=3).to_dict())
    old_pnp = {k: v for k, v in pnp.items() if k not in flower_names}
    fl = jsonable(flower_spec("pmf", num_flower_steps=3).to_dict())
    old_fl = {k: v for k, v in fl.items() if k not in flower_names}
    results.append(report("pre-Flower artefacts still resume",
                          run._resolved_spec_matches(old_pnp, pnp)
                          and not run._resolved_spec_matches(old_fl, fl)
                          and pnp["seed_recipes"] == old_pnp["seed_recipes"]
                          and "flower" not in pnp["seed_recipes"],
                          "a PnP spec stored without the Flower fields matches; a Flower spec "
                          "cannot match one that lacks them"))
    sp = flower_spec("pmf", num_flower_steps=7)
    key_a = run.warmup_key(sp, make_problem("denoising"))
    key_b = run.warmup_key(dataclasses.replace(sp, num_flower_steps=8), make_problem("denoising"))
    key_c = run.warmup_key(dataclasses.replace(sp, flower_gamma=1), make_problem("denoising"))
    results.append(report("warm-up key separates N and gamma",
                          key_a != key_b and key_a != key_c, "N=7/N=8/gamma=1 give 3 keys"))
    rec_cols = set(run.RESULT_COLUMNS)
    results.append(report("result columns",
                          set(flower_names) | {"linear_solve_iterations"} <= rec_cols,
                          "flower hyperparameters and linear_solve_iterations are recorded"))


def check_configs(results):
    from src.config import load_config, resolve_run_plan, validate_config
    print("\nGenerated configurations")
    gen = load_script("make_flower_configs")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "screen.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            gen.main(["screen", "--out", str(out)])
        regenerated = [out.read_text(), Path(gen.matched.smoke_path(out)).read_text()]
    committed = [(REPO / gen.SCREEN_OUT).read_text(),
                 (REPO / gen.matched.smoke_path(gen.SCREEN_OUT)).read_text()]
    strip = lambda t: "\n".join(l for l in t.split("\n") if "python run.py --config" not in l)
    results.append(report("committed configs == generator output",
                          [strip(t) for t in regenerated] == [strip(t) for t in committed],
                          "%s and its smoke variant" % gen.SCREEN_OUT))

    def plan_of(rel_path):
        cfg = load_config(str(REPO / rel_path))
        return resolve_run_plan(cfg, validate_config(cfg), run_id="cfg"), cfg

    plan, cfg = plan_of(gen.SCREEN_OUT)
    specs = plan.specs
    cells = {}
    for s in specs:
        cells.setdefault((s.model, s.problem), []).append(s.num_flower_steps)
    results.append(report("screen: 40 Flower jobs, pMF + JiT x 5 tasks x N",
                          len(specs) == 40 and {s.method for s in specs} == {"flower"}
                          and {s.model for s in specs} == {"pmf", "jit"}
                          and all(sorted(v) == [50, 100, 200, 500] for v in cells.values())
                          and len(cells) == 10 and all(s.num_images == 4 and s.t0 == 1.0
                                                       and s.flower_gamma == 0 for s in specs)
                          and len({s.job_id for s in specs}) == 40
                          and cfg["runtime"]["max_atomic_jobs"] >= 40,
                          "%d jobs, %d cells, 4 images, t0 = 1, gamma = 0" % (len(specs), len(cells))))
    results.append(report("screen: every job Eval. = N, BP = 0",
                          all(s.expected_model_evals == s.num_flower_steps
                              and s.expected_backprops == 0 for s in specs),
                          "per-chunk evals %s"
                          % sorted({s.expected_model_evals for s in specs})))
    splan, _ = plan_of(gen.matched.smoke_path(gen.SCREEN_OUT))
    fl = [s for s in splan.specs if s.method == "flower"]
    sd = [s for s in splan.specs if s.method == "sdedit"]
    pairs = {(s.experiment, s.model) for s in fl} == {(s.experiment, s.model) for s in sd}
    results.append(report("smoke: 2 images, each Flower job has an SDEdit partner",
                          len(fl) == 10 and len(sd) == 10 and pairs
                          and all(s.num_images == 2 and s.num_flower_steps == 50 and s.t0 == 1.0
                                  for s in fl) and all(s.t0 == 1.0 for s in sd),
                          "10 Flower + 10 SDEdit at t0 = 1, for the shared_initial_state check"))

    # extending.md step 7, executed with the toy adapter on the smoke config's own specs
    from src.checks import check_shared_initial_state
    ok_all, details = True, []
    for model, adapter_cls in (("pmf", ToyMeanFlow), ("jit", ToyJiT)):
        problem = make_problem("deblur")
        sel = [s for s in splan.specs if s.model == model and s.problem == "deblur"]
        ok, msg = check_shared_initial_state(adapter_cls(problem.ground_truth), problem, sel,
                                             None)
        ok_all &= ok and {s.method for s in sel} == {"sdedit", "flower"}
        details.append("%s: %s" % (model, msg.split(":")[0]))
    results.append(report("shared_initial_state passes for flower + sdedit", ok_all,
                          "; ".join(details)))


def write_results(path, rows):
    cols = ["run_id", "job_id", "problem", "method", "model", "status", "lpips", "psnr",
            "num_images", "num_flower_steps", "flower_gamma", "flower_cg_max_iters",
            "flower_cg_tol", "num_pnp_steps", "gamma0", "alpha", "rhso_terminal_mode"]
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict({"run_id": path.name, "job_id": "j%d" % i, "method": "flower",
                             "status": "ok", "psnr": 25.0, "flower_gamma": 0,
                             "flower_cg_max_iters": 50, "flower_cg_tol": 1e-05}, **r))


def check_freeze_and_hpo(results):
    print("\nhpo.py and make_flower_configs.py freeze")
    gen = load_script("make_flower_configs")
    hpo = gen.hpo
    results.append(report("hpo.py knows the flower knob",
                          hpo.KNOBS["flower"] == ["num_flower_steps"] and "flower" not in hpo.METHODS
                          and "flower" in hpo.SEARCHED,
                          "KNOBS['flower'] = %s; finalize still requires %s"
                          % (hpo.KNOBS["flower"], hpo.METHODS)))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        screen_rows, r2_rows = [], []
        lp = {50: 0.40, 100: 0.30, 200: 0.25, 500: 0.20}         # winner on the N = 500 edge
        for model in ("pmf", "jit"):
            for problem in hpo.PROBLEMS:
                for n, v in lp.items():
                    screen_rows.append(dict(model=model, problem=problem, num_flower_steps=n,
                                            lpips=v, num_images=4))
                # round 2 at 8 images: 200 wins for pMF, 500 for JiT; an off-grid 1000 is
                # better still but must not be chosen
                for n, v in ((200, 0.21 if model == "pmf" else 0.26),
                             (500, 0.23 if model == "pmf" else 0.22), (1000, 0.10)):
                    r2_rows.append(dict(model=model, problem=problem, num_flower_steps=n,
                                        lpips=v, num_images=8))
        write_results(tmp / "flower_screen", screen_rows)
        write_results(tmp / "flower_r2", r2_rows)

        def propose(lock):
            out = tmp / ("round2_%s.yaml" % bool(lock))
            argv = ["propose", "--from", str(tmp / "flower_screen"), "--models", "pmf,jit",
                    "--images", "8", "--no-paper", "--top", "2", "--out", str(out)]
            if lock:
                argv += ["--lock", "flower=num_flower_steps"]
            sys_argv, sys.argv = sys.argv, ["hpo.py"] + argv
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    hpo.main()
            finally:
                sys.argv = sys_argv
            return out.read_text()

        free, locked = propose(False), propose(True)
        pushed = [l for l in free.split("\n") if "flower:" in l and "edge" in l.lower()]
        results.append(report("propose: flower edge pushed without --lock, never with it",
                              "flower:" in free and "num_flower_steps: 1077" in free
                              and "num_flower_steps: 1077" not in locked
                              and locked.count("flower:") == 20,
                              "unlocked proposes N=1077 past the 500 edge; locked re-runs the "
                              "top 2 of each of 10 cells only"))

        out = tmp / "frozen.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            gen.main(["freeze", "--from", str(tmp / "flower_r2"), "--grid",
                      str(tmp / "flower_screen"), "--out", str(out)])
        from src.config import load_config, resolve_run_plan, validate_config
        cfg = load_config(str(out))
        plan = resolve_run_plan(cfg, validate_config(cfg), run_id="frozen")
        chosen = {(s.model, s.problem): s.num_flower_steps for s in plan.specs}
        smoke_cfg = load_config(gen.matched.smoke_path(out))
        smoke = resolve_run_plan(smoke_cfg, validate_config(smoke_cfg), run_id="fs").specs
        results.append(report("freeze: lowest LPIPS inside the screened grid",
                              all(v == (200 if m == "pmf" else 500) for (m, _), v in chosen.items())
                              and len(chosen) == 10 and all(s.num_images == 100 for s in plan.specs)
                              and all(s.expected_backprops == 0 for s in plan.specs)
                              and len(smoke) == 20 and all(s.num_images == 2 for s in smoke),
                              "pMF -> N=200, JiT -> N=500 on every task; the off-grid N=1000 "
                              "(LPIPS 0.10) is refused; 100-image config + 2-image smoke"))

        bad = list(r2_rows)
        bad[0] = dict(bad[0], flower_gamma=1, lpips=0.01, num_flower_steps=100)
        write_results(tmp / "flower_bad", bad)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gen.main(["freeze", "--from", str(tmp / "flower_bad"), "--grid",
                          str(tmp / "flower_screen"), "--out", str(tmp / "bad.yaml")])
            refused, msg = False, "accepted"
        except SystemExit as exc:
            refused, msg = "flower_gamma=1" in str(exc), str(exc)[:80]
        results.append(report("freeze refuses an off-protocol winner", refused, msg))

        # finalize still demands all six benchmark methods; Flower rows do not satisfy it
        try:
            sys_argv, sys.argv = sys.argv, ["hpo.py", "finalize", "--from",
                                            str(tmp / "flower_r2"), "--models", "pmf",
                                            "--out", str(tmp / "final.yaml")]
            with contextlib.redirect_stdout(io.StringIO()):
                hpo.main()
            ok, msg = False, "finalize accepted Flower-only input"
        except SystemExit as exc:
            ok, msg = "no winner for cells" in str(exc) and "flower" not in str(exc), str(exc)[:70]
        finally:
            sys.argv = sys_argv
        results.append(report("finalize unchanged: needs the six methods", ok, msg))


def main():
    results = []
    check_math(results)
    check_cg(results)
    check_end_to_end(results)
    check_rejections(results)
    check_identity(results)
    check_configs(results)
    check_freeze_and_hpo(results)
    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
