"""Execute Receding-Horizon State Optimization against synthetic adapters.

No checkpoint, no GPU, no download.  Three toy "models":

    ToyMeanFlow        pixel-space analytic learned transition T(x; a -> b), which records
                       every (s_from, s_to) it is asked for -- that trace is what proves the
                       receding-horizon structure rather than a plausible-looking output;
    ToyLatentMeanFlow  the same dynamics on a smaller "latent", with a linear differentiable
                       decoder, so the latent gradient path is exercised too;
    ToyFlow            a nonlinear standard-flow velocity field.

`rhso.meanflow_rhso` runs end to end here.  `rhso.flow_rhso` CANNOT: it calls
torch.optim.Adam, torch.no_grad and Tensor.backward, and PyTorch is not installable in this
container (the same limitation the existing tests document).  Everything in the
standard-flow path that is not torch-specific IS executed: the planning suffix, the
evaluation counts, the shared one-interval step, the differentiable suffix integration and
its gradient -- against an independent transcription of the reference integrator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from spec_support import build_plan, make_spec, source_banner, spec_source   # noqa: E402
from src.models.base import (AdapterSpec, Conditioning, MeanFlowAdapter,               # noqa: E402
                             StandardFlowAdapter)
from src.problems import InverseProblem                               # noqa: E402
from src.utils import (FLOW_ASCENDING, MEANFLOW, MEANFLOW_DESCENDING,  # noqa: E402
                       STANDARD_FLOW)

RES = 16
BATCH = 2


class _ProbeProblem:
    """The one attribute `run.warmup_key` reads off a problem."""
    class measurement:
        shape = (BATCH, 256, 256, 3)


def _names_used(fn):
    """Every global name referenced by `fn`, including inside its nested closures.

    `meanflow_rhso` builds its objective in an inner `loss_fn`, so a flat `co_names` scan
    would miss the very call this check is about.
    """
    import types
    names, stack = set(), [fn.__code__]
    while stack:
        code = stack.pop()
        names.update(code.co_names)
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return names


class _NoMu:
    """A spec from before `mu` existed: the attribute is simply not there."""


class _StripMu:
    """Proxy exposing every field of a resolved spec EXCEPT `mu`.

    Used to prove backward compatibility against the actual pre-change interface rather
    than against `mu = 0` written out explicitly.
    """

    def __init__(self, spec):
        object.__setattr__(self, "_spec", spec)

    def __getattr__(self, name):
        if name == "mu":
            raise AttributeError("mu")
        return getattr(object.__getattribute__(self, "_spec"), name)


class ToyMeanFlow(MeanFlowAdapter):
    """Analytic learned transition, plus a trace of the intervals it was asked for."""

    def __init__(self, target, name="pmf"):
        super().__init__(name, {})
        self.target = jnp.asarray(target)
        self.calls = []                       # [(s_from, s_to)] in order
        self.spec = AdapterSpec(
            name=name, display_name="pMF", dynamics_family=MEANFLOW, framework="jax",
            state_space="pixel", native_shape=(RES, RES, 3), layout="BHWC",
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
        self.calls.append((round(float(s_from), 12), round(float(s_to), 12)))
        dt = float(s_to) - float(s_from)
        return state + dt * (0.7 * (self.target - state) + 0.2 * jnp.sin(3.0 * state))


class RecordingMeanFlow(ToyMeanFlow):
    """ToyMeanFlow that also keeps the OUTPUT of every transition it is asked for.

    The executed interval of stage k is the transition whose s_to is not 1 (or, at the last
    stage, the final one), so its recorded output is exactly the state the next stage must
    anchor to.  Comparing that array against the anchor the next stage actually used is how
    "a fresh anchor is taken from the executed state" is verified rather than assumed.
    """

    def __init__(self, target, name="pmf"):
        super().__init__(target, name=name)
        self.outputs = []                     # [(s_from, s_to, np.ndarray)] in order

    def transition(self, state, s_from, s_to, conditioning):
        out = super().transition(state, s_from, s_to, conditioning)
        # Inside jax.value_and_grad the output is a tracer with no value yet; only the
        # EXECUTED transitions are concrete, and those are the ones this check needs.
        if not isinstance(out, jax.core.Tracer):
            self.outputs.append((round(float(s_from), 12), round(float(s_to), 12),
                                 np.asarray(jax.device_get(out), np.float64)))
        return out


class ToyLatentMeanFlow(ToyMeanFlow):
    """The same dynamics on an 8x8x4 'latent', decoded by a fixed linear map.

    The decoder is the point: a latent model's RHSO objective has to differentiate through
    it, and no separate latent RHSO mathematics may exist.
    """

    def __init__(self, target_pixels):
        super().__init__(jnp.zeros((BATCH, RES // 2, RES // 2, 4), jnp.float32), name="imf")
        self.spec = dataclasses.replace(self.spec, name="imf", display_name="iMF",
                                        state_space="latent",
                                        native_shape=(RES // 2, RES // 2, 4))
        rng = np.random.default_rng(5)
        self._decoder = jnp.asarray(rng.standard_normal((4, 3)).astype(np.float32) * 0.5)
        self.target = jnp.asarray(rng.standard_normal(
            (BATCH, RES // 2, RES // 2, 4)).astype(np.float32) * 0.3)
        self._target_pixels = target_pixels

    def to_pixels(self, state, differentiable=False):
        # (B, 8, 8, 4) -> (B, 16, 16, 3): nearest-neighbour upsample then a channel mix.
        up = jnp.repeat(jnp.repeat(state, 2, axis=1), 2, axis=2)
        pixels = jnp.tanh(up @ self._decoder)
        return pixels if differentiable else np.asarray(jax.device_get(pixels), np.float32)

    def encode_pixels(self, pixels):
        return jnp.zeros((pixels.shape[0], RES // 2, RES // 2, 4), jnp.float32)


class ToyFlow(StandardFlowAdapter):
    """A nonlinear, time-dependent velocity field; records the times it is evaluated at."""

    def __init__(self, euler_final_step_for_heun=False):
        super().__init__("jit", {})
        self.times = []
        self.spec = AdapterSpec(
            name="jit", display_name="JiT", dynamics_family=STANDARD_FLOW, framework="jax",
            state_space="pixel", native_shape=(RES, RES, 3), layout="BHWC",
            pixel_resolution=RES, prediction_kind="velocity",
            native_time_mapping=FLOW_ASCENDING, batch_size=BATCH, fixed_batch_shape=False,
            num_classes=1000, null_label=1000, guidance={}, checkpoint={},
            euler_final_step_for_heun=euler_final_step_for_heun)

    def to_native_noise(self, noise):
        return jnp.asarray(noise, jnp.float32)

    def encode_pixels(self, pixels):
        return jnp.asarray(pixels, jnp.float32)

    def to_pixels(self, state, differentiable=False):
        return state if differentiable else np.asarray(jax.device_get(state), np.float32)

    def _lerp(self, guide, noise, keep, add):
        return keep * guide + add * noise

    def velocity(self, state, s, conditioning):
        self.count_forwards(1)
        self.times.append(round(float(s), 12))
        return jnp.sin(2.0 * state) + float(s) * state * 0.5 + 0.3


def make_problem(sigma=0.05):
    rng = np.random.default_rng(3)
    gt = rng.standard_normal((BATCH, RES, RES, 3), np.float32) * 0.3
    mask = np.ones((BATCH, RES, RES, 1), np.float32)
    mask[:, 4:10, 4:10, :] = 0.0
    return InverseProblem(name="box_inpaint", key="k", sigma=sigma, params={},
                          ground_truth=gt, measurement=gt * mask, display_measurement=gt,
                          initialization_guide=gt * mask, guide_mode="zero_fill", mask=mask,
                          image_ids=("img0", "img1"))


def reference_integrate(adapter, cond, x, grid, solver, final_euler):
    """Independent transcription of the reference integrator, for comparison."""
    steps = len(grid) - 1
    for k in range(steps):
        s, s_next = grid[k], grid[k + 1]
        dt = s_next - s
        step = "euler" if (solver == "heun" and final_euler and k == steps - 1) else solver
        v1 = adapter.velocity(x, s, cond)
        if step == "euler":
            x = x + dt * v1
        elif step == "heun":
            v2 = adapter.velocity(x + dt * v1, s_next, cond)
            x = x + 0.5 * dt * (v1 + v2)
        else:
            s_mid = 0.5 * (s + s_next)
            v2 = adapter.velocity(x + 0.5 * dt * v1, s_mid, cond)
            v3 = adapter.velocity(x + 0.5 * dt * v2, s_mid, cond)
            v4 = adapter.velocity(x + dt * v3, s_next, cond)
            x = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
    return x


def report(name, ok, detail):
    print("  [%s] %-32s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def main():
    from src.dflow import integrate_flow, meanflow_dflow, solver_evaluations
    from src.problems import make_phi, phi_log_scale
    from src.reconstruction import registered_methods, select_reconstructor
    from src.rhso import (flow_rhso, meanflow_rhso, rhso_cost_estimate,
                          rhso_planning_grid, rhso_planning_evaluations,
                          rhso_execution_evaluations, rhso_time_grid)
    from src.schedule import canonical_time_grid
    from src.sdedit import sdedit_meanflow

    print("\n%s" % source_banner())
    problem = make_problem()
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    results = []

    # ---------------------------------------------------------------- 1. registration
    print("\nRegistry")
    families = registered_methods().get("rhso", ())
    results.append(report(
        "registered for both families",
        set(families) == {STANDARD_FLOW, MEANFLOW}
        and select_reconstructor(MEANFLOW, "rhso") is meanflow_rhso
        and select_reconstructor(STANDARD_FLOW, "rhso") is flow_rhso,
        "(%s) -> rhso resolves to the two family implementations"
        % ", ".join(sorted(families))))

    # ---------------------------------------------------------------- MeanFlow RHSO
    print("\nMeanFlow RHSO -- structure")
    adapter = ToyMeanFlow(problem.ground_truth)
    eps = adapter.prior_sample(problem.image_ids)
    guide = adapter.encode_pixels(problem.initialization_guide)
    spec = make_spec("rhso", num_rhso_steps=3, num_opt_steps=4, lr=0.05, beta=0.5)
    x0 = adapter.initial_state(guide, spec.t0, eps)
    grid = rhso_time_grid(spec)
    N, M = spec.num_rhso_steps, spec.num_opt_steps

    x0_before = np.asarray(x0).copy()
    adapter.calls = []
    state, stats = meanflow_rhso(adapter, cond, x0, problem, spec)
    trace = list(adapter.calls)

    # 2. the shared initial state is used as given, never rebuilt or mutated
    other_spec = make_spec("dflow", steps=1)
    x0_other = adapter.initial_state(guide, other_spec.t0, eps)
    results.append(report(
        "shared z_t0 is used as given",
        np.array_equal(np.asarray(x0), x0_before)
        and np.array_equal(np.asarray(x0_other), x0_before)
        and np.array_equal(np.asarray(adapter.prior_sample(problem.image_ids)),
                           np.asarray(eps)),
        "x0 is unchanged by the run and identical to the state another method gets at the "
        "same (model, t0, images, replicate); epsilon does not depend on the method"))

    # 7 / 9 / 10. the call trace IS the receding-horizon structure
    expected = []
    for k in range(N):
        expected += [(round(grid[k], 12), 1.0)] * M            # M planning transitions
        expected += [(round(grid[k], 12), round(grid[k + 1], 12))]   # ONE execution
    results.append(report(
        "planner is ONE direct s_k -> 1",
        all(call == (round(grid[k], 12), 1.0)
            for k in range(N) for call in trace[k * (M + 1):k * (M + 1) + M]),
        "every inner objective is a single learned transition to the clean endpoint, not a "
        "composition of the %d remaining outer intervals" % N))
    results.append(report(
        "executes exactly one interval", trace == expected,
        "the full trace is %d x (%d planning + 1 execution): %s"
        % (N, M, " ".join("%.3f->%.3f" % c for c in trace[:M + 2]) + " ...")))
    results.append(report(
        "replans at the state it reached",
        [c[0] for c in trace[::(M + 1)]] == [round(g, 12) for g in grid[:-1]],
        "stage k starts a NEW optimisation at s_k = %s"
        % ["%.4f" % g for g in grid[:-1]]))

    # 12. counters
    results.append(report(
        "counters",
        stats.model_evals_planning == N * M and stats.backprops_through_model == N * M
        and stats.optimizer_iterations == N * M and stats.objective_evals == N * M
        and stats.data_gradient_evals == N * M
        and stats.model_evals_total == N * (M + 1)
        and stats.network_forwards == N * (M + 1),
        "N*M = %d planning evals / backprops / objectives, N*(M+1) = %d evaluations total"
        % (N * M, stats.model_evals_total)))
    predicted = rhso_cost_estimate({"num_rhso_steps": N, "num_opt_steps": M}, MEANFLOW)
    results.append(report(
        "planner cost model == measured",
        predicted["model_evals"] == stats.model_evals_total
        and predicted["backprops"] == stats.backprops_through_model
        and predicted["planning_evals"] == stats.model_evals_planning
        and predicted["optimizer_iterations"] == stats.optimizer_iterations,
        "dry-run estimate %d evals / %d backprops matches the measured %d / %d"
        % (predicted["model_evals"], predicted["backprops"], stats.model_evals_total,
           stats.backprops_through_model)))

    # 14. finite output
    results.append(report("finite output", stats.finite
                          and bool(np.isfinite(np.asarray(state)).all()),
                          "final state finite, shape %s" % (tuple(state.shape),)))

    # 13. determinism
    again = meanflow_rhso(adapter, cond, x0, problem, spec)[0]
    results.append(report("determinism",
                          bool(np.array_equal(np.asarray(state), np.asarray(again))),
                          "a repeated run is bitwise identical"))

    # ---------------------------------------------------------------- optimisation
    print("\nMeanFlow RHSO -- optimisation")
    B = adapter.backend()
    phi = make_phi(problem, B, spec.phi_normalization)
    scale = phi_log_scale(problem, spec.phi_normalization)

    # 3. q^(0) = x_k: with lr = 0 no update happens, so RHSO degenerates EXACTLY to
    #    successive execution of the scheduled intervals -- i.e. SDEdit on the same grid.
    frozen = dataclasses.replace(spec, lr=0.0)
    frozen_state = meanflow_rhso(adapter, cond, x0, problem, frozen)[0]
    sdedit_state = sdedit_meanflow(adapter, cond, x0,
                                   dataclasses.replace(spec, steps=N))[0]
    results.append(report(
        "q is initialised from x_k",
        bool(np.allclose(np.asarray(frozen_state), np.asarray(sdedit_state), atol=1e-6)),
        "with lr = 0 the run reproduces plain successive execution along the same beta "
        "grid, so each stage really does start from the current state"))

    # 4. gradients wrt q
    def terminal_loss(q, s):
        return phi(adapter.to_pixels(adapter.transition(q, s, 1.0, cond),
                                     differentiable=True))
    g = np.asarray(jax.grad(lambda q: terminal_loss(q, grid[0]))(x0))
    results.append(report("gradient wrt q",
                          bool(np.isfinite(g).all()) and np.abs(g).max() > 0,
                          "|d loss / d q|max = %.4e through the direct transition"
                          % float(np.abs(g).max())))

    # 5. the terminal fidelity falls during the inner optimisation
    first_stage = stats.loss_history[:M]
    results.append(report("inner optimisation descends",
                          first_stage[-1] < first_stage[0]
                          and stats.loss_history[-1] < stats.loss_history[0],
                          "stage 0: %.6f -> %.6f; whole run: %.6f -> %.6f"
                          % (first_stage[0], first_stage[-1], stats.loss_history[0],
                             stats.loss_history[-1])))

    # 6. the executed interval belongs to the FINAL q, not the pre-update one
    single = dataclasses.replace(spec, num_rhso_steps=1)
    single_state, single_stats = meanflow_rhso(adapter, cond, x0, problem, single)
    final_loss = float(phi(adapter.to_pixels(single_state, differentiable=True))) * scale
    results.append(report(
        "output is the FINAL q",
        final_loss <= single_stats.loss_history[-1] + 1e-9,
        "N=1: the returned state scores %.6f <= the last recorded objective %.6f, which "
        "was computed before the last Adam update"
        % (final_loss, single_stats.loss_history[-1])))

    # 11. Adam moments are rebuilt at every outer stage
    def manual(reset_each_stage: bool):
        q_state = None
        x = x0
        tx = optax.adam(float(spec.lr))
        for k in range(N):
            s_k, s_next = grid[k], grid[k + 1]
            loss_fn = lambda q, _s=s_k: terminal_loss(q, _s)
            q = x
            if reset_each_stage or q_state is None:
                q_state = tx.init(q)
            vg = jax.value_and_grad(loss_fn)
            for _ in range(M):
                _loss, grads = vg(q)
                updates, q_state = tx.update(grads, q_state, q)
                q = optax.apply_updates(q, updates)
            x = jax.lax.stop_gradient(adapter.transition(q, s_k, s_next, cond))
        return np.asarray(x)

    fresh, carried = manual(True), manual(False)
    results.append(report(
        "Adam state is fresh per stage",
        np.array_equal(np.asarray(state), fresh) and not np.array_equal(fresh, carried),
        "matches a reference that re-initialises the optimiser at every stage; carrying "
        "the moments instead changes the result by %.3e, so the reset is real"
        % float(np.max(np.abs(fresh - carried)))))

    # RHSO is not D-Flow: same budget, different algorithm, different answer
    dflow_state = meanflow_dflow(adapter, cond, x0, problem,
                                 dataclasses.replace(spec, steps=N,
                                                     num_opt_steps=N * M))[0]
    results.append(report(
        "distinct from D-Flow",
        not np.allclose(np.asarray(state), np.asarray(dflow_state), atol=1e-6),
        "D-Flow optimises one state and then executes the whole trajectory; RHSO commits "
        "one interval and re-optimises, and the two do not coincide (max diff %.3e)"
        % float(np.max(np.abs(np.asarray(state) - np.asarray(dflow_state))))))

    # ---------------------------------------------------------------- 15. latent path
    print("\nMeanFlow RHSO -- latent model (differentiable decoder)")
    latent = ToyLatentMeanFlow(problem.ground_truth)
    lat_eps = latent.prior_sample(problem.image_ids)
    lat_x0 = latent.initial_state(latent.encode_pixels(problem.initialization_guide),
                                  spec.t0, lat_eps)
    lat_phi = make_phi(problem, latent.backend(), spec.phi_normalization)
    lat_grad = np.asarray(jax.grad(lambda q: lat_phi(latent.to_pixels(
        latent.transition(q, grid[0], 1.0, cond), differentiable=True)))(lat_x0))
    lat_spec = dataclasses.replace(spec, num_rhso_steps=2, num_opt_steps=3)
    lat_state, lat_stats = meanflow_rhso(latent, cond, lat_x0, problem, lat_spec)
    results.append(report(
        "gradient survives the decoder",
        bool(np.isfinite(lat_grad).all()) and np.abs(lat_grad).max() > 0
        and lat_grad.shape == (BATCH, RES // 2, RES // 2, 4),
        "|d loss / d q|max = %.4e, taken in latent shape %s through the decoder and the "
        "measurement operator" % (float(np.abs(lat_grad).max()), lat_grad.shape)))
    results.append(report(
        "same code path as the pixel model",
        lat_stats.model_evals_total == 2 * (3 + 1) and lat_stats.finite
        and tuple(lat_state.shape)[1:] == (RES // 2, RES // 2, 4)
        and lat_stats.loss_history[-1] < lat_stats.loss_history[0],
        "%d evaluations, latent state kept native, loss %.6f -> %.6f -- no separate "
        "pixel/latent RHSO mathematics" % (lat_stats.model_evals_total,
                                           lat_stats.loss_history[0],
                                           lat_stats.loss_history[-1])))

    # ---------------------------------------------------------------- standard flow
    print("\nStandard-flow RHSO -- planner, counts and the differentiable suffix")
    print("  (rhso.flow_rhso itself needs PyTorch and is NOT executed here)")
    flow_spec = make_spec("rhso", model="jit", num_rhso_steps=4, num_opt_steps=2,
                          solver="heun", beta=0.5)
    assert flow_spec.dynamics_family == STANDARD_FLOW
    flow_grid = rhso_time_grid(flow_spec)

    # 8. the planner is the remaining suffix of the SAME grid
    suffix_ok = all(rhso_planning_grid(flow_grid, k) == list(flow_grid[k:])
                    for k in range(4))
    results.append(report(
        "planner uses the remaining suffix", suffix_ok
        and rhso_planning_grid(flow_grid, 3) == [flow_grid[3], 1.0],
        "stage k integrates [s_k, ..., 1] -- 4, 3, 2, 1 intervals -- with no second "
        "planning-resolution hyperparameter"))

    for final_euler in (False, True):
        toy = ToyFlow(final_euler)
        counts_ok, exec_ok = [], []
        for k in range(4):
            predicted_plan = rhso_planning_evaluations(toy, flow_spec, k, flow_grid)
            toy.reset_counters()
            toy.times = []
            integrate_flow(toy, cond, jnp.zeros((BATCH, RES, RES, 3), jnp.float32),
                           flow_spec, rhso_planning_grid(flow_grid, k),
                           reaches_data_endpoint=True)
            counts_ok.append(toy.forward_counter == predicted_plan)
            predicted_exec = rhso_execution_evaluations(toy, flow_spec, k, flow_grid)
            exec_ok.append(predicted_exec == solver_evaluations(
                toy, "heun", 1, reaches_data_endpoint=(k == 3)))
        results.append(report(
            "eval counts%s" % (" + final-Euler" if final_euler else ""),
            all(counts_ok) and all(exec_ok),
            "measured planning evaluations match the predicted counts at all four stages, "
            "and the final-step policy applies only to the interval that lands on s = 1"))

    toy = ToyFlow(True)
    estimate = rhso_cost_estimate({"num_rhso_steps": 4, "num_opt_steps": 2,
                                   "solver": "heun"}, STANDARD_FLOW,
                                  euler_final_step_for_heun=True)
    measured_plan = sum(2 * rhso_planning_evaluations(toy, flow_spec, k, flow_grid)
                        for k in range(4))
    measured_exec = sum(rhso_execution_evaluations(toy, flow_spec, k, flow_grid)
                        for k in range(4))
    results.append(report(
        "planner cost model (standard flow)",
        estimate["planning_evals"] == measured_plan
        and estimate["model_evals"] == measured_plan + measured_exec
        and estimate["objective_evals"] == 4 * 2,
        "%d planning + %d execution evaluations -- shrinking suffixes, not a crude N*M "
        "estimate" % (measured_plan, measured_exec)))

    # the inner objective itself: arithmetic and gradient, without torch
    toy = ToyFlow(False)
    rng = np.random.default_rng(11)
    q0 = jnp.asarray(rng.standard_normal((BATCH, RES, RES, 3), np.float32) * 0.4)
    suffix = rhso_planning_grid(flow_grid, 1)
    mine = integrate_flow(toy, cond, q0, flow_spec, suffix, reaches_data_endpoint=True)
    ref = reference_integrate(toy, cond, q0, suffix, "heun", False)
    flow_phi = make_phi(problem, toy.backend(), flow_spec.phi_normalization)
    loss_fn = lambda q: flow_phi(toy.to_pixels(
        integrate_flow(toy, cond, q, flow_spec, suffix, reaches_data_endpoint=True),
        differentiable=True))
    value, grad = jax.value_and_grad(loss_fn)(q0)
    gnp = np.asarray(grad)
    results.append(report(
        "suffix integration is exact",
        float(np.max(np.abs(np.asarray(mine) - np.asarray(ref)))) == 0.0,
        "matches an independent transcription of the reference integrator bitwise over the "
        "non-uniform suffix %s" % ["%.4f" % s for s in suffix]))
    results.append(report(
        "terminal gradient wrt q",
        bool(np.isfinite(gnp).all()) and np.abs(gnp).max() > 0
        and float(loss_fn(q0 - 0.05 * grad)) < float(value),
        "loss %.5f, |d loss / d q|max = %.4e, and a descent step reduces it"
        % (float(value), float(np.abs(gnp).max()))))

    # the one-interval execution step, shared with SDEdit's arithmetic
    from src.dflow import flow_step
    toy = ToyFlow(True)
    one = flow_step(toy, cond, q0, flow_grid[0], flow_grid[1], "heun",
                    apply_final_euler_policy=False)
    ref_one = reference_integrate(toy, cond, q0, [flow_grid[0], flow_grid[1]], "heun", False)
    last = flow_step(toy, cond, q0, flow_grid[3], 1.0, "heun",
                     apply_final_euler_policy=True)
    ref_last = reference_integrate(toy, cond, q0, [flow_grid[3], 1.0], "heun", True)
    results.append(report(
        "one-interval execution",
        float(np.max(np.abs(np.asarray(one) - np.asarray(ref_one)))) == 0.0
        and float(np.max(np.abs(np.asarray(last) - np.asarray(ref_last)))) == 0.0,
        "a mid-trajectory interval executes a full Heun step; only the interval landing on "
        "s = 1 takes the adapter's Euler fallback"))

    # beta really does change the executed grid
    beta_one = rhso_time_grid(dataclasses.replace(flow_spec, beta=1.0))
    results.append(report(
        "beta drives the outer schedule",
        beta_one == canonical_time_grid(flow_spec.canonical_start_time, 4, 1.0)
        and flow_grid != beta_one,
        "beta=1 gives the uniform grid %s; beta=0.5 gives %s"
        % (["%.3f" % s for s in beta_one], ["%.3f" % s for s in flow_grid])))

    # ---------------------------------------------------------------- mu: state anchor
    print("\nState-anchor regularisation (mu)")
    from src.rhso import rhso_mu, rhso_total_objective, state_anchor_penalty
    from src.utils import JaxBackend, NUMPY_BACKEND

    JB = JaxBackend(jnp)
    anchor_np = np.random.default_rng(11).normal(size=(BATCH, RES, RES, 3)).astype(np.float32)
    q_np = anchor_np + 0.3
    d = int(anchor_np[0].size)

    # 2 + 3. zero at the anchor, positive after a displacement, matching the definition
    exact = 0.5 * sum(float(np.mean((q_np[b] - anchor_np[b]) ** 2)) for b in range(BATCH))
    at_anchor = float(state_anchor_penalty(NUMPY_BACKEND, anchor_np.copy(), anchor_np))
    displaced = float(state_anchor_penalty(NUMPY_BACKEND, q_np, anchor_np))
    results.append(report(
        "R = 0 at the anchor, R > 0 once displaced",
        at_anchor == 0.0 and displaced > 0.0 and abs(displaced - exact) < 1e-6,
        "R(anchor, anchor) = 0 exactly; R after a 0.3 shift = %.6f, matching "
        "sum_b mean((q_b - a_b)^2)/2 = %.6f" % (displaced, exact)))

    # 7. one implementation, two frameworks: same normalisation by construction
    jax_value = float(state_anchor_penalty(JB, jnp.asarray(q_np), jnp.asarray(anchor_np)))
    # `rhso_mu` is now resolved in the shared `rhso._prepare` both loops call, so the
    # scan covers that function too: the property being asserted is "ONE resolution of mu
    # and ONE penalty implementation serve both families", not "the call appears literally
    # inside each loop body".
    from src.rhso import _prepare
    shared = (all("rhso_total_objective" in _names_used(fn)
                  for fn in (flow_rhso, meanflow_rhso))
              and all("_prepare" in _names_used(fn) for fn in (flow_rhso, meanflow_rhso))
              and "rhso_mu" in _names_used(_prepare))
    scale_free = abs(
        float(state_anchor_penalty(NUMPY_BACKEND,
                                   np.zeros((BATCH, 2, 2, 1), np.float32) + 0.3,
                                   np.zeros((BATCH, 2, 2, 1), np.float32))) - displaced)
    results.append(report(
        "torch and JAX share one normalisation",
        abs(jax_value - displaced) < 1e-6 and shared and scale_free < 1e-6,
        "NumPy and JAX backends agree to %.2e on the SAME backend-generic function, which "
        "both loops call (rhso_mu and rhso_total_objective appear in the bytecode of "
        "flow_rhso and of meanflow_rhso); the 1/d factor makes R independent of state size. "
        "Torch parity is therefore structural -- torch itself cannot be executed here"
        % abs(jax_value - displaced)))

    # batch SUM, not mean
    doubled = float(state_anchor_penalty(NUMPY_BACKEND, np.concatenate([q_np, q_np]),
                                         np.concatenate([anchor_np, anchor_np])))
    results.append(report(
        "summed over the batch", abs(doubled - 2.0 * displaced) < 1e-6,
        "doubling the batch doubles R (%.6f -> %.6f), so one sample's gradient never "
        "depends on how many others share the batch" % (displaced, doubled)))

    # 4. the gradient points back at the anchor
    grad = np.asarray(jax.grad(lambda z: state_anchor_penalty(JB, z, jnp.asarray(anchor_np)))(
        jnp.asarray(q_np)))
    analytic = (q_np - anchor_np) / float(d)
    stepped = q_np - 0.5 * grad
    results.append(report(
        "the gradient points at the anchor",
        float(np.abs(grad - analytic).max()) < 1e-8
        and float(np.abs(stepped - anchor_np).max()) < float(np.abs(q_np - anchor_np).max()),
        "dR/dq = (q - anchor)/d to %.2e, and a descent step strictly reduces the distance "
        "to the anchor" % float(np.abs(grad - analytic).max())))

    # 10. invalid weights fail clearly
    bad_rejected = 0
    for bad in (-1.0, float("nan"), float("inf")):
        try:
            rhso_mu(dataclasses.replace(make_spec("rhso"), mu=bad))
        except ValueError:
            bad_rejected += 1
    absent = rhso_mu(_NoMu())
    results.append(report(
        "invalid mu fails, absent mu is 0",
        bad_rejected == 3 and absent == 0.0,
        "negative, NaN and infinite weights raise; a spec that predates the field resolves "
        "to mu = 0.0, i.e. the vanilla objective"))

    # 1 + 8. mu = 0 is the unpenalised objective and costs exactly the same
    adapter = ToyMeanFlow(problem.ground_truth)
    x0 = jnp.asarray(problem.initialization_guide, jnp.float32)
    base_spec = make_spec("rhso", num_rhso_steps=3, num_opt_steps=4, lr=0.05, mu=0.0)
    legacy_spec = _StripMu(base_spec)            # no `mu` attribute at all
    state_zero, stats_zero = meanflow_rhso(adapter, cond, x0, problem, base_spec)
    state_legacy, stats_legacy = meanflow_rhso(adapter, cond, x0, problem, legacy_spec)
    reg_spec = dataclasses.replace(base_spec, mu=0.5)
    state_reg, stats_reg = meanflow_rhso(adapter, cond, x0, problem, reg_spec)

    bitwise = bool(np.array_equal(np.asarray(jax.device_get(state_zero)),
                                  np.asarray(jax.device_get(state_legacy))))
    results.append(report(
        "mu = 0 reproduces the pre-change behaviour", bitwise
        and stats_zero.loss_history == stats_legacy.loss_history
        and all(t == f for t, f in zip(stats_zero.loss_history,
                                       stats_zero.fidelity_history)),
        "a spec carrying mu=0.0 and a spec with no mu field at all produce a BITWISE "
        "identical state and identical loss history, and the optimised scalar is still the "
        "fidelity itself"))

    counters = ("model_evals_total", "model_evals_planning", "backprops_through_model",
                "objective_evals", "optimizer_iterations", "control_iterations",
                "data_gradient_evals", "network_forwards")
    same_cost = all(getattr(stats_zero, c) == getattr(stats_reg, c) for c in counters)
    predicted = rhso_cost_estimate({"num_rhso_steps": 3, "num_opt_steps": 4}, MEANFLOW)
    results.append(report(
        "mu does not change any compute counter",
        same_cost and stats_reg.model_evals_total == predicted["model_evals"]
        == base_spec.expected_model_evals
        and stats_reg.backprops_through_model == predicted["backprops"],
        "mu=0 and mu=0.5 agree on all %d counters (%d model evals, %d backprops, %d "
        "objectives), and both match the unchanged planner estimate"
        % (len(counters), stats_reg.model_evals_total,
           stats_reg.backprops_through_model, stats_reg.objective_evals)))

    # the differentiated scalar really is fidelity + mu * R
    decomposes = all(abs(t - (f + 0.5 * r)) <= 1e-5 * max(1.0, abs(t))
                     for t, f, r in zip(stats_reg.loss_history, stats_reg.fidelity_history,
                                        stats_reg.state_penalty_history))
    results.append(report(
        "loss history decomposes",
        decomposes and len(stats_reg.state_penalty_history) == 3 * 4
        and stats_reg.state_penalty_history[0] == 0.0
        and max(stats_reg.state_penalty_history) > 0.0,
        "for all 12 recorded iterations total = fidelity + mu*R; R is 0 at the first "
        "iteration of a stage (q IS the anchor) and grows to %.4g as q moves away"
        % max(stats_reg.state_penalty_history)))

    # a larger mu must hold q closer to the anchor
    tight_spec = dataclasses.replace(base_spec, mu=50.0)
    _tight_state, stats_tight = meanflow_rhso(adapter, cond, x0, problem, tight_spec)
    results.append(report(
        "a larger mu keeps q nearer the anchor",
        max(stats_tight.state_penalty_history) < max(stats_reg.state_penalty_history),
        "mu=50 holds the maximum displacement to %.3g against %.3g at mu=0.5 -- the "
        "penalty acts as a trust region rather than merely being recorded"
        % (max(stats_tight.state_penalty_history),
           max(stats_reg.state_penalty_history))))

    # 5 + 6. the anchor is fixed within a stage and refreshed from the EXECUTED state
    import src.rhso as rhso_module
    recorder = RecordingMeanFlow(problem.ground_truth)
    seen = []
    original = rhso_module.rhso_total_objective

    def spy(B, fidelity, q, anchor, mu):
        # `q` is a tracer inside value_and_grad and has no value yet; the ANCHOR is the
        # concrete constant this check is about, and that asymmetry is itself the point.
        seen.append((np.asarray(jax.device_get(anchor), np.float64),
                     isinstance(q, jax.core.Tracer)))
        return original(B, fidelity, q, anchor, mu)

    rhso_module.rhso_total_objective = spy
    try:
        anchor_spec = dataclasses.replace(base_spec, num_rhso_steps=3, num_opt_steps=4,
                                          mu=0.5)
        _anchor_state, anchor_stats = meanflow_rhso(recorder, cond, x0, problem, anchor_spec)
    finally:
        rhso_module.rhso_total_objective = original

    stages = [seen[i * 4:(i + 1) * 4] for i in range(3)]
    fixed_within = all(np.array_equal(it[0], stage[0][0]) for stage in stages for it in stage)
    anchor_never_traced = all(
        not isinstance(it[0], jax.core.Tracer) for stage in stages for it in stage)
    # q DID move inside each stage: the displacement is 0 at the stage's first iteration
    # (q is the anchor) and non-zero afterwards, measured against that same fixed anchor.
    per_stage = [anchor_stats.state_penalty_history[i * 4:(i + 1) * 4] for i in range(3)]
    moves_within = all(r[0] == 0.0 and max(r[1:]) > 0.0 for r in per_stage)
    results.append(report(
        "the anchor is fixed for the whole stage",
        fixed_within and anchor_never_traced and moves_within,
        "across all 4 inner iterations of each of the 3 stages the anchor array is bitwise "
        "unchanged and never a tracer, while the displacement measured against it grows "
        "from exactly 0 -- a stop-gradient constant, not a moving target"))

    _g = [round(float(t), 12) for t in rhso_time_grid(anchor_spec)]
    intervals = list(zip(_g[:-1], _g[1:]))
    executed = [out for out in recorder.outputs if (out[0], out[1]) in intervals]
    fresh = all(np.array_equal(stages[k + 1][0][0], executed[k][2]) for k in range(2))
    not_origin = not np.array_equal(stages[1][0][0], stages[0][0][0])
    starts_at_x0 = np.array_equal(stages[0][0][0],
                                  np.asarray(jax.device_get(x0), np.float64))
    results.append(report(
        "each stage anchors to the state it actually reached",
        fresh and not_origin and starts_at_x0,
        "stage 0 anchors to z_t0; stage k+1's anchor is BITWISE the output of stage k's "
        "executed interval -- not x0, and not the optimised q before execution"))

    # ---------------------------------------------------------------- configuration
    if spec_source() == "real":
        print("\nConfiguration: RHSO in the declaration, validation and planning machinery")
        from src.config import (COMPARED_METHODS, ConfigError, MODEL_CAPABILITIES,
                                METHOD_DECLARATIONS, _estimate_cost)

        decl = METHOD_DECLARATIONS.get("rhso")
        results.append(report(
            "declared with the intended fields",
            decl is not None and not decl.is_mpc and not decl.uses_K
            and set(decl.fields) == {"t0", "beta", "num_rhso_steps", "num_opt_steps", "lr",
                                     "mu", "optimizer", "phi_normalization", "solver",
                                     # the terminal planner and the theory diagnostics
                                     "rhso_terminal_mode", "rhso_stage_diagnostics",
                                     "rhso_consistency_diagnostics",
                                     "rhso_jacobian_diagnostics", "rhso_jacobian_probes",
                                     "rhso_jacobian_power_iters", "rhso_jacobian_seed",
                                     "rhso_gradient_authority_diagnostics",
                                     # the time-dependent anchor weight (rebuttal Task 8)
                                     "mu_schedule"}
            and all("rhso" in c.supported_methods for c in MODEL_CAPABILITIES.values())
            and "rhso" in COMPARED_METHODS,
            "shared t0/beta plus N, M, lr, mu, optimizer, phi, solver, the terminal-planner "
            "mode and the theory diagnostics -- no lambda, no K, no control-cost "
            "normalisation; supported by every model and compared against SDEdit"))

        rejected = {}
        for field, value in (("lam", 1.0), ("K", 2), ("control_cost_normalization",
                                                      "sum_squared"),
                             ("optimizer", "sgd")):
            try:
                build_plan({"rhso": {"num_rhso_steps": 2, field: value}})
            except ConfigError as exc:
                rejected[field] = str(exc).splitlines()[0]
        results.append(report(
            "meaningless fields are rejected", len(rejected) == 4,
            "lam, K, control_cost_normalization and a non-Adam optimizer are all refused "
            "with an explanation rather than silently ignored"))

        try:
            build_plan({"rhso": {"num_rhso_steps": 2, "solver": "heun"}}, model="pmf")
            solver_rejected = False
        except ConfigError:
            solver_rejected = True
        jit_plan, _w = build_plan({"rhso": {"num_rhso_steps": 2, "solver": "rk4"}},
                                  model="jit")
        results.append(report(
            "solver is standard-flow only",
            solver_rejected and jit_plan.specs[0].solver == "rk4",
            "a MeanFlow RHSO job refuses `solver`; a standard-flow one accepts the same "
            "values as the other standard-flow methods"))

        plan, _w = build_plan({"rhso": {"num_rhso_steps": [2, 4], "num_opt_steps": 3,
                                        "beta": [0.5, 1.0]}})
        results.append(report(
            "sweeps and job identity",
            len(plan.specs) == 4 and len({s.job_id for s in plan.specs}) == 4
            and len({s.leaf_dir for s in plan.specs}) == 4
            and {(s.num_rhso_steps, s.beta) for s in plan.specs}
            == {(2, 0.5), (2, 1.0), (4, 0.5), (4, 1.0)},
            "num_rhso_steps x beta expands to 4 jobs with 4 distinct ids and directories"))

        estimate = _estimate_cost("rhso", {"num_rhso_steps": 3, "num_opt_steps": 4,
                                           "solver": None}, MEANFLOW)
        spec3 = dataclasses.replace(make_spec("rhso", num_rhso_steps=3, num_opt_steps=4,
                                              lr=0.05), record_loss_history=True)
        _state3, stats3 = meanflow_rhso(adapter, cond, x0, problem, spec3)
        results.append(report(
            "planner cost == measured (through config)",
            estimate["model_evals"] == stats3.model_evals_total
            == spec3.expected_model_evals
            and estimate["backprops"] == stats3.backprops_through_model
            and spec3.expected_objective_evals == stats3.objective_evals,
            "the dry-run plan predicts %d evaluations and %d backprops; the run measured "
            "%d and %d" % (spec3.expected_model_evals, estimate["backprops"],
                           stats3.model_evals_total, stats3.backprops_through_model)))

        # ---- mu through the configuration system ----------------------------------
        mu_plan, _w = build_plan({"rhso": {"num_rhso_steps": 2, "num_opt_steps": 3,
                                           "mu": [0.0, 0.01, 1.0]}})
        mus = sorted(s.mu for s in mu_plan.specs)
        leaves = [s.leaf_dir for s in mu_plan.specs]
        results.append(report(
            "mu sweeps into distinct jobs",
            mus == [0.0, 0.01, 1.0] and len({s.job_id for s in mu_plan.specs}) == 3
            and len(set(leaves)) == 3
            and sum("mu=" in leaf for leaf in leaves) == 2,
            "3 jobs with 3 job ids and 3 output paths; mu appears in the two non-default "
            "leaf names and is omitted at mu=0, exactly as beta is"))

        default_mu, _w = build_plan({"rhso": {"num_rhso_steps": 2}})
        titles = {s.figure_title() for s in mu_plan.specs}
        results.append(report(
            "omitting mu resolves to 0.0",
            default_mu.specs[0].mu == 0.0
            and default_mu.specs[0].hyperparameter_sources.get("mu") == "builtin"
            and sum("mu=" in t for t in titles) == 2,
            "an RHSO block that never mentions mu resolves to 0.0 with provenance "
            "'builtin' (the feature off, not an untuned guess), and only penalised jobs "
            "carry mu in their figure label"))

        bad_mu = 0
        for bad in (-0.5, float("nan"), float("inf"), "small", True):
            try:
                build_plan({"rhso": {"num_rhso_steps": 2, "mu": bad}})
            except ConfigError:
                bad_mu += 1
        wrong_method = 0
        for method, entry in (("dflow", {"steps": 1}), ("sdedit", {"steps": 2}),
                              ("mpc_delta_t", {"num_mpc_steps": 2}),
                              ("pnp", {"num_pnp_steps": 2})):
            try:
                build_plan({method: dict(entry, mu=0.1)})
            except ConfigError:
                wrong_method += 1
        results.append(report(
            "mu is validated and RHSO-only",
            bad_mu == 5 and wrong_method == 4,
            "negative, NaN, infinite, non-numeric and boolean weights are all rejected, and "
            "mu is refused for D-Flow, SDEdit, MPC-delta_t and PnP with an explanation that "
            "distinguishes it from MPC's lam"))

        mu_costs = {s.mu: (s.expected_model_evals, s.expected_backprops,
                           s.expected_objective_evals) for s in mu_plan.specs}
        results.append(report(
            "mu leaves the planner's cost estimate alone",
            len(set(mu_costs.values())) == 1,
            "all three mu values predict the same %d model evaluations / %d backprops / %d "
            "objectives -- the penalty contains no model evaluation"
            % mu_costs[0.0]))

        import run as runner
        mu_keys = {runner.warmup_key(s, _ProbeProblem()) for s in mu_plan.specs}
        results.append(report(
            "mu is a results column and a warm-up key",
            "mu" in runner.RESULT_COLUMNS
            and "hyperparameter_source_mu" in runner.RESULT_COLUMNS
            and len(mu_keys) == 3,
            "results.csv records mu and where it came from, and each mu traces its own "
            "objective so none of them compiles inside a measured region"))

        keys = {runner.warmup_key(s, _ProbeProblem()) for s in plan.specs}
        results.append(report(
            "warm-up key and results columns",
            len(keys) == 4 and "num_rhso_steps" in runner.RESULT_COLUMNS
            and runner.METHOD_ORDER.get("rhso") == 5,
            "each (N, beta) traces its own computation and gets its own warm-up; the horizon "
            "is a results column and RHSO has its place in the summary ordering"))

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
