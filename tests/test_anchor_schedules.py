"""RHSO anchor schedules (rebuttal Task 8): mu_k = mu0 * m(t_k).

Run with:  python tests/test_anchor_schedules.py        (needs jax + optax + torch + pyyaml + git)

No checkpoint, no GPU.  What is checked:

  * the multipliers at N = 8 and N = 4 against the task's table, and decreasing = 1/increasing;
  * the stage times of the pMF frozen RHSO entries are k/N (t0 = 1, beta = 1);
  * DEFAULT UNCHANGED: with mu_schedule absent, None or "constant", RHSO reproduces the
    pre-change implementation (src/rhso.py at commit 347a4a1, loaded from git) BIT FOR BIT on
    synthetic adapters, on the JAX MeanFlow loop and on the PyTorch standard-flow loop;
  * explicitly asking for "constant" resolves to the same job (id, directory, spec) as the
    default and gives the same output;
  * evaluation / backprop / forward counts and the planner's cost estimate do not depend on the
    schedule;
  * the non-constant schedules really use mu_k: total = fidelity + mu_k * R at every stage, and
    on the JAX loop mu_k reaches the objective as an array argument; mu0 = 0 is identical
    under every schedule;
  * the field is declared (validation, warm-up key, results column, resume rule, job identity);
  * the tuning configuration (22 per task, 110 jobs, fields copied from the frozen entries), the
    freeze script on a planted tuning run, and the table script on planted 100-image runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib
import csv
import dataclasses
import importlib.util
import io
import subprocess
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import yaml

from spec_support import build_plan, make_spec, source_banner
from test_flower import ToyJiT, ToyMeanFlow, make_problem
from src.models.base import Conditioning

REPO = Path(__file__).resolve().parents[1]
PRE_CHANGE_COMMIT = "347a4a1"          # Task 7, the last commit before mu_schedule existed
RES, BATCH = 16, 2

INCREASING_N8 = [1, 1.0204, 1.1111, 1.3600, 2, 3.7778, 10, 50]
INCREASING_N4 = [1, 1.1111, 2, 10]


def report(name, ok, detail=""):
    print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pre_change_rhso():
    """src/rhso.py as it was before this task, imported as a sibling module of src."""
    source = subprocess.run(["git", "-C", str(REPO), "show", "%s:src/rhso.py" % PRE_CHANGE_COMMIT],
                            capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / "rhso_before_task8.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("src._rhso_before_task8", tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StripField:
    """A resolved spec WITHOUT `mu_schedule`, i.e. the interface before the field existed."""

    def __init__(self, spec, name="mu_schedule"):
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, name):
        if name == object.__getattribute__(self, "_name"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_spec"), name)


class _ProbeProblem:
    class measurement:
        shape = (BATCH, 256, 256, 3)


COUNTERS = ("model_evals_total", "model_evals_planning", "backprops_through_model",
            "network_forwards", "objective_evals", "optimizer_iterations",
            "control_iterations", "data_gradient_evals")


def counters(stats):
    return tuple(getattr(stats, k) for k in COUNTERS)


def histories(stats):
    return (list(stats.loss_history), list(stats.fidelity_history),
            list(stats.state_penalty_history))


def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(jax.device_get(x))


# =====================================================================================
# runners on the synthetic adapters
# =====================================================================================
def pmf_setup(n_steps=8):
    problem = make_problem("denoising")
    target = problem.ground_truth * 0.9
    noise = np.random.default_rng(11).standard_normal((BATCH, RES, RES, 3)).astype(np.float32)
    return problem, target, noise


def run_meanflow(module, spec, problem, target, noise):
    adapter = ToyMeanFlow(target)
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    x0 = adapter.to_native_noise(noise)
    state, stats = module.meanflow_rhso(adapter, cond, x0, problem, spec)
    return to_np(state), stats


def run_flow(module, spec, problem, target, noise):
    adapter = ToyJiT(target)
    cond = Conditioning(labels=np.zeros((BATCH,), np.int32), guidance={})
    x0 = adapter.to_native_noise(noise)
    state, stats = module.flow_rhso(adapter, cond, x0, problem, spec)
    return to_np(state), stats


def pmf_spec(**over):
    base = dict(t0=1.0, num_rhso_steps=8, num_opt_steps=3, lr=0.05, mu=0.5,
                record_loss_history=True)
    base.update(over)
    return make_spec("rhso", model="pmf", **base)


def jit_spec(**over):
    base = dict(t0=1.0, num_rhso_steps=4, num_opt_steps=3, lr=0.05, mu=0.5, solver="heun",
                record_loss_history=True)
    base.update(over)
    return make_spec("rhso", model="jit", **base)


def same(a, b):
    return np.array_equal(a[0], b[0]) and counters(a[1]) == counters(b[1]) \
        and histories(a[1]) == histories(b[1])


# =====================================================================================
# checks
# =====================================================================================
def check_multipliers(results):
    from src.rhso import mu_schedule_multiplier, rhso_stage_mus, rhso_time_grid
    print("\nmultipliers")
    s8 = pmf_spec(mu=1.0, mu_schedule="increasing")
    d8 = pmf_spec(mu=1.0, mu_schedule="decreasing")
    inc8, dec8 = rhso_stage_mus(s8), rhso_stage_mus(d8)
    results.append(report(
        "N = 8 stage times are k/8", rhso_time_grid(s8)[:-1] == [k / 8 for k in range(8)],
        str(rhso_time_grid(s8)[:-1])))
    results.append(report(
        "N = 8 increasing multipliers match the table",
        [round(v, 4) for v in inc8] == INCREASING_N8, ", ".join("%.4f" % v for v in inc8)))
    results.append(report(
        "N = 8 decreasing multipliers are their reciprocals",
        all(abs(d - 1.0 / i) <= 1e-12 for d, i in zip(dec8, inc8)),
        ", ".join("%.4f" % v for v in dec8)))
    inc4 = rhso_stage_mus(pmf_spec(mu=1.0, num_rhso_steps=4, mu_schedule="increasing"))
    results.append(report(
        "N = 4 increasing multipliers are 1, 1.1111, 2, 10",
        [round(v, 4) for v in inc4] == INCREASING_N4, ", ".join("%.4f" % v for v in inc4)))
    const = rhso_stage_mus(pmf_spec(mu=0.3))
    results.append(report(
        "constant gives mu0 at every stage", const == [0.3] * 8, str(const)))
    scaled = rhso_stage_mus(pmf_spec(mu=0.3, mu_schedule="increasing"))
    results.append(report(
        "mu_k = mu0 * m(t_k)", all(abs(a - 0.3 * b) <= 1e-12 for a, b in zip(scaled, inc8))))
    try:
        mu_schedule_multiplier("increasing", 1.0)
        ok = False
    except ValueError:
        ok = True
    results.append(report("t = 1 is refused (RHSO never optimises there)", ok))


def check_frozen_stage_times(results):
    from src.config import load_config, resolve_run_plan, validate_config
    from src.rhso import rhso_time_grid
    print("\nstage times of the pMF frozen RHSO entries")
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_config(str(REPO / "configs" / "experiments_final_frozen100.yaml"))
        plan = resolve_run_plan(cfg, validate_config(cfg), run_id="t")
    specs = [s for s in plan.specs if s.model == "pmf" and s.method == "rhso"]
    ok = len(specs) == 5 and all(
        rhso_time_grid(s)[:-1] == [k / s.num_rhso_steps for k in range(s.num_rhso_steps)]
        and s.t0 == 1.0 and float(s.beta) == 1.0 for s in specs)
    results.append(report(
        "t_k = k/N on all five (N = 4: 0..0.75, N = 8: 0..0.875)", ok,
        ", ".join("%s N=%d" % (s.problem, s.num_rhso_steps) for s in specs)))


def check_default_unchanged(results, legacy):
    import src.rhso as rhso
    print("\ndefault reproduces the pre-change implementation (commit %s)" % PRE_CHANGE_COMMIT)
    problem, target, noise = pmf_setup()
    for mu in (0.5, 0.0):
        spec = pmf_spec(mu=mu)
        before = run_meanflow(legacy, _StripField(spec), problem, target, noise)
        variants = {"absent": _StripField(spec), "None": dataclasses.replace(spec, mu_schedule=None),
                    "default": spec, "explicit constant": pmf_spec(mu=mu, mu_schedule="constant")}
        ok = all(same(before, run_meanflow(rhso, v, problem, target, noise))
                 for v in variants.values())
        results.append(report(
            "JAX MeanFlow loop, mu0=%g: bitwise identical" % mu, ok,
            "state, %d-entry loss history and counters; absent / None / default / constant"
            % len(before[1].loss_history)))

    jproblem = make_problem("box_inpaint")
    jtarget = jproblem.ground_truth * 0.9
    for mu in (0.5, 0.0):
        spec = jit_spec(mu=mu)
        before = run_flow(legacy, _StripField(spec), jproblem, jtarget, noise)
        variants = [_StripField(spec), dataclasses.replace(spec, mu_schedule=None), spec,
                    jit_spec(mu=mu, mu_schedule="constant")]
        ok = all(same(before, run_flow(rhso, v, jproblem, jtarget, noise)) for v in variants)
        results.append(report(
            "PyTorch standard-flow loop, mu0=%g: bitwise identical" % mu, ok,
            "Heun suffix planner; state, loss history and counters"))


def check_explicit_constant(results):
    print("\nexplicit constant = default")
    default_plan, _ = build_plan({"rhso": {"t0": 1.0, "num_rhso_steps": 4, "mu": 0.3}})
    explicit_plan, _ = build_plan({"rhso": {"t0": 1.0, "num_rhso_steps": 4, "mu": 0.3,
                                            "mu_schedule": "constant"}})
    a, b = default_plan.specs[0], explicit_plan.specs[0]
    da, db = a.to_dict(), b.to_dict()
    sa, sb = da.pop("hyperparameter_sources"), db.pop("hyperparameter_sources")
    # The only difference is provenance: where the value came from (builtin vs method).
    results.append(report(
        "same job id, directory and resolved spec (provenance aside)",
        a.job_id == b.job_id and a.leaf_dir == b.leaf_dir and da == db
        and a.mu_schedule == "constant"
        and {k for k in sa if sa[k] != sb.get(k)} == {"mu_schedule"}
        and (sa["mu_schedule"], sb["mu_schedule"]) == ("builtin", "method"),
        "%s %s" % (a.job_id, a.leaf_dir)))
    inc_plan, _ = build_plan({"rhso": {"t0": 1.0, "num_rhso_steps": 4, "mu": 0.3,
                                       "mu_schedule": "increasing"}})
    c = inc_plan.specs[0]
    results.append(report(
        "a non-constant schedule is a different job",
        c.job_id != a.job_id and "musched=increasing" in c.leaf_dir, c.leaf_dir))


def check_counts(results):
    import src.rhso as rhso
    print("\ncounts do not depend on the schedule")
    problem, target, noise = pmf_setup()
    runs = {s: run_meanflow(rhso, pmf_spec(mu=0.5, mu_schedule=s), problem, target, noise)
            for s in ("constant", "increasing", "decreasing")}
    results.append(report(
        "JAX: evaluations, backprops, forwards, objectives",
        len({counters(r[1]) for r in runs.values()}) == 1,
        str(dict(zip(COUNTERS, counters(runs["constant"][1]))))))
    jproblem = make_problem("box_inpaint")
    jruns = {s: run_flow(rhso, jit_spec(mu=0.5, mu_schedule=s), jproblem,
                         jproblem.ground_truth * 0.9, noise)
             for s in ("constant", "increasing", "decreasing")}
    results.append(report(
        "PyTorch: evaluations, backprops, forwards, objectives",
        len({counters(r[1]) for r in jruns.values()}) == 1))
    costs = set()
    for s in ("constant", "increasing", "decreasing"):
        spec = pmf_spec(mu=0.5, mu_schedule=s)
        costs.add((spec.expected_model_evals, spec.expected_backprops,
                   spec.expected_objective_evals, spec.expected_data_gradients))
    results.append(report("planner cost estimate", len(costs) == 1, str(costs)))
    return runs, jruns


def check_schedule_effect(results, runs, jruns):
    import src.rhso as rhso
    from src.rhso import rhso_stage_mus
    print("\nthe non-constant schedules use mu_k")
    m = 3
    for name, rs, make in (("JAX", runs, pmf_spec), ("PyTorch", jruns, jit_spec)):
        ok = True
        for s in ("increasing", "decreasing"):
            stats = rs[s][1]
            mus = rhso_stage_mus(make(mu=0.5, mu_schedule=s))
            for i, (loss, fid, pen) in enumerate(zip(*histories(stats))):
                mu_k = mus[i // m]
                ok &= abs(loss - (fid + mu_k * pen)) <= 1e-5 * max(1.0, abs(loss))
            ok &= not np.array_equal(rs[s][0], rs["constant"][0])
        results.append(report(
            "%s: total = fidelity + mu_k R per stage; output differs from constant" % name, ok))

    seen = []
    original = rhso.rhso_scheduled_objective

    def spy(B, fidelity, q, anchor, mu_stage):
        seen.append(mu_stage)
        return original(B, fidelity, q, anchor, mu_stage)

    problem, target, noise = pmf_setup()
    rhso.rhso_scheduled_objective = spy
    try:
        run_meanflow(rhso, pmf_spec(mu=0.5, mu_schedule="increasing"), problem, target, noise)
        n_scheduled = len(seen)
        run_meanflow(rhso, pmf_spec(mu=0.5), problem, target, noise)
        n_after_constant = len(seen)
    finally:
        rhso.rhso_scheduled_objective = original
    values = sorted({round(float(jax.device_get(v)), 6) for v in seen})
    results.append(report(
        "JAX: mu_k is an array argument; constant never takes the new path",
        n_scheduled == 8 * 3 and n_after_constant == n_scheduled
        and all(isinstance(v, jax.Array) for v in seen)
        and all(v.dtype == jnp.float32 for v in seen) and len(values) == 8,
        "%d calls, %d distinct stage values" % (n_scheduled, len(values))))

    zero = {s: run_meanflow(rhso, pmf_spec(mu=0.0, mu_schedule=s), problem, target, noise)
            for s in ("constant", "increasing", "decreasing")}
    results.append(report(
        "mu0 = 0: every schedule gives the same output",
        all(np.array_equal(zero[s][0], zero["constant"][0]) for s in zero)))


def check_declaration(results):
    import run as runner
    from src.config import ConfigError
    print("\ndeclaration")
    for bad, method in (("linear", "rhso"), (1, "rhso")):
        try:
            build_plan({method: {"t0": 1.0, "mu_schedule": bad}})
            ok = False
        except ConfigError:
            ok = True
        results.append(report("mu_schedule=%r is rejected" % (bad,), ok))
    try:
        build_plan({"sdedit": {"t0": 1.0, "mu_schedule": "increasing"}})
        ok = False
    except ConfigError:
        ok = True
    results.append(report("mu_schedule on a non-RHSO method is rejected", ok))
    plan, _ = build_plan({"rhso": {"t0": 1.0, "mu": 0.3,
                                   "mu_schedule": ["constant", "increasing", "decreasing"]}})
    keys = {runner.warmup_key(s, _ProbeProblem()) for s in plan.specs}
    results.append(report(
        "sweepable; each schedule is its own warm-up key",
        len(plan.specs) == 3 and len(keys) == 3
        and {s.mu_schedule for s in plan.specs} == {"constant", "increasing", "decreasing"}))
    results.append(report(
        "results column and record field",
        "mu_schedule" in runner.RESULT_COLUMNS
        and runner.RESULT_COLUMNS.index("mu_schedule") == runner.RESULT_COLUMNS.index("mu") + 1))
    current = {"mu_schedule": "constant", "dynamics_family": "meanflow"}
    results.append(report(
        "resume: a pre-field job matches constant only",
        "mu_schedule" in runner.NEW_SPEC_FIELDS
        and runner._describes_legacy_behaviour("mu_schedule", current)
        and runner._describes_legacy_behaviour("mu_schedule", {"mu_schedule": None})
        and not runner._describes_legacy_behaviour("mu_schedule", {"mu_schedule": "increasing"})))
    sd, _ = build_plan({"sdedit": {"t0": 1.0}})
    results.append(report("None on every other method", sd.specs[0].mu_schedule is None))


def check_tuning_config(results):
    from src.config import load_config, resolve_run_plan, validate_config
    tune = _load_script("make_anchor_schedule_tuning")
    print("\ntuning configuration")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tuning.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            tune.write_tuning(tune.BENCHMARK, str(out))
        regenerated = out.read_text()
        smoke_regenerated = (Path(tmp) / "tuning_smoke.yaml").read_text()
    committed = (REPO / tune.TUNING_OUT).read_text()
    results.append(report(
        "committed config = generator output",
        regenerated.split("runtime:", 1)[1] == committed.split("runtime:", 1)[1]))
    results.append(report(
        "smoke body matches the committed smoke",
        smoke_regenerated.split("runtime:", 1)[1]
        == (REPO / "configs" / "experiments_anchor_schedule_tuning_smoke.yaml").read_text()
        .split("runtime:", 1)[1]))

    cfg = yaml.safe_load(committed)
    frozen = yaml.safe_load((REPO / tune.BENCHMARK).read_text())
    sweep = yaml.safe_load((REPO / "configs" / "experiments_hpo_sweep_checkpoints.yaml").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        c = load_config(str(REPO / tune.TUNING_OUT))
        plan = resolve_run_plan(c, validate_config(c), run_id="t")
    per_task = {}
    for s in plan.specs:
        per_task.setdefault(s.problem, []).append(s)
    results.append(report(
        "22 configurations per task, 110 in total, pMF RHSO only",
        len(plan.specs) == 110 and all(len(v) == 22 for v in per_task.values())
        and len(per_task) == 5 and {(s.model, s.method) for s in plan.specs} == {("pmf", "rhso")}))
    grid_ok = True
    for problem, specs in per_task.items():
        pairs = sorted((s.mu, s.mu_schedule) for s in specs)
        want = sorted([(0.0, "constant")] + [(float(m), sch) for m in tune.MU_GRID
                                             for sch in tune.SCHEDULES])
        grid_ok &= pairs == want
    results.append(report("grid: 7 mu0 x 3 schedules + mu0 = 0 once per task", grid_ok))
    fields_ok = True
    for name, exp in cfg["experiments"].items():
        task = name.split("__")[0]
        src_exp = frozen["experiments"][task + "__rhso"]
        src = src_exp["models"]["pmf"]["methods"]["rhso"]
        mine = exp["models"]["pmf"]["methods"]["rhso"]
        fields_ok &= {k: v for k, v in mine.items() if k not in ("mu", "mu_schedule")} == \
            {k: v for k, v in src.items() if k != "mu"}
        fields_ok &= exp["degradation"] == src_exp["degradation"] and exp["problem"] == src_exp["problem"]
    fields_ok &= cfg["defaults"] == frozen["defaults"] and cfg["metrics"] == frozen["metrics"]
    results.append(report("every other field copied from the frozen pMF __rhso entry", fields_ok))
    first = next(iter(sweep["experiments"].values()))
    results.append(report(
        "pool, 8 images and batch size of the tuning protocol",
        cfg["data"]["local_folder"] == sweep["data"]["local_folder"]
        and all(e["num_images"] == 8 == first["num_images"] for e in cfg["experiments"].values())
        and all(e["models"]["pmf"]["batch_size"] == first["models"]["jit"]["batch_size"] == 4
                for e in cfg["experiments"].values()),
        cfg["data"]["local_folder"]))


def plant_tuning(path, lpips_of):
    """A results.csv like run.py's, for the 110 tuning jobs; lpips_of(task, sched, mu)."""
    tune = _load_script("make_anchor_schedule_tuning")
    frozen = tune.matched.load_yaml(tune.BENCHMARK)
    rows = []
    for problem in tune.hpo.PROBLEMS:
        fixed = dict(tune.benchmark_entry(frozen, problem)[0])
        exp = tune.hpo.EXP_NAME[problem]
        cells = [(exp + tune.MU0_SUFFIX, "constant", 0.0)] + [
            (exp + tune.TUNE_SUFFIX, s, float(m)) for m in tune.MU_GRID for s in tune.SCHEDULES]
        for i, (e, s, mu) in enumerate(cells):
            lp = lpips_of(problem, s, mu)
            rows.append(dict(job_id="%s_%d" % (problem, i), experiment=e, problem=problem,
                             model="pmf", method="rhso", status="ok", num_images=8,
                             t0=fixed["t0"], num_rhso_steps=fixed["num_rhso_steps"],
                             num_opt_steps=fixed["num_opt_steps"], lr=fixed["lr"], mu=mu,
                             mu_schedule=s, lpips=lp, psnr=30 - 10 * lp, ssim=0.9 - lp))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_freeze(results, tmp):
    from src.config import load_config, resolve_run_plan, validate_config
    freeze = _load_script("freeze_anchor_schedules")
    print("\nfreeze script")

    def lpips_of(problem, s, mu):
        if problem == "denoising":            # every schedule prefers mu0 = 0
            return 0.2 + 0.01 * mu
        if problem == "deblur":               # constant and decreasing pick 0, increasing 0.3
            return 0.15 - (0.02 if (s == "increasing" and mu == 0.3) else 0) + 0.001 * mu
        best = {"constant": 1.0, "increasing": 0.1, "decreasing": 3.0}[s]
        return 0.1 + 0.01 * abs(np.log10(mu / best)) if mu > 0 else 0.13

    run = tmp / "tuning_run"
    run.mkdir()
    plant_tuning(run / "results.csv", lpips_of)
    out = tmp / "frozen.yaml"
    with contextlib.redirect_stdout(io.StringIO()):
        freeze.main(["--from", str(run), "--out", str(out), "--out-dir", str(tmp / "curves")])
    cfg = yaml.safe_load(out.read_text())
    names = sorted(cfg["experiments"])
    results.append(report(
        "selection and shared mu0 = 0 blocks",
        "denoising__anchor_constant_increasing_decreasing" in names
        and "deblurring__anchor_constant_decreasing" in names
        and cfg["experiments"]["deblurring__anchor_increasing"]["models"]["pmf"]["methods"]
        ["rhso"]["mu"] == 0.3
        and cfg["experiments"]["super_resolution__anchor_decreasing"]["models"]["pmf"]
        ["methods"]["rhso"]["mu"] == 3
        and len(names) == 1 + 2 + 3 * 3, ", ".join(names[:4]) + ", ..."))
    text = out.read_text()
    results.append(report(
        "15 (task, schedule) entries; comments name schedule, mu0 and tuning LPIPS",
        text.count("schedule ") >= 15 and "15 (task, schedule) entries" in text
        and "schedule increasing: selected mu0 = 0.3, tuning LPIPS 0.1303" in text))
    with contextlib.redirect_stdout(io.StringIO()):
        c = load_config(str(out))
        plan = resolve_run_plan(c, validate_config(c), run_id="t")
    frozen = yaml.safe_load((REPO / "configs" / "experiments_final_frozen100.yaml").read_text())
    ok = all(s.num_images == 100 and c["data"]["local_folder"] == frozen["data"]["local_folder"]
             for s in plan.specs)
    for name, exp in cfg["experiments"].items():
        src = frozen["experiments"][name.split("__")[0] + "__rhso"]["models"]["pmf"]
        ok &= exp["models"]["pmf"]["batch_size"] == src["batch_size"]
        mine = exp["models"]["pmf"]["methods"]["rhso"]
        ok &= {k: v for k, v in mine.items() if k not in ("mu", "mu_schedule")} == \
            {k: v for k, v in src["methods"]["rhso"].items() if k != "mu"}
        if mine["mu"] == 0:
            ok &= mine["mu_schedule"] == "constant"
    results.append(report(
        "frozen config resolves: 100 images, benchmark pool and batch sizes, frozen fields",
        ok and len(plan.specs) == 12, "%d jobs" % len(plan.specs)))
    curves = list(csv.DictReader(open(tmp / "curves" / "anchor_schedule_tuning_curves.csv")))
    tex = (tmp / "curves" / "anchor_schedule_tuning_curves.tex").read_text()
    results.append(report(
        "tuning curves: 8 points per (task, schedule), one selected, CSV + LaTeX",
        len(curves) == 5 * 3 * 8
        and sum(r["selected"] == "True" for r in curves) == 15 and "\\textbf{" in tex))
    bad = tmp / "bad_run"
    bad.mkdir()
    rows = list(csv.DictReader(open(run / "results.csv")))
    with open(bad / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[:-1])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            freeze.main(["--from", str(bad), "--out", str(tmp / "x.yaml"),
                         "--out-dir", str(tmp / "x")])
        ok = False
    except SystemExit as exc:
        ok = "incomplete" in str(exc)
    results.append(report("an incomplete grid is refused", ok))
    return out


def plant_per_image(path, jobs):
    """jobs: [(experiment, problem, mu, schedule, job_id, {image: (psnr, ssim, lpips)}, rmse)]."""
    cols = ["run_id", "job_id", "experiment", "problem", "image_id", "model", "method", "t0",
            "num_rhso_steps", "num_opt_steps", "lr", "mu", "mu_schedule", "psnr", "ssim",
            "lpips", "measurement_rmse", "status"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for exp, problem, mu, sched, job_id, images, rmse in jobs:
            for image, (p, s, l) in images.items():
                w.writerow(dict(run_id="r", job_id=job_id, experiment=exp, problem=problem,
                                image_id=image, model="pmf", method="rhso", t0=1.0,
                                num_rhso_steps=4, num_opt_steps=40, lr=0.03, mu=mu,
                                mu_schedule=sched, psnr=p, ssim=s, lpips=l,
                                measurement_rmse=rmse, status="ok"))


def check_table(results, tmp):
    table = _load_script("anchor_schedule_table")
    ps = _load_script("paired_stats")
    print("\ntable script")
    rng = np.random.default_rng(5)
    ids = ["img%03d" % i for i in range(100)]
    base = {i: (28 + rng.normal(), 0.8 + 0.01 * rng.normal(), 0.1 + 0.01 * rng.normal())
            for i in ids}
    shift = {i: (v[0] + 0.3, v[1], v[2] - 0.01) for i, v in base.items()}
    final = tmp / "final_run"
    final.mkdir()
    plant_per_image(final / "results_per_image.csv", [
        ("denoising__anchor_constant_increasing", "denoising", 0.0, "constant", "j0", base, 0.2),
        ("denoising__anchor_decreasing", "denoising", 3.0, "decreasing", "j1", shift, 0.21),
    ])
    ref = tmp / "ref_run"
    ref.mkdir()
    plant_per_image(ref / "results_per_image.csv", [
        ("denoising__rhso", "denoising", 1.0, "", "p0", shift, 0.19),
        ("denoising__core", "denoising", 1.0, "", "p1", base, 0.5),
    ])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        table.main(["--final", str(final), "--reference", str(ref), "--out-dir", str(tmp / "t")])
    rows = list(csv.DictReader(open(tmp / "t" / "anchor_schedule_table.csv")))
    by = {(r["role"], r["schedule"]): r for r in rows}
    want = ps.bootstrap({m: np.subtract([shift[i][k] for i in ids], [base[i][k] for i in ids])
                         for k, m in enumerate(ps.METRICS)})
    dec = by[("schedule", "decreasing")]
    ok = (len(rows) == 4 and [r["schedule"] or "ref" for r in rows]
          == ["constant", "increasing", "decreasing", "ref"]
          and abs(float(dec["lpips_diff"]) - want["lpips"][0]) < 1e-12
          and abs(float(dec["lpips_ci_lo"]) - want["lpips"][1]) < 1e-12
          and abs(float(dec["psnr_ci_hi"]) - want["psnr"][2]) < 1e-12
          and dec["resamples"] == "10000" and dec["seed"] == "20260903")
    results.append(report("rows per schedule; paired_stats bootstrap (10,000, seed 20260903)",
                          ok, "dLPIPS %s" % dec["lpips_diff"]))
    inc = by[("schedule", "increasing")]
    refrow = by[("reference", "")]
    results.append(report(
        "shared mu0 = 0 run labelled for both, zero difference, marked same run",
        inc["same_run_as_constant"] == "True" and float(inc["lpips_diff"]) == 0.0
        and inc["job_id"] == by[("schedule", "constant")]["job_id"]))
    results.append(report(
        "reference row unpaired (means only) with RMSE next to sigma",
        refrow["lpips_diff"] == "" and refrow["job_id"] == "p0" and refrow["sigma"] == "0.2"
        and float(refrow["measurement_rmse"]) == 0.19
        and (tmp / "t" / "anchor_schedule_table.tex").read_text().count("\\\\") >= 5))


def main():
    print("RHSO anchor schedules (Task 8); %s" % "specs from the real planner")
    results = []
    check_multipliers(results)
    check_frozen_stage_times(results)
    try:
        legacy = load_pre_change_rhso()
    except Exception as exc:                                     # git missing, shallow clone
        results.append(report("load src/rhso.py at %s from git" % PRE_CHANGE_COMMIT, False,
                              "%s: %s" % (type(exc).__name__, exc)))
        legacy = None
    if legacy is not None:
        check_default_unchanged(results, legacy)
    check_explicit_constant(results)
    runs, jruns = check_counts(results)
    check_schedule_effect(results, runs, jruns)
    check_declaration(results)
    check_tuning_config(results)
    with tempfile.TemporaryDirectory() as tmp:
        check_freeze(results, Path(tmp))
        check_table(results, Path(tmp))
    print("\n" + source_banner())
    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
