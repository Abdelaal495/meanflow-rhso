"""Timing and peak-memory run (rebuttal Task 1): configs, measurement mode and summary.

`scripts/make_timing_config.py` writes one timing configuration per prior from the three
frozen 100-image configurations.  This suite resolves every file through the REAL validator
and planner -- the code `run.py --dry-run` runs -- and checks:

  1. regenerating gives the committed files byte for byte;
  2. every block resolves to exactly one job of its own prior, at 3 images (2 in the smoke
     variant), batch size 1, with gpu_memory_profiling and timing_detail on;
  3. every block's resolved job equals its source entry's resolved job in EVERY field except
     the identity fields (job id, experiment name, image count, batch size);
  4. every (source, experiment, prior, method) entry of the three sources is covered exactly
     once, and only entries identical in every field are ever merged;
  5. the smoke variant has the same blocks; no comment claims reuse across checkpoints;
  6. runtime.timing_detail is opt-in: absent by default, a bool when present;
  7. src/timing.py: the device-synchronised chunk timer, the allocator-peak scope rule, the
     allocator settings and hardware records;
  8. run.run_job with a TimingDetail on a synthetic JAX prior records one outer time per
     image and returns bit-identical pixels to the run without it;
  9. scripts/run_timing.py builds a model-only baseline, one process per block with one
     shared run id, and an aggregate, and sets the allocator only for the children;
 10. scripts/summarize_timing.py on synthetic run directories: per-image mean and spread,
     seconds per forward-equivalent and its spread, model-only baselines, GPU warnings, and
     the three LaTeX tables in Table 29's layout.

Runs on CPU with no checkpoint:

    python tests/test_timing_configs.py
"""

import contextlib
import copy
import csv
import dataclasses
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from src.config import ConfigError, resolve_run_plan, validate_config     # noqa: E402

PASSED = [0]
FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mtc = load_module("make_timing_config", REPO_ROOT / "scripts" / "make_timing_config.py")
rt_driver = load_module("run_timing", REPO_ROOT / "scripts" / "run_timing.py")
summ = load_module("summarize_timing", REPO_ROOT / "scripts" / "summarize_timing.py")

PRIORS = ["pmf", "jit", "imf", "sit"]
# Fields that identify a job rather than define its computation.
IDENTITY_FIELDS = {"job_id", "experiment", "num_images", "image_ids", "batch_size"}
SOURCE_RE = re.compile(r"^\s*# (source|also covers \(identical in every field\)): "
                       r"(\S+) :: (\S+) / (\S+) / (\S+)\s*$")


def resolve(cfg):
    cfg = copy.deepcopy(cfg)
    warnings_ = validate_config(cfg)
    return resolve_run_plan(cfg, warnings_, run_id="test")


def spec_fields(spec):
    return {k: v for k, v in json.loads(json.dumps(spec.to_dict(), default=str)).items()
            if k not in IDENTITY_FIELDS}


def source_specs():
    """{(path, experiment, model, method): resolved spec} for the three sources."""
    out = {}
    for _, path in mtc.SOURCES:
        plan = resolve(yaml.safe_load((REPO_ROOT / path).read_text()))
        for spec in plan.specs:
            key = (path, spec.experiment, spec.model, spec.method)
            assert key not in out, key
            out[key] = spec
    return out


# =====================================================================================
def check_generated_files():
    print("\n[1-5] generated configurations")
    files = mtc.outputs()
    stale = [p for p, text in files.items()
             if not (REPO_ROOT / p).exists() or (REPO_ROOT / p).read_text() != text]
    check("regenerating gives the committed files", not stale and len(files) == 8,
          "%d files%s" % (len(files), (" STALE: %s" % stale) if stale else ""))

    sources = source_specs()
    covered = {}
    field_mismatch, shape_errors, merge_errors = [], [], []
    for prior in PRIORS:
        for smoke in (False, True):
            path = mtc.out_path(prior, smoke)
            text = (REPO_ROOT / path).read_text()
            cfg = yaml.safe_load(text)
            plan = resolve(cfg)
            images = mtc.SMOKE_IMAGES if smoke else mtc.TIMING_IMAGES
            rt = cfg["runtime"]
            if not (rt.get("gpu_memory_profiling") is True and rt.get("timing_detail") is True
                    and list(cfg["models"]) == [prior]):
                shape_errors.append("%s runtime/models" % path)
            by_exp = {}
            for spec in plan.specs:
                by_exp.setdefault(spec.experiment, []).append(spec)
            if set(by_exp) != set(cfg["experiments"]):
                shape_errors.append("%s: blocks without jobs" % path)
            for exp, specs in by_exp.items():
                if len(specs) != 1:
                    shape_errors.append("%s/%s: %d jobs" % (path, exp, len(specs)))
                    continue
                spec = specs[0]
                if (spec.model, spec.num_images, spec.batch_size) != (prior, images, 1):
                    shape_errors.append("%s/%s: %s %d %d" % (path, exp, spec.model,
                                                             spec.num_images, spec.batch_size))
            # Map every block to the source entries its comments name.
            lines = text.splitlines()
            pending = []
            for line in lines:
                m = SOURCE_RE.match(line)
                if m:
                    pending.append((m.group(1), (m.group(2), m.group(3), m.group(4),
                                                 m.group(5))))
                    continue
                block = re.match(r"^  (\w+):\s*$", line)
                if block and pending:
                    exp = block.group(1)
                    kept = [k for kind, k in pending if kind == "source"]
                    merged = [k for kind, k in pending if kind != "source"]
                    pending = []
                    if len(kept) != 1 or exp not in by_exp:
                        shape_errors.append("%s/%s: comment" % (path, exp))
                        continue
                    spec = by_exp[exp][0]
                    src = sources.get(kept[0])
                    if src is None or spec_fields(spec) != spec_fields(src):
                        diff = sorted(k for k in spec_fields(spec)
                                      if src is None or spec_fields(spec)[k] != spec_fields(src).get(k))
                        field_mismatch.append("%s/%s %s" % (path, exp, diff[:4]))
                    for dup in merged:
                        d = sources.get(dup)
                        if d is None or spec_fields(d) != spec_fields(src) or \
                                d.problem_params != src.problem_params:
                            merge_errors.append("%s merged into %s" % (dup, kept[0]))
                    if not smoke:
                        for key in kept + merged:
                            covered[key] = covered.get(key, 0) + 1
            if not smoke and prior == "pmf":
                pmf_text = text
    check("every block: one job of its own prior, batch 1, 3 / 2 images, timing on",
          not shape_errors, "; ".join(shape_errors[:3]))
    check("every block equals its source entry in every non-identity field",
          not field_mismatch, "; ".join(field_mismatch[:3]))
    twice = [k for k, n in covered.items() if n != 1]
    missing = sorted(set(sources) - set(covered))
    extra = sorted(set(covered) - set(sources))
    check("every source entry covered exactly once",
          not twice and not missing and not extra,
          "%d source entries; missing %s twice %s extra %s"
          % (len(sources), missing[:2], twice[:2], extra[:2]))
    check("only entries identical in every field are merged", not merge_errors,
          "; ".join(merge_errors[:2]))
    counts = {p: len(mtc.plan(p)) for p in PRIORS}
    expected = {"pmf": 45, "jit": 40, "imf": 30, "sit": 30}
    check("block counts: final 30 per prior, + Flower 5 and matched (pMF 10, JiT 5)",
          counts == expected, str(counts))
    tasks = {p: sorted({e.problem for e, _ in mtc.plan(p)}) for p in PRIORS}
    check("no deduplication across tasks: all five tasks for every method",
          all(len({(e.experiment.split("__")[1], e.method) for e, _ in mtc.plan(p)
                   if e.problem == t}) == len({(e.experiment.split("__")[1], e.method)
                                               for e, _ in mtc.plan(p) if e.problem == "denoising"})
              for p in PRIORS for t in tasks[p]) and all(len(t) == 5 for t in tasks.values()),
          "per-task method sets identical")

    smoke_ok = all(set(yaml.safe_load((REPO_ROOT / mtc.out_path(p, True)).read_text())["experiments"])
                   == set(yaml.safe_load((REPO_ROOT / mtc.out_path(p)).read_text())["experiments"])
                   for p in PRIORS)
    check("smoke variant has exactly the same blocks", smoke_ok)
    forbidden = re.compile(r"reus|carried over|borrow|selected on|tuned on|from another "
                           r"(checkpoint|prior)|same as", re.IGNORECASE)
    bad = [p for p in files if any(forbidden.search(line) for line in files[p].splitlines()
                                   if line.lstrip().startswith("#"))]
    check("no comment claims hyperparameters were reused across checkpoints", not bad, str(bad))
    block = re.search(r"  # source: configs/experiments_final_frozen100\.yaml :: "
                      r"deblurring__rhso / pmf / rhso\n  deblurring__rhso__rhso:\n", pmf_text)
    check("block comment names the source config and entry", block is not None)


def check_opt_in():
    print("\n[6] runtime.timing_detail is opt-in")
    base = yaml.safe_load((REPO_ROOT / mtc.SOURCES[0][1]).read_text())
    try:
        validate_config(copy.deepcopy(base))
        ok_default = "timing_detail" not in base["runtime"]
    except ConfigError:
        ok_default = False
    check("frozen benchmark config validates and has no timing_detail", ok_default)
    on = copy.deepcopy(base)
    on["runtime"]["timing_detail"] = True
    bad = copy.deepcopy(base)
    bad["runtime"]["timing_detail"] = "yes"
    try:
        validate_config(on)
        accepted = True
    except ConfigError:
        accepted = False
    try:
        validate_config(bad)
        rejected = False
    except ConfigError:
        rejected = True
    check("timing_detail: true accepted, a non-bool rejected", accepted and rejected)
    import run
    args = run.parse_args(["--config", "x.yaml"])
    check("run.py --memory-baseline-only defaults to off",
          getattr(args, "memory_baseline_only", None) is False)
    import inspect
    param = inspect.signature(run.run_job).parameters.get("timing")
    check("run.run_job(timing=None) by default", param is not None and param.default is None)


def check_timing_module():
    print("\n[7] src/timing.py")
    from src import timing
    import time as _time
    import jax
    import jax.numpy as jnp
    big = jnp.ones((1024, 1024))
    pending = big @ big @ big                       # dispatched asynchronously
    method = timing.device_sync("jax")
    started = _time.perf_counter()
    jax.block_until_ready(pending)                  # must already be finished
    waited = _time.perf_counter() - started
    check("device_sync('jax') waits for work already queued", method.startswith("jax")
          and waited < 0.005, "%s; residual wait %.2f ms" % (method, waited * 1e3))
    check("device_sync(None) is a no-op", timing.device_sync(None).startswith("none"))

    import jax.numpy as jnp
    timer = timing.ChunkTimer("jax")
    for size in (1, 1, 1):
        timer.start()
        x = jnp.ones((64, 64)) @ jnp.ones((64, 64))
        timer.stop(x, method_seconds=0.001, size=size)
    d = timer.to_dict()
    check("ChunkTimer records one outer and one method time per chunk",
          len(d["chunk_seconds_outer_synced"]) == 3 and d["chunk_sizes"] == [1, 1, 1]
          and all(v >= 0 for v in d["chunk_seconds_outer_synced"]) and d["sync_method"])

    def snap(peak, in_use=1 << 30, pool=None):
        stats = {"peak_bytes_in_use": peak, "bytes_in_use": in_use}
        if pool is not None:
            stats.update(pool_bytes=pool, bytes_limit=30 << 30)
        return {"jax": stats, "torch": None, "nvml_process_gib": None}

    g = 1 << 30
    cases = [((2 * g, 3 * g, 5 * g), "timed_region", 5.0),
             ((2 * g, 4 * g, 4 * g), "warm_up_of_this_configuration (timed region <= this)", 4.0),
             ((4 * g, 4 * g, 4 * g), "model_load (timed region <= this)", 4.0)]
    results = []
    for (load, pre, post), scope, value in cases:
        td = timing.TimingDetail("jax", after_load=snap(load))
        td.snapshots["before_timed_region"] = snap(pre, pool=16 * g)
        td.snapshots["after_timed_region"] = snap(post, pool=30 * g)
        rep = td.allocator_peak({})
        results.append(rep["allocator_peak_scope"] == scope and rep["allocator_peak_gib"] == value)
    check("JAX allocator peak: scope = timed region / warm-up / model load", all(results))
    torch_rep = timing.TimingDetail("torch").allocator_peak(
        {"gpu_memory_source": "torch.cuda.max_memory_allocated", "gpu_peak_gib": 7.6,
         "gpu_baseline_gib": 0.33})
    check("PyTorch allocator peak is the timed-region max_memory_allocated",
          torch_rep["allocator_peak_gib"] == 7.6 and torch_rep["allocator_peak_scope"] == "timed_region")
    td = timing.TimingDetail("jax", after_load=snap(g))
    td.snapshots["before_timed_region"] = snap(g, pool=16 * g)
    td.snapshots["after_timed_region"] = snap(2 * g, pool=30 * g)
    rep = td.report({"gpu_process_peak_gib": 32.56, "gpu_process_baseline_gib": 32.56})
    check("report: pool size before/after, pool limit, NVML labelled as reserved pool",
          rep["jax_pool_bytes_before_timed_region"] == 16 * g
          and rep["jax_pool_bytes_after_timed_region"] == 30 * g
          and rep["jax_bytes_limit_after_timed_region"] == 30 << 30
          and rep["nvml_process_reserved_peak_gib"] == 32.56 and "pool" in rep["nvml_meaning"])

    saved = {k: os.environ.pop(k, None) for k in timing.ALLOCATOR_ENV}
    try:
        unset = timing.allocator_settings()
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        set_ = timing.allocator_settings()
    finally:
        os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    check("allocator settings: memory fraction recorded, default when unset",
          "0.75" in unset["effective"]["memory_fraction"]
          and unset["environment"]["XLA_PYTHON_CLIENT_ALLOCATOR"] is None
          and set_["effective"]["memory_fraction"] == "0.9")
    env = timing.process_environment()
    hw = env["hardware"]
    check("environment records hardware, versions and allocator settings",
          {"gpu_name", "gpu_memory_total_mib", "driver_version", "cuda_driver_version"} <= set(hw)
          and env["versions"].get("jax") and "allocator" in env,
          "gpu=%s jax=%s" % (hw.get("gpu_name"), env["versions"].get("jax")))


def check_run_job():
    print("\n[8] run.run_job with and without timing detail")
    import numpy as np
    import run
    from src import timing
    tf = load_module("test_flower_support", REPO_ROOT / "tests" / "test_flower.py")
    problem = tf.make_problem("denoising")
    spec = dataclasses.replace(tf.make_spec("sdedit", model="pmf", t0=1.0, steps=2),
                               num_images=2, batch_size=1)
    plain = run.run_job(tf.ToyMeanFlow(problem.ground_truth), spec, problem, None)
    detail = timing.TimingDetail("jax")
    timed = run.run_job(tf.ToyMeanFlow(problem.ground_truth), spec, problem, None,
                        timing=detail)
    rep = detail.report({})
    check("one outer synchronised time per image at batch size 1",
          len(rep["chunk_seconds_outer_synced"]) == 2 and rep["chunk_sizes"] == [1, 1]
          and len(rep["chunk_seconds_method"]) == 2,
          "%s" % rep["chunk_seconds_outer_synced"])
    check("outer time >= the method's own timer",
          all(o + 1e-4 >= m for o, m in zip(rep["chunk_seconds_outer_synced"],
                                             rep["chunk_seconds_method"])))
    check("snapshots taken before and after the timed region",
          detail.snapshots["before_timed_region"] is not None
          and detail.snapshots["after_timed_region"] is not None)
    check("pixels bit-identical with and without timing detail",
          np.array_equal(plain["pixels"], timed["pixels"])
          and plain["stats"].model_evals_total == timed["stats"].model_evals_total)


def check_driver():
    print("\n[9] scripts/run_timing.py")
    cfg = mtc.out_path("pmf")
    args = rt_driver.main.__globals__["argparse"].Namespace(
        config=cfg, run_id="timing_pmf", output_root=None, cache_root=None, num_images=None,
        experiments=None, with_checks=False, python="python")
    cmds = rt_driver.build_commands(args)
    blocks = list(yaml.safe_load((REPO_ROOT / cfg).read_text())["experiments"])
    labels = [label for label, _ in cmds]
    per_block = cmds[1:-1]
    check("model-only baseline first, one process per block, aggregate last",
          labels[0] == "model_only_baseline" and labels[-1] == "aggregate"
          and [l for l, _ in per_block] == blocks
          and "--memory-baseline-only" in cmds[0][1] and "--aggregate" in cmds[-1][1])
    check("every process shares one --run-id and selects one block",
          all(argv[argv.index("--run-id") + 1] == "timing_pmf" for _, argv in cmds)
          and all(argv[argv.index("--experiments") + 1] == label for label, argv in per_block))
    check("measured processes skip the GPU-allocating checks",
          all("--no-check" in argv for _, argv in cmds[:-1]))
    before = dict(os.environ)
    env = rt_driver.child_env({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"})
    check("children get the pooling allocator without preallocation; parent untouched",
          env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
          and env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and dict(os.environ) == before)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = rt_driver.main(["--config", cfg, "--run-id", "x", "--dry-run"])
    check("--dry-run prints the commands and runs nothing", code == 0
          and out.getvalue().count("run.py") == len(cmds))


# =====================================================================================
def _job(model, problem, method, experiment, values, evals, bps, peak, scope="timed_region",
         gpu="NVIDIA A100-SXM4-40GB", nvml=10.0, detail=True):
    payload = {"job_id": "%s_%s_%s_%s" % (model, problem, method, experiment),
               "model": model, "problem": problem, "method": method, "experiment": experiment,
               "status": "ok", "num_images": len(values), "batch_size": 1,
               "model_evaluations": evals * len(values),
               "backprops_through_model": bps * len(values),
               "runtime_per_image": sum(values) / len(values),
               "gpu_process_peak_gib": nvml, "gpu_peak_gib": peak,
               "gpu_memory_source": "torch.cuda.max_memory_allocated"}
    if detail:
        payload["timing_detail"] = {
            "chunk_seconds_outer_synced": values, "chunk_seconds_method": values,
            "chunk_sizes": [1] * len(values), "allocator_peak_gib": peak,
            "allocator_peak_scope": scope, "allocator_peak_source": "test",
            "sync_method": "test",
            "environment": {"hardware": {"gpu_name": gpu, "gpu_memory_total_mib": 40960}}}
    return payload


def _write_run(root, name, jobs, baselines):
    run_dir = Path(root) / name
    for job in jobs:
        d = run_dir / job["experiment"] / job["model"] / job["method"] / "leaf"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps(job))
    for model, (alloc, nvml) in baselines.items():
        (run_dir / ("memory_baseline_%s.json" % model)).write_text(json.dumps(
            {"model": model, "allocator_in_use_gib": alloc, "nvml_process_reserved_gib": nvml}))
    return str(run_dir)


def check_summary():
    print("\n[10] scripts/summarize_timing.py")
    tasks = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
    exps = ["denoising", "deblurring", "super_resolution", "random_inpainting", "box_inpainting"]
    means = [12.97, 17.98, 13.23, 13.19, 14.01]
    with tempfile.TemporaryDirectory() as tmp:
        pmf_jobs = []
        for task, exp, m in zip(tasks, exps, means):
            pmf_jobs.append(_job("pmf", task, "rhso", exp + "__rhso", [m - 0.1, m, m + 0.1],
                                 160, 160, 3.0, scope=("timed_region" if task != "deblur"
                                                       else "model_load (timed region <= this)")))
            pmf_jobs.append(_job("pmf", task, "rhso", exp + "__rhso_matched", [m / 2] * 3,
                                 140, 140, 2.5))
            pmf_jobs.append(_job("pmf", task, "sdedit", exp + "__core", [0.03] * 3, 2, 0, 1.2))
        pmf_jobs.append(_job("pmf", "denoising", "pnp", "denoising__core", [1.0], 10, 0, 1.5,
                             detail=False))
        jit_jobs = [_job("jit", "denoising", "rhso", "denoising__rhso", [2.0, 2.2, 2.4], 320, 160,
                         4.0, gpu="NVIDIA A100-SXM4-40GB")]
        runs = [_write_run(tmp, "timing_pmf", pmf_jobs, {"pmf": (1.0, 2.0)}),
                _write_run(tmp, "timing_jit", jit_jobs, {"jit": (0.5, 1.0)})]
        out = Path(tmp) / "summary"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = summ.main(runs + ["--out", str(out)])
        text = buf.getvalue()
        rows = list(csv.DictReader(open(out / "timing_summary.csv")))
        tex = (out / "timing_tables.tex").read_text()
        spread = list(csv.DictReader(open(out / "timing_fe_spread.csv")))

        r = next(r for r in rows if r["prior"] == "pmf" and r["task"] == "denoising"
                 and r["method"] == "rhso" and r["variant"] == "base")
        check("per-image mean and SD over the 3 images",
              code == 0 and abs(float(r["seconds_per_image_mean"]) - 12.97) < 1e-9
              and abs(float(r["seconds_per_image_sd"]) - 0.1) < 1e-9
              and r["seconds_basis"] == "per_image_outer_synced")
        check("seconds per forward-equivalent = s / (Eval + 2 BP)",
              abs(float(r["seconds_per_forward_equivalent"]) - 12.97 / 480) < 1e-12)
        check("allocator and NVML increments over the model-only baseline",
              abs(float(r["allocator_incremental_over_model_gib"]) - 2.0) < 1e-9
              and abs(float(r["nvml_incremental_over_model_gib"]) - 8.0) < 1e-9)
        job_mean = next(r for r in rows if r["method"] == "pnp")
        check("no per-image times -> the job mean, labelled",
              job_mean["seconds_basis"] == "job_mean" and job_mean["seconds_per_image_sd"] == "")
        pmf_spread = next(s for s in spread if s["prior"] == "pmf")
        fe_values = [float(r["seconds_per_forward_equivalent"]) for r in rows if r["prior"] == "pmf"]
        check("per-prior spread of seconds per forward-equivalent",
              abs(float(pmf_spread["fe_max_over_min"]) - max(fe_values) / min(fe_values)) < 1e-9
              and int(pmf_spread["rows"]) == len(fe_values))
        check("Table 29 row: prior multirow, one column per task, two decimals",
              "\\multirow{" in tex and "{pMF-L/16}" in tex
              and " & RHSO & 12.97 & 17.98 & 13.23 & 13.19 & 14.01 \\\\" in tex)
        check("runtime caption names the GPU as in Table 29",
              "\\caption{Runtime per image (seconds/image, batch size one, A100-SXM4-40GB)}" in tex)
        check("three tables: seconds, allocator memory with model-only row, forward-equivalents",
              tex.count("\\begin{table}") == 3 and "Model only & \\multicolumn{5}{c}{1.00}" in tex
              and "3.00$^\\dagger$" in tex and "RHSO (matched BP)" in tex)
        check("GPU name printed per prior, no warning when all match",
              "GPU: NVIDIA A100-SXM4-40GB" in text and "DIFFERENT GPU" not in text)

        jit_jobs[0] = _job("jit", "denoising", "rhso", "denoising__rhso", [2.0], 320, 160, 4.0,
                           gpu="NVIDIA H100 80GB HBM3")
        runs[1] = _write_run(tmp, "timing_jit_h100", jit_jobs, {"jit": (0.5, 1.0)})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summ.main(runs + ["--out", str(out)])
        check("warns when the priors ran on different GPU types",
              "DIFFERENT GPU types" in buf.getvalue())


def main():
    check_generated_files()
    check_opt_in()
    check_timing_module()
    check_run_job()
    check_driver()
    check_summary()
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
