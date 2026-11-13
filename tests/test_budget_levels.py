"""Cheap and mid budget levels on the pixel priors (rebuttal Task 6): configs, timing, table.

`scripts/make_budget_level_config.py` writes the 100-image quality configuration, its 2-image
smoke variant and a timing configuration (3 images, batch size 1) with its smoke variant.
This suite resolves every file through the REAL validator and planner -- the code
`run.py --dry-run` runs -- and checks:

  1. regenerating gives the committed files byte for byte, from the task's table (written
     out here independently of the generator);
  2. the quality config is exactly the 66 entries (36 cheap, 30 mid) at 100 images (2 in the
     smoke); block names <task>__budget_<level> plus _rhc / _rhso only where the benchmark
     batch size differs; every job equals its source entry in every field except its
     identity, its step knob(s) and what the planner derives from them; JiT RHSO keeps the
     direct planner and Heun;
  3. backprops: RHSO N x M, MPC-delta_t (N - 1) x n_ctrl, D-Flow num_opt_steps, MPC-RHC
     num_mpc_steps x n_ctrl x (K - 1);
  4. every entry's comment names its source entry and its level; no comment claims a value
     was reused from another checkpoint;
  5. the timing config: one block per entry, one job each, 3 images (2 in the smoke), batch
     size one, timing_detail on, otherwise equal to the quality job;
  6. scripts/run_timing.py: one baseline per prior, 66 blocks, one run id; submit_timing.sh
     and slurm/run_timing.sh accept budget_levels; summarize_timing.py labels the levels;
  7. scripts/budget_level_table.py on synthetic runs: the hard-coded budgets, the selection
     rule (1.1 x budget, cheapest when none qualifies, JiT RHSO direct only, untimed jobs
     excluded), the RHSO-time level, the paired bootstrap of paired_stats.py against the
     best in-budget method, wins and ties, the outputs, and its refusals.

Runs on CPU with no checkpoint:

    python tests/test_budget_levels.py
"""

import contextlib
import copy
import csv
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import resolve_run_plan, validate_config         # noqa: E402

PASSED = [0]
FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %-66s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_module("make_budget_level_config", REPO_ROOT / "scripts" / "make_budget_level_config.py")
rt_driver = load_module("run_timing", REPO_ROOT / "scripts" / "run_timing.py")
summ = load_module("summarize_timing", REPO_ROOT / "scripts" / "summarize_timing.py")
table = load_module("budget_level_table", REPO_ROOT / "scripts" / "budget_level_table.py")
ps = load_module("paired_stats", REPO_ROOT / "scripts" / "paired_stats.py")

QUALITY = "configs/experiments_budget_levels_pixel.yaml"
QUALITY_SMOKE = "configs/experiments_budget_levels_pixel_smoke.yaml"
TIMING = "configs/experiments_rebuttal_timing_budget_levels.yaml"
TIMING_SMOKE = "configs/experiments_rebuttal_timing_budget_levels_smoke.yaml"
FINAL = "configs/experiments_final_frozen100.yaml"
MATCHED = "configs/experiments_matched_frozen100.yaml"

# The task's tables, verbatim ("--" = no new run).  RHSO is N,M.
TABLE = """
cheap pmf denoising        4,1   1   6  1
cheap pmf deblur           8,8   9  69 --
cheap pmf super_resolution 4,1   1   7  1
cheap pmf random_inpaint   4,1   1   6  1
cheap pmf box_inpaint      8,1   1   8 --
cheap jit denoising        7,2   3  25  6
cheap jit deblur           8,12 14 103 --
cheap jit super_resolution 4,5   3  24  5
cheap jit random_inpaint   7,2   3  25  6
cheap jit box_inpaint      4,5   3  25 --
mid   pmf denoising        4,7   4  27  6
mid   pmf super_resolution 4,7   4  29  3
mid   pmf random_inpaint   4,7   3  27  3
mid   pmf box_inpaint      8,4   4  34 --
mid   jit denoising        7,13 14 102 25
mid   jit super_resolution 4,23 13  99 24
mid   jit random_inpaint   7,14 14 102 25
mid   jit box_inpaint      4,24 14 101 --
"""
EXPECTED = {}          # (level, prior, task, method) -> {knob: value}
for line in TABLE.strip().splitlines():
    level, prior, task, rhso, dt, dflow, rhc = line.split()
    n, m = rhso.split(",")
    EXPECTED[(level, prior, task, "rhso")] = {"num_rhso_steps": int(n), "num_opt_steps": int(m)}
    EXPECTED[(level, prior, task, "mpc_delta_t")] = {"n_ctrl": int(dt)}
    EXPECTED[(level, prior, task, "dflow")] = {"num_opt_steps": int(dflow)}
    if rhc != "--":
        EXPECTED[(level, prior, task, "mpc_rhc")] = {"n_ctrl": int(rhc)}

BUDGET_TABLE = """
pmf cheap 0.522 6.447 0.574 0.530 0.624
pmf mid   2.088 --    2.296 2.120 2.496
jit cheap 1.417 5.855 1.372 1.430 1.410
jit mid   5.668 --    5.488 5.720 5.640
"""
TASKS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
EXP_OF = {"denoising": "denoising", "deblur": "deblurring", "super_resolution": "super_resolution",
          "random_inpaint": "random_inpainting", "box_inpaint": "box_inpainting"}
SOURCE = {("rhso", "pmf"): (FINAL, "__rhso"), ("rhso", "jit"): (MATCHED, "__rhso_matched"),
          ("mpc_delta_t", "pmf"): (FINAL, "__core"), ("mpc_delta_t", "jit"): (FINAL, "__core"),
          ("dflow", "pmf"): (FINAL, "__core"), ("dflow", "jit"): (FINAL, "__core"),
          ("mpc_rhc", "pmf"): (FINAL, "__rhc"), ("mpc_rhc", "jit"): (FINAL, "__rhc")}
IDENTITY = {"job_id", "experiment", "num_images", "image_ids", "batch_size"}
# What the planner derives from a budget knob (src/config.py _estimate_cost and the time grid).
DERIVED = {"expected_model_evals", "expected_backprops", "expected_control_iterations",
           "expected_objective_evals", "expected_data_gradients", "expected_denoiser_samples",
           "delta_min", "delta_max", "delta_nominal_uniform"}


def resolve_file(path):
    cfg = yaml.safe_load((REPO_ROOT / path).read_text())
    return resolve_run_plan(cfg, validate_config(copy.deepcopy(cfg)), run_id="test")


def fields(spec):
    return json.loads(json.dumps(spec.to_dict(), default=str))


def level_of(spec):
    return re.search(r"__budget_(cheap|mid)", spec.experiment).group(1)


def source_specs():
    out = {}
    for path in (FINAL, MATCHED):
        for spec in resolve_file(path).specs:
            out[(path, spec.experiment, spec.model, spec.method)] = spec
    return out


# =====================================================================================
def check_generated():
    print("\n[1] regeneration")
    files = gen.outputs()
    check("generator owns exactly the four files",
          sorted(files) == sorted([QUALITY, QUALITY_SMOKE, TIMING, TIMING_SMOKE]))
    stale = [p for p, text in files.items() if (REPO_ROOT / p).read_text() != text]
    check("committed files equal a fresh generation", not stale, str(stale))
    got = {(e["level"], e["prior"], e["task"], e["method"]): {k: new for k, _, new in e["changes"]}
           for e in gen.entries()}
    check("the generator's values are the task's table", got == EXPECTED,
          "%d entries" % len(EXPECTED))
    check("66 entries: 36 cheap, 30 mid",
          len(EXPECTED) == 66 and sum(k[0] == "cheap" for k in EXPECTED) == 36)


def check_quality(src):
    print("\n[2-3] quality configuration")
    plan = resolve_file(QUALITY)
    specs = plan.specs
    cells = [(level_of(s), s.model, s.problem, s.method) for s in specs]
    check("66 jobs, one per table entry", len(specs) == 66 and len(set(cells)) == 66
          and set(cells) == set(EXPECTED), "%d jobs" % len(specs))
    check("pMF 33 jobs, JiT 33; cheap 36, mid 30",
          sum(s.model == "pmf" for s in specs) == 33 and sum(s.model == "jit" for s in specs) == 33
          and sum(c[0] == "cheap" for c in cells) == 36)
    check("every job at 100 images", all(s.num_images == 100 for s in specs))
    check("no mid deblurring", not any(c[0] == "mid" and c[2] == "deblur" for c in cells))
    smoke = resolve_file(QUALITY_SMOKE).specs
    check("smoke: the same 66 entries at 2 images",
          sorted((s.experiment, s.model, s.method) for s in smoke)
          == sorted((s.experiment, s.model, s.method) for s in specs)
          and all(s.num_images == 2 for s in smoke))
    names = {s.experiment for s in specs}
    check("block names: <task>__budget_<level>[_rhc|_rhso]",
          all(re.fullmatch(r"[a-z_]+__budget_(cheap|mid)(_rhc|_rhso)?", n) for n in names))
    check("_rhc holds MPC-RHC only, and every MPC-RHC (pMF 5 vs 10, JiT 10 vs 20)",
          all(s.experiment.endswith("_rhc") == (s.method == "mpc_rhc") for s in specs))
    check("_rhso holds JiT RHSO only; pMF RHSO shares the main block (batch 10)",
          all(s.experiment.endswith("_rhso") == (s.method == "rhso" and s.model == "jit")
              for s in specs))

    knob_ok, bs_ok, rest_ok, bad = True, True, True, []
    for s in specs:
        want = EXPECTED[(level_of(s), s.model, s.problem, s.method)]
        path, suffix = SOURCE[(s.method, s.model)]
        source = src[(path, EXP_OF[s.problem] + suffix, s.model, s.method)]
        knob_ok &= all(getattr(s, k) == v for k, v in want.items())
        bs_ok &= s.batch_size == source.batch_size
        a, b = fields(s), fields(source)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - IDENTITY - DERIVED - set(want)
        if diff:
            rest_ok = False
            bad.append((s.experiment, s.model, s.method, sorted(diff)))
    check("every step knob has the table's value", knob_ok)
    check("every batch size is the source entry's", bs_ok)
    check("only the step knob (and what the planner derives from it) changed", rest_ok,
          str(bad[:2]))
    check("RHSO N equals its source's N in every entry (only M moves)",
          all(s.num_rhso_steps == src[(SOURCE[("rhso", s.model)][0],
                                       EXP_OF[s.problem] + SOURCE[("rhso", s.model)][1],
                                       s.model, "rhso")].num_rhso_steps
              for s in specs if s.method == "rhso"))
    jit_rhso = [s for s in specs if s.method == "rhso" and s.model == "jit"]
    check("JiT RHSO: direct planner and Heun kept (9 jobs: 5 cheap, 4 mid)",
          len(jit_rhso) == 9 and all(s.rhso_terminal_mode == "direct" and s.solver == "heun"
                                      for s in jit_rhso))
    dt = [s for s in specs if s.method == "mpc_delta_t"]
    check("MPC-delta_t keeps num_mpc_steps N (= its source's)",
          all(s.num_mpc_steps == src[(FINAL, EXP_OF[s.problem] + "__core", s.model,
                                      "mpc_delta_t")].num_mpc_steps for s in dt))

    rhso = [s for s in specs if s.method == "rhso"]
    dflow = [s for s in specs if s.method == "dflow"]
    rhc = [s for s in specs if s.method == "mpc_rhc"]
    check("RHSO BP = N x M (18 jobs)",
          len(rhso) == 18 and all(s.expected_backprops == s.num_rhso_steps * s.num_opt_steps
                                  for s in rhso))
    check("MPC-delta_t BP = (N - 1) x n_ctrl (18 jobs)",
          len(dt) == 18 and all(s.expected_backprops == (s.num_mpc_steps - 1) * s.n_ctrl
                                for s in dt),
          "e.g. %s" % ", ".join("%s/%s %dx%d=%d" % (s.model, s.problem, s.num_mpc_steps - 1,
                                                    s.n_ctrl, s.expected_backprops) for s in dt[:2]))
    check("D-Flow BP = num_opt_steps (18 jobs)",
          len(dflow) == 18 and all(s.expected_backprops == s.num_opt_steps for s in dflow))
    check("MPC-RHC BP = num_mpc_steps x n_ctrl x (K - 1) (12 jobs)",
          len(rhc) == 12 and all(s.expected_backprops == s.num_mpc_steps * s.n_ctrl * (s.K - 1)
                                 for s in rhc),
          "K in %s" % sorted({s.K for s in rhc}))
    return specs


def check_comments():
    print("\n[4] comments")
    text = (REPO_ROOT / QUALITY).read_text()
    lines = [l for l in text.splitlines()
             if re.match(r"\s+(rhso|mpc_delta_t|dflow|mpc_rhc):\s+\{", l)]
    pattern = re.compile(r"# source: (configs/experiments_(final|matched)_frozen100\.yaml) :: "
                         r"(\S+) / (pmf|jit) / (rhso|mpc_delta_t|dflow|mpc_rhc); "
                         r"(cheap|mid) level \(.*\); [a-z_]+ \d+ -> \d+(, [a-z_]+ \d+ -> \d+)?$")
    check("each of the 66 entries names its source and its level",
          len(lines) == 66 and all(pattern.search(l) for l in lines))
    level_ok = True
    blocks = yaml.safe_load(text)["experiments"]
    for name in blocks:
        level = re.search(r"__budget_(cheap|mid)", name).group(1)
        start = text.index("\n  %s:\n" % name)
        end = text.find("\n\n", start + 1)
        for l in text[start:end if end > 0 else None].splitlines():
            if "# source:" in l and ("; %s level" % level) not in l:
                level_ok = False
    check("each comment's level is its block's level", level_ok)
    forbidden = re.compile(r"reus|carried over|copied from|borrow|selected on|tuned on|"
                           r"from another (checkpoint|prior)|same as", re.IGNORECASE)
    bad = []
    for path in (QUALITY, QUALITY_SMOKE, TIMING, TIMING_SMOKE):
        for line in (REPO_ROOT / path).read_text().splitlines():
            if "#" in line and forbidden.search(line.split("#", 1)[1]):
                bad.append((path, line.strip()))
    check("no comment claims a value was reused across checkpoints", not bad, str(bad[:1]))
    check("JiT RHSO sources are the JiT matched entries, pMF's the pMF frozen ones",
          all(("experiments_matched_frozen100.yaml" in l) == (" / jit / rhso;" in l)
              for l in lines if " / rhso;" in l))


def check_timing(quality_specs):
    print("\n[5] timing configuration")
    by_entry = {(level_of(s), s.model, s.problem, s.method): s for s in quality_specs}
    for path, n in ((TIMING, 3), (TIMING_SMOKE, 2)):
        cfg = yaml.safe_load((REPO_ROOT / path).read_text())
        plan = resolve_file(path)
        blocks = list(cfg["experiments"])
        per_block = {}
        for s in plan.specs:
            per_block.setdefault(s.experiment, []).append(s)
        check("%s: 66 blocks, one job each" % Path(path).name,
              len(blocks) == 66 and all(len(per_block.get(b, [])) == 1 for b in blocks))
        specs = [per_block[b][0] for b in blocks]
        check("%s: %d images, batch size 1, timing_detail and memory profiling on"
              % (Path(path).name, n),
              all(s.num_images == n and s.batch_size == 1 for s in specs)
              and cfg["runtime"].get("timing_detail") is True
              and cfg["runtime"].get("gpu_memory_profiling") is True)
        same = all({k: v for k, v in fields(s).items() if k not in IDENTITY}
                   == {k: v for k, v in fields(by_entry[(level_of(s), s.model, s.problem,
                                                         s.method)]).items()
                       if k not in IDENTITY} for s in specs)
        check("%s: every job equals its quality job but for identity" % Path(path).name, same)
    check("timing block names carry the level, method and prior",
          all(re.fullmatch(r"[a-z_]+__budget_(cheap|mid)(_rhc|_rhso)?__"
                           r"(rhso|mpc_delta_t|dflow|mpc_rhc)__(pmf|jit)", b)
              for b in yaml.safe_load((REPO_ROOT / TIMING).read_text())["experiments"]))
    check("the timing config names run id timing_budget_levels",
          "--run-id timing_budget_levels" in (REPO_ROOT / TIMING).read_text()
          and "submit_timing.sh budget_levels" in (REPO_ROOT / TIMING).read_text())


def check_driver():
    print("\n[6] run_timing.py, submit_timing.sh, slurm/run_timing.sh, summarize_timing.py")
    with tempfile.TemporaryDirectory() as tmp:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rt_driver.main(["--config", TIMING, "--run-id", "timing_budget_levels",
                                   "--output-root", tmp, "--dry-run"])
        text = out.getvalue()
        check("--dry-run: a baseline per prior, 66 blocks and the aggregate",
              code == 0 and "[model_only_baseline_pmf]" in text
              and "[model_only_baseline_jit]" in text and text.count("run.py") == 69
              and "66 block(s) selected" in text)
        check("... all under run id timing_budget_levels",
              text.count("--run-id timing_budget_levels") == 69)

    slurm = (REPO_ROOT / "slurm" / "run_timing.sh").read_text()
    check("slurm/run_timing.sh resolves budget_levels to the committed configs",
          'CONFIG="configs/experiments_rebuttal_timing_${PRIOR}.yaml"' in slurm
          and 'CONFIG="configs/experiments_rebuttal_timing_${PRIOR}_smoke.yaml"' in slurm
          and (REPO_ROOT / TIMING).exists() and (REPO_ROOT / TIMING_SMOKE).exists())

    bash = shutil.which("bash")
    if bash is None:
        check("bash available to run submit_timing.sh --dry-run", False)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            # An LF copy: this checkout may have CRLF line endings (core.autocrlf).
            script = Path(tmp) / "submit_timing.sh"
            script.write_bytes((REPO_ROOT / "submit_timing.sh").read_bytes().replace(b"\r\n", b"\n"))

            def run(*args):
                p = subprocess.run([bash, script.as_posix(), *args], capture_output=True,
                                   text=True, cwd=tmp)
                return p.returncode, p.stdout + p.stderr
            code, text = run("budget_levels", "--account", "def-test", "--gpu", "h100:1",
                             "--time", "02:00:00", "--dry-run")
            ok = (code == 0 and "run id: timing_budget_levels" in text
                  and "PRIOR=budget_levels" in text and "--gpus-per-node=h100:1" in text)
            check("submit_timing.sh budget_levels --dry-run: accepted, run id timing_budget_levels",
                  ok, "" if ok else text[-200:])
            code, text = run("budget_levels", "--smoke", "--account", "def-test", "--dry-run")
            check("... --smoke uses timing_budget_levels_smoke",
                  code == 0 and "run id: timing_budget_levels_smoke" in text)
            code, _ = run("no_such_prior", "--account", "def-test", "--dry-run")
            check("... an unknown name is still refused", code == 2)

    check("summarize_timing: budget-level blocks are their own variants",
          summ.variant_of("denoising__budget_cheap__rhso__pmf") == "budget_cheap"
          and summ.variant_of("denoising__budget_mid_rhc__mpc_rhc__jit") == "budget_mid"
          and summ.variant_of("deblurring__budget_cheap_rhso__rhso__jit") == "budget_cheap"
          and summ.variant_of("deblurring__matched_time__pnp__pmf") == "matched_time"
          and summ.variant_of("deblurring__rhso_matched__rhso") == "matched"
          and summ.variant_of("deblurring__core__pnp") == "base")
    check("... with their own row labels",
          summ.ROW_LABEL[("rhso", "budget_cheap")] == "RHSO (cheap level)"
          and summ.ROW_LABEL[("mpc_rhc", "budget_mid")] == "MPC-RHC (mid level)"
          and summ.ROW_LABEL[("dflow", "budget_mid")] == "D-Flow (mid level)")


# ------------------------------------------------------------------ synthetic runs
IMAGES = ["img%03d" % i for i in range(12)]
BASE = np.linspace(0.06, 0.10, len(IMAGES))          # the planted best's per-image LPIPS


def lp(offset=0.0, jitter=0.0, seed=0):
    return BASE + offset + jitter * np.random.default_rng(seed).normal(size=len(IMAGES))


# Mean 0.0002 above the best, per-image differences +-0.001: the CI straddles 0.
TIE = BASE + 0.0002 + 0.001 * np.array([1, -1] * (len(IMAGES) // 2))


def qrows(job_id, model, problem, method, experiment, lpips, params, images=IMAGES):
    rows = []
    for i, image in enumerate(images):
        row = {"job_id": job_id, "experiment": experiment, "model": model, "problem": problem,
               "method": method, "status": "ok", "image_id": image, "num_images": len(images),
               "batch_size": 4, "expected_backprops": 0, "psnr": 25.0 + i * 0.1,
               "ssim": 0.8 + i * 0.001, "lpips": float(lpips[i])}
        row.update(params)
        rows.append(row)
    return rows


def write_quality(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path / "results_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def write_timing(path, jobs):
    """jobs: (model, problem, method, experiment, seconds per image, params)."""
    for model, problem, method, experiment, seconds, params in jobs:
        payload = {"job_id": "t_%s_%s_%s" % (experiment, model, method), "model": model,
                   "problem": problem, "method": method, "experiment": experiment,
                   "status": "ok", "num_images": 3, "batch_size": 1, "model_evaluations": 30,
                   "backprops_through_model": 0, "runtime_per_image": seconds,
                   "timing_detail": {"chunk_seconds_outer_synced": [seconds] * 3,
                                     "chunk_seconds_method": [seconds] * 3,
                                     "chunk_sizes": [1, 1, 1], "allocator_peak_exact": True,
                                     "environment": {"hardware": {"gpu_name": "NVIDIA H100"}}}}
        payload.update(params)
        d = Path(path) / experiment / model / method
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps(payload))
    return str(path)


P = {"rhso": {"num_rhso_steps": 4, "num_opt_steps": 40, "rhso_terminal_mode": "direct"},
     "rhso_m": {"num_rhso_steps": 4, "num_opt_steps": 35, "rhso_terminal_mode": "direct"},
     "rhso_c": {"num_rhso_steps": 4, "num_opt_steps": 1, "rhso_terminal_mode": "direct"},
     "rhso_mid": {"num_rhso_steps": 4, "num_opt_steps": 7, "rhso_terminal_mode": "direct"},
     "dt": {"num_mpc_steps": 8, "n_ctrl": 20}, "dt_c": {"num_mpc_steps": 8, "n_ctrl": 1},
     "dt_mid": {"num_mpc_steps": 8, "n_ctrl": 4},
     "df_t": {"num_opt_steps": 155}, "df_c": {"num_opt_steps": 6},
     "rhc": {"num_mpc_steps": 4, "n_ctrl": 40, "K": 2},
     "pnp": {"num_pnp_steps": 10}, "pnp_t": {"num_pnp_steps": 978},
     "fl": {"num_flower_steps": 50},
     "j_rhso_m": {"num_rhso_steps": 7, "num_opt_steps": 20, "rhso_terminal_mode": "direct",
                  "solver": "heun"},
     "j_rhso_f": {"num_rhso_steps": 4, "num_opt_steps": 40, "solver": "heun",
                  "rhso_terminal_mode": "rollout"},
     "j_rhso_c": {"num_rhso_steps": 7, "num_opt_steps": 2, "rhso_terminal_mode": "direct",
                  "solver": "heun"},
     "deb_rhso": {"num_rhso_steps": 8, "num_opt_steps": 20, "rhso_terminal_mode": "direct"},
     "deb_pnp": {"num_pnp_steps": 200}}


def synthetic(tmp):
    """Planted outcomes on pMF denoising, pMF deblurring and JiT denoising."""
    D, B = "denoising", "deblur"
    tuned = []
    tuned += qrows("p_rhso", "pmf", D, "rhso", "denoising__rhso", lp(0.03), P["rhso"])
    tuned += qrows("p_dt", "pmf", D, "mpc_delta_t", "denoising__core", lp(0.09), P["dt"])
    tuned += qrows("p_rhc", "pmf", D, "mpc_rhc", "denoising__rhc", lp(-0.03), P["rhc"])
    tuned += qrows("p_pnp", "pmf", D, "pnp", "denoising__core", BASE, P["pnp"])
    tuned += qrows("p_fl", "pmf", D, "flower", "denoising__flower", TIE, P["fl"])
    tuned += qrows("p_deb_rhso", "pmf", B, "rhso", "deblurring__rhso", lp(0.03), P["deb_rhso"])
    tuned += qrows("p_deb_pnp", "pmf", B, "pnp", "deblurring__core", lp(0.10), P["deb_pnp"])
    tuned += qrows("j_rhso_f", "jit", D, "rhso", "denoising__rhso", lp(-0.05), P["j_rhso_f"])
    tuned += qrows("j_pnp", "jit", D, "pnp", "denoising__core", lp(0.09, 0.002, 1), P["pnp"])
    tuned += qrows("j_fl", "jit", D, "flower", "denoising__flower", lp(0.06, 0.002, 2), P["fl"])
    matched = []
    matched += qrows("p_rhso_m", "pmf", D, "rhso", "denoising__rhso_matched", lp(0.06),
                     P["rhso_m"])
    matched += qrows("j_rhso_m", "jit", D, "rhso", "denoising__rhso_matched", lp(0.04),
                     P["j_rhso_m"])
    mt = []
    mt += qrows("p_df_t", "pmf", D, "dflow", "denoising__matched_time", lp(0.28), P["df_t"])
    mt += qrows("p_pnp_t", "pmf", D, "pnp", "denoising__matched_time", lp(-0.05), P["pnp_t"])
    new = []
    new += qrows("p_rhso_c", "pmf", D, "rhso", "denoising__budget_cheap", lp(0.14, 0.003, 3),
                 P["rhso_c"])
    new += qrows("p_rhso_mid", "pmf", D, "rhso", "denoising__budget_mid", lp(0.05, 0.003, 4),
                 P["rhso_mid"])
    new += qrows("p_dt_c", "pmf", D, "mpc_delta_t", "denoising__budget_cheap", lp(0.24),
                 P["dt_c"])
    new += qrows("p_dt_mid", "pmf", D, "mpc_delta_t", "denoising__budget_mid", lp(0.16),
                 P["dt_mid"])
    new += qrows("p_df_c", "pmf", D, "dflow", "denoising__budget_cheap", lp(0.34, 0.003, 5),
                 P["df_c"])
    new += qrows("j_rhso_c", "jit", D, "rhso", "denoising__budget_cheap_rhso",
                 lp(0.19, 0.003, 6), P["j_rhso_c"])
    runs = {"tuned": write_quality(tmp / "tuned", tuned),
            "matched": write_quality(tmp / "matched", matched),
            "mt": write_quality(tmp / "mt", mt), "new": write_quality(tmp / "new", new)}
    timing = write_timing(tmp / "timing", [
        ("pmf", D, "rhso", "denoising__rhso__rhso__pmf", 12.0, P["rhso"]),
        ("pmf", D, "rhso", "denoising__rhso_matched__rhso__pmf", 11.0, P["rhso_m"]),
        ("pmf", D, "rhso", "denoising__budget_cheap__rhso__pmf", 0.55, P["rhso_c"]),
        ("pmf", D, "rhso", "denoising__budget_mid__rhso__pmf", 2.088 * 1.10, P["rhso_mid"]),
        ("pmf", D, "mpc_delta_t", "denoising__core__mpc_delta_t__pmf", 10.0, P["dt"]),
        ("pmf", D, "mpc_delta_t", "denoising__budget_cheap__mpc_delta_t__pmf", 0.60, P["dt_c"]),
        ("pmf", D, "mpc_delta_t", "denoising__budget_mid__mpc_delta_t__pmf", 2.0, P["dt_mid"]),
        ("pmf", D, "dflow", "denoising__matched_time__dflow__pmf", 13.38, P["df_t"]),
        ("pmf", D, "dflow", "denoising__budget_cheap__dflow__pmf", 0.50, P["df_c"]),
        ("pmf", D, "mpc_rhc", "denoising__rhc__mpc_rhc__pmf", 13.78, P["rhc"]),
        ("pmf", D, "pnp", "denoising__core__pnp__pmf", 0.13, P["pnp"]),
        ("pmf", D, "flower", "denoising__flower__flower__pmf", 0.52, P["fl"]),
        ("pmf", B, "rhso", "deblurring__rhso__rhso__pmf", 16.0, P["deb_rhso"]),
        ("pmf", B, "pnp", "deblurring__core__pnp__pmf", 6.4, P["deb_pnp"]),
        ("jit", D, "rhso", "denoising__rhso__rhso__jit", 1.0, P["j_rhso_f"]),
        ("jit", D, "rhso", "denoising__rhso_matched__rhso__jit", 8.0, P["j_rhso_m"]),
        ("jit", D, "rhso", "denoising__budget_cheap_rhso__rhso__jit", 1.40, P["j_rhso_c"]),
        ("jit", D, "pnp", "denoising__core__pnp__jit", 0.5, P["pnp"]),
        ("jit", D, "flower", "denoising__flower__flower__jit", 1.0, P["fl"]),
    ])                                      # p_pnp_t deliberately has no timing job
    return runs, timing, tuned + matched + mt + new


def check_table():
    print("\n[7] scripts/budget_level_table.py")
    budgets = {}
    for line in BUDGET_TABLE.strip().splitlines():
        prior, level, *values = line.split()
        budgets[(prior, level)] = {t: (None if v == "--" else float(v))
                                   for t, v in zip(TASKS, values)}
    check("hard-coded budgets are the task's table", table.BUDGETS == budgets)
    doc = table.__doc__
    check("... and the table is in the script as a comment",
          "pMF    cheap  0.522      6.447       0.574  0.530        0.624" in doc
          and "JiT    mid    5.668      --          5.488  5.720        5.640" in doc)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runs, timing, q = synthetic(tmp)
        quality = table.mtt.read_quality(list(runs.values()))
        rows, notes = table.build(quality, table.mtt.read_timing([timing]), resamples=2000)
        get = {(r["prior"], r["task"], r["level"], r["method"]): r for r in rows}
        pd = lambda level, m: get.get(("pmf", "denoising", level, m))     # noqa: E731
        jd = lambda level, m: get.get(("jit", "denoising", level, m))     # noqa: E731

        check("levels per cell: cheap, mid, RHSO time; no mid on deblurring",
              all(pd(l, "pnp") for l in ("cheap", "mid", "rhso"))
              and get.get(("pmf", "deblur", "cheap", "rhso")) is not None
              and get.get(("pmf", "deblur", "rhso", "rhso")) is not None
              and not any(k[1] == "deblur" and k[2] == "mid" for k in get))
        check("budgets: cheap/mid from the table, RHSO time = the reference's measured time",
              pd("cheap", "rhso")["budget_seconds"] == 0.522
              and pd("mid", "rhso")["budget_seconds"] == 2.088
              and pd("rhso", "rhso")["budget_seconds"] == 12.0
              and jd("rhso", "pnp")["budget_seconds"] == 8.0
              and get[("pmf", "deblur", "rhso", "pnp")]["budget_seconds"] == 16.0)
        check("cheap: the lowest-LPIPS run within 1.1 x (RHSO cheap at 1.05 x)",
              pd("cheap", "rhso")["quality_job_id"] == "p_rhso_c"
              and pd("cheap", "rhso")["within_budget"] is True)
        check("mid: a run at exactly 1.10 x the budget qualifies and wins on LPIPS",
              pd("mid", "rhso")["quality_job_id"] == "p_rhso_mid"
              and abs(pd("mid", "rhso")["budget_ratio"] - 1.10) < 1e-9)
        check("RHSO time: all of a method's runs compete (tuned RHSO beats matched BP)",
              pd("rhso", "rhso")["quality_job_id"] == "p_rhso"
              and abs(pd("rhso", "rhso")["budget_ratio"] - 1.0) < 1e-12)
        check("mid: MPC-dt's mid run (lower LPIPS) is picked over its cheap run",
              pd("mid", "mpc_delta_t")["quality_job_id"] == "p_dt_mid")
        r = pd("cheap", "mpc_delta_t")
        check("none within 1.1 x: the cheapest run, marked over budget (x ratio)",
              r["quality_job_id"] == "p_dt_c" and r["within_budget"] is False
              and r["budget_note"] == "over budget (x%.2f)" % (0.60 / 0.522)
              and r["outcome"] == "over budget", r["budget_note"])
        rhc = pd("cheap", "mpc_rhc")
        check("an over-budget run is never the best, even with the lowest LPIPS",
              rhc["lpips"] < pd("cheap", "pnp")["lpips"] and not rhc["is_best"]
              and pd("cheap", "pnp")["is_best"])
        check("untimed quality jobs are not candidates, and are listed",
              all(get[k]["quality_job_id"] != "p_pnp_t" for k in get if k[3] == "pnp")
              and any("p_pnp_t" in n and "no timing job" in n for n in notes))
        check("JiT: a non-direct RHSO is never a candidate (1.0 s, lowest LPIPS)",
              all(get[k]["quality_job_id"] != "j_rhso_f" for k in get if k[0] == "jit")
              and jd("cheap", "rhso")["quality_job_id"] == "j_rhso_c"
              and jd("rhso", "rhso")["quality_job_id"] == "j_rhso_m")
        check("JiT RHSO time: the matched direct-planner RHSO is the reference and the best",
              jd("rhso", "rhso")["is_best"] and jd("rhso", "rhso")["budget_ratio"] == 1.0)

        def lookup(job):
            return {r["image_id"]: r for r in q if r["job_id"] == job}
        a, b = lookup("p_df_c"), lookup("p_pnp")
        diffs = {m: np.subtract([float(a[i][m]) for i in IMAGES],
                                [float(b[i][m]) for i in IMAGES]) for m in ps.METRICS}
        expect = ps.bootstrap(diffs, ps.SEED, 2000)
        r = pd("cheap", "dflow")
        check("differences are method - best with paired_stats.bootstrap's CI",
              all(abs(r["%s_diff" % m] - expect[m][0]) < 1e-12
                  and abs(r["%s_ci_lo" % m] - expect[m][1]) < 1e-12
                  and abs(r["%s_ci_hi" % m] - expect[m][2]) < 1e-12 for m in ps.METRICS)
              and r["best_method"] == "pnp")
        check("significance: D-Flow clearly worse; Flower (+0.0002 mean) ties with the best",
              r["outcome"] == "worse" and r["lpips_ci_lo"] > 0
              and pd("cheap", "flower")["outcome"] == "tie"
              and pd("cheap", "flower")["lpips_ci_lo"] <= 0 <= pd("cheap", "flower")["lpips_ci_hi"])
        check("the best: 'best (tie)' when someone ties, else 'best (significant)'",
              pd("cheap", "pnp")["outcome"] == "best (tie)"
              and jd("rhso", "rhso")["outcome"] == "best (significant)")
        check("the best row has no difference of its own",
              pd("cheap", "pnp")["lpips_diff"] is None)

        per_task, per_method = table.summary(rows)
        s = {(x["prior"], x["level"], x["task"]): x for x in per_task}
        check("summary per task: best, significance and who ties",
              s[("pmf", "cheap", "denoising")]["best_method"] == "pnp"
              and s[("pmf", "cheap", "denoising")]["significant"] is False
              and s[("pmf", "cheap", "denoising")]["tied_with_best"] == "flower"
              and s[("jit", "rhso", "denoising")]["significant"] is True)
        c = {(x["prior"], x["level"], x["method"]): x for x in per_method}
        check("wins and ties per method",
              c[("pmf", "cheap", "pnp")]["ties"] == 1 and c[("pmf", "cheap", "pnp")]["wins"] == 1
              and c[("pmf", "cheap", "flower")]["ties"] == 1
              and c[("pmf", "cheap", "mpc_delta_t")]["over_budget"] == 1
              and c[("jit", "rhso", "rhso")]["wins"] == 1,
              "pMF cheap PnP: wins %d (deblurring) ties %d (denoising)"
              % (c[("pmf", "cheap", "pnp")]["wins"], c[("pmf", "cheap", "pnp")]["ties"]))

        out = tmp / "out"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            code = table.main(sum([["--quality", r] for r in runs.values()], [])
                              + ["--timing", timing, "--out-dir", str(out),
                                 "--resamples", "500"])
        files = ["budget_levels.csv", "budget_levels_summary.csv", "budget_levels_wins.csv",
                 "budget_levels.tex"]
        check("CLI writes the three CSVs and the LaTeX",
              code == 0 and all((out / f).exists() for f in files))
        with open(out / "budget_levels.csv", newline="") as fh:
            written = list(csv.DictReader(fh))
        check("one CSV row per entry, with seconds, ratio, metrics and the CI",
              len(written) == len(rows)
              and {"seconds_per_image", "budget_ratio", "psnr", "ssim", "lpips", "lpips_diff",
                   "lpips_ci_lo", "lpips_ci_hi", "budget_note", "outcome"} <= set(written[0]))
        tex = (out / "budget_levels.tex").read_text()
        check("LaTeX: a table per prior and level plus the summary; over budget marked",
              tex.count("\\begin{table}") == len({(r["prior"], r["level"]) for r in rows}) + 1
              and "$^\\dagger$" in tex and "tab:rebuttal-budget-summary" in tex
              and tex.count("\\begin{tabular}") == tex.count("\\end{tabular}"))

        def refuses(fn):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    fn()
            except SystemExit:
                return True
            return False
        copy_of_new = str(tmp / "new_copy")
        shutil.copytree(runs["new"], copy_of_new)
        check("refuses a (prior, task, method, variant) given by two quality jobs",
              refuses(lambda: table.build(table.mtt.read_quality([runs["new"], copy_of_new]),
                                          table.mtt.read_timing([timing]), resamples=50)))
        check("refuses a quality job that matches two timing jobs",
              refuses(lambda: table.build(quality, table.mtt.read_timing([timing, timing]),
                                          resamples=50)))
        short = write_quality(tmp / "short", qrows("p_df_c", "pmf", "denoising", "dflow",
                                                   "denoising__budget_cheap", lp(0.34),
                                                   P["df_c"], images=IMAGES[:-1]))
        check("refuses sides that cover different images",
              refuses(lambda: table.build(table.mtt.read_quality(
                  [runs["tuned"], runs["matched"], runs["mt"], short]),
                  table.mtt.read_timing([timing]), resamples=50)))
        other = dict(P["dt_c"], n_ctrl=2)
        mism = write_timing(tmp / "mism", [
            ("pmf", "denoising", "mpc_delta_t", "denoising__budget_cheap__mpc_delta_t__pmf",
             0.3, other)])
        r2, n2 = table.build(table.mtt.read_quality([runs["new"]]),
                             table.mtt.read_timing([mism]), resamples=50)
        check("a timing job with another knob value is not paired with the quality job",
              not any(r["quality_job_id"] == "p_dt_c" for r in r2)
              and any("p_dt_c" in n for n in n2))


def main():
    check_generated()
    specs = check_quality(source_specs())
    check_comments()
    check_timing(specs)
    check_driver()
    check_table()
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
