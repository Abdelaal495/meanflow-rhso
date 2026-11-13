"""Matched-time comparison on the pixel priors (rebuttal Task 5): configs, timing, table.

`scripts/make_matched_time_config.py` writes the 100-image quality configuration, its 2-image
smoke variant and a timing configuration (3 images, batch size 1) with its smoke variant.
This suite resolves every file through the REAL validator and planner -- the code
`run.py --dry-run` runs -- and checks:

  1. regenerating gives the committed files byte for byte;
  2. the quality config is exactly the 33 (prior, task, method) entries of the table below
     (written out here independently of the generator), at 100 images (2 in the smoke);
  3. every job equals its source frozen job in every field except its identity, its budget
     knob and the fields the planner derives from that knob; the knob has the table's value;
     the batch size is the benchmark's; Flower's deblurring CG caps are pMF 100, JiT 50;
  4. backprops: D-Flow = num_opt_steps; MPC-RHC = num_mpc_steps x n_ctrl x (K - 1), which is
     num_mpc_steps x n_ctrl for K = 2; PnP-Flow and Flower none;
  5. every entry's comment names its source entry and says "matched to RHSO's measured
     time"; no comment claims a value was reused from another checkpoint;
  6. the timing config: one block per entry, one job each, 3 images (2 in the smoke), batch
     size one, timing_detail on, otherwise equal to the quality job;
  7. scripts/run_timing.py on it: one model-only baseline process per prior (--models),
     one process per block, one run id; the one-prior configs keep their single baseline;
  8. scripts/summarize_timing.py labels the matched-time rows;
  9. scripts/matched_time_table.py on synthetic runs: the RHSO and MPC-delta_t roles per
     prior, the paired bootstrap of paired_stats.py, the 10% time rule, and its refusals.

Runs on CPU with no checkpoint:

    python tests/test_matched_time.py
"""

import contextlib
import copy
import csv
import importlib.util
import io
import json
import re
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
    print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_module("make_matched_time_config", REPO_ROOT / "scripts" / "make_matched_time_config.py")
rt_driver = load_module("run_timing", REPO_ROOT / "scripts" / "run_timing.py")
summ = load_module("summarize_timing", REPO_ROOT / "scripts" / "summarize_timing.py")
table = load_module("matched_time_table", REPO_ROOT / "scripts" / "matched_time_table.py")
ps = load_module("paired_stats", REPO_ROOT / "scripts" / "paired_stats.py")

QUALITY = "configs/experiments_matched_time_pixel.yaml"
QUALITY_SMOKE = "configs/experiments_matched_time_pixel_smoke.yaml"
TIMING = "configs/experiments_rebuttal_timing_matched_time.yaml"
TIMING_SMOKE = "configs/experiments_rebuttal_timing_matched_time_smoke.yaml"

# The task's table, verbatim ("--" = no new run).
TABLE = """
pmf denoising        978  1218 155 --
pmf deblur           513    13 162 --
pmf super_resolution 962  1240 168 20
pmf random_inpaint   888  1210 151 18
pmf box_inpaint     1053  1155 166 --
jit denoising        312   323 145 --
jit deblur           606   171 286 --
jit super_resolution 600   636 287 71
jit random_inpaint   309   324 145 --
jit box_inpaint      309   321 143 --
"""
COLUMNS = [("pnp", "num_pnp_steps"), ("flower", "num_flower_steps"), ("dflow", "num_opt_steps"),
           ("mpc_rhc", "n_ctrl")]
EXPECTED = {}
for line in TABLE.strip().splitlines():
    prior, task, *values = line.split()
    for (method, knob), value in zip(COLUMNS, values):
        if value != "--":
            EXPECTED[(prior, task, method)] = (knob, int(value))

SOURCE_EXP = {"pnp": "__core", "dflow": "__core", "mpc_rhc": "__rhc", "flower": "__flower"}
EXP_OF = {"denoising": "denoising", "deblur": "deblurring", "super_resolution": "super_resolution",
          "random_inpaint": "random_inpainting", "box_inpaint": "box_inpainting"}
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


def source_specs():
    out = {}
    for path in ("configs/experiments_final_frozen100.yaml",
                 "configs/experiments_flower_frozen100.yaml"):
        for spec in resolve_file(path).specs:
            out[(spec.experiment, spec.model, spec.method)] = spec
    return out


# =====================================================================================
def check_generated():
    print("\n[1] regeneration")
    files = gen.outputs()
    check("generator owns exactly the four files",
          sorted(files) == sorted([QUALITY, QUALITY_SMOKE, TIMING, TIMING_SMOKE]))
    stale = [p for p, text in files.items() if (REPO_ROOT / p).read_text() != text]
    check("committed files equal a fresh generation", not stale, str(stale))
    check("the generator's table is the task's table",
          {(e["prior"], e["task"], e["method"]): (e["knob"], e["new"]) for e in gen.entries()}
          == EXPECTED, "%d entries" % len(EXPECTED))


def check_quality(src):
    print("\n[2-4] quality configuration")
    plan = resolve_file(QUALITY)
    specs = plan.specs
    cells = [(s.model, s.problem, s.method) for s in specs]
    check("33 jobs, one per table entry", len(specs) == 33 and len(set(cells)) == 33
          and set(cells) == set(EXPECTED), "%d jobs" % len(specs))
    check("pMF 17 jobs, JiT 16", sum(s.model == "pmf" for s in specs) == 17
          and sum(s.model == "jit" for s in specs) == 16)
    check("every job at 100 images", all(s.num_images == 100 for s in specs))
    smoke = resolve_file(QUALITY_SMOKE).specs
    check("smoke: the same 33 entries at 2 images",
          sorted((s.experiment, s.model, s.method) for s in smoke)
          == sorted((s.experiment, s.model, s.method) for s in specs)
          and all(s.num_images == 2 for s in smoke))
    names = {s.experiment for s in specs}
    check("block names: <task>__matched_time (+ _rhc for MPC-RHC)",
          all(re.fullmatch(r"[a-z_]+__matched_time(_rhc)?", n) for n in names)
          and all(s.experiment.endswith("_rhc") == (s.method == "mpc_rhc") for s in specs))

    knob_ok, bs_ok, rest_ok, bad = True, True, True, []
    for s in specs:
        knob, value = EXPECTED[(s.model, s.problem, s.method)]
        source = src[(EXP_OF[s.problem] + SOURCE_EXP[s.method], s.model, s.method)]
        knob_ok &= getattr(s, knob) == value
        bs_ok &= s.batch_size == source.batch_size
        a, b = fields(s), fields(source)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - IDENTITY - DERIVED - {knob}
        if diff:
            rest_ok = False
            bad.append((s.experiment, s.model, s.method, sorted(diff)))
    check("every knob has the table's value", knob_ok)
    check("every batch size is the benchmark's", bs_ok)
    check("only the knob (and what the planner derives from it) changed", rest_ok, str(bad[:2]))
    caps = {(s.model, s.problem): s.flower_cg_max_iters for s in specs if s.method == "flower"}
    check("Flower deblurring keeps its CG cap (pMF 100, JiT 50)",
          caps[("pmf", "deblur")] == 100 and caps[("jit", "deblur")] == 50)

    dflow = [s for s in specs if s.method == "dflow"]
    rhc = [s for s in specs if s.method == "mpc_rhc"]
    check("D-Flow BP = num_opt_steps (10 jobs)",
          len(dflow) == 10 and all(s.expected_backprops == s.num_opt_steps for s in dflow))
    check("MPC-RHC BP = num_mpc_steps x n_ctrl x (K-1) (3 jobs)",
          len(rhc) == 3 and all(s.expected_backprops == s.num_mpc_steps * s.n_ctrl * (s.K - 1)
                                for s in rhc),
          ", ".join("%s/%s K=%d BP=%d" % (s.model, s.problem, s.K, s.expected_backprops)
                    for s in rhc))
    k2 = [s for s in rhc if s.K == 2]
    check("... = num_mpc_steps x n_ctrl where K = 2 (JiT SR: 4 x 71 = 284)",
          len(k2) == 1 and k2[0].expected_backprops == k2[0].num_mpc_steps * k2[0].n_ctrl == 284)
    check("PnP-Flow and Flower have no backprops",
          all(s.expected_backprops == 0 for s in specs if s.method in ("pnp", "flower")))
    return specs


def check_comments():
    print("\n[5] comments")
    text = (REPO_ROOT / QUALITY).read_text()
    lines = [l for l in text.splitlines() if re.match(r"\s+(pnp|flower|dflow|mpc_rhc):\s+\{", l)]
    pattern = re.compile(r"# source: (configs/experiments_(final|flower)_frozen100\.yaml) :: "
                         r"(\S+) / (pmf|jit) / (pnp|flower|dflow|mpc_rhc); \S+ \S+ -> \d+, "
                         r"matched to RHSO's measured time$")
    check("each of the 33 entries names its source and says it is matched to RHSO's time",
          len(lines) == 33 and all(pattern.search(l) for l in lines))
    forbidden = re.compile(r"reus|carried over|copied from|borrow|selected on|tuned on|"
                           r"from another (checkpoint|prior)|same as", re.IGNORECASE)
    bad = []
    for path in (QUALITY, QUALITY_SMOKE, TIMING, TIMING_SMOKE):
        for line in (REPO_ROOT / path).read_text().splitlines():
            if "#" in line and forbidden.search(line.split("#", 1)[1]):
                bad.append((path, line.strip()))
    check("no comment claims a value was reused across checkpoints", not bad, str(bad[:1]))


def check_timing(quality_specs):
    print("\n[6] timing configuration")
    by_entry = {(s.model, s.problem, s.method): s for s in quality_specs}
    for path, n in ((TIMING, 3), (TIMING_SMOKE, 2)):
        cfg = yaml.safe_load((REPO_ROOT / path).read_text())
        plan = resolve_file(path)
        blocks = list(cfg["experiments"])
        per_block = {}
        for s in plan.specs:
            per_block.setdefault(s.experiment, []).append(s)
        check("%s: 33 blocks, one job each" % Path(path).name,
              len(blocks) == 33 and all(len(per_block.get(b, [])) == 1 for b in blocks))
        specs = [per_block[b][0] for b in blocks]
        check("%s: %d images, batch size 1, timing_detail and memory profiling on"
              % (Path(path).name, n),
              all(s.num_images == n and s.batch_size == 1 for s in specs)
              and cfg["runtime"].get("timing_detail") is True
              and cfg["runtime"].get("gpu_memory_profiling") is True)
        same = all({k: v for k, v in fields(s).items() if k not in IDENTITY}
                   == {k: v for k, v in fields(by_entry[(s.model, s.problem, s.method)]).items()
                       if k not in IDENTITY} for s in specs)
        check("%s: every job equals its quality job but for identity" % Path(path).name, same)
    check("timing block names carry the method and prior",
          all(re.fullmatch(r"[a-z_]+__matched_time(_rhc)?__(pnp|flower|dflow|mpc_rhc)__(pmf|jit)", b)
              for b in yaml.safe_load((REPO_ROOT / TIMING).read_text())["experiments"]))


def check_driver():
    print("\n[7] scripts/run_timing.py on the two-prior config")

    class Args:
        config, run_id, output_root, cache_root, num_images = TIMING, "timing_matched_time", None, None, None
        python, with_checks = "python", False
    blocks = list(yaml.safe_load((REPO_ROOT / TIMING).read_text())["experiments"])
    cmds = rt_driver.build_commands(Args, blocks, baseline_models=["jit", "pmf"])
    labels = [l for l, _ in cmds]
    check("one model-only baseline process per prior, each with --models",
          labels[:2] == ["model_only_baseline_jit", "model_only_baseline_pmf"]
          and all("--memory-baseline-only" in a and a[a.index("--models") + 1] == m
                  for (_, a), m in zip(cmds[:2], ["jit", "pmf"])))
    check("then one process per block and the aggregate, all under one run id",
          labels[2:-1] == blocks and labels[-1] == "aggregate"
          and all(a[a.index("--run-id") + 1] == "timing_matched_time" for _, a in cmds))
    single = rt_driver.build_commands(Args, blocks[:1])
    check("without baseline_models: the single baseline of the one-prior configs",
          [l for l, _ in single] == ["model_only_baseline", blocks[0], "aggregate"]
          and "--models" not in single[0][1])
    import run
    ns = run.parse_args(["--config", TIMING, "--run-id", "x", "--models", "jit"])
    from src.config import load_config
    cfg = run.apply_cli_overrides(load_config(TIMING), ns)
    plan = resolve_run_plan(cfg, validate_config(cfg), run_id="x")
    check("--models jit leaves only JiT for that baseline process",
          set(plan.scheduled_models()) == {"jit"})
    with tempfile.TemporaryDirectory() as tmp:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rt_driver.main(["--config", TIMING, "--run-id", "x", "--output-root", tmp,
                                   "--dry-run"])
        text = out.getvalue()
        check("--dry-run lists both baselines, 33 blocks and the aggregate",
              code == 0 and "[model_only_baseline_pmf]" in text
              and "[model_only_baseline_jit]" in text and text.count("run.py") == 36)
        base = Path(tmp) / "x"
        base.mkdir(parents=True)
        (base / "memory_baseline_pmf.json").write_text("{}")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rt_driver.main(["--config", TIMING, "--run-id", "x", "--output-root", tmp, "--dry-run"])
        text = out.getvalue()
        check("a prior whose baseline exists is not measured again",
              "[model_only_baseline_jit]" in text and "[model_only_baseline_pmf]" not in text)


def check_summary_labels():
    print("\n[8] scripts/summarize_timing.py")
    check("matched-time blocks are their own variant",
          summ.variant_of("deblurring__matched_time__pnp__pmf") == "matched_time"
          and summ.variant_of("super_resolution__matched_time_rhc__mpc_rhc__jit") == "matched_time"
          and summ.variant_of("deblurring__rhso_matched__rhso") == "matched"
          and summ.variant_of("deblurring__core__pnp") == "base")
    check("... with their own row labels",
          summ.ROW_LABEL[("pnp", "matched_time")] == "PnP-Flow (matched time)"
          and summ.ROW_LABEL[("mpc_rhc", "matched_time")] == "MPC-RHC (matched time)")


# ------------------------------------------------------------------ synthetic runs
IMAGES = ["img%03d" % i for i in range(12)]


def _quality_rows(job_id, model, problem, method, experiment, lpips, params, seed, n=IMAGES):
    rng = np.random.default_rng(seed)
    rows = []
    for image in n:
        row = {"job_id": job_id, "experiment": experiment, "model": model, "problem": problem,
               "method": method, "status": "ok", "image_id": image, "num_images": len(n),
               "batch_size": 4, "expected_backprops": 0,
               "psnr": 25 + rng.normal(), "ssim": 0.8 + 0.01 * rng.normal(),
               "lpips": lpips + 0.01 * rng.normal()}
        row.update(params)
        rows.append(row)
    return rows


def _write_quality(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path / "results_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def _write_timing(path, jobs):
    """jobs: (model, problem, method, experiment, seconds per image, params)."""
    for model, problem, method, experiment, seconds, params in jobs:
        payload = {"job_id": "t_%s_%s" % (experiment, model), "model": model, "problem": problem,
                   "method": method, "experiment": experiment, "status": "ok", "num_images": 3,
                   "batch_size": 1, "model_evaluations": 30, "backprops_through_model": 0,
                   "runtime_per_image": seconds,
                   "timing_detail": {"chunk_seconds_outer_synced": [seconds] * 3,
                                     "chunk_seconds_method": [seconds] * 3,
                                     "chunk_sizes": [1, 1, 1], "allocator_peak_exact": True,
                                     "environment": {"hardware": {"gpu_name": "NVIDIA H100"}}}}
        payload.update(params)
        d = Path(path) / experiment / model / method
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps(payload))
    return str(path)


def synthetic(tmp):
    """Quality and timing runs for pMF and JiT denoising, with planted outcomes."""
    P = {"rhso": {"num_rhso_steps": 4, "num_opt_steps": 40},
         "rhso_m": {"num_rhso_steps": 7, "num_opt_steps": 20, "rhso_terminal_mode": "direct"},
         "dt_b": {"num_mpc_steps": 5, "n_ctrl": 40}, "dt": {"num_mpc_steps": 8, "n_ctrl": 20},
         "pnp": {"num_pnp_steps": 10}, "pnp_t": {"num_pnp_steps": 978},
         "fl": {"num_flower_steps": 50}, "fl_t": {"num_flower_steps": 1218}}
    q = []
    q += _quality_rows("p_rhso", "pmf", "denoising", "rhso", "denoising__rhso", 0.10, P["rhso"], 1)
    q += _quality_rows("p_rhso_m", "pmf", "denoising", "rhso", "denoising__rhso_matched", 0.30,
                       P["rhso_m"], 2)                                   # not pMF's reference
    q += _quality_rows("p_dt_b", "pmf", "denoising", "mpc_delta_t",
                       "denoising__mpc_delta_t_matched", 0.20, P["dt_b"], 3)
    q += _quality_rows("p_dt", "pmf", "denoising", "mpc_delta_t", "denoising__core", 0.05,
                       P["dt"], 4)                                       # not pMF's MPC-dt
    q += _quality_rows("p_pnp", "pmf", "denoising", "pnp", "denoising__core", 0.25, P["pnp"], 5)
    q += _quality_rows("p_fl", "pmf", "denoising", "flower", "denoising__flower", 0.12, P["fl"], 7)
    q += _quality_rows("j_rhso", "jit", "denoising", "rhso", "denoising__rhso_matched", 0.11,
                       P["rhso_m"], 9)
    q += _quality_rows("j_dt", "jit", "denoising", "mpc_delta_t", "denoising__core", 0.18,
                       P["dt"], 10)
    q += _quality_rows("j_pnp", "jit", "denoising", "pnp", "denoising__core", 0.30, P["pnp"], 11)
    bench = _write_quality(tmp / "bench", q)
    mt = []
    mt += _quality_rows("p_pnp_t", "pmf", "denoising", "pnp", "denoising__matched_time", 0.15,
                        P["pnp_t"], 6)
    mt += _quality_rows("p_fl_t", "pmf", "denoising", "flower", "denoising__matched_time", 0.08,
                        P["fl_t"], 8)
    mt += _quality_rows("j_pnp_t", "jit", "denoising", "pnp", "denoising__matched_time", 0.09,
                        P["pnp_t"], 12)
    matched_time = _write_quality(tmp / "mt", mt)
    timing = _write_timing(tmp / "timing", [
        ("pmf", "denoising", "rhso", "denoising__rhso__rhso", 10.0, P["rhso"]),
        ("pmf", "denoising", "mpc_delta_t", "denoising__mpc_delta_t_matched__mpc_delta_t", 9.0,
         P["dt_b"]),
        ("pmf", "denoising", "pnp", "denoising__core__pnp", 1.0, P["pnp"]),
        ("pmf", "denoising", "pnp", "denoising__matched_time__pnp__pmf", 11.0, P["pnp_t"]),
        ("pmf", "denoising", "flower", "denoising__flower__flower", 1.0, P["fl"]),
        ("pmf", "denoising", "flower", "denoising__matched_time__flower__pmf", 11.5, P["fl_t"]),
        ("jit", "denoising", "rhso", "denoising__rhso_matched__rhso", 4.0, P["rhso_m"]),
        ("jit", "denoising", "mpc_delta_t", "denoising__core__mpc_delta_t", 6.0, P["dt"]),
        ("jit", "denoising", "pnp", "denoising__core__pnp", 0.5, P["pnp"]),
        ("jit", "denoising", "pnp", "denoising__matched_time__pnp__jit", 4.2, P["pnp_t"]),
    ])
    return bench, matched_time, timing, q, mt


def check_table():
    print("\n[9] scripts/matched_time_table.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bench, mt, timing, q, mtq = synthetic(tmp)
        rows, notes = table.build(table.read_quality([mt, bench]), table.read_timing([timing]),
                                  resamples=2000)
        get = {(r["prior"], r["method"], r["variant"]): r for r in rows}
        check("RHSO reference: frozen on pMF, matched-BP on JiT",
              get[("pmf", "rhso", "frozen")]["quality_job_id"] == "p_rhso"
              and get[("jit", "rhso", "matched")]["quality_job_id"] == "j_rhso"
              and ("pmf", "rhso", "matched") not in get)
        check("MPC-delta_t: Direction B on pMF, frozen on JiT",
              get[("pmf", "mpc_delta_t", "matched")]["quality_job_id"] == "p_dt_b"
              and get[("jit", "mpc_delta_t", "frozen")]["quality_job_id"] == "j_dt"
              and ("pmf", "mpc_delta_t", "frozen") not in get)
        check("frozen and matched-time entries both shown",
              all((p, m, v) in get for p, m in (("pmf", "pnp"), ("pmf", "flower"), ("jit", "pnp"))
                  for v in ("frozen", "matched_time")))
        sel = {(r["prior"], r["method"]): r["variant"] for r in rows if r["selected"]}
        check("at 1.10 x RHSO the lower-LPIPS matched-time PnP is marked (pMF)",
              sel.get(("pmf", "pnp")) == "matched_time"
              and abs(get[("pmf", "pnp", "matched_time")]["time_ratio"] - 1.10) < 1e-12)
        check("at 1.15 x RHSO matched-time Flower is over; the frozen one is marked",
              sel.get(("pmf", "flower")) == "frozen"
              and get[("pmf", "flower", "matched_time")]["within_tolerance"] is False)
        check("JiT: matched-time PnP at 1.05 x RHSO with lower LPIPS is marked",
              sel.get(("jit", "pnp")) == "matched_time")
        check("a single entry within the tolerance is marked (MPC-delta_t)",
              sel.get(("pmf", "mpc_delta_t")) == "matched" and sel.get(("jit", "mpc_delta_t")) is None,
              "JiT MPC-dt is 1.5 x RHSO")
        check("the reference row is never marked", not any(r["selected"] for r in rows
                                                           if r["role"] == "reference"))

        # paired bootstrap, recomputed with paired_stats.py's own function
        def lookup(rows_, job):
            return {r["image_id"]: r for r in rows_ if r["job_id"] == job}
        ref, other = lookup(q, "p_rhso"), lookup(mtq, "p_pnp_t")
        diffs = {m: np.subtract([float(ref[i][m]) for i in IMAGES],
                                [float(other[i][m]) for i in IMAGES]) for m in ps.METRICS}
        expect = ps.bootstrap(diffs, ps.SEED, 2000)
        r = get[("pmf", "pnp", "matched_time")]
        check("differences are RHSO - method with paired_stats.bootstrap's CI",
              all(abs(r["%s_diff" % m] - expect[m][0]) < 1e-12
                  and abs(r["%s_ci_lo" % m] - expect[m][1]) < 1e-12
                  and abs(r["%s_ci_hi" % m] - expect[m][2]) < 1e-12 for m in ps.METRICS))
        check("a row does not depend on which other rows were computed",
              table.paired(table.read_quality([bench])[("pmf", "denoising", "rhso", "frozen")][0],
                           table.read_quality([mt])[("pmf", "denoising", "pnp", "matched_time")][0],
                           ps.SEED, 2000)["lpips"] == expect["lpips"])

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = table.main(["--quality", mt, "--quality", bench, "--timing", timing,
                               "--out-dir", str(tmp / "out"), "--resamples", "500"])
        written = list(csv.DictReader(open(tmp / "out" / "matched_time.csv")))
        tex = (tmp / "out" / "matched_time.tex").read_text()
        check("CSV and LaTeX written, one row per entry", code == 0 and len(written) == len(rows)
              and "\\begin{tabular}" in tex)
        check("LaTeX marks exactly the selected entries",
              tex.count("$^\\dagger$ &") == sum(r["selected"] for r in rows))

        # refusals
        def refuses(fn):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    fn()
            except SystemExit:
                return True
            return False
        dup = _write_quality(tmp / "dup", _quality_rows("p_pnp_again", "pmf", "denoising", "pnp",
                                                        "denoising__core", 0.2, {"num_pnp_steps": 10}, 5))
        check("refuses a cell that two runs both provide",
              refuses(lambda: table.build(table.read_quality([bench, dup]),
                                          table.read_timing([timing]), resamples=50)))
        bad_t = _write_timing(tmp / "bad_timing", [
            ("pmf", "denoising", "rhso", "denoising__rhso__rhso", 10.0,
             {"num_rhso_steps": 8, "num_opt_steps": 20})])
        check("refuses a timing job whose configuration differs",
              refuses(lambda: table.build(table.read_quality([bench]),
                                          table.read_timing([bad_t]), resamples=50)))
        short = _write_quality(tmp / "short", _quality_rows(
            "p_pnp_t", "pmf", "denoising", "pnp", "denoising__matched_time", 0.15,
            {"num_pnp_steps": 978}, 6, n=IMAGES[:-1]))
        check("refuses sides that cover different images",
              refuses(lambda: table.build(table.read_quality([short, bench]),
                                          table.read_timing([timing]), resamples=50)))


def main():
    check_generated()
    specs = check_quality(source_specs())
    check_comments()
    check_timing(specs)
    check_driver()
    check_summary_labels()
    check_table()
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
