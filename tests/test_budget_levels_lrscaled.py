"""Budget levels with scaled learning rates (rebuttal Task 7): configs and table pairing.

`scripts/make_budget_level_lrscaled_config.py` rewrites Task 6's 66 budget-level entries with

    lr_new = lr_tuned x (tuned knob / new knob)

(knob: RHSO num_opt_steps, MPC-delta_t n_ctrl, D-Flow num_opt_steps, MPC-RHC n_ctrl; tuned
values from the entry's source entry).  This suite resolves the files through the REAL
validator and planner -- the code `run.py --dry-run` runs -- and checks:

  1. regenerating gives the committed files byte for byte; the learning rates equal the
     task's table (written out here independently) to 4 significant figures;
  2. 66 jobs (2 images in the smoke), block names Task 6's plus "_lr"; every job equals its
     Task 6 job in every field except its identity and lr, so batch sizes, step counts and
     backprop counts are Task 6's;
  3. every lr follows the rule, recomputed here from the source configs;
  4. every comment names its source entry, its level and "learning rate scaled by tuned/new
     steps"; no comment claims a value was reused from another checkpoint;
  5. scripts/budget_level_table.py: the "_lr" blocks are the variants "cheap level, lr
     scaled" / "mid level, lr scaled"; Task 6 names keep their variants; without
     --timing-ignore lr the lr-scaled jobs find no timing job (the default is unchanged);
     with it they take Task 6's timing job (also when another variant's timing job agrees),
     the CLI accepts the option and rejects any other field.

Runs on CPU with no checkpoint:

    python tests/test_budget_levels_lrscaled.py
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


gen = load_module("make_budget_level_lrscaled_config",
                  REPO_ROOT / "scripts" / "make_budget_level_lrscaled_config.py")
table = load_module("budget_level_table", REPO_ROOT / "scripts" / "budget_level_table.py")

QUALITY = "configs/experiments_budget_levels_lrscaled_pixel.yaml"
QUALITY_SMOKE = "configs/experiments_budget_levels_lrscaled_pixel_smoke.yaml"
TASK6 = "configs/experiments_budget_levels_pixel.yaml"
FINAL = "configs/experiments_final_frozen100.yaml"
MATCHED = "configs/experiments_matched_frozen100.yaml"

# The task's tables, verbatim (4 significant figures; "--" = no entry).
LR_TABLE = """
cheap pmf denoising        1.2      3       5.333  4
cheap pmf deblur           0.025    0.4444  0.2783 --
cheap pmf super_resolution 0.4      6       2.743  2
cheap pmf random_inpaint   0.4      6       3.2    2
cheap pmf box_inpaint      0.2      12      8      --
cheap jit denoising        0.1      2       7.68   0.3333
cheap jit deblur           0.01458  0.2857  1.864  --
cheap jit super_resolution 0.07     0.6667  8      0.4
cheap jit random_inpaint   0.1      0.3333  7.68   0.1333
cheap jit box_inpaint      0.07     0.6667  7.68   --
mid   pmf denoising        0.1714   0.75    1.185  0.6667
mid   pmf super_resolution 0.05714  1.5     0.6621 0.6667
mid   pmf random_inpaint   0.05714  2       0.7111 0.6667
mid   pmf box_inpaint      0.05     3       1.882  --
mid   jit denoising        0.01538  0.4286  1.882  0.08
mid   jit super_resolution 0.01522  0.1538  1.939  0.08333
mid   jit random_inpaint   0.01429  0.07143 1.882  0.032
mid   jit box_inpaint      0.01458  0.1429  1.901  --
"""
EXPECTED_LR = {}       # (level, prior, task, method) -> lr at 4 significant figures
for line in LR_TABLE.strip().splitlines():
    level, prior, task, *values = line.split()
    for method, v in zip(["rhso", "mpc_delta_t", "dflow", "mpc_rhc"], values):
        if v != "--":
            EXPECTED_LR[(level, prior, task, method)] = float(v)

EXP_OF = {"denoising": "denoising", "deblur": "deblurring", "super_resolution": "super_resolution",
          "random_inpaint": "random_inpainting", "box_inpaint": "box_inpainting"}
SOURCE = {("rhso", "pmf"): (FINAL, "__rhso"), ("rhso", "jit"): (MATCHED, "__rhso_matched"),
          ("mpc_delta_t", "pmf"): (FINAL, "__core"), ("mpc_delta_t", "jit"): (FINAL, "__core"),
          ("dflow", "pmf"): (FINAL, "__core"), ("dflow", "jit"): (FINAL, "__core"),
          ("mpc_rhc", "pmf"): (FINAL, "__rhc"), ("mpc_rhc", "jit"): (FINAL, "__rhc")}
KNOB = {"rhso": "num_opt_steps", "mpc_delta_t": "n_ctrl", "dflow": "num_opt_steps",
        "mpc_rhc": "n_ctrl"}
IDENTITY = {"job_id", "experiment"}


def resolve_file(path):
    cfg = yaml.safe_load((REPO_ROOT / path).read_text())
    return resolve_run_plan(cfg, validate_config(copy.deepcopy(cfg)), run_id="test")


def fields(spec):
    return json.loads(json.dumps(spec.to_dict(), default=str))


def level_of(spec):
    return re.search(r"__budget_(cheap|mid)", spec.experiment).group(1)


def sig4(x):
    return float("%.4g" % x)


# =====================================================================================
def check_generated():
    print("\n[1] regeneration and the task's table")
    files = gen.outputs()
    check("generator owns exactly the two files", sorted(files) == sorted([QUALITY, QUALITY_SMOKE]))
    stale = [p for p, text in files.items() if (REPO_ROOT / p).read_text() != text]
    check("committed files equal a fresh generation", not stale, str(stale))
    got = {(e["level"], e["prior"], e["task"], e["method"]): e["params"]["lr"]
           for e in gen.entries()}
    check("66 table entries (36 cheap, 30 mid)",
          len(EXPECTED_LR) == 66 and sum(k[0] == "cheap" for k in EXPECTED_LR) == 36)
    check("the generator has exactly the table's entries", set(got) == set(EXPECTED_LR))
    off = [(k, got[k], v) for k, v in EXPECTED_LR.items() if k in got and sig4(got[k]) != v]
    check("every lr equals the table to 4 significant figures", not off, str(off[:3]))


def check_quality():
    print("\n[2] quality configuration against Task 6")
    specs = resolve_file(QUALITY).specs
    task6 = {(s.experiment, s.model, s.method): s for s in resolve_file(TASK6).specs}
    check("66 jobs, one per table entry",
          len(specs) == 66 and {(level_of(s), s.model, s.problem, s.method) for s in specs}
          == set(EXPECTED_LR), "%d jobs" % len(specs))
    check("every job at 100 images", all(s.num_images == 100 for s in specs))
    smoke = resolve_file(QUALITY_SMOKE).specs
    check("smoke: the same 66 entries at 2 images",
          sorted((s.experiment, s.model, s.method) for s in smoke)
          == sorted((s.experiment, s.model, s.method) for s in specs)
          and all(s.num_images == 2 for s in smoke))
    names = {s.experiment for s in specs}
    check("block names: <task>__budget_<level>[_rhc|_rhso]_lr",
          all(re.fullmatch(r"[a-z_]+__budget_(cheap|mid)(_rhc|_rhso)?_lr", n) for n in names))
    check("blocks are Task 6's blocks plus _lr, one for one",
          {n[:-3] for n in names} == {k[0] for k in task6} and len(names) == len(
              {k[0] for k in task6}))
    pairs = [(s, task6.get((s.experiment[:-3], s.model, s.method))) for s in specs]
    check("every job has its Task 6 job", all(t is not None for _, t in pairs))
    pairs = [(s, t) for s, t in pairs if t is not None]
    bad = []
    for s, t in pairs:
        a, b = fields(s), fields(t)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - IDENTITY - {"lr"}
        if diff:
            bad.append((s.experiment, s.model, s.method, sorted(diff)))
    check("only lr differs from the Task 6 job (batch size, steps, all else equal)", not bad,
          str(bad[:2]))
    check("every lr differs from Task 6's (every knob was reduced)",
          all(s.lr != t.lr for s, t in pairs))
    check("backprops per job equal Task 6's",
          all(s.expected_backprops == t.expected_backprops for s, t in pairs),
          "total %d" % sum(s.expected_backprops for s in specs))
    check("batch sizes equal Task 6's", all(s.batch_size == t.batch_size for s, t in pairs))
    return specs


def check_rule(specs):
    print("\n[3] lr_new = lr_tuned x tuned knob / new knob")
    src = {}
    for path in (FINAL, MATCHED):
        for spec in resolve_file(path).specs:
            src[(path, spec.experiment, spec.model, spec.method)] = spec
    bad, n = [], 0
    for s in specs:
        path, suffix = SOURCE[(s.method, s.model)]
        source = src[(path, EXP_OF[s.problem] + suffix, s.model, s.method)]
        knob = KNOB[s.method]
        want = source.lr * getattr(source, knob) / getattr(s, knob)
        n += 1
        if abs(s.lr - want) > 1e-9 * abs(want):
            bad.append((s.experiment, s.model, s.method, s.lr, want))
    check("every one of the 66 lrs follows the rule (relative 1e-9)", n == 66 and not bad,
          str(bad[:2]))
    rhso = [s for s in specs if s.method == "rhso"]
    check("RHSO: M scales lr, N does not (18 jobs)", len(rhso) == 18 and all(
        abs(s.lr - src[(SOURCE[("rhso", s.model)][0], EXP_OF[s.problem]
                        + SOURCE[("rhso", s.model)][1], s.model, "rhso")].lr
            * src[(SOURCE[("rhso", s.model)][0], EXP_OF[s.problem]
                   + SOURCE[("rhso", s.model)][1], s.model, "rhso")].num_opt_steps
            / s.num_opt_steps) < 1e-9 * s.lr for s in rhso))


def check_comments():
    print("\n[4] comments")
    text = (REPO_ROOT / QUALITY).read_text()
    lines = [l for l in text.splitlines()
             if re.match(r"\s+(rhso|mpc_delta_t|dflow|mpc_rhc):\s+\{", l)]
    pattern = re.compile(r"# source: configs/experiments_(final|matched)_frozen100\.yaml :: "
                         r"\S+ / (pmf|jit) / (rhso|mpc_delta_t|dflow|mpc_rhc); "
                         r"(cheap|mid) level \(.*\); .*learning rate scaled by tuned/new steps"
                         r": lr [0-9.e-]+ x \d+/\d+ -> [0-9.e-]+; Task 6 entry \S+$")
    check("each of the 66 entries names its source, level and the lr scaling",
          len(lines) == 66 and all(pattern.search(l) for l in lines), "%d lines" % len(lines))
    block, levels = None, []
    for l in text.splitlines():
        m = re.match(r"  ([a-z_]+__budget_(cheap|mid)\S*):$", l)
        if m:
            block = m.group(2)
        elif re.match(r"\s+(rhso|mpc_delta_t|dflow|mpc_rhc):\s+\{", l):
            levels.append((block, re.search(r"; (cheap|mid) level", l).group(1)))
    check("each comment's level matches its block's",
          len(levels) == 66 and all(b == c for b, c in levels))
    check("no comment says a value was reused from another checkpoint",
          not re.search(r"reus|borrow|copied from|transferred", text, re.I))


# ------------------------------------------------------------------ synthetic runs
IMAGES = ["img%03d" % i for i in range(12)]
BASE = np.linspace(0.06, 0.10, len(IMAGES))


def qrows(job_id, model, problem, method, experiment, offset, params):
    rows = []
    for i, image in enumerate(IMAGES):
        row = {"job_id": job_id, "experiment": experiment, "model": model, "problem": problem,
               "method": method, "status": "ok", "image_id": image, "num_images": len(IMAGES),
               "batch_size": 4, "expected_backprops": 0, "psnr": 25.0 + i * 0.1,
               "ssim": 0.8 + i * 0.001, "lpips": float(BASE[i] + offset)}
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


def P(**kw):
    return kw


TUNED_RHSO = P(num_rhso_steps=4, num_opt_steps=40, lr=0.03, rhso_terminal_mode="direct")
T6_RHSO = P(num_rhso_steps=4, num_opt_steps=1, lr=0.03, rhso_terminal_mode="direct")
LR_RHSO = P(num_rhso_steps=4, num_opt_steps=1, lr=1.2, rhso_terminal_mode="direct")
T6_DF = P(num_opt_steps=6, lr=0.1)
LR_DF = P(num_opt_steps=6, lr=5.333333333)
MT_DF = P(num_opt_steps=6, lr=0.05)               # another variant's timing job, same knob
T6_DT_MID = P(num_mpc_steps=8, n_ctrl=4, lam=30, lr=0.15)
LR_DT_MID = P(num_mpc_steps=8, n_ctrl=4, lam=30, lr=0.75)
T6_J_RHSO = P(num_rhso_steps=7, num_opt_steps=2, lr=0.01, solver="heun",
              rhso_terminal_mode="direct")
LR_J_RHSO = P(num_rhso_steps=7, num_opt_steps=2, lr=0.1, solver="heun",
              rhso_terminal_mode="direct")
PNP = P(num_pnp_steps=10)


def synthetic(tmp):
    D = "denoising"
    other = write_quality(tmp / "tuned", (
        qrows("p_rhso", "pmf", D, "rhso", "denoising__rhso", 0.03, TUNED_RHSO)
        + qrows("p_pnp", "pmf", D, "pnp", "denoising__core", 0.0, PNP)))
    new = write_quality(tmp / "lr", (
        qrows("p_rhso_lr", "pmf", D, "rhso", "denoising__budget_cheap_lr", 0.02, LR_RHSO)
        + qrows("p_df_lr", "pmf", D, "dflow", "denoising__budget_cheap_lr", 0.05, LR_DF)
        + qrows("p_dt_mid_lr", "pmf", D, "mpc_delta_t", "denoising__budget_mid_lr", 0.01,
                LR_DT_MID)
        + qrows("j_rhso_lr", "jit", D, "rhso", "denoising__budget_cheap_rhso_lr", 0.02,
                LR_J_RHSO)))
    timing = write_timing(tmp / "timing", [
        ("pmf", D, "rhso", "denoising__rhso__rhso__pmf", 12.0, TUNED_RHSO),
        ("pmf", D, "pnp", "denoising__core__pnp__pmf", 0.13, PNP),
        ("pmf", D, "rhso", "denoising__budget_cheap__rhso__pmf", 0.55, T6_RHSO),
        ("pmf", D, "dflow", "denoising__budget_cheap__dflow__pmf", 0.50, T6_DF),
        ("pmf", D, "dflow", "denoising__matched_time__dflow__pmf", 0.90, MT_DF),
        ("pmf", D, "mpc_delta_t", "denoising__budget_mid__mpc_delta_t__pmf", 2.0, T6_DT_MID),
        ("jit", D, "rhso", "denoising__budget_cheap_rhso__rhso__jit", 1.40, T6_J_RHSO),
    ])
    return [other, new], timing


def refuses(fn):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            fn()
    except SystemExit:
        return True
    return False


def check_table():
    print("\n[5] scripts/budget_level_table.py --timing-ignore lr")
    v = table.variant_of
    check("_lr blocks are the lr-scaled variants",
          v("denoising__budget_cheap_lr") == "budget_cheap_lr"
          and v("denoising__budget_mid_rhc_lr") == "budget_mid_lr"
          and v("denoising__budget_cheap_rhso_lr") == "budget_cheap_lr"
          and v("denoising__budget_mid_lr__dflow__pmf") == "budget_mid_lr")
    check("Task 6 and earlier names keep their variants",
          v("denoising__budget_cheap") == "budget_cheap"
          and v("denoising__budget_mid_rhc__mpc_rhc__pmf") == "budget_mid"
          and v("denoising__budget_cheap_rhso") == "budget_cheap"
          and v("denoising__matched_time") == "matched_time"
          and v("denoising__rhso_matched") == "matched" and v("denoising__core") == "frozen")
    check("labels: 'cheap level, lr scaled' / 'mid level, lr scaled'",
          table.label("dflow", "budget_cheap_lr") == "D-Flow (cheap level, lr scaled)"
          and table.label("mpc_rhc", "budget_mid_lr") == "MPC-RHC (mid level, lr scaled)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        quality_runs, timing_run = synthetic(tmp)
        quality = table.mtt.read_quality(quality_runs)
        timing = table.mtt.read_timing([timing_run])
        lr_ids = {"p_rhso_lr", "p_df_lr", "p_dt_mid_lr", "j_rhso_lr"}

        rows, notes = table.build(quality, timing, resamples=50)
        check("default: no lr-scaled job is paired (lr differs from every timing job)",
              not any(r["quality_job_id"] in lr_ids for r in rows))
        check("default: each lr-scaled job is reported untimed",
              all(any(i in n and "no timing job" in n for n in notes) for i in lr_ids))
        check("default: the other jobs still pair as before",
              {r["quality_job_id"] for r in rows} == {"p_rhso", "p_pnp"})

        rows, notes = table.build(quality, timing, resamples=50, timing_ignore=["lr"])
        got = {r["quality_job_id"]: r for r in rows}
        check("with lr ignored: all four lr-scaled jobs are paired", lr_ids <= set(got),
              str(sorted(set(got))))
        want_timing = {"p_rhso_lr": ("denoising__budget_cheap__rhso__pmf", 0.55),
                       "p_df_lr": ("denoising__budget_cheap__dflow__pmf", 0.50),
                       "p_dt_mid_lr": ("denoising__budget_mid__mpc_delta_t__pmf", 2.0),
                       "j_rhso_lr": ("denoising__budget_cheap_rhso__rhso__jit", 1.40)}
        ok = all(i in got and want_timing[i][0] in str(got[i]["timing_job_id"])
                 and abs(got[i]["seconds_per_image"] - want_timing[i][1]) < 1e-9
                 for i in want_timing)
        check("... each with its Task 6 timing job and time", ok)
        check("... D-Flow picks the Task 6 timing job over a matched-time one with the same "
              "knob", "p_df_lr" in got and "budget_cheap__dflow" in got["p_df_lr"]["timing_job_id"])
        check("... labelled 'cheap level, lr scaled' / 'mid level, lr scaled'",
              got.get("p_df_lr", {}).get("label") == "D-Flow (cheap level, lr scaled)"
              and got.get("p_dt_mid_lr", {}).get("label")
              == "MPC-$\\Delta t$ (mid level, lr scaled)")
        check("... and a note says how many jobs were paired across lr",
              any("paired with a timing job that differs in lr" in n for n in notes))
        cheap_df = [r for r in rows if r["quality_job_id"] == "p_df_lr" and r["level"] == "cheap"]
        check("... and the lr-scaled D-Flow enters the pMF denoising cheap level",
              len(cheap_df) == 1 and cheap_df[0]["within_budget"])

        check("build refuses a field other than lr",
              refuses(lambda: table.build(quality, timing, resamples=50,
                                          timing_ignore=["mu"])))
        out = tmp / "out"
        args = sum([["--quality", q] for q in quality_runs], []) + [
            "--timing", timing_run, "--out-dir", str(out), "--resamples", "50"]
        with contextlib.redirect_stdout(io.StringIO()) as buf, \
                contextlib.redirect_stderr(io.StringIO()):
            code = table.main(args + ["--timing-ignore", "lr"])
        with open(out / "budget_levels.csv", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        check("CLI --timing-ignore lr: exit 0, lr-scaled rows in the CSV",
              code == 0 and any(r["variant"] == "budget_cheap_lr" for r in csv_rows)
              and "lr scaled" in buf.getvalue())
        check("CLI rejects --timing-ignore mu",
              refuses(lambda: table.main(args + ["--timing-ignore", "mu"])))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            table.main(args)
        with open(out / "budget_levels.csv", newline="") as fh:
            plain = list(csv.DictReader(fh))
        check("CLI without the option: same columns, no lr-scaled rows",
              list(plain[0]) == list(csv_rows[0])
              and not any(r["variant"].endswith("_lr") for r in plain))


def main():
    check_generated()
    specs = check_quality()
    check_rule(specs)
    check_comments()
    check_table()
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
