#!/usr/bin/env python
"""RHSO anchor schedules on pMF-L/16 (rebuttal Task 8): the 100-image table.

    python scripts/anchor_schedule_table.py \
        --final $MPCFLOW_OUTPUT_ROOT/run_<final id> \
        --reference $MPCFLOW_OUTPUT_ROOT/run_9187465 \
        --out-dir $MPCFLOW_OUTPUT_ROOT/anchor_schedule_table

FINAL runs are runs of configs/experiments_anchor_schedule_frozen100.yaml, read from
`results_per_image.csv` (aggregate a sharded run first).  Each block's name says which
schedule(s) it stands for (`<task>__anchor_<schedule>[_<schedule>...]`; a block named for
several schedules is one mu0 = 0 run that each of them selected, and is reported once per
schedule).

Rows, per task: constant, increasing, decreasing (each with its retuned mu0), then the
paper's frozen RHSO from --reference (the task's pMF `<task>__rhso` job of run_9187465).

Columns: mu0; mean PSNR / SSIM / LPIPS over the images; measurement RMSE next to the task's
noise level sigma; and, for increasing and decreasing, the paired difference

    schedule - retuned constant      (PSNR, SSIM: positive favours the schedule; LPIPS: negative)

with a 95% percentile bootstrap over the paired images, exactly as scripts/paired_stats.py
(10,000 resamples, seed 20260903, a fresh generator per comparison, one index matrix shared
by the three metrics).  A schedule whose selected job IS the constant's (both chose mu0 = 0)
differs by exactly zero and is marked "same run".

The reference row is UNPAIRED: its means only.  run_9187465 ran on a different GPU type than
the final run, so per-image differences against it would mix the method with the hardware.
Measurement RMSE is recorded once per job (pooled over the job's images), so it is reported
as a job-level value with no interval, as in paired_stats.py.

Outputs in --out-dir: anchor_schedule_table.csv and anchor_schedule_table.tex.
"""
import argparse
import collections
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ps = _load("paired_stats")
freeze = _load("freeze_anchor_schedules")
hpo = freeze.hpo

SEED, RESAMPLES, METRICS = ps.SEED, ps.RESAMPLES, ps.METRICS
SCHEDULES = freeze.SCHEDULES
TASKS = list(hpo.PROBLEMS)
TASK_NAME = freeze.TASK_NAME
SIGMA = {p: float(yaml.safe_load(hpo.DEGRADATION[p])["sigma"]) for p in TASKS}
MODEL = "pmf"
REFERENCE_LABEL = "paper (frozen RHSO, run_9187465)"


# ------------------------------------------------------------------ readers
def _jobs(paths, accept):
    """job key -> {job_id, run, experiment, row fields, images}; `accept(row)` filters rows."""
    jobs = collections.OrderedDict()
    for path in paths:
        csv_path = ps.per_image_csv(path)
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("model") != MODEL or r.get("method") != "rhso" \
                        or r.get("status", "ok") not in ("ok", "", None) or not accept(r):
                    continue
                key = (r["job_id"], str(csv_path))
                job = jobs.get(key)
                if job is None:
                    job = jobs[key] = {"job_id": r["job_id"], "run": str(path),
                                       "experiment": r["experiment"], "problem": r["problem"],
                                       "mu": ps.fnum(r.get("mu")),
                                       "mu_schedule": freeze.schedule_of(r),
                                       "fields": {k: r.get(k) for k in freeze.CHECKED_FIELDS},
                                       "rmse": ps.fnum(r.get("measurement_rmse")),
                                       "images": {}}
                if r["image_id"] in job["images"]:
                    raise SystemExit("%s: image %s appears twice in job %s"
                                     % (path, r["image_id"], r["job_id"]))
                job["images"][r["image_id"]] = r
    return list(jobs.values())


def read_final(paths):
    """(task, schedule) -> job, from blocks named <task>__anchor_<schedule>[_...]."""
    out = {}
    for job in _jobs(paths, lambda r: freeze.schedules_of_block(r["experiment"]) is not None):
        schedules = freeze.schedules_of_block(job["experiment"])
        if len(schedules) > 1 and job["mu"] != 0.0:
            raise SystemExit("%s (job %s) stands for %s but ran mu0=%s; only a mu0 = 0 run "
                             "can be shared" % (job["experiment"], job["job_id"],
                                                schedules, job["mu"]))
        if len(schedules) == 1 and job["mu"] != 0.0 and job["mu_schedule"] != schedules[0]:
            raise SystemExit("%s (job %s) ran mu_schedule=%s" % (job["experiment"],
                                                                 job["job_id"], job["mu_schedule"]))
        for s in schedules:
            cell = (job["problem"], s)
            if cell in out:
                raise SystemExit("%s/%s has two jobs (%s in %s, %s in %s); pass runs with "
                                 "exactly one" % (cell[0], s, out[cell]["job_id"], out[cell]["run"],
                                                  job["job_id"], job["run"]))
            out[cell] = job
    return out


def read_reference(paths):
    """task -> the paper's frozen pMF RHSO job (<task>__rhso)."""
    names = {hpo.EXP_NAME[p] + "__rhso": p for p in TASKS}
    out = {}
    for job in _jobs(paths, lambda r: r["experiment"] in names):
        task = names[job["experiment"]]
        if task in out:
            raise SystemExit("reference %s has two jobs (%s, %s)" % (task, out[task]["job_id"],
                                                                     job["job_id"]))
        out[task] = job
    return out


# ------------------------------------------------------------------ the table
def means(job):
    out = {}
    for metric in METRICS:
        vals = [ps.fnum(r.get(metric)) for r in job["images"].values()]
        if any(v is None for v in vals):
            raise SystemExit("job %s: missing %s values" % (job["job_id"], metric))
        out[metric] = float(np.mean(vals))
    return out


def paired(job, ref, seed, resamples):
    ids_a, ids_b = set(job["images"]), set(ref["images"])
    if ids_a != ids_b:
        raise SystemExit("%s vs %s: the two sides cover different images (%d only in the "
                         "schedule, %d only in constant)"
                         % (job["job_id"], ref["job_id"], len(ids_a - ids_b), len(ids_b - ids_a)))
    ids = sorted(ids_a)
    diffs = {m: np.subtract([ps.fnum(job["images"][i][m]) for i in ids],
                            [ps.fnum(ref["images"][i][m]) for i in ids]) for m in METRICS}
    return ps.bootstrap(diffs, seed, resamples)


def _row(task, role, schedule, job, notes=""):
    row = {"task": task, "task_name": TASK_NAME[task], "role": role, "schedule": schedule,
           "mu0": job["mu"], "n_images": len(job["images"]), "sigma": SIGMA[task],
           "measurement_rmse": job["rmse"],
           "rmse_over_sigma": (job["rmse"] / SIGMA[task]) if job["rmse"] is not None else None,
           "num_rhso_steps": job["fields"].get("num_rhso_steps"),
           "num_opt_steps": job["fields"].get("num_opt_steps"), "lr": job["fields"].get("lr"),
           "same_run_as_constant": False, "job_id": job["job_id"], "run": job["run"],
           "experiment": job["experiment"], "notes": notes}
    row.update(means(job))
    for metric in METRICS:
        row.update({"%s_diff" % metric: None, "%s_ci_lo" % metric: None,
                    "%s_ci_hi" % metric: None})
    return row


def build(final, reference, seed=SEED, resamples=RESAMPLES):
    rows, notes = [], []
    for task in TASKS:
        constant = final.get((task, "constant"))
        if constant is None:
            if any(k[0] == task for k in final):
                notes.append("%s: no constant-schedule job; task skipped" % task)
            continue
        for s in SCHEDULES:
            job = final.get((task, s))
            if job is None:
                notes.append("%s: no %s job" % (task, s))
                continue
            row = _row(task, "schedule", s, job)
            if s != "constant":
                stats = paired(job, constant, seed, resamples)
                for metric in METRICS:
                    mean, lo, hi = stats[metric]
                    row.update({"%s_diff" % metric: mean, "%s_ci_lo" % metric: lo,
                                "%s_ci_hi" % metric: hi})
                if job["job_id"] == constant["job_id"] and job["run"] == constant["run"]:
                    row["same_run_as_constant"] = True
                    row["notes"] = "same run as constant (both selected mu0 = 0)"
            rows.append(row)
        ref = reference.get(task)
        if ref is not None:
            row = _row(task, "reference", None, ref,
                       "unpaired means; different GPU type from the final run")
            rows.append(row)
            for k in ("num_rhso_steps", "num_opt_steps", "lr"):
                if ps.fnum(ref["fields"].get(k)) != ps.fnum(constant["fields"].get(k)):
                    notes.append("WARNING %s: reference %s=%s differs from the final run's %s"
                                 % (task, k, ref["fields"].get(k), constant["fields"].get(k)))
            if set(ref["images"]) != set(constant["images"]):
                notes.append("WARNING %s: the reference covers different images" % task)
        elif reference:
            notes.append("%s: no reference job" % task)
    for r in rows:
        r.update(resamples=resamples, seed=seed)
    return rows, notes


COLUMNS = (["task", "task_name", "role", "schedule", "mu0", "n_images"] + METRICS
           + ["measurement_rmse", "sigma", "rmse_over_sigma"]
           + ["%s_%s" % (m, s) for m in METRICS for s in ("diff", "ci_lo", "ci_hi")]
           + ["same_run_as_constant", "num_rhso_steps", "num_opt_steps", "lr", "resamples",
              "seed", "job_id", "experiment", "run", "notes"])


# ------------------------------------------------------------------ outputs
def _fmt(v, f):
    return "--" if v is None else f % v


def _ci(r, metric, digits):
    if r["%s_diff" % metric] is None:
        return "--"
    f = "%+." + str(digits) + "f"
    return (f + " [" + f + ", " + f + "]") % (r["%s_diff" % metric], r["%s_ci_lo" % metric],
                                              r["%s_ci_hi" % metric])


def _mu(v):
    return "--" if v is None else ("0" if v == 0.0 else "%g" % v)


def latex(rows):
    resamples = rows[0]["resamples"] if rows else RESAMPLES
    seed = rows[0]["seed"] if rows else SEED
    lines = [
        "% Generated by scripts/anchor_schedule_table.py; needs \\usepackage{booktabs}.",
        "%% Differences are schedule - retuned constant, mean over paired images [95%% "
        "bootstrap CI, %d resamples, seed %d]." % (resamples, seed),
        "% Measurement RMSE is job-level (pooled over the images); no CI.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{RHSO with a time-dependent state anchor on pMF-L/16 (100 images). "
        "$\\mu_k = \\mu_0\\, m(t_k)$; each schedule's $\\mu_0$ retuned on 8 tuning images. "
        "$\\Delta$ = schedule $-$ retuned constant, 95\\%% bootstrap CI. RMSE is the "
        "measurement RMSE (job level) next to the noise level $\\sigma$. Paper: the "
        "frozen RHSO of the benchmark, unpaired means from a different GPU type.}",
        "\\label{tab:rebuttal-anchor-schedules}", "\\small",
        "\\begin{tabular}{llrrrrrrlll}", "\\toprule",
        "Task & Schedule & $\\mu_0$ & PSNR & SSIM & LPIPS & RMSE & $\\sigma$ & "
        "$\\Delta$PSNR & $\\Delta$SSIM & $\\Delta$LPIPS \\\\", "\\midrule"]
    previous = None
    for r in rows:
        first = r["task"] != previous
        if previous is not None and first:
            lines.append("\\midrule")
        previous = r["task"]
        name = "paper (frozen)" if r["role"] == "reference" else r["schedule"]
        if r["same_run_as_constant"]:
            name += "$^\\ast$"
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            r["task_name"] if first else "", name, _mu(r["mu0"]), _fmt(r["psnr"], "%.2f"),
            _fmt(r["ssim"], "%.4f"), _fmt(r["lpips"], "%.4f"),
            _fmt(r["measurement_rmse"], "%.4f"), "%g" % r["sigma"],
            _ci(r, "psnr", 2), _ci(r, "ssim", 4), _ci(r, "lpips", 4)))
    lines += ["\\bottomrule", "\\end{tabular}"]
    if any(r["same_run_as_constant"] for r in rows):
        lines.append("\\\\[2pt]{\\footnotesize $^\\ast$ same run as constant: both selected "
                     "$\\mu_0 = 0$, where every schedule is the same objective.}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write(rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out / "anchor_schedule_table.csv", out / "anchor_schedule_table.tex"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})
    tex_path.write_text(latex(rows))
    return csv_path, tex_path


def print_rows(rows):
    previous = None
    for r in rows:
        if r["task"] != previous:
            previous = r["task"]
            print("\n%s  (sigma %g)" % (r["task"], r["sigma"]))
        name = REFERENCE_LABEL if r["role"] == "reference" else r["schedule"]
        print("  %-34s mu0 %-5s PSNR %6.2f SSIM %.4f LPIPS %.4f RMSE %s  dLPIPS %s%s"
              % (name, _mu(r["mu0"]), r["psnr"], r["ssim"], r["lpips"],
                 _fmt(r["measurement_rmse"], "%.4f"), _ci(r, "lpips", 4),
                 "  (same run as constant)" if r["same_run_as_constant"] else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final", action="append", required=True, metavar="RUN",
                   help="run of experiments_anchor_schedule_frozen100.yaml, directory or "
                        "results_per_image.csv (repeatable)")
    p.add_argument("--reference", action="append", default=[], metavar="RUN",
                   help="the paper's benchmark run (run_9187465), directory or "
                        "results_per_image.csv (repeatable)")
    p.add_argument("--out-dir", default="outputs/anchor_schedule_table")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    rows, notes = build(read_final(a.final), read_reference(a.reference) if a.reference else {},
                        a.seed, a.resamples)
    if not rows:
        raise SystemExit("no task has a constant-schedule job in the final runs given")
    print_rows(rows)
    for note in notes:
        print("! " + note, file=sys.stderr)
    csv_path, tex_path = write(rows, a.out_dir)
    print("\nwrote %s and %s: %d rows" % (csv_path, tex_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
