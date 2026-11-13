#!/usr/bin/env python
"""Matched-time comparison on the pixel priors (rebuttal Task 5): time, quality, paired CIs.

    python scripts/matched_time_table.py \
        --quality outputs/<matched-time run> --quality outputs/run_9187465 \
        --quality outputs/run_9191132 --quality outputs/run_9187499 --quality outputs/run_9187476 \
        --timing outputs/timing_pmf --timing outputs/timing_jit --timing outputs/timing_matched_time \
        --out-dir outputs/matched_time_table

QUALITY runs are read from `results_per_image.csv` (aggregate a sharded run first), TIMING
runs from the per-job `metadata.json` that scripts/run_timing.py leaves (the same reader as
scripts/summarize_timing.py: device-synchronised seconds per image at batch size one).

Which entry plays which role, per prior:

    RHSO (reference)   pMF: the frozen RHSO          JiT: the matched-BP RHSO (direct planner)
    MPC-delta_t        pMF: Direction B (matched BP) JiT: the frozen MPC-delta_t
    PnP-Flow, D-Flow,  each: the frozen entry AND the matched-time entry, whichever exist
    MPC-RHC, Flower

An entry's variant comes from its experiment name: `..__matched_time..` is matched-time,
any other `.._matched..` is matched-BP, everything else is frozen.  A (prior, task, method,
variant) cell must come from exactly one job across all the runs given, and the timing job
paired with a quality job must have the same budget and step-size fields, or the script stops.

Per (prior, task), every entry gets its measured seconds per image, the ratio to RHSO's,
mean PSNR / SSIM / LPIPS over the images, and the paired difference

    RHSO - method        (PSNR, SSIM: positive favours RHSO; LPIPS: negative does)

with a 95% percentile bootstrap over the paired images exactly as scripts/paired_stats.py
(10,000 resamples, seed 20260903, a fresh generator per comparison, one index matrix shared
by the three metrics).  For each method, the entry with the lowest LPIPS among those that do
not exceed RHSO's time by more than 10% (--tolerance) is marked as that method's matched-time
entry; a method with no entry inside the tolerance has none marked.

Outputs in --out-dir: matched_time.csv (one row per entry) and matched_time.tex.
"""
import argparse
import collections
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ps = _load("paired_stats", REPO_ROOT / "scripts" / "paired_stats.py")
summ = _load("summarize_timing", REPO_ROOT / "scripts" / "summarize_timing.py")

SEED, RESAMPLES, TOLERANCE = ps.SEED, ps.RESAMPLES, 0.10
METRICS = ps.METRICS
PRIOR_NAME = ps.PRIOR_NAME
PRIORS = ["pmf", "jit"]
TASK_NAME = ps.TASK_NAME
TASK_ORDER = ps.TASK_ORDER
# method -> prior -> the variants that enter the table, in display order
ROLES = {
    "rhso":        {"pmf": ["frozen"], "jit": ["matched"]},
    "mpc_delta_t": {"pmf": ["matched"], "jit": ["frozen"]},
    "mpc_rhc":     {"pmf": ["frozen", "matched_time"], "jit": ["frozen", "matched_time"]},
    "pnp":         {"pmf": ["frozen", "matched_time"], "jit": ["frozen", "matched_time"]},
    "dflow":       {"pmf": ["frozen", "matched_time"], "jit": ["frozen", "matched_time"]},
    "flower":      {"pmf": ["frozen", "matched_time"], "jit": ["frozen", "matched_time"]},
}
METHOD_ORDER = list(ROLES)
METHOD_NAME = {"rhso": "RHSO", "mpc_delta_t": "MPC-$\\Delta t$", "mpc_rhc": "MPC-RHC",
               "pnp": "PnP-Flow", "dflow": "D-Flow", "flower": "Flower"}
VARIANT_NAME = {("rhso", "frozen"): "", ("rhso", "matched"): " (matched BP, direct)",
                ("mpc_delta_t", "matched"): " (Direction B)", ("mpc_delta_t", "frozen"): "",
                "frozen": " (frozen)", "matched_time": " (matched time)",
                "matched": " (matched BP)"}
# Fields that define the computation; a timing job must agree with its quality job on all
# of them that both record.
PARAM_FIELDS = ("problem", "t0", "steps", "solver", "num_mpc_steps", "n_ctrl", "K", "lam", "lr",
                "num_pnp_steps", "gamma0", "alpha", "noise_samples", "num_opt_steps",
                "num_rhso_steps", "mu", "rhso_terminal_mode", "num_flower_steps",
                "flower_gamma", "flower_cg_max_iters", "flower_cg_tol")
BUDGET = {"rhso": ("num_rhso_steps", "num_opt_steps"), "mpc_delta_t": ("num_mpc_steps", "n_ctrl"),
          "mpc_rhc": ("num_mpc_steps", "n_ctrl"), "pnp": ("num_pnp_steps",),
          "dflow": ("num_opt_steps",), "flower": ("num_flower_steps",)}


def variant_of(experiment):
    name = str(experiment or "")
    if "__matched_time" in name:
        return "matched_time"
    return "matched" if "_matched" in name else "frozen"


def label(method, variant):
    suffix = VARIANT_NAME.get((method, variant), VARIANT_NAME.get(variant, " (%s)" % variant))
    return METHOD_NAME.get(method, method) + suffix


def norm(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "nan", "NaN", "null"):
        return None
    try:
        return round(float(text), 9)
    except ValueError:
        return text


def budget_text(method, fields):
    keys = BUDGET.get(method, ())
    return " ".join("%s=%s" % (k, _num(fields.get(k))) for k in keys)


def _num(v):
    v = norm(v)
    if isinstance(v, float) and v.is_integer():
        return "%d" % v
    return "--" if v is None else str(v)


# ------------------------------------------------------------------ readers
def read_quality(paths):
    """{(model, problem, method, variant): [job, ...]}; job = {job_id, run, images, fields, ...}."""
    jobs = collections.OrderedDict()
    for path in paths:
        csv_path = ps.per_image_csv(path)
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("method") not in ROLES or r.get("status", "ok") not in ("ok", "", None):
                    continue
                key = (r["job_id"], str(csv_path))
                job = jobs.get(key)
                if job is None:
                    job = jobs[key] = {"job_id": r["job_id"], "run": str(path),
                                       "experiment": r.get("experiment"),
                                       "cell": (r["model"], r["problem"], r["method"],
                                                variant_of(r.get("experiment"))),
                                       "fields": {k: r.get(k) for k in PARAM_FIELDS},
                                       "bp": ps.bp_per_image(r), "images": {}}
                if r["image_id"] in job["images"]:
                    raise SystemExit("%s: image %s appears twice in job %s"
                                     % (path, r["image_id"], r["job_id"]))
                job["images"][r["image_id"]] = r
    cells = collections.defaultdict(list)
    for job in jobs.values():
        cells[job["cell"]].append(job)
    return cells


def read_timing(run_dirs):
    """{(model, problem, method, variant): [summary row + fields]} from timing runs."""
    cells = collections.defaultdict(list)
    for run_dir in run_dirs:
        payloads, baselines = summ.load_run(run_dir)
        for payload in payloads:
            if payload.get("status") != "ok":
                continue
            row = summ.summarise_job(payload, baselines)
            row["fields"] = {k: payload.get(k) for k in PARAM_FIELDS}
            row["run"] = str(run_dir)
            cells[(payload.get("model"), payload.get("problem"), payload.get("method"),
                   variant_of(payload.get("experiment")))].append(row)
    return cells


def one(cells, key, what):
    found = cells.get(key) or []
    if len(found) > 1:
        raise SystemExit("%s %s has %d jobs (%s); pass runs with exactly one"
                         % (what, "/".join(key), len(found),
                            ", ".join("%s in %s" % (j["job_id"], j["run"]) for j in found)))
    return found[0] if found else None


def check_same_config(quality, timing):
    diffs = []
    for k in PARAM_FIELDS:
        a, b = norm(quality["fields"].get(k)), norm(timing["fields"].get(k))
        if a is not None and b is not None and a != b:
            diffs.append("%s %s vs %s" % (k, a, b))
    if diffs:
        raise SystemExit("timing job %s (%s) is not the configuration of quality job %s (%s): %s"
                         % (timing["job_id"], timing["run"], quality["job_id"], quality["run"],
                            "; ".join(diffs)))


# ------------------------------------------------------------------ the comparison
def paired(ref, job, seed, resamples):
    ids_r, ids_m = set(ref["images"]), set(job["images"])
    if ids_r != ids_m:
        raise SystemExit("%s vs %s: the two sides cover different images (%d only in RHSO, %d "
                         "only in the method); a paired comparison needs the same set"
                         % (ref["job_id"], job["job_id"], len(ids_r - ids_m), len(ids_m - ids_r)))
    ids = sorted(ids_r)
    diffs = {}
    for metric in METRICS:
        a = [ps.fnum(ref["images"][i].get(metric)) for i in ids]
        b = [ps.fnum(job["images"][i].get(metric)) for i in ids]
        if any(v is None for v in a + b):
            raise SystemExit("%s vs %s: missing %s values" % (ref["job_id"], job["job_id"], metric))
        diffs[metric] = np.subtract(a, b)
    return ps.bootstrap(diffs, seed, resamples)


def means(job):
    out = {}
    for metric in METRICS:
        vals = [ps.fnum(r.get(metric)) for r in job["images"].values()]
        vals = [v for v in vals if v is not None]
        out[metric] = float(np.mean(vals)) if vals else None
    return out


def build(quality, timing, tolerance=TOLERANCE, seed=SEED, resamples=RESAMPLES):
    """(rows, notes).  Rows in prior, task, method, variant order."""
    rows, notes = [], []
    for prior in PRIORS:
        for task in TASK_ORDER:
            ref_variant = ROLES["rhso"][prior][0]
            ref = one(quality, (prior, task, "rhso", ref_variant), "quality")
            if ref is None:
                if any(k[0] == prior and k[1] == task for k in quality):
                    notes.append("%s/%s: no %s RHSO quality job; cell skipped"
                                 % (prior, task, ref_variant))
                continue
            ref_t = one(timing, (prior, task, "rhso", ref_variant), "timing")
            if ref_t is not None:
                check_same_config(ref, ref_t)
            else:
                notes.append("%s/%s: no timing for the RHSO reference; nothing can be marked"
                             % (prior, task))
            ref_s = ref_t["seconds_per_image_mean"] if ref_t else None
            cell_rows = [_row(prior, task, "rhso", ref_variant, ref, ref_t, ref_s, None, True)]
            for method in METHOD_ORDER[1:]:
                mine = []
                for variant in ROLES[method][prior]:
                    job = one(quality, (prior, task, method, variant), "quality")
                    if job is None:
                        continue
                    t = one(timing, (prior, task, method, variant), "timing")
                    if t is not None:
                        check_same_config(job, t)
                    else:
                        notes.append("%s/%s %s: no timing job" % (prior, task,
                                                                  label(method, variant)))
                    stats = paired(ref, job, seed, resamples)
                    mine.append(_row(prior, task, method, variant, job, t, ref_s, stats, False,
                                     tolerance))
                eligible = [r for r in mine if r["within_tolerance"] is True
                            and r["lpips"] is not None]
                if eligible:
                    min(eligible, key=lambda r: r["lpips"])["selected"] = True
                elif any(r["time_ratio"] is not None for r in mine):
                    notes.append("%s/%s %s: no entry within %.0f%% of RHSO's time; none marked"
                                 % (prior, task, METHOD_NAME[method], 100 * tolerance))
                cell_rows += mine
            for r in cell_rows:
                r.update(resamples=resamples, seed=seed, tolerance=tolerance)
            rows += cell_rows
    return rows, notes


def _row(prior, task, method, variant, job, t, ref_s, stats, reference, tolerance=TOLERANCE):
    seconds = t["seconds_per_image_mean"] if t else None
    ratio = seconds / ref_s if seconds is not None and ref_s else None
    row = {"prior": prior, "prior_name": PRIOR_NAME.get(prior, prior), "task": task,
           "method": method, "variant": variant, "label": label(method, variant),
           "role": "reference" if reference else "comparison",
           "quality_job_id": job["job_id"], "quality_run": job["run"],
           "timing_job_id": t["job_id"] if t else None, "timing_run": t["run"] if t else None,
           "budget": budget_text(method, job["fields"]), "bp_per_image": job["bp"],
           "n_images": len(job["images"]), "seconds_per_image": seconds,
           "seconds_per_image_sd": t["seconds_per_image_sd"] if t else None,
           "seconds_basis": t["seconds_basis"] if t else None,
           "rhso_seconds_per_image": ref_s, "time_ratio": ratio,
           "within_tolerance": (None if ratio is None else
                                (True if reference else ratio <= 1.0 + tolerance + 1e-12)),
           "selected": False, "gpu_name": t["gpu_name"] if t else None}
    row.update({m: v for m, v in means(job).items()})
    for metric in METRICS:
        mean, lo, hi = stats[metric] if stats else (None, None, None)
        row.update({"%s_diff" % metric: mean, "%s_ci_lo" % metric: lo,
                    "%s_ci_hi" % metric: hi})
    return row


COLUMNS = (["prior", "prior_name", "task", "method", "variant", "label", "role", "selected",
            "budget", "bp_per_image", "n_images", "seconds_per_image", "seconds_per_image_sd",
            "seconds_basis", "rhso_seconds_per_image", "time_ratio", "within_tolerance",
            "tolerance"]
           + METRICS
           + ["%s_%s" % (m, s) for m in METRICS for s in ("diff", "ci_lo", "ci_hi")]
           + ["resamples", "seed", "quality_job_id", "quality_run", "timing_job_id",
              "timing_run", "gpu_name"])


# ------------------------------------------------------------------ outputs
def _fmt(v, f):
    return "--" if v is None else f % v


def _ci(r, metric, digits):
    if r["%s_diff" % metric] is None:
        return "--"
    f = "%+." + str(digits) + "f"
    return (f + " [" + f + ", " + f + "]") % (r["%s_diff" % metric], r["%s_ci_lo" % metric],
                                              r["%s_ci_hi" % metric])


def latex(rows, tolerance=TOLERANCE):
    gpus = sorted({summ.short_gpu(r["gpu_name"]) for r in rows if r["gpu_name"]})
    gpu = gpus[0] if len(gpus) == 1 else ("mixed GPUs" if gpus else "GPU unknown")
    resamples = rows[0]["resamples"] if rows else RESAMPLES
    seed = rows[0]["seed"] if rows else SEED
    lines = [
        "% Generated by scripts/matched_time_table.py; needs \\usepackage{booktabs}.",
        "%% Differences are RHSO - method, mean over paired images [95%% bootstrap CI, "
        "%d resamples, seed %d]." % (resamples, seed),
        "\\begin{table}[t]", "\\centering",
        "\\caption{Matched wall-clock comparison (seconds/image at batch size one, %s). "
        "$\\Delta$ = RHSO $-$ method with 95\\%% bootstrap CI. $^\\dagger$: the method's "
        "matched-time entry, the lowest LPIPS within %d\\%% of RHSO's time.}"
        % (gpu, round(100 * tolerance)),
        "\\label{tab:rebuttal-matched-time}", "\\small",
        "\\begin{tabular}{lllrrrrrlll}", "\\toprule",
        "Prior & Task & Method & s/img & $\\times$RHSO & PSNR & SSIM & LPIPS & "
        "$\\Delta$PSNR & $\\Delta$SSIM & $\\Delta$LPIPS \\\\", "\\midrule"]
    previous = None
    for r in rows:
        cell = (r["prior"], r["task"])
        if previous is not None and cell != previous:
            lines.append("\\midrule" if cell[0] != previous[0] else "\\cmidrule(lr){2-11}")
        first = cell != previous
        previous = cell
        name = r["label"] + ("$^\\dagger$" if r["selected"] else "")
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            r["prior_name"] if first else "", TASK_NAME.get(r["task"], r["task"]) if first else "",
            name, _fmt(r["seconds_per_image"], "%.2f"), _fmt(r["time_ratio"], "%.2f"),
            _fmt(r["psnr"], "%.2f"), _fmt(r["ssim"], "%.4f"), _fmt(r["lpips"], "%.4f"),
            _ci(r, "psnr", 2), _ci(r, "ssim", 4), _ci(r, "lpips", 4)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def write(rows, out_dir, tolerance=TOLERANCE):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "matched_time.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})
    (out / "matched_time.tex").write_text(latex(rows, tolerance))
    return out / "matched_time.csv", out / "matched_time.tex"


def print_rows(rows):
    previous = None
    for r in rows:
        if (r["prior"], r["task"]) != previous:
            previous = (r["prior"], r["task"])
            print("\n%s  %s" % (r["prior_name"], r["task"]))
        print("  %1s %-34s %-34s s/img %7s  x%5s  PSNR %6s SSIM %6s LPIPS %6s  dLPIPS %s"
              % ("*" if r["selected"] else "", r["label"].replace("$\\Delta t$", "dt"),
                 r["budget"], _fmt(r["seconds_per_image"], "%.2f"),
                 _fmt(r["time_ratio"], "%.2f"), _fmt(r["psnr"], "%.2f"),
                 _fmt(r["ssim"], "%.4f"), _fmt(r["lpips"], "%.4f"), _ci(r, "lpips", 4)))
    print("\n  * = the method's matched-time entry (lowest LPIPS within the time tolerance)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quality", action="append", required=True, metavar="RUN",
                   help="quality run directory or results_per_image.csv (repeatable)")
    p.add_argument("--timing", action="append", required=True, metavar="RUN",
                   help="timing run directory written by scripts/run_timing.py (repeatable)")
    p.add_argument("--out-dir", default="outputs/matched_time_table")
    p.add_argument("--tolerance", type=float, default=TOLERANCE,
                   help="largest allowed excess over RHSO's time, as a fraction (default 0.10)")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    rows, notes = build(read_quality(a.quality), read_timing(a.timing), a.tolerance, a.seed,
                        a.resamples)
    if not rows:
        raise SystemExit("no (prior, task) has an RHSO reference in the quality runs given")
    print_rows(rows)
    for prior in PRIORS:
        gpus = sorted({str(r["gpu_name"]) for r in rows if r["prior"] == prior and r["gpu_name"]})
        if len(gpus) > 1:
            notes.append("WARNING: %s timings come from different GPU types (%s); the ratios "
                         "to RHSO are not like for like" % (prior, ", ".join(gpus)))
    for note in notes:
        print("! " + note, file=sys.stderr)
    csv_path, tex_path = write(rows, a.out_dir, a.tolerance)
    print("\nwrote %s and %s: %d rows" % (csv_path, tex_path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
