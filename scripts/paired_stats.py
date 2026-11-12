#!/usr/bin/env python
"""Paired RHSO vs MPC-delta_t differences, per prior and task, with bootstrap intervals.

Reads `results_per_image.csv` of the matched-budget frozen run(s) and of the original
benchmark run(s) and pairs images by (prior, task, image_id).  Every difference is

    RHSO - MPC-delta_t        (PSNR, SSIM: positive favours RHSO; LPIPS: negative does)

for three comparisons, each emitted only when both of its sides are present:

    A    RHSO from the matched run     vs MPC-delta_t from the benchmark run  (equal BP)
    B    MPC-delta_t from the matched run vs RHSO from the benchmark run      (equal BP)
    ref  both from the benchmark run (the original, unmatched comparison)

The 95% intervals are nonparametric percentile bootstraps over the paired images: 10,000
resamples, seed 20260903 as in Appendix C.5.  Each comparison draws its OWN generator from
that seed, so a row does not depend on which other rows were computed, and all metrics of a
row share one set of resampled image indices.

MEASUREMENT RMSE HAS NO INTERVAL.  `run.py` records it once per job, over all of the job's
images pooled (metrics.measurement_consistency), and copies that one value onto every
per-image row, so it cannot be resampled over images.  It is reported as the job-level
difference, and the CSV and LaTeX outputs both say so.

    python scripts/paired_stats.py --matched outputs/matched_final_pmf --benchmark outputs/final_pmf \
        --out-dir outputs/paired_stats
"""
import argparse
import collections
import csv
import math
import sys
from pathlib import Path

import numpy as np

SEED = 20260903
RESAMPLES = 10000
METRICS = ["psnr", "ssim", "lpips"]
RMSE_NOTE = "job-level difference; no CI (measurement RMSE is not recorded per image)"
METHODS = ("rhso", "mpc_delta_t")
COMPARISONS = [("A", "matched", "benchmark"), ("B", "benchmark", "matched"),
               ("ref", "benchmark", "benchmark")]          # (name, RHSO source, MPC source)
PRIOR_NAME = {"pmf": "pMF-L/16", "jit": "JiT-B/16", "imf": "iMF-B/2", "sit": "SiT-XL/2"}
PRIOR_ORDER = ["pmf", "jit", "imf", "sit"]
TASK_NAME = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "SR $2\\times$",
             "random_inpaint": "Random inpainting", "box_inpaint": "Box inpainting"}
TASK_ORDER = list(TASK_NAME)


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def per_image_csv(path):
    p = Path(path)
    return p / "results_per_image.csv" if p.is_dir() else p


def read_side(paths, source):
    """(model, problem, method) -> one job: {job_id, bp, rmse, images: {image_id: row}}."""
    jobs = collections.defaultdict(dict)                 # cell -> job_id -> rows
    for path in paths:
        for r in csv.DictReader(open(per_image_csv(path), newline="")):
            if r.get("method") not in METHODS or r.get("status", "ok") not in ("ok", "", None):
                continue
            cell = (r["model"], r["problem"], r["method"])
            jobs[cell].setdefault(r["job_id"], []).append(r)
    out = {}
    for cell, by_job in jobs.items():
        if len(by_job) > 1:
            raise SystemExit("%s run has %d jobs for %s (%s); pass a run with exactly one"
                             % (source, len(by_job), "/".join(cell), ", ".join(sorted(by_job))))
        job_id, rows = next(iter(by_job.items()))
        images = {}
        for r in rows:
            if r["image_id"] in images:
                raise SystemExit("%s run: image %s appears twice in job %s"
                                 % (source, r["image_id"], job_id))
            images[r["image_id"]] = r
        first = rows[0]
        out[cell] = dict(job_id=job_id, bp=bp_per_image(first),
                         rmse=fnum(first.get("measurement_rmse")), images=images)
    return out


def bp_per_image(row):
    """`expected_backprops` is the JOB total (per batched pass x passes); one pass serves a
    whole batch, so a pass's count is the per-image count."""
    total = fnum(row.get("expected_backprops"))
    n, b = fnum(row.get("num_images")), fnum(row.get("batch_size"))
    if total is None or not n or not b:
        return None
    return total / math.ceil(n / b)


def bootstrap(diffs, seed=SEED, resamples=RESAMPLES):
    """{metric: (mean, lo, hi)} -- one index matrix shared by every metric of the row."""
    n = len(next(iter(diffs.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    out = {}
    for metric, d in diffs.items():
        d = np.asarray(d, np.float64)
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[metric] = (float(d.mean()), float(lo), float(hi))
    return out


def compare(name, rhso, mpc, cell_key, seed=SEED, resamples=RESAMPLES):
    model, problem = cell_key
    ids_r, ids_m = set(rhso["images"]), set(mpc["images"])
    if ids_r != ids_m:
        raise SystemExit("%s %s/%s: the two sides cover different images (%d only in RHSO, "
                         "%d only in MPC-delta_t); a paired comparison needs the same set"
                         % (name, model, problem, len(ids_r - ids_m), len(ids_m - ids_r)))
    ids = sorted(ids_r)
    diffs, means = {}, {}
    for metric in METRICS:
        a = [fnum(rhso["images"][i].get(metric)) for i in ids]
        b = [fnum(mpc["images"][i].get(metric)) for i in ids]
        if any(v is None for v in a + b):
            raise SystemExit("%s %s/%s: missing %s values" % (name, model, problem, metric))
        diffs[metric] = np.subtract(a, b)
        means[metric] = (float(np.mean(a)), float(np.mean(b)))
    stats = bootstrap(diffs, seed, resamples)
    row = dict(comparison=name, model=model, prior=PRIOR_NAME.get(model, model), problem=problem,
               rhso_job_id=rhso["job_id"], mpc_job_id=mpc["job_id"],
               rhso_bp=rhso["bp"], mpc_bp=mpc["bp"],
               bp_matched=(rhso["bp"] is not None and rhso["bp"] == mpc["bp"]),
               n_images=len(ids), resamples=resamples, seed=seed)
    for metric in METRICS:
        mean, lo, hi = stats[metric]
        row.update({"%s_rhso" % metric: means[metric][0], "%s_mpc" % metric: means[metric][1],
                    "%s_diff" % metric: mean, "%s_ci_lo" % metric: lo, "%s_ci_hi" % metric: hi})
    rr, rm = rhso["rmse"], mpc["rmse"]
    row.update(measurement_rmse_rhso=rr, measurement_rmse_mpc=rm,
               measurement_rmse_diff=(rr - rm) if rr is not None and rm is not None else None,
               measurement_rmse_note=RMSE_NOTE)
    return row


def paired_rows(matched, benchmark, seed=SEED, resamples=RESAMPLES):
    sides = {"matched": matched, "benchmark": benchmark}
    cells = sorted({(m, p) for side in sides.values() for (m, p, _) in side},
                   key=lambda c: (PRIOR_ORDER.index(c[0]) if c[0] in PRIOR_ORDER else 99,
                                  TASK_ORDER.index(c[1]) if c[1] in TASK_ORDER else 99))
    rows = []
    for model, problem in cells:
        for name, rhso_src, mpc_src in COMPARISONS:
            rhso = sides[rhso_src].get((model, problem, "rhso"))
            mpc = sides[mpc_src].get((model, problem, "mpc_delta_t"))
            if rhso is None or mpc is None:
                continue
            rows.append(compare(name, rhso, mpc, (model, problem), seed, resamples))
    return rows


COLUMNS = (["comparison", "model", "prior", "problem", "rhso_job_id", "mpc_job_id", "rhso_bp",
            "mpc_bp", "bp_matched", "n_images", "resamples", "seed"]
           + ["%s_%s" % (m, s) for m in METRICS for s in ("rhso", "mpc", "diff", "ci_lo", "ci_hi")]
           + ["measurement_rmse_rhso", "measurement_rmse_mpc", "measurement_rmse_diff",
              "measurement_rmse_note"])


def latex_rows(rows):
    def cell(r, metric, digits):
        f = "%+." + str(digits) + "f"
        return (f + " [" + f + ", " + f + "]") % (
            r["%s_diff" % metric], r["%s_ci_lo" % metric], r["%s_ci_hi" % metric])

    def bp(v):
        return "--" if v is None else "%d" % round(v)

    lines = ["%% RHSO - MPC-delta_t, mean over paired images [95%% bootstrap CI, "
             "%d resamples, seed %d]." % (rows[0]["resamples"] if rows else RESAMPLES,
                                         rows[0]["seed"] if rows else SEED),
             "% Measurement RMSE: " + RMSE_NOTE + ".",
             "% Prior & Task & Comparison & BP (RHSO / MPC) & dPSNR & dSSIM & dLPIPS & dRMSE (job-level) \\\\"]
    for r in rows:
        rmse = "--" if r["measurement_rmse_diff"] is None else "%+.4f" % r["measurement_rmse_diff"]
        lines.append("%s & %s & %s & %s / %s & %s & %s & %s & %s \\\\" % (
            r["prior"], TASK_NAME.get(r["problem"], r["problem"]), r["comparison"],
            bp(r["rhso_bp"]), bp(r["mpc_bp"]), cell(r, "psnr", 2), cell(r, "ssim", 4),
            cell(r, "lpips", 4), rmse))
    return "\n".join(lines) + "\n"


def write(rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "paired_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})
    (out / "paired_stats.tex").write_text(latex_rows(rows))
    return out / "paired_stats.csv", out / "paired_stats.tex"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matched", action="append", required=True, metavar="RUN",
                   help="matched-budget frozen run directory or results_per_image.csv (repeatable)")
    p.add_argument("--benchmark", action="append", required=True, metavar="RUN",
                   help="original benchmark run directory or results_per_image.csv (repeatable)")
    p.add_argument("--out-dir", default="outputs/paired_stats")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    rows = paired_rows(read_side(a.matched, "matched"), read_side(a.benchmark, "benchmark"),
                       a.seed, a.resamples)
    if not rows:
        raise SystemExit("no (prior, task) has both an RHSO and an MPC-delta_t side")
    csv_path, tex_path = write(rows, a.out_dir)
    for r in rows:
        if r["comparison"] in ("A", "B") and not r["bp_matched"]:
            print("! %s %s/%s: BP per image differs (RHSO %s, MPC-delta_t %s)"
                  % (r["comparison"], r["model"], r["problem"], r["rhso_bp"], r["mpc_bp"]),
                  file=sys.stderr)
    print("wrote %s and %s: %d comparisons" % (csv_path, tex_path, len(rows)))
    print("note: measurement RMSE is a %s" % RMSE_NOTE)


if __name__ == "__main__":
    main()
