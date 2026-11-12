"""`scripts/paired_stats.py`: paired RHSO - MPC-delta_t differences with bootstrap intervals.

Executes the script on synthetic `results_per_image.csv` files with known per-image
differences and checks:

  1. images are paired by image_id, not by row position, and the sign is RHSO - MPC-delta_t;
  2. the interval is the documented percentile bootstrap (10,000 resamples, seed 20260903),
     reproduced here independently, contains the mean, has the expected width, and
     collapses to a point for a constant difference;
  3. a row does not depend on which other rows were computed, and reruns are identical;
  4. comparisons A, B and ref take each side from the right run, and BP per image is the
     job total divided by the number of batched passes;
  5. measurement RMSE is the job-level difference with NO interval, and both outputs say so;
  6. different image sets and ambiguous runs are refused.

Runs on CPU in seconds:

    python tests/test_paired_stats.py
"""

import csv
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PASSED = [0]
FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


spec = importlib.util.spec_from_file_location("paired_stats", REPO_ROOT / "scripts" / "paired_stats.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

COLUMNS = ["job_id", "model", "problem", "method", "status", "image_id", "psnr", "ssim", "lpips",
           "measurement_rmse", "expected_backprops", "num_images", "batch_size"]
N = 100


def job_rows(job_id, model, problem, method, values, rmse, bp_per_image, batch):
    """values: {metric: array over images img000..img099}."""
    chunks = math.ceil(N / batch)
    return [dict(job_id=job_id, model=model, problem=problem, method=method, status="ok",
                 image_id="img%03d" % i, psnr=values["psnr"][i], ssim=values["ssim"][i],
                 lpips=values["lpips"][i], measurement_rmse=rmse,
                 expected_backprops=bp_per_image * chunks, num_images=N, batch_size=batch)
            for i in range(N)]


def write(path, rows, shuffle_seed=None):
    rows = list(rows)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def base(rng):
    return {"psnr": rng.normal(26, 2, N), "ssim": rng.uniform(0.6, 0.9, N),
            "lpips": rng.uniform(0.05, 0.3, N)}


def build(tmp):
    rng = np.random.default_rng(0)
    mpc_bench = base(rng)
    d = {"psnr": rng.normal(0.5, 1.0, N), "ssim": rng.normal(0.01, 0.02, N),
         "lpips": rng.normal(-0.02, 0.01, N)}
    rhso_matched = {k: mpc_bench[k] + d[k] for k in d}
    rhso_bench = base(rng)
    mpc_matched = {k: rhso_bench[k] - 0.3 for k in rhso_bench}           # constant difference
    matched = (job_rows("m_rhso", "pmf", "deblur", "rhso", rhso_matched, 0.050, 280, 10)
               + job_rows("m_mpc", "pmf", "deblur", "mpc_delta_t", mpc_matched, 0.061, 160, 10))
    bench = (job_rows("b_rhso", "pmf", "deblur", "rhso", rhso_bench, 0.055, 160, 10)
             + job_rows("b_mpc", "pmf", "deblur", "mpc_delta_t", mpc_bench, 0.070, 280, 10)
             + job_rows("b_rhso_den", "pmf", "denoising", "rhso", base(rng), 0.1, 160, 10)
             + job_rows("b_mpc_den", "pmf", "denoising", "mpc_delta_t", base(rng), 0.12, 140, 10))
    return (write(tmp / "matched" / "results_per_image.csv", matched, shuffle_seed=1),
            write(tmp / "bench" / "results_per_image.csv", bench), d)


def reference_ci(diff):
    """An independent transcription of the documented protocol."""
    idx = np.random.default_rng(20260903).integers(0, N, size=(10000, N))
    lo, hi = np.percentile(np.asarray(diff)[idx].mean(axis=1), [2.5, 97.5])
    return lo, hi


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        matched_csv, bench_csv, d = build(tmp)
        rows = ps.paired_rows(ps.read_side([matched_csv.parent], "matched"),
                              ps.read_side([bench_csv], "benchmark"))
        by = {(r["comparison"], r["problem"]): r for r in rows}

        print("\n1. pairing and sign")
        a = by[("A", "deblur")]
        check("diff = mean of per-image RHSO - MPC", all(
            abs(a["%s_diff" % m] - float(np.mean(d[m]))) < 1e-9 for m in ps.METRICS),
            "PSNR %+.4f, SSIM %+.4f, LPIPS %+.4f" % (a["psnr_diff"], a["ssim_diff"], a["lpips_diff"]))
        check("paired by image_id (matched file shuffled)", a["n_images"] == N,
              "%d pairs from a row-shuffled file" % a["n_images"])

        print("\n2. bootstrap interval")
        lo, hi = reference_ci(d["psnr"])
        check("== independent transcription (seed 20260903)",
              abs(a["psnr_ci_lo"] - lo) < 1e-12 and abs(a["psnr_ci_hi"] - hi) < 1e-12,
              "[%.4f, %.4f]" % (lo, hi))
        width = a["psnr_ci_hi"] - a["psnr_ci_lo"]
        normal = 2 * 1.96 * np.std(d["psnr"], ddof=1) / math.sqrt(N)
        check("contains the mean, normal-theory width",
              a["psnr_ci_lo"] < a["psnr_diff"] < a["psnr_ci_hi"] and abs(width / normal - 1) < 0.15,
              "width %.4f vs 1.96-sigma %.4f" % (width, normal))
        b = by[("B", "deblur")]
        check("constant difference -> zero-width interval",
              all(abs(b["%s_ci_hi" % m] - b["%s_ci_lo" % m]) < 1e-9
                  and abs(b["%s_diff" % m] - 0.3) < 1e-9 for m in ps.METRICS),
              "RHSO - MPC = +0.3 on every image and metric")

        print("\n3. independence and determinism")
        alone = ps.paired_rows(ps.read_side([matched_csv], "matched"),
                               {k: v for k, v in ps.read_side([bench_csv], "benchmark").items()
                                if k[1] == "deblur" and k[2] == "mpc_delta_t"})
        again = ps.paired_rows(ps.read_side([matched_csv], "matched"),
                               ps.read_side([bench_csv], "benchmark"))
        check("a row does not depend on the other rows",
              len(alone) == 1 and alone[0] == a, "A computed alone == A computed with 4 others")
        check("reruns are identical", again == rows)

        print("\n4. comparisons and budgets")
        got = sorted((r["comparison"], r["problem"], r["rhso_job_id"], r["mpc_job_id"]) for r in rows)
        want = sorted([("A", "deblur", "m_rhso", "b_mpc"), ("B", "deblur", "b_rhso", "m_mpc"),
                       ("ref", "deblur", "b_rhso", "b_mpc"), ("ref", "denoising", "b_rhso_den", "b_mpc_den")])
        check("A / B / ref take the right sides", got == want, str(got))
        check("BP per image = job total / passes",
              (a["rhso_bp"], a["mpc_bp"], a["bp_matched"]) == (280, 280, True)
              and (b["rhso_bp"], b["mpc_bp"], b["bp_matched"]) == (160, 160, True)
              and not by[("ref", "deblur")]["bp_matched"],
              "A 280/280, B 160/160, ref 160/280 (unmatched, as expected)")

        print("\n5. measurement RMSE")
        check("job-level difference, no interval",
              abs(a["measurement_rmse_diff"] - (0.050 - 0.070)) < 1e-12
              and not any("measurement_rmse_ci" in k for k in ps.COLUMNS)
              and a["measurement_rmse_note"] == ps.RMSE_NOTE,
              "%+.3f; note: %s" % (a["measurement_rmse_diff"], ps.RMSE_NOTE))
        out = tmp / "out"
        ps.main(["--matched", str(matched_csv.parent), "--benchmark", str(bench_csv.parent),
                 "--out-dir", str(out)])
        written = list(csv.DictReader(open(out / "paired_stats.csv")))
        tex = (out / "paired_stats.tex").read_text()
        check("CSV and LaTeX both say no CI for RMSE",
              len(written) == 4 and all(r["measurement_rmse_note"] == ps.RMSE_NOTE for r in written)
              and ps.RMSE_NOTE in tex and tex.count("\\\\\n") == 4 + 1
              and "10000 resamples, seed 20260903" in tex,
              "%d CSV rows, %d LaTeX rows" % (len(written), tex.count("\\\\\n") - 1))

        print("\n6. refusals")
        rows_m = list(csv.DictReader(open(matched_csv)))
        short = write(tmp / "short.csv", [r for r in rows_m if r["image_id"] != "img007"])
        try:
            ps.paired_rows(ps.read_side([short], "matched"), ps.read_side([bench_csv], "benchmark"))
            msg = ""
        except SystemExit as e:
            msg = str(e)
        check("different image sets are refused", "different images" in msg, msg[:80])
        dup = write(tmp / "dup.csv", rows_m + [dict(r, job_id="m_rhso_2") for r in rows_m
                                               if r["method"] == "rhso"])
        try:
            ps.read_side([dup], "matched")
            msg = ""
        except SystemExit as e:
            msg = str(e)
        check("two jobs for one cell are refused", "exactly one" in msg, msg[:80])

    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
