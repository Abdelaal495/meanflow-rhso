#!/usr/bin/env python
"""Freeze the RHSO anchor schedules on pMF-L/16 (rebuttal Task 8) from the tuning run.

Reads `results.csv` of the run(s) of configs/experiments_anchor_schedule_tuning.yaml and,
for every (task, schedule) in

    tasks      denoising, deblurring, super-resolution, random inpainting, box inpainting
    schedules  constant, increasing, decreasing   (src/rhso.py, mu_schedule)

selects the mu0 with the lowest LPIPS on the 8 tuning images.  The candidates of a schedule
are its 7 grid values {0.01, ..., 10} AND the task's single mu0 = 0 job, which every schedule
shares (m(t) multiplies mu0, so all three objectives coincide at 0).  An exact LPIPS tie
goes to the smaller mu0.  The grid must be complete, every row must carry the task's frozen
fields (N, M, lr, t0) and 8 images, or the script stops.

Outputs

    configs/experiments_anchor_schedule_frozen100.yaml   (and a 2-image _smoke variant)
        One entry per (task, schedule) -- 15 -- on the 100 frozen evaluation images with
        the benchmark pool and the pMF __rhso batch sizes.  When several schedules of a
        task select mu0 = 0, that configuration runs ONCE, in a block named for all of them
        (`<task>__anchor_constant_increasing`), so the table can label it for each.  A mu0 = 0
        entry runs the default (constant) code path.
    <out-dir>/anchor_schedule_tuning_curves.csv / .tex
        LPIPS and PSNR against mu0 per task and schedule (mu0 = 0 in every schedule's curve),
        the selected point marked.

    python scripts/freeze_anchor_schedules.py --from $MPCFLOW_OUTPUT_ROOT/run_<tuning id> \
        --out-dir $MPCFLOW_OUTPUT_ROOT/anchor_schedule_tuning_curves
"""
import argparse
import collections
import csv
import importlib.util
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tune = _load("make_anchor_schedule_tuning")
matched, hpo = tune.matched, tune.hpo

SCHEDULES = tune.SCHEDULES
MU_GRID = tune.MU_GRID
TASK_NAME = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "SR $2\\times$",
             "random_inpaint": "Random inpainting", "box_inpaint": "Box inpainting"}
CHECKED_FIELDS = ("t0", "num_rhso_steps", "num_opt_steps", "lr")


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def schedule_of(row):
    s = (row.get("mu_schedule") or "").strip()
    return "constant" if s in ("", "None", "null") else s


# =====================================================================================
# reading the tuning run
# =====================================================================================
def read_tuning(runs, benchmark):
    """task -> {"mu0": row or None, schedule: {mu0: row}}; refuses anything off-protocol."""
    expected = {p: dict(tune.benchmark_entry(benchmark, p)[0]) for p in hpo.PROBLEMS}
    exps = {}
    for p in hpo.PROBLEMS:
        exps[hpo.EXP_NAME[p] + tune.TUNE_SUFFIX] = (p, "grid")
        exps[hpo.EXP_NAME[p] + tune.MU0_SUFFIX] = (p, "mu0")
    cells = {p: {"mu0": None, **{s: {} for s in SCHEDULES}} for p in hpo.PROBLEMS}
    seen = 0
    for run in runs:
        path = Path(run)
        path = path / "results.csv" if path.is_dir() else path
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("experiment") not in exps or r.get("model") != tune.MODEL \
                        or r.get("method") != tune.METHOD:
                    continue
                problem, kind = exps[r["experiment"]]
                where = "%s job %s (%s)" % (r["experiment"], r.get("job_id"), path)
                if r.get("status", "ok") not in ("ok", "", None):
                    raise SystemExit("%s has status %s" % (where, r.get("status")))
                if int(round(fnum(r.get("num_images")) or 0)) != tune.TUNING_IMAGES:
                    raise SystemExit("%s ran on %s images, not %d"
                                     % (where, r.get("num_images"), tune.TUNING_IMAGES))
                for k, v in expected[problem].items():
                    if k in CHECKED_FIELDS and fnum(r.get(k)) != float(v):
                        raise SystemExit("%s has %s=%s; the task's frozen entry has %s"
                                         % (where, k, r.get(k), v))
                if fnum(r.get("lpips")) is None:
                    raise SystemExit("%s has no LPIPS" % where)
                mu, schedule = fnum(r.get("mu")), schedule_of(r)
                if mu is None or schedule not in SCHEDULES:
                    raise SystemExit("%s: mu=%s mu_schedule=%s" % (where, r.get("mu"), schedule))
                row = dict(r, _mu=mu, _schedule=schedule, _run=str(path))
                if kind == "mu0" or mu == 0.0:
                    if mu != 0.0:
                        raise SystemExit("%s: the mu0 = 0 block ran mu=%s" % (where, mu))
                    if cells[problem]["mu0"] is not None:
                        raise SystemExit("%s: a second mu0 = 0 job (first: %s)"
                                         % (where, cells[problem]["mu0"]["job_id"]))
                    cells[problem]["mu0"] = row
                else:
                    if mu in cells[problem][schedule]:
                        raise SystemExit("%s: a second %s job at mu0=%g (first: %s)"
                                         % (where, schedule, mu,
                                            cells[problem][schedule][mu]["job_id"]))
                    cells[problem][schedule][mu] = row
                seen += 1
    for problem in hpo.PROBLEMS:
        missing = [] if cells[problem]["mu0"] is not None else ["mu0=0"]
        for s in SCHEDULES:
            missing += ["%s mu0=%g" % (s, m) for m in MU_GRID if float(m) not in cells[problem][s]]
            extra = sorted(set(cells[problem][s]) - {float(m) for m in MU_GRID})
            if extra:
                raise SystemExit("%s/%s: mu0 values outside the grid: %s" % (problem, s, extra))
        if missing:
            raise SystemExit("%s: the tuning grid is incomplete; missing %s"
                             % (problem, ", ".join(missing)))
    return cells, seen


def candidates(cell, schedule):
    """[(mu0, row)] of one schedule, mu0 = 0 first."""
    return [(0.0, cell["mu0"])] + sorted(cell[schedule].items())


def select(cells):
    """(task, schedule) -> (mu0, row, number of candidates); lowest LPIPS, ties to smaller mu0."""
    out = {}
    for problem in hpo.PROBLEMS:
        for s in SCHEDULES:
            cands = candidates(cells[problem], s)
            mu, row = min(cands, key=lambda c: (fnum(c[1]["lpips"]), c[0]))
            out[(problem, s)] = (mu, row, len(cands))
    return out


# =====================================================================================
# the frozen configuration
# =====================================================================================
def groups(selected, problem):
    """[(schedules, mu0, row)] for one task: schedules that selected mu0 = 0 share ONE job."""
    zero = [s for s in SCHEDULES if selected[(problem, s)][0] == 0.0]
    out = []
    for s in SCHEDULES:
        mu, row, _ = selected[(problem, s)]
        if mu == 0.0:
            if s == zero[0]:
                out.append((zero, 0.0, row))
        else:
            out.append(([s], mu, row))
    return out


def frozen_block_name(problem, schedules):
    return hpo.EXP_NAME[problem] + tune.FROZEN_PREFIX + "_".join(schedules)


def schedules_of_block(experiment):
    """Inverse of frozen_block_name: '<task>__anchor_constant_increasing' -> [...]."""
    if tune.FROZEN_PREFIX not in experiment:
        return None
    tail = experiment.split(tune.FROZEN_PREFIX, 1)[1]
    names = tail.split("_")
    return names if names and all(n in SCHEDULES for n in names) else None


def mu_text(mu):
    return "0" if mu == 0.0 else ("%g" % mu)


def entry_comment(schedules, mu, selected, problem):
    parts = []
    for s in schedules:
        _, row, n = selected[(problem, s)]
        parts.append("schedule %s: selected mu0 = %s, tuning LPIPS %.4f (PSNR %.2f) at %d "
                     "tuning images, best of %d" % (s, mu_text(mu), fnum(row["lpips"]),
                                                    fnum(row.get("psnr")) or 0.0,
                                                    tune.TUNING_IMAGES, n))
    text = "; ".join(parts)
    if mu == 0.0:
        text += " (mu0 = 0: the same objective under every schedule; one run)"
    return text


def frozen_lines(benchmark, selected, smoke):
    lines, n_jobs, n_entries = [], 0, 0
    n_images = tune.SMOKE_IMAGES if smoke else tune.FROZEN_IMAGES
    for problem in hpo.PROBLEMS:
        fixed, _, batch = tune.benchmark_entry(benchmark, problem)
        for schedules, mu, _row in groups(selected, problem):
            schedule_field = "constant" if mu == 0.0 else schedules[0]
            fields = fixed + [("mu", 0 if mu == 0.0 else mu), ("mu_schedule", schedule_field)]
            lines += tune.block(frozen_block_name(problem, schedules), problem, n_images,
                                min(batch, n_images), fields,
                                entry_comment(schedules, mu, selected, problem))
            n_jobs += 1
            n_entries += len(schedules)
    return lines, n_jobs, n_entries


FROZEN_TEXT = """RHSO with a time-dependent state anchor on pMF-L/16 (rebuttal Task 8): mu_k = mu0 * m(t_k),
constant m = 1, increasing m = (t^2 + (1-t)^2) / (1-t)^2, decreasing = 1 / increasing.
Each entry is, for its (task, schedule), the lowest-LPIPS mu0 among the schedule's grid
%s and the task's mu0 = 0 run, on the 8 tuning images of
pMF-L/16 itself (trailing comment).  N, M, lr, t0 and the degradation are the task's pMF
__rhso entry of %s; the batch sizes are its pMF ones.
A block named for several schedules (<task>__anchor_constant_increasing) is one mu0 = 0
run that each of them selected.  %d (task, schedule) entries in %d jobs.
Generated by scripts/freeze_anchor_schedules.py from %s."""


def write_frozen(benchmark_path, selected, runs, out):
    benchmark = matched.load_yaml(benchmark_path)
    stems = ", ".join(sorted(Path(r).name for r in runs))
    paths = []
    for smoke in (False, True):
        target = matched.smoke_path(out) if smoke else Path(out).as_posix()
        lines, n_jobs, n_entries = frozen_lines(benchmark, selected, smoke)
        title = ("SMOKE: RHSO anchor schedules, frozen entries at 2 images" if smoke else
                 "RHSO ANCHOR SCHEDULES FINAL on pMF-L/16, frozen ImageNet-100 "
                 "(class seed 42 / image seed 43)")
        usage = ("#   python scripts/build_local_imagenet_pool.py --frozen-manifest "
                 "benchmarks/imagenet100_c42_i43/manifest.csv\n"
                 "#   python run.py --config %s --models pmf --run-id <id> --no-figures\n" % target)
        text = FROZEN_TEXT % ("{%s}" % ", ".join(tune.fmt(m) for m in MU_GRID), benchmark_path,
                              n_entries, n_jobs, stems)
        paths.append(tune.write(target, lines, n_jobs, title, usage, text,
                                matched.FROZEN_POOL, not smoke))
    return paths


# =====================================================================================
# tuning curves
# =====================================================================================
CURVE_COLUMNS = ["task", "schedule", "mu0", "lpips", "psnr", "ssim", "n_images", "selected",
                 "job_id", "experiment", "run"]


def curve_rows(cells, selected):
    rows = []
    for problem in hpo.PROBLEMS:
        for s in SCHEDULES:
            chosen = selected[(problem, s)][0]
            for mu, r in candidates(cells[problem], s):
                rows.append({"task": problem, "schedule": s, "mu0": mu,
                             "lpips": fnum(r["lpips"]), "psnr": fnum(r.get("psnr")),
                             "ssim": fnum(r.get("ssim")),
                             "n_images": int(round(fnum(r.get("num_images")) or 0)),
                             "selected": mu == chosen, "job_id": r["job_id"],
                             "experiment": r["experiment"], "run": r["_run"]})
    return rows


def curves_latex(rows):
    by = {(r["task"], r["schedule"], r["mu0"]): r for r in rows}
    mus = [0.0] + [float(m) for m in MU_GRID]
    lines = ["% Generated by scripts/freeze_anchor_schedules.py; needs \\usepackage{booktabs}.",
             "% LPIPS / PSNR on the 8 tuning images; bold = the schedule's selected mu0.",
             "\\begin{table}[t]", "\\centering",
             "\\caption{RHSO anchor schedules on pMF-L/16: tuning curves (8 tuning images). "
             "Each cell is LPIPS / PSNR at $\\mu_0$; $\\mu_0 = 0$ is one run shared by the "
             "three schedules. Bold: the selected $\\mu_0$.}",
             "\\label{tab:rebuttal-anchor-schedule-tuning}", "\\small",
             "\\begin{tabular}{ll" + "c" * len(mus) + "}", "\\toprule",
             "Task & Schedule & " + " & ".join("$%s$" % mu_text(m) for m in mus) + " \\\\",
             "\\midrule"]
    for i, problem in enumerate(hpo.PROBLEMS):
        if i:
            lines.append("\\midrule")
        for j, s in enumerate(SCHEDULES):
            cells = []
            for m in mus:
                r = by[(problem, s, m)]
                text = "%.4f / %.2f" % (r["lpips"], r["psnr"] or 0.0)
                cells.append("\\textbf{%s}" % text if r["selected"] else text)
            lines.append("%s & %s & %s \\\\" % (TASK_NAME[problem] if j == 0 else "", s,
                                                " & ".join(cells)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def write_curves(rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out / "anchor_schedule_tuning_curves.csv", out / "anchor_schedule_tuning_curves.tex"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CURVE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tex_path.write_text(curves_latex(rows))
    return csv_path, tex_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="runs", action="append", required=True, metavar="RUN",
                   help="tuning run directory or its results.csv (repeatable)")
    p.add_argument("--benchmark", default=tune.BENCHMARK,
                   help="frozen benchmark the fixed fields come from (default: %(default)s)")
    p.add_argument("--out", default=tune.FROZEN_OUT,
                   help="frozen config path; the smoke variant is <out>_smoke.yaml")
    p.add_argument("--out-dir", default="outputs/anchor_schedule_tuning_curves",
                   help="where the tuning-curve CSV and LaTeX go")
    a = p.parse_args(argv)
    benchmark = matched.load_yaml(a.benchmark)
    cells, seen = read_tuning(a.runs, benchmark)
    selected = select(cells)
    print("read %d tuning jobs" % seen)
    for problem in hpo.PROBLEMS:
        print("  %-17s %s" % (problem, "  ".join(
            "%s mu0=%s (LPIPS %.4f)" % (s, mu_text(selected[(problem, s)][0]),
                                        fnum(selected[(problem, s)][1]["lpips"]))
            for s in SCHEDULES)))
    write_frozen(a.benchmark, selected, a.runs, a.out)
    csv_path, tex_path = write_curves(curve_rows(cells, selected), a.out_dir)
    print("wrote %s and %s" % (csv_path, tex_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
