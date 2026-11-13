#!/usr/bin/env python
"""Budget-level comparison on the pixel priors (rebuttal Task 6): cheap, mid and RHSO time.

    python scripts/budget_level_table.py \
        --quality outputs/<budget-level run> --quality outputs/run_26548623 \
        --quality outputs/run_26548625 --quality outputs/run_9187465 --quality outputs/run_9191132 \
        --quality outputs/run_9187499 --quality outputs/run_9187476 \
        --timing outputs/timing_pmf --timing outputs/timing_jit \
        --timing outputs/timing_matched_time --timing outputs/timing_budget_levels \
        --out-dir outputs/budget_level_table

    (Task 7: pass the lr-scaled run as a --quality run and add --timing-ignore lr.)

Quality and timing runs are read with scripts/matched_time_table.py's readers (per-image CSV;
per-job metadata.json from scripts/run_timing.py, device-synchronised seconds per image at
batch size one).  Any number of each may be given.

THE BUDGETS, seconds per image (cheap = the cheap methods' measured time, mid = 4 x cheap):

    Prior  Level  Denoising  Deblurring  SR     Random inp.  Box inp.
    pMF    cheap  0.522      6.447       0.574  0.530        0.624
    pMF    mid    2.088      --          2.296  2.120        2.496
    JiT    cheap  1.417      5.855       1.372  1.430        1.410
    JiT    mid    5.668      --          5.488  5.720        5.640

and a third level, "rhso": RHSO's own measured time from Task 5 (pMF: the frozen RHSO; JiT:
the matched-BP direct-planner RHSO), taken from the timing runs.

SELECTION.  For every (prior, task, level), each method (RHSO, MPC-delta_t, D-Flow, MPC-RHC,
PnP-Flow, Flower) gets ONE entry: among ALL its quality jobs for that prior and task (tuned,
matched-BP, Task 5 matched-time, cheap and mid), the lowest mean LPIPS whose measured time is
at most 1.1 x the budget (--tolerance).  If none qualifies, its cheapest job is shown and
marked "over budget (xRATIO)".  On JiT only direct-planner RHSO jobs are candidates.  A
quality job is paired with the timing job of the same (prior, task, method) whose
computation-defining fields agree with it (budget knobs required on both sides; the same
variant breaks a tie); a quality job with no such timing job cannot be placed and is listed
on stderr.

LR-SCALED RUNS (Task 7, scripts/make_budget_level_lrscaled_config.py).  Blocks named
<task>__budget_<level>[_rhc|_rhso]_lr are the variants "cheap level, lr scaled" and "mid
level, lr scaled".  They differ from Task 6's entries in lr only, which does not change the
time per image, so they have no timing run of their own: with --timing-ignore lr the pairing
skips lr (and, if several timing jobs then agree, prefers the one of the same variant, then
the Task 6 variant without "_lr").  Without the option nothing changes.

DIFFERENCES.  The level's best method is the lowest-LPIPS entry WITHIN budget (an
over-budget entry is never the best).  Every other entry gets

    method - best        (LPIPS: positive = worse than the best; PSNR/SSIM: negative is)

with the paired 95% percentile bootstrap of scripts/paired_stats.py (10,000 resamples, seed
20260903, a fresh generator per comparison, one index matrix for the three metrics).  An
in-budget entry whose LPIPS interval lies above 0 is significantly worse; otherwise it TIES
with the best.  The best WINS (significantly) when no in-budget entry ties with it.
Over-budget entries are shown but neither win nor tie.

Outputs in --out-dir: budget_levels.csv (one row per entry), budget_levels_summary.csv (per
prior, level and task: the best method and who ties), budget_levels_wins.csv (per prior,
level and method: tasks won and tied) and budget_levels.tex.
"""
import argparse
import collections
import csv
import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mtt = _load("matched_time_table", REPO_ROOT / "scripts" / "matched_time_table.py")
ps, summ = mtt.ps, mtt.summ

SEED, RESAMPLES, TOLERANCE = ps.SEED, ps.RESAMPLES, 0.10
METRICS = ps.METRICS
PRIORS = ["pmf", "jit"]
PRIOR_NAME = ps.PRIOR_NAME
TASK_ORDER = ps.TASK_ORDER
TASK_NAME = ps.TASK_NAME
LEVELS = ["cheap", "mid", "rhso"]
LEVEL_NAME = {"cheap": "cheap", "mid": "mid", "rhso": "RHSO time"}
METHOD_ORDER = ["rhso", "mpc_delta_t", "dflow", "mpc_rhc", "pnp", "flower"]
METHOD_NAME = mtt.METHOD_NAME
METHOD_PLAIN = {"rhso": "RHSO", "mpc_delta_t": "MPC-dt", "dflow": "D-Flow",
                "mpc_rhc": "MPC-RHC", "pnp": "PnP-Flow", "flower": "Flower"}
VARIANT_NAME = {"frozen": "tuned", "matched": "matched BP", "matched_time": "matched time",
                "budget_cheap": "cheap level", "budget_mid": "mid level",
                "budget_cheap_lr": "cheap level, lr scaled", "budget_mid_lr": "mid level, lr scaled"}
# Fields --timing-ignore may drop from the quality/timing pairing: ones that do not change the
# computation, so not the time per image.  Task 7's runs differ from Task 6's in lr only.
TIMING_IGNORABLE = ("lr",)

# Seconds per image (the user's table above).  None = no budget at that level.
BUDGETS = {
    ("pmf", "cheap"): {"denoising": 0.522, "deblur": 6.447, "super_resolution": 0.574,
                       "random_inpaint": 0.530, "box_inpaint": 0.624},
    ("pmf", "mid"):   {"denoising": 2.088, "deblur": None, "super_resolution": 2.296,
                       "random_inpaint": 2.120, "box_inpaint": 2.496},
    ("jit", "cheap"): {"denoising": 1.417, "deblur": 5.855, "super_resolution": 1.372,
                       "random_inpaint": 1.430, "box_inpaint": 1.410},
    ("jit", "mid"):   {"denoising": 5.668, "deblur": None, "super_resolution": 5.488,
                       "random_inpaint": 5.720, "box_inpaint": 5.640},
}
# The job that defines the "rhso" level: Task 5's RHSO reference.
REFERENCE_VARIANT = {p: mtt.ROLES["rhso"][p][0] for p in PRIORS}     # pmf frozen, jit matched


_LR_SCALED = re.compile(r"__budget_(cheap|mid)(?:_rhc|_rhso)?_lr(?:__|$)")


def variant_of(experiment):
    name = str(experiment or "")
    lr_scaled = _LR_SCALED.search(name)          # Task 7: <task>__budget_<level>[_rhc|_rhso]_lr
    if lr_scaled:
        return "budget_%s_lr" % lr_scaled.group(1)
    for level in ("cheap", "mid"):
        if "__budget_%s" % level in name:
            return "budget_%s" % level
    return mtt.variant_of(name)


def label(method, variant):
    return "%s (%s)" % (METHOD_NAME[method], VARIANT_NAME.get(variant, variant))


def plain(text):
    return text.replace("MPC-$\\Delta t$", "MPC-dt")


def is_candidate(prior, method, job):
    """On JiT only direct-planner RHSO counts."""
    if prior == "jit" and method == "rhso":
        return mtt.norm(job["fields"].get("rhso_terminal_mode")) == "direct"
    return True


# ------------------------------------------------------------------ pairing quality and timing
def quality_jobs(quality_cells):
    """{(prior, task, method): [job, ...]}; each job gets its variant.  One job per variant."""
    seen, out = {}, collections.defaultdict(list)
    for jobs in quality_cells.values():
        for job in jobs:
            prior, task, method, _ = job["cell"]
            if prior not in PRIORS:
                continue
            job = dict(job, variant=variant_of(job["experiment"]))
            key = (prior, task, method, job["variant"])
            if key in seen:
                other = seen[key]
                raise SystemExit("quality %s has two jobs (%s in %s, %s in %s); pass runs with "
                                 "exactly one" % ("/".join(key), other["job_id"], other["run"],
                                                  job["job_id"], job["run"]))
            seen[key] = job
            out[(prior, task, method)].append(job)
    return out


def timing_rows(timing_cells):
    return [dict(r, variant=variant_of(r.get("experiment")))
            for rows in timing_cells.values() for r in rows]


def agrees(job, row, method, ignore=()):
    for k in mtt.BUDGET[method]:
        if mtt.norm(job["fields"].get(k)) is None or mtt.norm(row["fields"].get(k)) is None:
            return False
    for k in mtt.PARAM_FIELDS:
        if k in ignore:
            continue
        a, b = mtt.norm(job["fields"].get(k)), mtt.norm(row["fields"].get(k))
        if a is not None and b is not None and a != b:
            return False
    return True


def _base_variant(variant):
    """budget_cheap_lr -> budget_cheap: the Task 6 variant whose timing a Task 7 job reuses."""
    return variant[:-3] if variant.endswith("_lr") else variant


def match_timing(job, rows, prior, task, method, ignore=()):
    found = [r for r in rows if r["prior"] == prior and r["task"] == task
             and r["method"] == method and agrees(job, r, method, ignore)]
    if len(found) > 1:
        same = [r for r in found if r["variant"] == job["variant"]]
        found = same if same else found
    if len(found) > 1:
        same = [r for r in found if r["variant"] == _base_variant(job["variant"])]
        found = same if same else found
    if len(found) > 1:
        raise SystemExit("quality job %s (%s) matches %d timing jobs: %s"
                         % (job["job_id"], job["run"], len(found),
                            ", ".join("%s in %s" % (r["job_id"], r["run"]) for r in found)))
    return found[0] if found else None


# ------------------------------------------------------------------ the comparison
def paired(job, best, seed, resamples):
    """method - best, per metric, with paired_stats' bootstrap (via matched_time_table)."""
    if set(job["images"]) != set(best["images"]):
        raise SystemExit("%s (%s) vs the best %s (%s): different image sets (%d vs %d images); "
                         "a paired comparison needs the same set"
                         % (job["job_id"], job["run"], best["job_id"], best["run"],
                            len(job["images"]), len(best["images"])))
    return mtt.paired(job, best, seed, resamples)


def candidates(jobs, timing, prior, task, notes, ignore=()):
    """{method: [(job, timing row, seconds, mean metrics)]} for the timed candidate jobs."""
    out = {}
    for method in METHOD_ORDER:
        mine = []
        for job in jobs.get((prior, task, method), []):
            if not is_candidate(prior, method, job):
                continue
            t = match_timing(job, timing, prior, task, method, ignore)
            if t is None or t.get("seconds_per_image_mean") is None:
                notes.append("%s/%s %s: quality job %s (%s) has no timing job; not a candidate"
                             % (prior, task, plain(label(method, job["variant"])), job["job_id"],
                                job["run"]))
                continue
            mine.append((job, t, t["seconds_per_image_mean"], mtt.means(job)))
        if mine:
            out[method] = mine
    return out


def budget_for(prior, task, level, cands, notes):
    if level != "rhso":
        return BUDGETS[(prior, level)].get(task)
    for job, t, seconds, _ in cands.get("rhso", []):
        if job["variant"] == REFERENCE_VARIANT[prior]:
            return seconds
    notes.append("%s/%s: no timed %s RHSO; the RHSO-time level is skipped"
                 % (prior, task, VARIANT_NAME[REFERENCE_VARIANT[prior]]))
    return None


def build(quality, timing, tolerance=TOLERANCE, seed=SEED, resamples=RESAMPLES,
          timing_ignore=()):
    """(rows, notes).  One row per (prior, task, level, method) that has a timed candidate.

    timing_ignore: fields (from TIMING_IGNORABLE) left out of the quality/timing pairing."""
    bad = [k for k in timing_ignore if k not in TIMING_IGNORABLE]
    if bad:
        raise SystemExit("--timing-ignore accepts only %s, not %s"
                         % (", ".join(TIMING_IGNORABLE), ", ".join(bad)))
    ignore = tuple(timing_ignore)
    jobs, trows = quality_jobs(quality), timing_rows(timing)
    rows, notes = [], []
    for prior in PRIORS:
        per_task = {}
        for task in TASK_ORDER:
            cands = candidates(jobs, trows, prior, task, notes, ignore)
            if cands:
                per_task[task] = cands
        if ignore:
            reused = sum(1 for cands in per_task.values() for pool in cands.values()
                         for job, t, _, _ in pool
                         if any(mtt.norm(job["fields"].get(k)) != mtt.norm(t["fields"].get(k))
                                for k in ignore))
            if reused:
                notes.append("%s: %d quality job(s) paired with a timing job that differs in %s "
                             "(--timing-ignore)" % (prior, reused, ", ".join(ignore)))
        for level in LEVELS:
            for task, cands in per_task.items():
                budget = budget_for(prior, task, level, cands, notes)
                if budget is None:
                    continue
                limit = budget * (1.0 + tolerance) + 1e-12
                level_rows = []
                for method in METHOD_ORDER:
                    if method not in cands:
                        notes.append("%s/%s %s: %s has no timed candidate"
                                     % (prior, task, level, METHOD_PLAIN[method]))
                        continue
                    pool = cands[method]
                    within = [c for c in pool if c[2] <= limit and c[3]["lpips"] is not None]
                    if within:
                        pick, over = min(within, key=lambda c: c[3]["lpips"]), False
                    else:
                        pick, over = min(pool, key=lambda c: c[2]), True
                    level_rows.append(_row(prior, task, level, budget, method, pick, over,
                                           len(pool), len(within)))
                _compare(level_rows, notes, seed, resamples)
                for r in level_rows:
                    r.update(tolerance=tolerance, seed=seed, resamples=resamples)
                rows += level_rows
    return rows, notes


def _row(prior, task, level, budget, method, pick, over, n_candidates, n_within):
    job, t, seconds, means = pick
    ratio = seconds / budget
    row = {"prior": prior, "prior_name": PRIOR_NAME.get(prior, prior), "task": task,
           "level": level, "budget_seconds": budget, "method": method,
           "variant": job["variant"], "label": label(method, job["variant"]),
           "knobs": mtt.budget_text(method, job["fields"]), "bp_per_image": job["bp"],
           "n_images": len(job["images"]), "seconds_per_image": seconds,
           "seconds_per_image_sd": t.get("seconds_per_image_sd"),
           "seconds_basis": t.get("seconds_basis"), "budget_ratio": ratio,
           "within_budget": not over,
           "budget_note": "over budget (x%.2f)" % ratio if over else "",
           "n_candidates": n_candidates, "n_within_budget": n_within,
           "is_best": False, "best_method": None, "outcome": None,
           "quality_job_id": job["job_id"], "quality_run": job["run"],
           "timing_job_id": t.get("job_id"), "timing_run": t.get("run"),
           "gpu_name": t.get("gpu_name"), "_job": job}
    row.update(means)
    for metric in METRICS:
        row.update({"%s_diff" % metric: None, "%s_ci_lo" % metric: None,
                    "%s_ci_hi" % metric: None})
    return row


def _compare(level_rows, notes, seed, resamples):
    inside = [r for r in level_rows if r["within_budget"]]
    if not inside:
        if level_rows:
            r0 = level_rows[0]
            notes.append("%s/%s %s: no method within budget; no best, no differences"
                         % (r0["prior"], r0["task"], r0["level"]))
        for r in level_rows:
            r["outcome"] = "over budget"
        return
    best = min(inside, key=lambda r: r["lpips"])
    best["is_best"] = True
    for r in level_rows:
        r["best_method"] = best["method"]
        if r is best:
            continue
        stats = paired(r["_job"], best["_job"], seed, resamples)
        for metric in METRICS:
            mean, lo, hi = stats[metric]
            r.update({"%s_diff" % metric: mean, "%s_ci_lo" % metric: lo,
                      "%s_ci_hi" % metric: hi})
        if not r["within_budget"]:
            r["outcome"] = "over budget"
        else:
            r["outcome"] = "worse" if r["lpips_ci_lo"] > 0 else "tie"
    tied = any(r["outcome"] == "tie" for r in level_rows)
    best["outcome"] = "best (tie)" if tied else "best (significant)"


# ------------------------------------------------------------------ summaries
def summary(rows):
    """(per-task rows, per-method rows) for each (prior, level)."""
    cells = collections.OrderedDict()
    for r in rows:
        cells.setdefault((r["prior"], r["level"], r["task"]), []).append(r)
    per_task = []
    counts = collections.OrderedDict()
    for (prior, level, task), mine in cells.items():
        best = next((r for r in mine if r["is_best"]), None)
        ties = [r["method"] for r in mine if r["outcome"] == "tie"]
        per_task.append({"prior": prior, "prior_name": PRIOR_NAME.get(prior, prior),
                         "level": level, "task": task,
                         "budget_seconds": mine[0]["budget_seconds"],
                         "best_method": best["method"] if best else None,
                         "best_label": best["label"] if best else None,
                         "best_lpips": best["lpips"] if best else None,
                         "significant": bool(best) and not ties,
                         "tied_with_best": ";".join(ties),
                         "over_budget": ";".join(r["method"] for r in mine
                                                 if not r["within_budget"])})
        for r in mine:
            c = counts.setdefault((prior, level, r["method"]),
                                  {"prior": prior, "prior_name": PRIOR_NAME.get(prior, prior),
                                   "level": level, "method": r["method"], "tasks": 0,
                                   "wins": 0, "ties": 0, "worse": 0, "over_budget": 0})
            c["tasks"] += 1
            if r["outcome"] == "best (significant)":
                c["wins"] += 1
            elif r["outcome"] in ("best (tie)", "tie"):
                c["ties"] += 1
            elif r["outcome"] == "worse":
                c["worse"] += 1
            else:
                c["over_budget"] += 1
    per_method = sorted(counts.values(), key=lambda c: (PRIORS.index(c["prior"]),
                                                         LEVELS.index(c["level"]),
                                                         METHOD_ORDER.index(c["method"])))
    return per_task, per_method


COLUMNS = (["prior", "prior_name", "level", "task", "budget_seconds", "tolerance", "method",
            "variant", "label", "knobs", "bp_per_image", "n_images", "seconds_per_image",
            "seconds_per_image_sd", "seconds_basis", "budget_ratio", "within_budget",
            "budget_note", "n_candidates", "n_within_budget", "is_best", "best_method",
            "outcome"]
           + METRICS
           + ["%s_%s" % (m, s) for m in METRICS for s in ("diff", "ci_lo", "ci_hi")]
           + ["resamples", "seed", "quality_job_id", "quality_run", "timing_job_id",
              "timing_run", "gpu_name"])
SUMMARY_COLUMNS = ["prior", "prior_name", "level", "task", "budget_seconds", "best_method",
                   "best_label", "best_lpips", "significant", "tied_with_best", "over_budget"]
WIN_COLUMNS = ["prior", "prior_name", "level", "method", "tasks", "wins", "ties", "worse",
               "over_budget"]


# ------------------------------------------------------------------ outputs
_fmt = mtt._fmt


def _lpips_ci(r):
    if r["is_best"]:
        return "best"
    return mtt._ci(r, "lpips", 4)


def _gpu(rows):
    gpus = sorted({summ.short_gpu(r["gpu_name"]) for r in rows if r["gpu_name"]})
    return gpus[0] if len(gpus) == 1 else ("mixed GPUs" if gpus else "GPU unknown")


def latex(rows, per_task, per_method, tolerance=TOLERANCE):
    resamples = rows[0]["resamples"] if rows else RESAMPLES
    seed = rows[0]["seed"] if rows else SEED
    lines = ["% Generated by scripts/budget_level_table.py; needs \\usepackage{booktabs}.",
             "%% Differences are method - best, mean over paired images [95%% bootstrap CI, "
             "%d resamples, seed %d]." % (resamples, seed)]
    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault((r["prior"], r["level"]), []).append(r)
    for (prior, level), mine in groups.items():
        lines += [
            "", "\\begin{table}[t]", "\\centering",
            "\\caption{%s, %s budget (seconds/image at batch size one, %s). Each method's "
            "lowest-LPIPS run within %d\\%% of the budget; $^\\dagger$ over budget (its "
            "cheapest run). Best LPIPS in bold; $\\Delta$LPIPS = method $-$ best with 95\\%% "
            "bootstrap CI.}" % (PRIOR_NAME.get(prior, prior), LEVEL_NAME[level], _gpu(mine),
                                round(100 * tolerance)),
            "\\label{tab:rebuttal-budget-%s-%s}" % (prior, level), "\\small",
            "\\begin{tabular}{llrrrrrl}", "\\toprule",
            "Task (budget) & Method (run) & s/img & $\\times$budget & PSNR & SSIM & LPIPS & "
            "$\\Delta$LPIPS \\\\", "\\midrule"]
        previous = None
        for r in mine:
            first = r["task"] != previous
            if previous is not None and first:
                lines.append("\\midrule")
            previous = r["task"]
            task = ("%s (%.2f s)" % (TASK_NAME.get(r["task"], r["task"]), r["budget_seconds"])
                    if first else "")
            lp = _fmt(r["lpips"], "%.4f")
            if r["is_best"]:
                lp = "\\textbf{%s}" % lp
            lines.append("%s & %s%s & %s & %s & %s & %s & %s & %s \\\\" % (
                task, r["label"], "" if r["within_budget"] else "$^\\dagger$",
                _fmt(r["seconds_per_image"], "%.2f"), _fmt(r["budget_ratio"], "%.2f"),
                _fmt(r["psnr"], "%.2f"), _fmt(r["ssim"], "%.4f"), lp, _lpips_ci(r)))
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tasks = [t for t in TASK_ORDER if any(s["task"] == t for s in per_task)]
    best = {(s["prior"], s["level"], s["task"]): s for s in per_task}
    width = max(len(tasks), len(METHOD_ORDER))
    lines += ["", "\\begin{table}[t]", "\\centering",
              "\\caption{Best method by LPIPS per prior, budget level and task ($^*$: "
              "significantly better than every in-budget method; otherwise tied), and per "
              "method the tasks won significantly / tied.}",
              "\\label{tab:rebuttal-budget-summary}", "\\small",
              "\\begin{tabular}{ll%s}" % ("l" * width), "\\toprule",
              "Prior & Level & %s \\\\" % " & ".join(TASK_NAME[t] for t in tasks), "\\midrule"]
    for prior in PRIORS:
        for level in LEVELS:
            if not any((prior, level, t) in best for t in tasks):
                continue
            cells = []
            for t in tasks:
                s = best.get((prior, level, t))
                if s is None or s["best_method"] is None:
                    cells.append("--")
                else:
                    cells.append(METHOD_NAME[s["best_method"]] + ("$^*$" if s["significant"]
                                                                  else ""))
            lines.append("%s & %s & %s \\\\" % (PRIOR_NAME.get(prior, prior), LEVEL_NAME[level],
                                                " & ".join(cells)))
    lines += ["\\midrule",
              "Prior & Level & %s \\\\" % " & ".join(
                  "%s w/t" % METHOD_NAME[m] for m in METHOD_ORDER),
              "\\midrule"]
    counts = {(c["prior"], c["level"], c["method"]): c for c in per_method}
    for prior in PRIORS:
        for level in LEVELS:
            if not any(k[:2] == (prior, level) for k in counts):
                continue
            cells = []
            for m in METHOD_ORDER:
                c = counts.get((prior, level, m))
                cells.append("--" if c is None else "%d/%d" % (c["wins"], c["ties"]))
            lines.append("%s & %s & %s \\\\" % (PRIOR_NAME.get(prior, prior), LEVEL_NAME[level],
                                                " & ".join(cells)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _write_csv(path, rows, columns):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def write(rows, out_dir, tolerance=TOLERANCE):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_task, per_method = summary(rows)
    _write_csv(out / "budget_levels.csv", rows, COLUMNS)
    _write_csv(out / "budget_levels_summary.csv", per_task, SUMMARY_COLUMNS)
    _write_csv(out / "budget_levels_wins.csv", per_method, WIN_COLUMNS)
    (out / "budget_levels.tex").write_text(latex(rows, per_task, per_method, tolerance))
    return [out / n for n in ("budget_levels.csv", "budget_levels_summary.csv",
                              "budget_levels_wins.csv", "budget_levels.tex")]


def print_rows(rows):
    previous = None
    for r in rows:
        cell = (r["prior"], r["level"], r["task"])
        if cell != previous:
            previous = cell
            print("\n%s  %s level  %s  budget %.3f s/img"
                  % (r["prior_name"], r["level"], r["task"], r["budget_seconds"]))
        print("  %1s %-30s %-34s s/img %7s x%5s  PSNR %6s SSIM %6s LPIPS %6s  dLPIPS %-28s %s"
              % ("*" if r["is_best"] else "", plain(r["label"]), r["knobs"],
                 _fmt(r["seconds_per_image"], "%.2f"), _fmt(r["budget_ratio"], "%.2f"),
                 _fmt(r["psnr"], "%.2f"), _fmt(r["ssim"], "%.4f"), _fmt(r["lpips"], "%.4f"),
                 _lpips_ci(r), r["budget_note"] or r["outcome"] or ""))
    print("\n  * = the level's best method (lowest LPIPS within budget)")


def print_summary(per_method):
    print("\nTasks won significantly / tied, per prior and level:")
    previous = None
    for c in per_method:
        if (c["prior"], c["level"]) != previous:
            previous = (c["prior"], c["level"])
            print("  %s %s:" % (c["prior_name"], c["level"]))
        print("      %-9s wins %d  ties %d  worse %d  over budget %d  (of %d)"
              % (METHOD_PLAIN[c["method"]], c["wins"], c["ties"], c["worse"], c["over_budget"],
                 c["tasks"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quality", action="append", required=True, metavar="RUN",
                   help="quality run directory or results_per_image.csv (repeatable)")
    p.add_argument("--timing", action="append", required=True, metavar="RUN",
                   help="timing run directory written by scripts/run_timing.py (repeatable)")
    p.add_argument("--out-dir", default="outputs/budget_level_table")
    p.add_argument("--tolerance", type=float, default=TOLERANCE,
                   help="largest allowed excess over the budget, as a fraction (default 0.10)")
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--timing-ignore", action="append", default=[], choices=TIMING_IGNORABLE,
                   metavar="FIELD",
                   help="pair quality and timing jobs ignoring this field (only 'lr': it does "
                        "not change the time per image; Task 7's lr-scaled runs reuse "
                        "timing_budget_levels this way)")
    a = p.parse_args(argv)
    rows, notes = build(mtt.read_quality(a.quality), mtt.read_timing(a.timing), a.tolerance,
                        a.seed, a.resamples, timing_ignore=a.timing_ignore)
    if not rows:
        raise SystemExit("no (prior, task) has a timed quality job in the runs given")
    print_rows(rows)
    print_summary(summary(rows)[1])
    for prior in PRIORS:
        gpus = sorted({str(r["gpu_name"]) for r in rows if r["prior"] == prior and r["gpu_name"]})
        if len(gpus) > 1:
            notes.append("WARNING: %s timings come from different GPU types (%s); the ratios "
                         "to the budgets are not like for like" % (prior, ", ".join(gpus)))
    for note in dict.fromkeys(notes):
        print("! " + note, file=sys.stderr)
    paths = write(rows, a.out_dir, a.tolerance)
    print("\nwrote %s: %d rows" % (", ".join(str(x) for x in paths), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
