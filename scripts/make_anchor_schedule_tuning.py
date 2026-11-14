#!/usr/bin/env python
"""RHSO anchor schedules on pMF-L/16 (rebuttal Task 8): the tuning configuration.

RHSO's stage-k objective is  Phi(P_k(x); y) + (mu_k / 2d) ||x - s_k||^2  with

    mu_k = mu0 * m(t_k),   t_k = grid[k]   (canonical clock, 0 = noise, 1 = data)

    constant     m(t) = 1                                (the benchmark's RHSO)
    increasing   m(t) = (t^2 + (1-t)^2) / (1-t)^2
    decreasing   m(t) = (1-t)^2 / (t^2 + (1-t)^2)

(src/rhso.py, `mu_schedule`).  This script writes the grid that retunes mu0 for each
schedule on pMF, every task:

    mu0 in {0.01, 0.03, 0.1, 0.3, 1, 3, 10} x the three schedules   (21 jobs per task)
    plus one mu0 = 0 job per task (all three schedules coincide at 0)  (1 job per task)

    22 configurations per task, 110 in total, ONE round on the first 8 images of the tuning
    pool (the pool, image count and pMF RHSO batch size of the tuning protocol, as in
    configs/experiments_hpo_sweep_checkpoints.yaml / configs/experiments_hpo_grid.yaml).

Every other field -- N, M, lr, t0, the degradation, the defaults block -- is copied from the
task's pMF `__rhso` entry of configs/experiments_final_frozen100.yaml; only mu and
mu_schedule vary.  The mu0 = 0 job runs the default (constant) code path.

    python scripts/make_anchor_schedule_tuning.py
    python run.py --config configs/experiments_anchor_schedule_tuning_smoke.yaml --models pmf --run-id anchor_smoke
    python run.py --config configs/experiments_anchor_schedule_tuning.yaml --models pmf --dry-run

Selection and the 100-image configuration: scripts/freeze_anchor_schedules.py.
"""
import argparse
import importlib.util
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matched = _load("make_matched_configs")
hpo = matched.hpo

MODEL = "pmf"
METHOD = "rhso"
SCHEDULES = ["constant", "increasing", "decreasing"]
MU_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SMOKE_MU = 1
BENCHMARK = "configs/experiments_final_frozen100.yaml"
TUNING_OUT = "configs/experiments_anchor_schedule_tuning.yaml"
FROZEN_OUT = "configs/experiments_anchor_schedule_frozen100.yaml"
TUNING_IMAGES, SMOKE_IMAGES, FROZEN_IMAGES = 8, 2, 100
TUNING_BATCH = hpo.BATCH[(MODEL, METHOD)]            # 4: pMF RHSO in the tuning protocol
TUNE_SUFFIX, MU0_SUFFIX = "__anchor_tune", "__anchor_tune_mu0"
FROZEN_PREFIX = "__anchor_"


# =====================================================================================
# the frozen benchmark entry each task copies
# =====================================================================================
def benchmark_entry(benchmark, problem):
    """(fixed fields in source order, without mu / mu_schedule; degradation; batch size)."""
    exp = benchmark["experiments"].get(hpo.EXP_NAME[problem] + "__rhso")
    if exp is None:
        raise SystemExit("%s has no %s__rhso experiment" % (benchmark["_path"], hpo.EXP_NAME[problem]))
    entry = matched._entry(benchmark, hpo.EXP_NAME[problem] + "__rhso", MODEL, METHOD)
    if exp.get("problem") != problem:
        raise SystemExit("%s__rhso is problem %r, expected %r"
                         % (hpo.EXP_NAME[problem], exp.get("problem"), problem))
    if yaml.safe_load(hpo.DEGRADATION[problem]) != exp.get("degradation"):
        raise SystemExit("%s__rhso: degradation %r differs from the benchmark protocol's %s"
                         % (hpo.EXP_NAME[problem], exp.get("degradation"), hpo.DEGRADATION[problem]))
    fixed = [(k, v) for k, v in entry.items() if k not in ("mu", "mu_schedule")]
    batch = int(exp["models"][MODEL]["batch_size"])
    return fixed, entry.get("mu"), batch


def fmt(v):
    """A YAML scalar that reads back as the same Python value (1.0 stays a float)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[%s]" % ", ".join(fmt(x) for x in v)
    return str(v)


def method_line(fields, comment):
    body = ", ".join("%s: %s" % (k, fmt(v)) for k, v in fields)
    return "          %-12s {%s}   # %s" % (METHOD + ":", body, comment)


def block(name, problem, num_images, batch, fields, comment):
    return ["", "  %s:" % name, "    enabled: true", "    problem: %s" % problem,
            "    num_images: %d" % num_images,
            "    degradation: %s" % hpo.DEGRADATION[problem], "    models:",
            "      %s:" % MODEL, "        batch_size: %d" % batch, "        methods:",
            method_line(fields, comment)]


def write(out, lines, n_jobs, title, usage, text, pool, save_images):
    body = matched.header(title, usage, text, matched.max_jobs_for(n_jobs), pool, [MODEL],
                          save_images)
    path = REPO_ROOT / out
    path.write_text(body + "\n".join(lines) + "\n")
    print("wrote %s: %d jobs" % (out, n_jobs))
    return path


# =====================================================================================
# tuning grid
# =====================================================================================
TUNING_TEXT = """Time-dependent state-anchor weight for RHSO on pMF-L/16 (rebuttal Task 8).
Stage k uses mu_k = mu0 * m(t_k), t_k = grid[k] (0 = noise, 1 = data):
  constant m = 1;  increasing m = (t^2 + (1-t)^2) / (1-t)^2;  decreasing = 1 / increasing.
At t0 = 1, beta = 1 the stage times are k/N: N = 4 -> 0, .25, .5, .75 (increasing: 1, 1.11,
2, 10); N = 8 -> 0, .125, ..., .875 (increasing: 1, 1.02, 1.11, 1.36, 2, 3.78, 10, 50).
Grid per task: mu0 in %s x mu_schedule in %s
(<task>__anchor_tune, 21 jobs) plus mu0 = 0 once (<task>__anchor_tune_mu0; every schedule
gives the same objective at 0): 22 configurations per task.
Every other field (N, M, lr, t0, degradation) is the task's pMF __rhso entry of
%s; only mu and mu_schedule vary.
One round on the first %d images of the tuning pool, pMF RHSO tuning batch size %d.
Selection: scripts/freeze_anchor_schedules.py (lowest LPIPS per task and schedule, mu0 = 0
a candidate of every schedule).
Generated by scripts/make_anchor_schedule_tuning.py; %d atomic jobs."""


def tuning_lines(benchmark, smoke):
    lines, n_jobs = [], 0
    n_images = SMOKE_IMAGES if smoke else TUNING_IMAGES
    batch = min(TUNING_BATCH, n_images)
    for problem in hpo.PROBLEMS:
        fixed, frozen_mu, _ = benchmark_entry(benchmark, problem)
        exp = hpo.EXP_NAME[problem]
        mus = [SMOKE_MU] if smoke else list(MU_GRID)
        jobs = len(mus) * len(SCHEDULES)
        source = ("%s fields of the frozen pMF entry (frozen mu0 = %s, constant)"
                  % ("/".join(k for k, _ in fixed), fmt(frozen_mu)))
        lines += block(exp + TUNE_SUFFIX, problem, n_images, batch,
                       fixed + [("mu", mus if len(mus) > 1 else mus[0]),
                                ("mu_schedule", list(SCHEDULES))],
                       "%d jobs; %s" % (jobs, source))
        lines += block(exp + MU0_SUFFIX, problem, n_images, batch,
                       fixed + [("mu", 0), ("mu_schedule", "constant")],
                       "1 job; mu0 = 0 is the same objective under every schedule")
        n_jobs += jobs + 1
    return lines, n_jobs


def write_tuning(benchmark_path=BENCHMARK, out=TUNING_OUT):
    benchmark = matched.load_yaml(benchmark_path)
    paths = []
    for smoke in (False, True):
        target = matched.smoke_path(out) if smoke else Path(out).as_posix()
        lines, n_jobs = tuning_lines(benchmark, smoke)
        title = ("SMOKE: RHSO anchor schedules on pMF, every task at 2 images" if smoke else
                 "RHSO ANCHOR SCHEDULES: mu0 tuning on pMF-L/16 at %d images" % TUNING_IMAGES)
        usage = ("#   python run.py --config %s --models pmf --run-id <id> --no-figures\n" % target)
        text = TUNING_TEXT % ("{%s}" % ", ".join(fmt(m) for m in MU_GRID),
                              "{%s}" % ", ".join(SCHEDULES), benchmark_path, TUNING_IMAGES, TUNING_BATCH,
                              n_jobs)
        if smoke:
            text += ("\nSmoke variant: mu0 = %s under each schedule plus mu0 = 0, on 2 images: it\n"
                     "checks that every task resolves and runs every code path, not which mu0 wins."
                     % fmt(SMOKE_MU))
        paths.append(write(target, lines, n_jobs, title, usage, text, matched.TUNING_POOL, False))
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default=BENCHMARK,
                   help="frozen benchmark the fixed fields are copied from (default: %(default)s)")
    p.add_argument("--out", default=TUNING_OUT,
                   help="output path; the smoke variant is <out>_smoke.yaml")
    a = p.parse_args(argv)
    write_tuning(a.benchmark, a.out)


if __name__ == "__main__":
    main()
