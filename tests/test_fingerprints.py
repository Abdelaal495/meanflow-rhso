"""scripts/check_fingerprints.py against realistic Flower and matched-budget runs.

The job sets are REAL: the benchmark is configs/experiments_final_frozen100.yaml resolved by
the planner, and the Flower and matched-budget runs are the frozen configurations that
make_flower_configs.py / make_matched_configs.py `freeze` write from planted tuning results.
The rows are written with run.py's own RESULT_COLUMNS writer, and every initial-state
fingerprint is computed the way run.run_job computes it -- pixel_fingerprint of the first
batch's z_t0 = prior_sample(image ids) at t0 = 1 -- on toy adapters, so a different batch,
replicate or image set really does produce a different fingerprint.

Checks: a correct Flower + matched run passes with every job compared (so the frozen configs'
batch sizes really do line up with the benchmark's); a changed measurement, guide or initial
state fails and names the job; a batch size or image count with no benchmark counterpart
fails; an inconsistent benchmark fails; an unrecorded fingerprint fails; an unaggregated run
fails with a hint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collections
import contextlib
import dataclasses
import csv
import importlib.util
import io
import tempfile

import numpy as np

from src.config import load_config, resolve_run_plan, validate_config
from src.utils import gaussian_noise, pixel_fingerprint
from test_flower import ToyMeanFlow, RES

REPO = Path(__file__).resolve().parents[1]
RESULTS = []


def report(name, ok, detail=""):
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    RESULTS.append(bool(ok))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


run = load("run_module", REPO / "run.py")
checker = load("check_fingerprints", REPO / "scripts" / "check_fingerprints.py")
mfc = load("make_flower_configs", REPO / "scripts" / "make_flower_configs.py")
mmc = mfc.matched


def plan_of(path):
    cfg = load_config(str(path))
    return resolve_run_plan(cfg, validate_config(cfg), run_id="fp").specs


# =====================================================================================
# fingerprints, computed as run.py computes them
# =====================================================================================
IMAGE_IDS = tuple("img%03d" % i for i in range(100))
_ADAPTERS = {}


def adapter_for(model):
    if model not in _ADAPTERS:
        _ADAPTERS[model] = ToyMeanFlow(np.zeros((1, RES, RES, 3), np.float32), name=model)
    return _ADAPTERS[model]


def x0_fingerprint(spec, replicate=None, ids=IMAGE_IDS):
    """run.run_job: pixel_fingerprint(adapter.to_numpy(x0)) of the FIRST batch, t0 = 1."""
    adapter = adapter_for(spec.model)
    first = list(ids[:min(int(spec.batch_size), int(spec.num_images))])
    x0 = adapter.initial_state(None, spec.t0, adapter.prior_sample(
        first, replicate=spec.replicate if replicate is None else replicate))
    return pixel_fingerprint(adapter.to_numpy(x0))


def problem_fingerprints(spec):
    """One y / g(y) per (problem_key, num_images), as ProblemStore builds one instance."""
    n = int(spec.num_images)
    y = gaussian_noise((n, 4, 4, 3), "y", spec.problem_key)
    g = gaussian_noise((n, 4, 4, 3), "g", spec.problem_key)
    return pixel_fingerprint(y), pixel_fingerprint(g)


def row_of(spec, **override):
    meas, guide = problem_fingerprints(spec)
    r = {k: getattr(spec, k, None) for k in run.RESULT_COLUMNS}
    r.update(run_id="fp", task=spec.experiment, status="ok", measurement_id=meas,
             guide_id=guide, initial_state_fingerprint=x0_fingerprint(spec),
             image_id="ALL")
    r.update(override)
    return r


def write_run(directory, rows):
    run.write_rows_csv(Path(directory) / "results.csv", rows)
    return str(directory)


def check(benchmark_dir, run_dirs):
    argv = ["--benchmark", benchmark_dir]
    for d in run_dirs:
        argv += ["--run", d]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = checker.main(argv)
    return code, out.getvalue(), err.getvalue()


# =====================================================================================
# frozen Flower and matched-budget configurations, from planted tuning results
# =====================================================================================
def tuning_row(s, lpips, n):
    row = {k: getattr(s, k, None) for k in run.RESULT_COLUMNS}
    row.update(status="ok", lpips=lpips, psnr=25.0, num_images=n)
    return row


def frozen_flower(tmp):
    screen = plan_of(REPO / mfc.SCREEN_OUT)
    rows = [tuning_row(s, 0.5 - 0.001 * i, 4) for i, s in enumerate(screen)]
    write_run(tmp / "flower_screen", rows)
    out = tmp / "flower_frozen.yaml"
    with contextlib.redirect_stdout(io.StringIO()):
        mfc.main(["freeze", "--from", str(tmp / "flower_screen"), "--grid",
                  str(tmp / "flower_screen"), "--out", str(out)])
    return plan_of(out), plan_of(mmc.smoke_path(out))


def frozen_matched(tmp):
    screen = plan_of(REPO / mmc.SCREEN_OUT)
    rows = [tuning_row(s, 0.5 - 0.001 * i, 4) for i, s in enumerate(screen)]
    write_run(tmp / "matched_screen", rows)
    out = tmp / "matched_frozen.yaml"
    with contextlib.redirect_stdout(io.StringIO()):
        mmc.main(["freeze", "--from", str(tmp / "matched_screen"), "--grid",
                  str(tmp / "matched_screen"), "--out", str(out)])
    return plan_of(out)


def main():
    bench_specs = plan_of(REPO / "configs" / "experiments_final_frozen100.yaml")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        flower, flower_smoke = frozen_flower(tmp)
        matched = frozen_matched(tmp)
        print("\njob sets: benchmark %d, Flower frozen %d (%s), matched frozen %d (%s)"
              % (len(bench_specs), len(flower),
                 dict(collections.Counter(s.model for s in flower)), len(matched),
                 dict(collections.Counter(s.method for s in matched))))

        bench = write_run(tmp / "benchmark", [row_of(s) for s in bench_specs])
        flower_rows = [row_of(s) for s in flower]
        matched_rows = [row_of(s) for s in matched]
        flower_dir = write_run(tmp / "flower_final", flower_rows)
        matched_dir = write_run(tmp / "matched_final", matched_rows)

        print("\nCorrect runs")
        code, out, err = check(bench, [flower_dir, matched_dir])
        n = len(flower_rows) + len(matched_rows)
        report("Flower + matched runs pass, every job compared",
               code == 0 and ("%d measurement and %d initial-state" % (n, n)) in out,
               out.strip().split("\n")[0][:110])
        # the comparison has teeth: Flower's first-batch z_t0 equals the benchmark's
        # core job of its prior, and differs from the RHSO group's where the batch differs
        jit_core = [s for s in bench_specs if s.model == "jit" and s.method == "sdedit"][0]
        jit_rhso = [s for s in bench_specs if s.model == "jit" and s.method == "rhso"][0]
        fl_jit = [s for s in flower if s.model == "jit"][0]
        report("first-batch z_t0 depends on batch size, not method",
               x0_fingerprint(fl_jit) == x0_fingerprint(jit_core)
               and x0_fingerprint(jit_rhso) != x0_fingerprint(jit_core)
               and fl_jit.batch_size == jit_core.batch_size == 20 and jit_rhso.batch_size == 8,
               "JiT Flower (batch 20) == JiT SDEdit (batch 20) != JiT RHSO (batch 8)")

        print("\nMismatches fail loudly")
        bad = [dict(r) for r in flower_rows]
        bad[3]["measurement_id"] = "0" * 16
        code, _, err = check(bench, [write_run(tmp / "bad_meas", bad)])
        report("changed measurement", code == 1 and "FINGERPRINT MISMATCH (measurement_id)" in err
               and bad[3]["job_id"] in err, err.strip().split("\n")[-1][:100])

        bad = [dict(r) for r in matched_rows]
        bad[0]["guide_id"] = "1" * 16
        code, _, err = check(bench, [write_run(tmp / "bad_guide", bad)])
        report("changed guide", code == 1 and "(guide_id)" in err and bad[0]["job_id"] in err)

        bad = [dict(r) for r in matched_rows]
        victim = matched[5]
        bad[5]["initial_state_fingerprint"] = x0_fingerprint(victim, replicate=1)
        code, _, err = check(bench, [flower_dir, write_run(tmp / "bad_x0", bad)])
        report("initial state from another noise draw",
               code == 1 and "FINGERPRINT MISMATCH (initial_state)" in err
               and victim.job_id in err and err.count("FINGERPRINT MISMATCH") == 1,
               "replicate-1 z_t0 on %s/%s %s" % (victim.model, victim.problem, victim.method))

        bad = [dict(r) for r in flower_rows]
        odd = [s for s in flower if s.model == "pmf"][0]
        bad[flower.index(odd)].update(batch_size=7, initial_state_fingerprint=x0_fingerprint(
            dataclasses.replace(odd, batch_size=7)))
        code, _, err = check(bench, [write_run(tmp / "bad_batch", bad)])
        report("batch size with no benchmark counterpart", code == 1
               and "NO BENCHMARK COUNTERPART" in err and "batch_size=7" in err)

        smoke_rows = [row_of(s) for s in flower_smoke]
        code, _, err = check(bench, [write_run(tmp / "smoke", smoke_rows)])
        report("2-image smoke run has no counterpart", code == 1
               and err.count("NO BENCHMARK COUNTERPART") == 2 * len(smoke_rows),
               "%d jobs, each unmatched on measurement and initial state" % len(smoke_rows))

        bench_rows = [row_of(s) for s in bench_specs]
        bench_rows[0]["initial_state_fingerprint"] = "f" * 16
        code, _, err = check(write_run(tmp / "bench_bad", bench_rows), [flower_dir])
        report("inconsistent benchmark", code == 1 and "BENCHMARK INCONSISTENT" in err)

        bad = [dict(r) for r in flower_rows]
        bad[0].update(initial_state_fingerprint="", status="failed")
        code, _, err = check(bench, [write_run(tmp / "unverifiable", bad)])
        report("job without fingerprints", code == 1 and "UNVERIFIABLE" in err)

        shard_dir = tmp / "sharded"
        shard_dir.mkdir()
        run.write_rows_csv(shard_dir / "results_shard00.csv", flower_rows)
        code, _, err = check(bench, [str(shard_dir)])
        report("unaggregated run", code == 2 and "--aggregate" in err, err.strip()[:90])

    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
