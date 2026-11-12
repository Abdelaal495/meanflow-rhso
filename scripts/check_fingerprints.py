#!/usr/bin/env python
"""Check that new runs share the benchmark's measurements and initial states, bit for bit.

The rebuttal compares Flower (Task 3) and the matched-budget RHSO / MPC-delta_t runs
(Task 2) against the original benchmark run image by image, so every one of their jobs must
have seen the SAME y and started from the SAME z_t0 as the benchmark.  run.py records both
in results.csv:

    measurement_id / guide_id    fingerprints of the problem instance's whole y / g(y)
                                 (all of its images, in order)
    initial_state_fingerprint    fingerprint of the job's FIRST batch of z_t0

This script compares them against the benchmark run and exits non-zero, naming every job,
on any mismatch.  A job with no benchmark counterpart is also an error: a check that
compares nothing must not pass.

    python scripts/check_fingerprints.py --benchmark outputs/<benchmark run> \
        --run outputs/flower_final_pmf --run outputs/matched_final_pmf

Matching rules
--------------
* measurement: the benchmark jobs with the same `problem_key` (problem, degradation
  parameters, seed and, if set, pairing group) and the same `num_images`.  Their
  measurement_id must be one value, and the run's job must have it; likewise guide_id.
* initial state: the benchmark jobs with the same model, problem_key, t0, seed, replicate,
  batch_size and num_images -- the jobs whose first batch is the same set of images at the
  same corruption.  Their initial_state_fingerprint must be one value, and the run's job
  must have it.  The frozen Flower and matched-budget configs use the benchmark's own batch
  sizes (scripts/hpo.py BATCH_FINAL), so every one of their jobs has such a counterpart.

Only the first batch's z_t0 is fingerprinted by run.py.  At t0 = 1 the remaining batches are
the same deterministic function of (model, image id, replicate), and the image ids and their
order are pinned by the measurement fingerprint, which covers every image.

Use `--aggregate` on a sharded run first: this reads each run's merged results.csv.
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

REQUIRED = ("job_id", "experiment", "model", "method", "problem", "problem_key", "t0", "seed",
            "replicate", "batch_size", "num_images", "measurement_id", "guide_id",
            "initial_state_fingerprint", "status")


class CheckError(Exception):
    pass


def read_rows(run_dir):
    run_dir = Path(run_dir)
    path = run_dir / "results.csv"
    if not path.exists():
        shards = sorted(run_dir.glob("results_shard*.csv"))
        hint = (" (found %d shard file(s): run `python run.py --aggregate %s` first)"
                % (len(shards), run_dir)) if shards else ""
        raise CheckError("%s does not exist%s" % (path, hint))
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise CheckError("%s lacks the columns %s" % (path, missing))
        rows = [dict(r, _run=run_dir.name) for r in reader]
    if not rows:
        raise CheckError("%s has no jobs" % path)
    return rows


def _num(v):
    return "%g" % float(v)


def measurement_key(r):
    return (r["problem_key"], int(float(r["num_images"])))


def initial_state_key(r):
    return (r["model"], r["problem_key"], _num(r["t0"]), int(float(r["seed"])),
            int(float(r["replicate"])), int(float(r["batch_size"])),
            int(float(r["num_images"])))


def _index(rows, key, fields):
    """key -> {field: set of values}, from rows that recorded those fields."""
    out = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if all(r.get(f) for f in fields):
            for f in fields:
                out[key(r)][f].add(r[f])
    return out


def describe(r):
    return "%s/%s %s %s/%s/%s (job %s)" % (r["_run"], r["experiment"], r["model"], r["method"],
                                           r["problem"], "t0=%s" % r["t0"], r["job_id"])


def check(benchmark_rows, run_rows):
    """Return (errors, number of measurement checks, number of initial-state checks)."""
    errors = []
    meas = _index(benchmark_rows, measurement_key, ("measurement_id", "guide_id"))
    init = _index(benchmark_rows, initial_state_key, ("initial_state_fingerprint",))

    for key, fields in sorted(meas.items()):
        for f, values in fields.items():
            if len(values) > 1:
                errors.append("BENCHMARK INCONSISTENT: %d different %s for problem %s at %d "
                              "images: %s" % (len(values), f, key[0], key[1], sorted(values)))
    for key, fields in sorted(init.items()):
        values = fields["initial_state_fingerprint"]
        if len(values) > 1:
            errors.append("BENCHMARK INCONSISTENT: %d different initial states for %s: %s"
                          % (len(values), key, sorted(values)))

    n_meas = n_init = 0
    for r in run_rows:
        who = describe(r)
        if not (r.get("measurement_id") and r.get("guide_id")
                and r.get("initial_state_fingerprint")):
            errors.append("UNVERIFIABLE: %s recorded no fingerprint (status %s)"
                          % (who, r.get("status")))
            continue
        bm = meas.get(measurement_key(r))
        if bm is None:
            errors.append("NO BENCHMARK COUNTERPART: %s -- no benchmark job measured problem "
                          "%s (%s) at %s images" % (who, r["problem_key"], r["problem"],
                                                    r["num_images"]))
        else:
            n_meas += 1
            for f in ("measurement_id", "guide_id"):
                if r[f] not in bm[f]:
                    errors.append("FINGERPRINT MISMATCH (%s): %s has %s, the benchmark has %s"
                                  % (f, who, r[f], sorted(bm[f])))
        bi = init.get(initial_state_key(r))
        if bi is None:
            errors.append("NO BENCHMARK COUNTERPART: %s -- no benchmark job with the same "
                          "(model, problem, t0, seed, replicate, batch_size=%s, num_images=%s), "
                          "so its first-batch z_t0 cannot be compared"
                          % (who, r["batch_size"], r["num_images"]))
        else:
            n_init += 1
            if r["initial_state_fingerprint"] not in bi["initial_state_fingerprint"]:
                errors.append("FINGERPRINT MISMATCH (initial_state): %s has %s, the benchmark "
                              "has %s" % (who, r["initial_state_fingerprint"],
                                          sorted(bi["initial_state_fingerprint"])))
    return errors, n_meas, n_init


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", action="append", required=True, metavar="RUN",
                   help="the original benchmark run directory (repeatable, e.g. one per prior)")
    p.add_argument("--run", action="append", required=True, metavar="RUN",
                   help="a Flower or matched-budget run directory to check (repeatable)")
    a = p.parse_args(argv)
    try:
        benchmark = [r for d in a.benchmark for r in read_rows(d)]
        runs = [r for d in a.run for r in read_rows(d)]
    except CheckError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    errors, n_meas, n_init = check(benchmark, runs)
    print("checked %d job(s) from %d run(s) against %d benchmark job(s): %d measurement and "
          "%d initial-state comparison(s)" % (len(runs), len(a.run), len(benchmark), n_meas,
                                               n_init))
    if errors:
        print("\nFAILED: %d problem(s)" % len(errors), file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    print("OK: every job saw the benchmark's measurement and started from its initial state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
