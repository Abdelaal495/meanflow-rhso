#!/usr/bin/env python
"""Run one prior's timing configuration with a FRESH PROCESS PER CONFIGURATION.

Why a process per configuration: GPU memory a framework has reserved is not handed back
while the process lives.  JAX's pooling (BFC) allocator keeps every region it has grown,
and PyTorch's caching allocator keeps its blocks, so in a long multi-job process the NVML
reading of a job is the high-water mark of every job before it (run_9187465: every pMF job
at ~32.56 GiB, JiT after pMF at ~33.4 GiB).  Here each process does only this:

    load data -> load the model -> warm up THIS configuration -> time THIS configuration

so every lifetime memory statistic belongs to that one configuration.  The steps:

    1. a model-only baseline process (run.py --memory-baseline-only), which loads the
       model and records what it alone costs -> <run dir>/memory_baseline_<prior>.json
    2. one run.py process per experiment block (--experiments <block>), all with ONE
       shared --run-id, with --no-check (the checks allocate GPU memory of their own; run
       the smoke config with checks first) and --no-figures
    3. run.py --aggregate, which merges the per-job metadata.json into results.csv

Allocator policy, for these child processes ONLY (the parent environment and every other
entry point are unchanged): JAX's default pooling allocator, explicitly
XLA_PYTHON_CLIENT_ALLOCATOR=bfc, with XLA_PYTHON_CLIENT_PREALLOCATE=false.  It is the only
JAX allocator with a `peak_bytes_in_use` statistic, which is the primary memory figure;
the NVML number is then the process's reserved pool and is labelled as such.  The memory
fraction is left at its default and recorded by src/timing.py.

    python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf.yaml --run-id timing_pmf
    python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf_smoke.yaml \
        --run-id timing_pmf_smoke --with-checks          # first, on the same GPU type
    python scripts/run_timing.py ... --dry-run             # print the commands only
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Set in the CHILD environment only.
TIMING_ENV = {"XLA_PYTHON_CLIENT_ALLOCATOR": "bfc", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def enabled_blocks(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    return [name for name, block in (cfg.get("experiments") or {}).items()
            if block.get("enabled", True)]


def child_env(base=None):
    env = dict(os.environ if base is None else base)
    env.update(TIMING_ENV)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_commands(args):
    """[(label, argv)] in execution order."""
    common = ["--config", args.config, "--run-id", args.run_id]
    for flag, value in (("--output-root", args.output_root), ("--cache-root", args.cache_root),
                        ("--num-images", args.num_images)):
        if value is not None:
            common += [flag, str(value)]
    run = [args.python, "-u", str(REPO_ROOT / "run.py")]
    checks = [] if args.with_checks else ["--no-check"]
    blocks = enabled_blocks(args.config)
    if args.experiments:
        wanted = [b.strip() for b in args.experiments.split(",") if b.strip()]
        unknown = sorted(set(wanted) - set(blocks))
        if unknown:
            raise SystemExit("--experiments names blocks not in %s: %s" % (args.config, unknown))
        blocks = [b for b in blocks if b in wanted]
    commands = [("model_only_baseline",
                 run + common + checks + ["--no-figures", "--memory-baseline-only"])]
    for block in blocks:
        commands.append((block, run + common + checks + ["--no-figures", "--experiments", block]))
    commands.append(("aggregate", run + common + ["--no-figures", "--aggregate"]))
    return commands


def run_dir_for(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = (args.output_root or os.environ.get("MPCFLOW_OUTPUT_ROOT")
            or (cfg.get("runtime") or {}).get("output_root", "outputs"))
    return Path(root) / args.run_id


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True, help="shared by every configuration's process")
    p.add_argument("--output-root", default=None)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--experiments", default=None, help="comma-separated subset of blocks")
    p.add_argument("--with-checks", action="store_true",
                   help="keep run.py's model checks (for the smoke run; they allocate GPU "
                        "memory, so never use this for the measured run)")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = p.parse_args(argv)

    commands = build_commands(args)
    log_dir = run_dir_for(args) / "timing_logs"
    print("run dir : %s" % run_dir_for(args))
    print("child environment: %s" % " ".join("%s=%s" % kv for kv in TIMING_ENV.items()))
    print("%d process(es): model-only baseline, %d configuration(s), aggregate"
          % (len(commands), len(commands) - 2))
    if args.dry_run:
        for label, argv_ in commands:
            print("  [%s] %s" % (label, " ".join(argv_)))
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    env = child_env()
    failed = []
    for index, (label, argv_) in enumerate(commands, 1):
        log = log_dir / ("%02d_%s.log" % (index, label))
        started = time.time()
        print("[%d/%d] %s -> %s" % (index, len(commands), label, log), flush=True)
        with open(log, "w") as fh:
            code = subprocess.call(argv_, cwd=str(REPO_ROOT), env=env, stdout=fh,
                                   stderr=subprocess.STDOUT)
        print("        exit %d after %.0f s" % (code, time.time() - started), flush=True)
        if code != 0:
            failed.append(label)
    if failed:
        print("FAILED: %s (see %s)" % (", ".join(failed), log_dir))
        return 1
    print("done. Summarise with: python scripts/summarize_timing.py %s" % run_dir_for(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
