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
       the smoke config with --with-checks first) and --no-figures
    3. run.py --aggregate, which merges the per-job metadata.json into results.csv

RESUMABLE: a block whose job is already complete in the shared run directory is skipped --
complete meaning run.py's own resume test passes (status ok, results.npz present, the stored
resolved spec equal to this one) AND its metadata.json carries the timing_detail block.  The
baseline is skipped when memory_baseline_<prior>.json exists.  A job resubmitted after a
timeout therefore only runs what is left (--rerun-all disables the skipping).

Allocator policy, for these child processes ONLY (the parent environment and every other
entry point are unchanged): JAX's default pooling allocator, explicitly
XLA_PYTHON_CLIENT_ALLOCATOR=bfc, with XLA_PYTHON_CLIENT_PREALLOCATE=false.  It is the only
JAX allocator with memory statistics, which give the primary memory figure; the NVML number
is then the process's reserved pool and is labelled as such.

WALL-CLOCK ESTIMATE (printed before every run, or alone with --estimate): per remaining
block, process start-up + model load + warm-up + images x seconds per image, from the jobs
logged in cluster_results/run_*/results.csv.  Those ran BATCHED, so a batch-1 image is
bracketed: at least the batched seconds per image, at most the seconds per whole batch
(a batch of one is never slower than a batch of B).  A block with no logged job of the same
cost-defining configuration is EXTRAPOLATED, not bracketed: its expected evaluations +
2 x backprops times the median logged seconds per forward-equivalent of the same method on
that prior (any task), or of all that prior's methods except Flower (whose deblurring cost
is its CG solve) when the method was never logged.  A prior with no logged job gets no
estimate unless --seconds-per-fe is given.

    python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf.yaml --run-id timing_pmf
    python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf_smoke.yaml \
        --run-id timing_pmf_smoke --with-checks          # first, on the same GPU type
    python scripts/run_timing.py ... --dry-run             # commands and estimate only
    python scripts/run_timing.py ... --estimate            # estimate only
"""
import argparse
import csv
import glob
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set in the CHILD environment only.
TIMING_ENV = {"XLA_PYTHON_CLIENT_ALLOCATOR": "bfc", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
JAX_PRIORS = {"pmf", "imf"}
# Fields that decide how much work a job does (the rest only change values).
COST_FIELDS = ("problem", "method", "t0", "steps", "solver", "num_mpc_steps", "n_ctrl", "K",
               "num_pnp_steps", "noise_samples", "num_opt_steps", "num_rhso_steps",
               "rhso_terminal_mode", "num_flower_steps", "flower_cg_max_iters")


def child_env(base=None):
    env = dict(os.environ if base is None else base)
    env.update(TIMING_ENV)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


# =====================================================================================
# The plan, resolved exactly as run.py resolves it
# =====================================================================================
def _run_argv(args):
    argv = ["--config", args.config, "--run-id", args.run_id]
    for flag, value in (("--output-root", args.output_root), ("--cache-root", args.cache_root),
                        ("--num-images", args.num_images)):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def resolve(args):
    """(plan, run_dir, {block: spec}) with run.py's own overrides and planner."""
    import run
    from src.config import load_config, resolve_run_plan, validate_config
    ns = run.parse_args(_run_argv(args))
    config = run.apply_cli_overrides(load_config(args.config), ns)
    plan = resolve_run_plan(config, validate_config(config), run_id=args.run_id)
    by_block = {}
    for spec in plan.specs:
        if spec.experiment in by_block:
            raise SystemExit("block %s resolves to more than one job; a timing block must be "
                             "exactly one configuration" % spec.experiment)
        by_block[spec.experiment] = spec
    timing_on = bool((config.get("runtime") or {}).get("timing_detail", False))
    return plan, Path(plan.output_dir).resolve(), by_block, timing_on


def param_loading_modes(config_path):
    """{model: param_loading} for the models whose config sets a non-default loader (Task 9)."""
    from src.config import load_config, resolve_model_registry
    config = load_config(config_path)
    modes = {}
    for model in (config.get("models") or {}):
        mode = resolve_model_registry(config, model).get("param_loading")
        if mode not in (None, "device"):
            modes[model] = mode
    return modes


def is_complete(run_dir, spec, timing_required=True):
    """run.py's resume test, plus the timing_detail block when the config asks for it."""
    import run
    try:
        finished = run.load_finished_job(Path(run_dir), spec)
    except Exception:
        return False
    if finished is None:
        return False
    return (not timing_required) or bool(finished["record"].get("timing_detail"))


def baseline_complete(run_dir, models):
    return all((Path(run_dir) / ("memory_baseline_%s.json" % m)).exists() for m in models)


def select_blocks(args, by_block):
    blocks = list(by_block)
    if args.experiments:
        wanted = [b.strip() for b in args.experiments.split(",") if b.strip()]
        unknown = sorted(set(wanted) - set(blocks))
        if unknown:
            raise SystemExit("--experiments names blocks not in %s: %s" % (args.config, unknown))
        blocks = [b for b in blocks if b in wanted]
    return blocks


def build_commands(args, blocks, skip_baseline=False, baseline_models=None):
    """[(label, argv)] in execution order, for the blocks still to run.

    `baseline_models` (a config with more than one prior, e.g. the matched-time timing
    config): one model-only baseline PROCESS per listed prior, each with `--models <prior>`,
    so no prior's baseline carries the pool another framework left behind.  Without it there
    is one baseline process, as for the one-prior configs.
    """
    common = _run_argv(args)
    run_ = [args.python, "-u", str(REPO_ROOT / "run.py")]
    checks = [] if args.with_checks else ["--no-check"]
    commands = []
    if baseline_models:
        for model in baseline_models:
            commands.append(("model_only_baseline_%s" % model,
                             run_ + common + checks + ["--no-figures", "--memory-baseline-only",
                                                       "--models", model]))
    elif not skip_baseline:
        commands.append(("model_only_baseline",
                         run_ + common + checks + ["--no-figures", "--memory-baseline-only"]))
    for block in blocks:
        commands.append((block, run_ + common + checks + ["--no-figures", "--experiments", block]))
    commands.append(("aggregate", run_ + common + ["--no-figures", "--aggregate"]))
    return commands


# =====================================================================================
# Wall-clock estimate
# =====================================================================================
def _norm(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "nan", "NaN"):
        return None
    try:
        return round(float(text), 9)
    except ValueError:
        return text


def _f(row, key):
    v = _norm(row.get(key))
    return v if isinstance(v, float) else None


def load_reference(source_dirs):
    """Logged jobs: [{prior, key, t_lo, t_hi, spfe_lo, spfe_hi, warm, load}].

    Every results.csv below each directory: the benchmark runs in cluster_results/ and, once
    one exists, a partial or smoke timing run, whose batch-1 rows give t_lo == t_hi.
    """
    refs = []
    if isinstance(source_dirs, (str, Path)):
        source_dirs = [source_dirs]
    paths = sorted({p for d in source_dirs
                    for p in glob.glob(str(Path(d) / "**" / "results.csv"), recursive=True)})
    for path in paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "ok":
                    continue
                n, bs = _f(row, "num_images"), _f(row, "batch_size")
                runtime, rpi = _f(row, "runtime"), _f(row, "runtime_per_image")
                evals, bps = _f(row, "model_evaluations"), _f(row, "backprops_through_model")
                if None in (n, bs, runtime, rpi, evals) or bs < 1:
                    continue
                chunks = math.ceil(n / bs)
                fe = (evals + 2.0 * (bps or 0.0)) / chunks     # counters are per batched call
                refs.append({"prior": row.get("model"),
                             "key": tuple(_norm(row.get(k)) for k in COST_FIELDS),
                             "t_lo": rpi, "t_hi": runtime / chunks,
                             "spfe_lo": rpi / fe if fe else None,
                             "spfe_hi": (runtime / chunks) / fe if fe else None,
                             "warm": _f(row, "warmup_seconds") or 0.0,
                             "load": _f(row, "model_load_seconds") or 0.0,
                             "source": path})
    return refs


def spec_key(spec):
    return tuple(_norm(getattr(spec, k, None)) for k in COST_FIELDS)


def estimate(specs, refs, overhead, jax_compile, include_baseline, fe_override=None):
    """{prior: summary} for the given remaining specs."""
    fe_override = fe_override or {}
    out = {}
    for prior in sorted({s.model for s in specs}):
        mine = [r for r in refs if r["prior"] == prior]
        # Flower's deblurring cost is its CG solve, not model passes: its seconds per
        # forward-equivalent says nothing about another method, so it never enters a
        # fallback pool other than Flower's own.
        pool = [r for r in mine if r["spfe_lo"] and not (r["key"][1] == "flower")]
        spfe = fe_override.get(prior)
        if spfe is not None:
            spfe_lo = spfe_hi = spfe
        elif pool:
            spfe_lo = statistics.median(r["spfe_lo"] for r in pool)
            spfe_hi = statistics.median(r["spfe_hi"] for r in pool)
        else:
            spfe_lo = spfe_hi = None

        def method_spfe(method):
            same = [r for r in mine if r["key"][1] == method and r["spfe_lo"]]
            if not same or spfe is not None:
                return spfe_lo, spfe_hi
            return (statistics.median(r["spfe_lo"] for r in same),
                    statistics.median(r["spfe_hi"] for r in same))
        load = max((r["load"] for r in mine), default=None)
        warm_default = max((r["warm"] for r in mine), default=0.0)
        compile_ = jax_compile if prior in JAX_PRIORS else 0.0
        blocks, matched, fallback, lo_total, hi_total, fe_total = [], 0, 0, 0.0, 0.0, 0.0
        ok = load is not None or spfe is not None
        load = load or 0.0
        for spec in (s for s in specs if s.model == prior):
            fe = float(spec.expected_model_evals) + 2.0 * float(spec.expected_backprops)
            fe_total += fe * spec.num_images
            hits = [r for r in mine if r["key"] == spec_key(spec)]
            if hits and spfe is None:
                t_lo = statistics.median(r["t_lo"] for r in hits)
                t_hi = max(r["t_hi"] for r in hits)
                warm = max(r["warm"] for r in hits)
                basis = "logged job"
                matched += 1
            elif spfe_lo is not None:
                m_lo, m_hi = method_spfe(spec.method)
                t_lo, t_hi, warm, basis = fe * m_lo, fe * m_hi, warm_default, "s/FE"
                fallback += 1
            else:
                ok = False
                continue
            fixed = overhead + load + warm + compile_
            lo = fixed + spec.num_images * t_lo
            hi = fixed + spec.num_images * t_hi
            lo_total += lo
            hi_total += hi
            blocks.append((spec.experiment, basis, fe, t_lo, t_hi, lo, hi))
        if include_baseline:
            lo_total += overhead + load
            hi_total += overhead + load
        lo_total += overhead                                        # aggregate
        hi_total += overhead
        out[prior] = {"ok": ok, "blocks": blocks, "matched": matched, "fallback": fallback,
                      "lower_s": lo_total if ok else None, "upper_s": hi_total if ok else None,
                      "load_s": load, "fe_total": fe_total,
                      "spfe": (spfe_lo, spfe_hi), "references": len(mine)}
    return out


def suggested_limit(upper_s, margin=1.1):
    """Upper bound x margin, rounded up to 15 minutes, as Slurm's HH:MM:SS."""
    minutes = int(math.ceil(upper_s * margin / 60.0 / 15.0) * 15) or 15
    return "%02d:%02d:00" % (minutes // 60, minutes % 60)


def print_estimate(est, args, n_remaining, n_total, verbose=False):
    print("\nWALL-CLOCK ESTIMATE  (%d of %d block(s) left; assumptions: %.0f s start-up per "
          "process, %.0f s JAX compile per block, from %s)"
          % (n_remaining, n_total, args.process_overhead, args.jax_compile,
             ", ".join(args.timing_source)))
    for prior, e in est.items():
        if not e["ok"]:
            print("  %-4s no logged job for this prior in %s: workload %.0f forward-equivalents;"
                  " pass --seconds-per-fe %s=<s> to estimate" % (prior, ", ".join(args.timing_source),
                                                                 e["fe_total"], prior))
            continue
        print("  %-4s %.1f to %.1f h  (model load %.0f s; %d block(s) from a logged job of the "
              "same configuration, %d from s/FE %.4f-%.4f)  -> suggested --time %s"
              % (prior, e["lower_s"] / 3600, e["upper_s"] / 3600, e["load_s"], e["matched"],
                 e["fallback"], e["spfe"][0] or 0, e["spfe"][1] or 0,
                 suggested_limit(e["upper_s"])))
        if verbose:
            for name, basis, fe, t_lo, t_hi, lo, hi in e["blocks"]:
                print("       %-42s %-10s FE/img %6.0f  s/img %7.3f-%7.3f  block %6.0f-%6.0f s"
                      % (name, basis, fe, t_lo, t_hi, lo, hi))


# =====================================================================================
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
    p.add_argument("--rerun-all", action="store_true",
                   help="do not skip blocks that are already complete in the run directory")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    p.add_argument("--estimate", action="store_true", help="print the estimate and exit")
    p.add_argument("--verbose", action="store_true", help="per-block estimate lines")
    p.add_argument("--timing-source", action="append", default=None,
                   help="directory whose results.csv files (any depth) are earlier runs to "
                        "estimate from; repeatable (default: cluster_results/)")
    p.add_argument("--process-overhead", type=float, default=45.0,
                   help="seconds per process for start-up, imports, data and problem building")
    p.add_argument("--jax-compile", type=float, default=60.0,
                   help="seconds of JAX compilation per block (batch 1 is a new shape)")
    p.add_argument("--seconds-per-fe", action="append", default=[], metavar="PRIOR=S",
                   help="override the seconds per forward-equivalent of a prior")
    args = p.parse_args(argv)
    args.timing_source = args.timing_source or [str(REPO_ROOT / "cluster_results")]

    plan, run_dir, by_block, timing_on = resolve(args)
    loading = param_loading_modes(args.config)
    if loading and args.with_checks:
        raise SystemExit("%s loads %s with an exact-peak loader (%s); run.py's checks would "
                         "allocate inside each measured process and void the exact JAX peak, "
                         "so --with-checks is refused here."
                         % (args.config, ", ".join(sorted(loading)),
                            ", ".join("%s=%s" % kv for kv in sorted(loading.items()))))
    blocks = select_blocks(args, by_block)
    models = sorted({by_block[b].model for b in blocks}) or sorted({s.model for s in plan.specs})
    if args.rerun_all:
        done, skip_baseline = [], False
    else:
        done = [b for b in blocks if is_complete(run_dir, by_block[b], timing_on)]
        skip_baseline = baseline_complete(run_dir, models)
    remaining = [b for b in blocks if b not in done]
    baseline_models = None
    if len(models) > 1:                     # one baseline process per prior still missing one
        baseline_models = [m for m in models if args.rerun_all
                           or not baseline_complete(run_dir, [m])]
    commands = build_commands(args, remaining, skip_baseline=skip_baseline,
                              baseline_models=baseline_models)

    print("run dir : %s" % run_dir)
    print("child environment: %s" % " ".join("%s=%s" % kv for kv in TIMING_ENV.items()))
    if loading:
        print("parameter loading: %s  (exact JAX peak per configuration; src/timing.py "
              "exact_jax_peak)" % ", ".join("%s=%s" % kv for kv in sorted(loading.items())))
    print("%d block(s) selected: %d already complete (skipped), %d to run; model-only "
          "baseline %s" % (len(blocks), len(done), len(remaining),
                           "already recorded (skipped)" if skip_baseline else "to run"))
    overrides = {}
    for item in args.seconds_per_fe:
        prior, _, value = item.partition("=")
        overrides[prior.strip()] = float(value)
    est = estimate([by_block[b] for b in remaining], load_reference(args.timing_source),
                   args.process_overhead, args.jax_compile, not skip_baseline, overrides)
    print_estimate(est, args, len(remaining), len(blocks), verbose=args.verbose)
    if args.estimate:
        return 0
    if args.dry_run:
        for label, argv_ in commands:
            print("  [%s] %s" % (label, " ".join(argv_)))
        return 0

    log_dir = run_dir / "timing_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = child_env()
    failed = []
    for index, (label, argv_) in enumerate(commands, 1):
        log = log_dir / ("%s.log" % label)
        started = time.time()
        print("[%d/%d] %s -> %s" % (index, len(commands), label, log), flush=True)
        with open(log, "a") as fh:
            fh.write("\n===== %s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv_)))
            fh.flush()
            code = subprocess.call(argv_, cwd=str(REPO_ROOT), env=env, stdout=fh,
                                   stderr=subprocess.STDOUT)
        print("        exit %d after %.0f s" % (code, time.time() - started), flush=True)
        if code != 0:
            failed.append(label)
    incomplete = [b for b in remaining if not is_complete(run_dir, by_block[b], timing_on)]
    if incomplete:
        print("NOT COMPLETE after this pass (rerun the same command to retry): %s"
              % ", ".join(incomplete))
    if failed or incomplete:
        print("FAILED: %s (see %s)" % (", ".join(failed) or "-", log_dir))
        return 1
    print("done. Summarise with: python scripts/summarize_timing.py %s" % run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
