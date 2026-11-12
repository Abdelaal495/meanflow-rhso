#!/usr/bin/env python
"""Matched-budget RHSO vs MPC-delta_t: the configurations for rebuttal Task 2.

Both methods are compared at EQUAL backpropagations through the prior per image (BP):

    RHSO          BP = N * M         (MeanFlow map, or `rhso_terminal_mode: direct` on JiT/SiT)
    MPC-delta_t   BP = (N - 1) * C   (its last replan does not differentiate the prior)

Direction A runs RHSO at the BP of each cell's frozen MPC-delta_t; direction B runs
MPC-delta_t at the BP of the frozen RHSO (pMF; iMF at 640 BP with --include-optional).  The
targets are read from configs/experiments_final_frozen100.yaml, so they are each prior's own
frozen configuration and nothing is carried across priors.

`src/config.py` expands list-valued fields as a Cartesian product, so `num_rhso_steps: [4, 8]`
with `num_opt_steps: [70, 35]` would also produce the off-budget (4, 35) and (8, 70).  Every
(N, M) or (N, C) pair therefore gets its OWN experiment block, named for its budget and N:
`deblurring__rhso_bp280_n4`, `denoising__mpc_delta_t_bp160_n5`.  Only the tuned knobs (RHSO
lr and mu, MPC-delta_t lam and lr) are lists, over the pre-registered values of
configs/experiments_hpo_grid.yaml.

    # 1. the screening grid (4 images) and its 2-image smoke variant
    python scripts/make_matched_configs.py screen
    python run.py --config configs/experiments_matched_screen_smoke.yaml --models pmf --run-id matched_smoke_pmf
    python run.py --config configs/experiments_matched_screen.yaml --models pmf --run-id matched_pmf --no-figures

    # 2. a refine round at 8 images that cannot leave the budget
    python scripts/hpo.py propose --from outputs/matched_pmf --models pmf --images 8 --no-paper \
        --lock rhso=num_rhso_steps,num_opt_steps --lock mpc_delta_t=num_mpc_steps,n_ctrl \
        --out configs/matched_round2_pmf.yaml
    python run.py --config configs/matched_round2_pmf.yaml --run-id matched_pmf_r2 --no-figures

    # 3. freeze: lowest LPIPS at 8 images inside the screened grid -> the 100-image evaluation
    python scripts/make_matched_configs.py freeze --from outputs/matched_pmf_r2 \
        --grid outputs/matched_pmf --models pmf

`freeze` is used instead of `hpo.py finalize`, which expects all six methods in every cell.
It applies the same rule (hpo.restrict + hpo.winners) and then refuses any winner whose
budget-defining pair or terminal planner differs from what its cell was screened with.
"""
import argparse
import collections
import importlib.util
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_hpo():
    spec = importlib.util.spec_from_file_location("hpo", REPO_ROOT / "scripts" / "hpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hpo = _load_hpo()

MODELS = ["jit", "sit", "pmf", "imf"]
STANDARD_FLOW = {"jit", "sit"}
METHODS = ["rhso", "mpc_delta_t"]
# The exact pairs per budget (rebuttal plan, Task 2).  Never a Cartesian product.
RHSO_PAIRS = {120: [(4, 30), (8, 15)], 140: [(4, 35), (7, 20)], 280: [(4, 70), (8, 35)]}
MPC_PAIRS = {160: [(5, 40), (9, 20)], 640: [(9, 80), (17, 40)]}
# direction B: MPC-delta_t at the frozen RHSO budget, for these priors only
DIRECTION_B = ["pmf"]
DIRECTION_B_OPTIONAL = ["imf"]

BENCHMARK = "configs/experiments_final_frozen100.yaml"
GRID = "configs/experiments_hpo_grid.yaml"
SCREEN_OUT = "configs/experiments_matched_screen.yaml"
FROZEN_OUT = "configs/experiments_matched_frozen100.yaml"
SCREEN_IMAGES, SMOKE_IMAGES, FROZEN_IMAGES = 4, 2, 100
TUNING_POOL = "cache/data/imagenet_val_100"
FROZEN_POOL = "cache/data/imagenet100_c42_i43_mirror"


def rhso_bp(n, m):
    return n * m


def mpc_bp(n, c):
    return (n - 1) * c


def smoke_path(path):
    p = Path(path)
    return p.with_name(p.stem + "_smoke" + p.suffix).as_posix()


# =====================================================================================
# targets and grids, read from the committed configurations
# =====================================================================================
def _entry(cfg, exp, model, method):
    try:
        return cfg["experiments"][exp]["models"][model]["methods"][method]
    except (KeyError, TypeError):
        raise SystemExit("%s has no %s/%s entry in experiment %s" % (cfg["_path"], model, method, exp))


def load_yaml(path):
    cfg = yaml.safe_load(open(REPO_ROOT / path))
    cfg["_path"] = path
    return cfg


def budget_targets(benchmark, include_optional=False):
    """[(model, problem, method, target BP, pairs)] -- every cell this task runs.

    Direction A: RHSO at the frozen MPC-delta_t budget of the SAME prior and task.
    Direction B: MPC-delta_t at the frozen RHSO budget of the SAME prior and task.
    """
    cells = []
    b_models = DIRECTION_B + (DIRECTION_B_OPTIONAL if include_optional else [])
    for problem in hpo.PROBLEMS:
        exp = hpo.EXP_NAME[problem]
        for model in MODELS:
            mpc = _entry(benchmark, exp + "__core", model, "mpc_delta_t")
            target = mpc_bp(int(mpc["num_mpc_steps"]), int(mpc["n_ctrl"]))
            if target not in RHSO_PAIRS:
                raise SystemExit("no RHSO (N, M) pairs for %d BP (%s/%s)" % (target, model, problem))
            cells.append((model, problem, "rhso", target, RHSO_PAIRS[target]))
            if model in b_models:
                rhso = _entry(benchmark, exp + "__rhso", model, "rhso")
                if model in STANDARD_FLOW:
                    raise SystemExit("direction B needs a MeanFlow RHSO budget (N*M)")
                target = rhso_bp(int(rhso["num_rhso_steps"]), int(rhso["num_opt_steps"]))
                if target not in MPC_PAIRS:
                    raise SystemExit("no MPC (N, C) pairs for %d BP (%s/%s)" % (target, model, problem))
                cells.append((model, problem, "mpc_delta_t", target, MPC_PAIRS[target]))
    return cells


def tuned_axes(grid, model, problem, method):
    """The pre-registered lists of the knobs this task tunes, and the fixed fields kept."""
    exp = hpo.EXP_NAME[problem] + "__" + method
    entry = _entry(grid, exp, model, method)
    tuned = ("lr", "mu") if method == "rhso" else ("lam", "lr")
    axes = {}
    for knob in tuned:
        values = entry[knob]
        axes[knob] = list(values) if isinstance(values, list) else [values]
    fixed = {"solver": entry["solver"]} if "solver" in entry else {}
    return axes, fixed


# =====================================================================================
# YAML writing
# =====================================================================================
def num(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return "%g" % v


def value(v):
    return "[%s]" % ", ".join(num(x) for x in v) if isinstance(v, list) else (
        v if isinstance(v, str) else num(v))


def method_line(method, fields, comment):
    body = ", ".join("%s: %s" % (k, value(v)) for k, v in fields)
    return "          %-12s {%s}   # %s" % (method + ":", body, comment)


def pair_fields(model, method, pair, fixed):
    """t0, the fixed fields, the planner and the budget-defining pair, in that order."""
    fields = [("t0", "1.0")] + sorted(fixed.items())
    if method == "rhso":
        if model in STANDARD_FLOW:
            # direct endpoint prediction: one evaluation per inner objective, so BP = N*M
            fields.append(("rhso_terminal_mode", "direct"))
        fields += [("num_rhso_steps", pair[0]), ("num_opt_steps", pair[1])]
    else:
        fields += [("num_mpc_steps", pair[0]), ("n_ctrl", pair[1])]
    return fields


def bp_of(method, pair):
    return rhso_bp(*pair) if method == "rhso" else mpc_bp(*pair)


def header(title, usage, description, max_jobs, pool, models, save_images):
    return """# =====================================================================================
# %s
# =====================================================================================
#
%s#
%s
# =====================================================================================

runtime:
  seed: 42
  accelerator: auto
  output_root: outputs
  cache_root: cache
  release_model_after_use: true
  continue_on_experiment_error: true
  progress_bars: false
  save_individual_images: %s
  verbose: true
  max_atomic_jobs: %d
  resume: true
  replicate: 0
  gpu_memory_profiling: true
  nvml_sample_interval: 0.02

data:
  # Written by scripts/build_local_imagenet_pool.py; point it at your copy.
  source: local_folder
  local_folder: "%s"
  image_size: 256

models:
%s
defaults:
  control_cost_normalization: sum_squared
  optimizer: adam
  warm_start: false
  grad_clip: null
  record_loss_history: false

metrics:
  lpips: true
  lpips_net: alex

experiments:
""" % (title, usage, "\n".join("# " + line if line else "#" for line in description.split("\n")),
       "true" if save_images else "false", max_jobs, pool, hpo.model_block(models))


def max_jobs_for(n_jobs):
    """The sweep guard, set just above the file's own total so no --max-jobs is needed."""
    return int(math.ceil(n_jobs * 1.1 / 64.0) * 64)


def experiment_block(name, problem, num_images, entries):
    lines = ["", "  %s:" % name, "    enabled: true", "    problem: %s" % problem,
             "    num_images: %d" % num_images,
             "    degradation: %s" % hpo.DEGRADATION[problem], "    models:"]
    for model, batch, method, fields, comment in entries:
        lines += ["      %s:" % model, "        batch_size: %d" % batch, "        methods:",
                  method_line(method, fields, comment)]
    return lines


# =====================================================================================
# screen
# =====================================================================================
def screen_blocks(cells, grid, smoke):
    """One block per (task, method, BP, N); returns (lines, number of atomic jobs)."""
    groups = collections.OrderedDict()
    for model, problem, method, target, pairs in cells:
        for pair in pairs:
            assert bp_of(method, pair) == target, (method, pair, target)
            name = "%s__%s_bp%d_n%d" % (hpo.EXP_NAME[problem], method, target, pair[0])
            groups.setdefault(name, (problem, []))[1].append((model, method, pair))
    lines, n_jobs = [], 0
    order = {m: i for i, m in enumerate(MODELS)}
    for name, (problem, members) in groups.items():
        entries = []
        for model, method, pair in sorted(members, key=lambda e: order[e[0]]):
            axes, fixed = tuned_axes(grid, model, problem, method)
            if smoke:
                axes = {k: v[0] for k, v in axes.items()}
            fields = pair_fields(model, method, pair, fixed) + list(axes.items())
            jobs = 1
            for v in axes.values():
                jobs *= len(v) if isinstance(v, list) else 1
            n_jobs += jobs
            entries.append((model, hpo.BATCH[(model, method)], method, fields,
                            "BP %d = %s; %d job%s" % (
                                bp_of(method, pair),
                                ("%d*%d" % pair) if method == "rhso" else ("(%d-1)*%d" % pair),
                                jobs, "" if jobs == 1 else "s")))
        lines += experiment_block(name, problem, SMOKE_IMAGES if smoke else SCREEN_IMAGES, entries)
    return lines, n_jobs


SCREEN_TEXT = """Screening grid for the matched-budget comparison of RHSO and MPC-delta_t.  Every
block holds ONE budget-defining pair -- (N, M) for RHSO, (N, C) for MPC-delta_t -- and its
name gives the backprops per image it was built for.  The target of each (prior, task) cell
is that prior's own frozen MPC-delta_t budget in %s (direction A), or
its own frozen RHSO budget (direction B, MPC-delta_t%s).  RHSO
tunes lr and mu, MPC-delta_t tunes lam and lr, over the pre-registered values in
%s.  JiT and SiT plan with `rhso_terminal_mode: direct`, so RHSO's
BP is N*M on every prior.  Selection: lowest LPIPS on 8 tuning images inside this grid
(scripts/hpo.py propose --lock ..., then scripts/make_matched_configs.py freeze).
Generated by scripts/make_matched_configs.py screen%s; %d atomic jobs over all four priors."""


def screen(a):
    benchmark, grid = load_yaml(a.benchmark), load_yaml(a.grid)
    cells = budget_targets(benchmark, a.include_optional)
    b_models = DIRECTION_B + (DIRECTION_B_OPTIONAL if a.include_optional else [])
    for smoke in (False, True):
        out = smoke_path(a.out) if smoke else Path(a.out).as_posix()
        lines, n_jobs = screen_blocks(cells, grid, smoke)
        title = ("SMOKE: matched-budget RHSO vs MPC-delta_t, every block at 2 images"
                 if smoke else "MATCHED-BUDGET SCREEN: RHSO vs MPC-delta_t at %d images" % SCREEN_IMAGES)
        usage = ("#   python run.py --config %s --models <prior> --run-id matched_<prior> --no-figures\n"
                 "#   one prior per job: every block carries per-model entries, and --models drops the rest\n"
                 % out)
        text = SCREEN_TEXT % (a.benchmark, " on %s" % "/".join(b_models), a.grid,
                              " --include-optional" if a.include_optional else "", n_jobs)
        if smoke:
            text += ("\nSmoke variant: the first value of every tuned list, so each block "
                     "runs one job;\nit checks that every pair resolves and runs, not which "
                     "value wins.")
        body = header(title, usage, text, max_jobs_for(n_jobs), TUNING_POOL, MODELS, False)
        Path(REPO_ROOT / out).write_text(body + "\n".join(lines) + "\n")
        print("wrote %s: %d jobs" % (out, n_jobs))


# =====================================================================================
# freeze
# =====================================================================================
def freeze(a):
    models = [m for m in a.models.split(",") if m]
    benchmark = load_yaml(a.benchmark)
    cells = [c for c in budget_targets(benchmark, a.include_optional) if c[0] in models]
    ranked, _, _ = hpo.load(getattr(a, "from"))
    grids = hpo.load(a.grid)[1]
    best = hpo.winners(hpo.restrict(ranked, models, grids))

    blocks = collections.OrderedDict()
    for model, problem, method, target, pairs in cells:
        cell = (model, problem, method)
        if cell not in best:
            raise SystemExit("no winner for %s inside the screened grid" % "/".join(cell))
        key, entry, n_candidates = best[cell]
        knobs = dict(zip(hpo.KNOBS[method], key))
        pair = ((int(knobs["num_rhso_steps"]), int(knobs["num_opt_steps"])) if method == "rhso"
                else (int(knobs["num_mpc_steps"]), int(knobs["n_ctrl"])))
        if pair not in pairs:
            raise SystemExit("%s: winner %s is not one of the %d-BP pairs %s"
                             % ("/".join(cell), pair, target, pairs))
        mode = entry["row"].get("rhso_terminal_mode") or None
        if method == "rhso" and model in STANDARD_FLOW and mode != "direct":
            raise SystemExit("%s: winner ran with rhso_terminal_mode=%s, not direct"
                             % ("/".join(cell), mode))
        _, fixed = tuned_axes(load_yaml(a.hpo_grid), model, problem, method)
        tuned = [("lr", knobs["lr"]), ("mu", knobs["mu"])] if method == "rhso" else \
            [("lam", knobs["lam"]), ("lr", knobs["lr"])]
        fields = pair_fields(model, method, pair, fixed) + [
            (k, float(v)) for k, v in tuned]
        comment = ("BP %d; LPIPS %.4f PSNR %.2f at %d tuning images, best of %d"
                   % (target, entry["lpips"], entry["psnr"] or 0.0, entry["n"], n_candidates))
        name = "%s__%s_matched" % (hpo.EXP_NAME[problem], method)
        blocks.setdefault(name, (problem, []))[1].append((model, method, fields, comment))

    stems = ", ".join(sorted(Path(r).name for r in getattr(a, "from")))
    for smoke in (False, True):
        out = smoke_path(a.out) if smoke else Path(a.out).as_posix()
        n_images = SMOKE_IMAGES if smoke else a.num_images
        lines = []
        for name, (problem, members) in blocks.items():
            entries = []
            for model, method, fields, comment in members:
                group = "rhso" if method == "rhso" else "core"
                batch = hpo.BATCH_FINAL[model][group]
                entries.append((model, min(batch, n_images), method, fields, comment))
            lines += experiment_block(name, problem, n_images, entries)
        n_jobs = sum(len(m) for _, m in blocks.values())
        title = ("SMOKE: matched-budget frozen configuration at 2 images" if smoke else
                 "MATCHED-BUDGET FINAL at t0 = 1.0 on the frozen ImageNet-100 (class seed 42 / image seed 43)")
        usage = ("#   python scripts/build_local_imagenet_pool.py --frozen-manifest "
                 "benchmarks/imagenet100_c42_i43/manifest.csv\n"
                 "#   python run.py --config %s --models <prior> --run-id <id> --no-figures\n" % out)
        text = ("Each entry is the lowest-LPIPS configuration of its (prior, task, method) cell among\n"
                "the candidates inside that prior's own screened grid, at the cell's own target\n"
                "backprops (the trailing comment).  Pair it with the benchmark run of %s\n"
                "in scripts/paired_stats.py.\n"
                "Generated by scripts/make_matched_configs.py freeze from %s."
                % (a.benchmark, stems))
        body = header(title, usage, text, max_jobs_for(n_jobs), FROZEN_POOL, models, not smoke)
        Path(REPO_ROOT / out).write_text(body + "\n".join(lines) + "\n")
        print("wrote %s: %d jobs at %d images" % (out, n_jobs, n_images))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, out):
        q.add_argument("--benchmark", default=BENCHMARK,
                       help="the frozen benchmark whose budgets are matched (default: %(default)s)")
        q.add_argument("--include-optional", action="store_true",
                       help="also run MPC-delta_t at iMF's 640-BP RHSO budget (direction B)")
        q.add_argument("--out", default=out, help="output path; the smoke variant is <out>_smoke.yaml")

    q = sub.add_parser("screen", help="write the screening grid and its smoke variant")
    common(q, SCREEN_OUT)
    q.add_argument("--grid", default=GRID, help="the pre-registered grid (default: %(default)s)")
    q.set_defaults(run=screen)

    q = sub.add_parser("freeze", help="write the frozen 100-image configuration and its smoke variant")
    common(q, FROZEN_OUT)
    q.add_argument("--from", dest="from", action="append", required=True, metavar="RUN",
                   help="run directory whose results.csv is ranked (repeatable)")
    q.add_argument("--grid", action="append", required=True, metavar="RUN",
                   help="the screening run directories: a winner must lie inside their grid")
    q.add_argument("--hpo-grid", default=GRID, help=argparse.SUPPRESS)
    q.add_argument("--models", default=",".join(MODELS))
    q.add_argument("--num-images", type=int, default=FROZEN_IMAGES)
    q.set_defaults(run=freeze)

    a = p.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
