#!/usr/bin/env python
"""Hyperparameter search for every (model, problem, method) cell.

One round at a time: run a configuration, `propose` the next one from its results, repeat
until no cell wants to move, then `finalize` the winners into the benchmark configuration.

    # round 1 -- the pre-registered grid
    python run.py --config configs/experiments_hpo_grid.yaml --models pmf --run-id hpo_pmf --no-figures

    # round 2 -- the top three of each cell, the manuscript's value, and one step past any
    #            axis whose winner sat on the edge of the grid, re-run on more images
    python scripts/hpo.py propose --from outputs/hpo_pmf --models pmf --images 8 --out configs/round2.yaml
    python run.py --config configs/round2.yaml --run-id hpo_pmf_r2 --no-figures

    # round 3+ -- only the cells still sitting on an edge of everything tried so far
    python scripts/hpo.py propose --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf outputs/hpo_pmf_r2 \
        --top 0 --no-paper --models pmf --images 8 --out configs/round3.yaml
    # `propose` prints "no cell needs another push" and writes nothing when the search is done

    # the benchmark configuration
    python scripts/hpo.py finalize --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf \
        --models pmf --out configs/experiments_final_frozen100.yaml

`--grid` is the union of every value tried so far and decides what counts as a grid edge;
`--from` is what the ranking reads, and defaults `--grid` to itself.  Restricting `finalize`
to a `--grid` keeps the winner inside the pre-registered ranges: on several axes (D-Flow and
PnP step counts, RHSO N and M) LPIPS keeps improving with compute far past the grid, which is
a compute-budget effect rather than a hyperparameter optimum.

For a matched-budget search, `--lock` keeps the budget-defining knobs where the grid put
them, so an edge push can never change a candidate's backprop count (see
scripts/make_matched_configs.py):

    python scripts/hpo.py propose --from outputs/matched_pmf --models pmf --images 8 \
        --no-paper --lock rhso=num_rhso_steps,num_opt_steps \
        --lock mpc_delta_t=num_mpc_steps,n_ctrl --out configs/matched_round2.yaml

An RHSO cell run with a non-default `rhso_terminal_mode` (JiT/SiT `direct`) carries that mode
into every candidate it produces, so a round never swaps the planner, and with it the cost.

Flower (rebuttal Task 3) is ranked and proposed like any other method, but it is not one of
the six benchmark methods `finalize` requires in every cell; its winners are frozen by
scripts/make_flower_configs.py freeze instead:

    python scripts/hpo.py propose --from outputs/flower_pmf --models pmf --images 8         --no-paper --lock flower=num_flower_steps --out configs/flower_round2_pmf.yaml
"""
import argparse
import collections
import csv
import math
from pathlib import Path

PROBLEMS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
EXP_NAME = {"denoising": "denoising", "deblur": "deblurring",
            "super_resolution": "super_resolution", "random_inpaint": "random_inpainting",
            "box_inpaint": "box_inpainting"}
DEGRADATION = {"denoising": "{sigma: 0.20}",
               "deblur": "{sigma: 0.05, blur_sigma: 1.0, kernel_size: 7, padding: reflect}",
               "super_resolution": "{sigma: 0.05, factor: 2}",
               "random_inpaint": "{sigma: 0.01, missing_fraction: 0.70}",
               "box_inpaint": "{sigma: 0.05, box: 40}"}
METHODS = ["sdedit", "pnp", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]
# Ranked and proposed like METHODS, but never required by `finalize` (see the docstring).
EXTRA_METHODS = ["flower"]
SEARCHED = METHODS + EXTRA_METHODS

# The knobs each method sweeps (CSV column names), and which are log-like (extend by the
# grid ratio) versus linear (extend by the grid step) when a winner sits on an edge.
KNOBS = {
    "sdedit":      ["steps"],
    "pnp":         ["num_pnp_steps", "gamma0", "alpha"],
    "dflow":       ["num_opt_steps", "lr"],
    "mpc_rhc":     ["K", "lam", "lr"],
    "mpc_delta_t": ["num_mpc_steps", "n_ctrl", "lam", "lr"],
    "rhso":        ["num_rhso_steps", "num_opt_steps", "lr", "mu"],
    "flower":      ["num_flower_steps"],
}
# counts and dimensions double rather than step: an extension of n_ctrl = 20 by the grid's
# linear step would propose 0, which is not a control dimension
LOG_KNOBS = {"gamma0", "lam", "lr", "num_pnp_steps", "num_opt_steps", "steps",
             "n_ctrl", "num_mpc_steps", "num_rhso_steps", "num_flower_steps"}
INTEGER = {"steps", "num_pnp_steps", "num_opt_steps", "K", "num_mpc_steps", "n_ctrl", "num_rhso_steps",
           "num_flower_steps"}
# hard limits a knob cannot be pushed past
BOUNDS = {"alpha": (0.0, 1.0), "K": (1, 4), "mu": (0.0, None), "steps": (1, None),
          "n_ctrl": (10, None), "num_mpc_steps": (2, None), "num_rhso_steps": (1, None),
          "num_pnp_steps": (5, None), "num_flower_steps": (1, None)}
# what `rhso_terminal_mode: auto` resolves to per model (src/rhso.py legacy_terminal_mode);
# only a mode that differs from this is written into a generated configuration
LEGACY_TERMINAL_MODE = {"jit": "suffix", "sit": "suffix", "pmf": "direct", "imf": "direct"}
# the YAML key each CSV column maps to (identical here, listed for clarity)
YAML_KEY = {k: k for m in KNOBS for k in KNOBS[m]}
# fixed fields every candidate carries, per method and model family
FIXED = {
    "sdedit":      {"jit": "solver: heun", "sit": "solver: heun", "pmf": "", "imf": ""},
    "pnp":         {m: "" for m in ("jit", "sit", "pmf", "imf")},
    "dflow":       {"jit": "steps: 1, solver: euler", "sit": "steps: 1, solver: euler", "pmf": "steps: 1", "imf": "steps: 1"},
    "mpc_rhc":     {m: "num_mpc_steps: 4, n_ctrl: 40" for m in ("jit", "sit", "pmf", "imf")},
    "mpc_delta_t": {m: "" for m in ("jit", "sit", "pmf", "imf")},
    "rhso":        {"jit": "solver: heun", "sit": "solver: heun", "pmf": "", "imf": ""},
    "flower":      {m: "" for m in ("jit", "sit", "pmf", "imf")},
}
# batch_size per (model, method) for a screening round
BATCH = {("jit", "sdedit"): 4, ("jit", "pnp"): 4, ("jit", "dflow"): 4, ("jit", "mpc_rhc"): 4,
         ("jit", "mpc_delta_t"): 4, ("jit", "rhso"): 4,
         ("sit", "sdedit"): 4, ("sit", "pnp"): 4, ("sit", "dflow"): 4, ("sit", "mpc_rhc"): 4,
         ("sit", "mpc_delta_t"): 4, ("sit", "rhso"): 2,
         ("pmf", "sdedit"): 4, ("pmf", "pnp"): 4, ("pmf", "dflow"): 4, ("pmf", "mpc_rhc"): 2,
         ("pmf", "mpc_delta_t"): 4, ("pmf", "rhso"): 4,
         ("imf", "sdedit"): 2, ("imf", "pnp"): 2, ("imf", "dflow"): 2, ("imf", "mpc_rhc"): 2,
         ("imf", "mpc_delta_t"): 2, ("imf", "rhso"): 2,
         # Flower: pixel priors only, no backprop through the model
         ("jit", "flower"): 4, ("pmf", "flower"): 4}
# standard-flow RHSO memory grows with N, so long-N candidates use smaller batches
BATCH_RHSO_LONG = {"jit": 2, "sit": 1, "pmf": 4, "imf": 2}

# The manuscript's hyperparameter tables, per problem in PROBLEMS order.  Only JiT and pMF are tabulated.
PAPER = {
    ("pmf", "pnp"): [dict(num_pnp_steps=50, gamma0=8e5, alpha=.75), dict(num_pnp_steps=100, gamma0=1.2e6, alpha=.50),
                     dict(num_pnp_steps=20, gamma0=1e5, alpha=.25), dict(num_pnp_steps=20, gamma0=1e5, alpha=.25),
                     dict(num_pnp_steps=100, gamma0=4e5, alpha=.25)],
    ("jit", "pnp"): [dict(num_pnp_steps=50, gamma0=2e5, alpha=.50), dict(num_pnp_steps=100, gamma0=1.2e6, alpha=.50),
                     dict(num_pnp_steps=20, gamma0=1e5, alpha=.25), dict(num_pnp_steps=20, gamma0=2e5, alpha=.25),
                     dict(num_pnp_steps=100, gamma0=4e5, alpha=.25)],
    ("pmf", "dflow"): [dict(num_opt_steps=320, lr=.03)] * 5,
    ("jit", "dflow"): [dict(num_opt_steps=320, lr=.03)] * 5,
    ("pmf", "mpc_rhc"): [dict(K=2, lam=15, lr=.10), dict(K=1, lam=60, lr=.05), dict(K=3, lam=240, lr=.05),
                         dict(K=3, lam=180, lr=.05), dict(K=1, lam=30, lr=.05)],
    ("jit", "mpc_rhc"): [dict(K=2, lam=15, lr=.05), dict(K=1, lam=60, lr=.10), dict(K=3, lam=240, lr=.05),
                         dict(K=3, lam=180, lr=.05), dict(K=1, lam=30, lr=.05)],
    ("pmf", "mpc_delta_t"): [dict(num_mpc_steps=8, n_ctrl=40, lam=30, lr=.10), dict(num_mpc_steps=8, n_ctrl=40, lam=540, lr=.10),
                             dict(num_mpc_steps=4, n_ctrl=20, lam=360, lr=.10), dict(num_mpc_steps=4, n_ctrl=20, lam=540, lr=.15),
                             dict(num_mpc_steps=8, n_ctrl=40, lam=180, lr=.10)],
    ("jit", "mpc_delta_t"): [dict(num_mpc_steps=8, n_ctrl=40, lam=30, lr=.10), dict(num_mpc_steps=8, n_ctrl=40, lam=540, lr=.10),
                             dict(num_mpc_steps=4, n_ctrl=20, lam=360, lr=.10), dict(num_mpc_steps=4, n_ctrl=20, lam=540, lr=.10),
                             dict(num_mpc_steps=8, n_ctrl=40, lam=180, lr=.10)],
    ("pmf", "rhso"): [dict(num_rhso_steps=4, num_opt_steps=40, lr=.03, mu=0.0), dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0),
                      dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0), dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0),
                      dict(num_rhso_steps=4, num_opt_steps=40, lr=.02, mu=0.0)],
    ("jit", "rhso"): [dict(num_rhso_steps=4, num_opt_steps=40, lr=.01, mu=0.0)] * 5,
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def key_of(row, method):
    return tuple(fnum(row.get(k)) for k in KNOBS[method])


def same(a, b):
    return all((x is None and y is None) or (x is not None and y is not None and
               abs(x - y) <= 1e-6 * max(1.0, abs(y))) for x, y in zip(a, b))


def fmt(v, knob):
    if v is None:
        return "null"
    if knob in INTEGER:
        return str(int(round(v)))
    return ("%g" % v) if abs(v) < 1e4 else ("%.1f" % v)


def load(runs, per_image=False):
    """cell -> {config key -> dict(lpips, psnr, n, row)}, the grid each (model, method) covers,
    and, optionally, the per-image LPIPS of every job.

    A configuration re-run in several rounds keeps its BEST row, so the result does not depend
    on the order the run directories are given in.
    """
    cells = collections.defaultdict(dict)
    grids = collections.defaultdict(lambda: collections.defaultdict(set))
    images = collections.defaultdict(list)
    for run in runs:
        for r in csv.DictReader(open(Path(run) / "results.csv")):
            if r["problem"] not in PROBLEMS or r["method"] not in SEARCHED:
                continue
            if r.get("status", "ok") != "ok":
                continue
            lp = fnum(r["lpips"])
            if lp is None:
                continue
            k = key_of(r, r["method"])
            cell = (r["model"], r["problem"], r["method"])
            if k not in cells[cell] or lp < cells[cell][k]["lpips"]:
                cells[cell][k] = dict(lpips=lp, psnr=fnum(r["psnr"]), n=int(r["num_images"]), row=r)
            for knob, v in zip(KNOBS[r["method"]], k):
                if v is not None:
                    grids[(r["model"], r["method"])][knob].add(v)
        if per_image:
            path = Path(run) / "results_per_image.csv"
            if path.exists():
                for r in csv.DictReader(open(path)):
                    lp = fnum(r.get("lpips"))
                    if lp is not None:
                        images[r["job_id"]].append(lp)
    return cells, grids, images


def extend(value, knob, grid, direction):
    """One grid step beyond `value` in `direction` (+1 / -1), or None if not allowed."""
    g = sorted(grid)
    if len(g) < 2:
        return None
    if knob in LOG_KNOBS:
        ratio = (g[-1] / g[0]) ** (1.0 / (len(g) - 1)) if g[0] > 0 else 2.0
        ratio = max(ratio, 1.5)
        new = value * ratio if direction > 0 else value / ratio
    else:
        step = (g[-1] - g[0]) / (len(g) - 1)
        new = value + direction * step
    lo, hi = BOUNDS.get(knob, (None, None))
    if lo is not None and new < lo - 1e-9:
        return None
    if hi is not None and new > hi + 1e-9:
        return None
    if knob in INTEGER:
        new = max(1, int(round(new)))
        if any(abs(new - x) < 0.5 for x in g):
            return None
        return new
    # two significant figures: 0.545136 is not a learning rate anyone would write down
    return float("%.2g" % new)


def edge_pushes(win_key, method, grid, locked=()):
    """Every (new key, why) that moves the winner one step off an edge of `grid`.

    A knob in `locked` is never moved: a winner on its edge is reported, not extended.
    """
    out = []
    for i, knob in enumerate(KNOBS[method]):
        v = win_key[i]
        axis = grid.get(knob, set())
        if v is None or len(axis) < 2:
            continue
        g = sorted(axis)
        if knob in locked:
            if any(abs(v - edge) <= 1e-9 * max(1, abs(edge)) for edge in (g[0], g[-1])):
                out.append((None, "%s=%s locked" % (knob, fmt(v, knob))))
            continue
        for direction, edge in ((-1, g[0]), (+1, g[-1])):
            if abs(v - edge) <= 1e-9 * max(1, abs(edge)):
                new = extend(v, knob, axis, direction)
                if new is None:
                    out.append((None, "%s=%s hard bound" % (knob, fmt(v, knob))))
                else:
                    k2 = list(win_key)
                    k2[i] = new
                    out.append((tuple(k2), "edge:%s%s" % (knob, "+" if direction > 0 else "-")))
    return out


def paper_key(model, method, problem):
    """The manuscript's configuration for a cell, as a knob tuple, or None."""
    table = PAPER.get((model, method))
    if not table:
        return None
    values = table[PROBLEMS.index(problem)]
    return tuple(float(values[k]) if k in values else None for k in KNOBS[method])


def batch_for(model, method, key):
    if method == "rhso" and (key[0] or 0) >= 8:
        return BATCH_RHSO_LONG[model]
    return BATCH[(model, method)]


def method_fields(method, model, key, terminal_mode=None):
    fields = ["t0: 1.0"] + ([FIXED[method][model]] if FIXED[method][model] else [])
    if method == "rhso" and terminal_mode and terminal_mode != LEGACY_TERMINAL_MODE[model]:
        fields.append("rhso_terminal_mode: %s" % terminal_mode)
    return fields + ["%s: %s" % (YAML_KEY[kn], fmt(v, kn))
                     for kn, v in zip(KNOBS[method], key) if v is not None]


def cell_terminal_mode(table, cell):
    """The one `rhso_terminal_mode` every row of an RHSO cell ran with (None if unrecorded).

    Two planners in one cell are two cost models ranked against each other, so that is
    refused rather than resolved by picking one.
    """
    if cell[2] != "rhso":
        return None
    modes = {e["row"].get("rhso_terminal_mode") or None for e in table.values()}
    if len(modes) > 1:
        raise SystemExit("cell %s mixes rhso_terminal_mode values %s; rank them separately"
                         % ("/".join(cell), sorted(m or "unset" for m in modes)))
    return modes.pop() if modes else None


def parse_locks(specs):
    """['rhso=num_rhso_steps,num_opt_steps', ...] -> {method: {knob, ...}}."""
    locks = collections.defaultdict(set)
    for spec in specs or []:
        method, sep, knobs = spec.partition("=")
        if not sep or method not in KNOBS:
            raise SystemExit("--lock %r: expected METHOD=KNOB[,KNOB] with METHOD one of %s"
                             % (spec, ", ".join(sorted(KNOBS))))
        for knob in filter(None, knobs.split(",")):
            if knob not in KNOBS[method]:
                raise SystemExit("--lock %r: %r is not a searched knob of %s (%s)"
                                 % (spec, knob, method, ", ".join(KNOBS[method])))
            locks[method].add(knob)
    return locks


HEADER = """# =====================================================================================
# %s
# =====================================================================================
#
%s#
# %s
# Generated by scripts/hpo.py %s from %s.
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
"""


def model_block(models):
    return "".join("  %s:%s\n" % (m, "\n    variant: JiT-B/16" if m == "jit" else " {}")
                   for m in models)


# =====================================================================================
# propose -- the next round of candidates
# =====================================================================================
def propose(a):
    models = [m for m in a.models.split(",") if m]
    locks = parse_locks(getattr(a, "lock", None))
    rank, _, _ = load(getattr(a, "from"))
    _, grids, _ = load(a.grid or getattr(a, "from"))
    candidates = collections.defaultdict(list)          # (problem, method) -> [(model, key, why)]
    modes = {}                                          # (model, problem, method) -> terminal mode
    print("%-4s %-17s %-12s %6s  %-44s %8s %s"
          % ("mdl", "problem", "method", "n_cfg", "winner", "LPIPS", "notes"))
    for model in models:
        for problem in PROBLEMS:
            for method in SEARCHED:
                table = rank.get((model, problem, method))
                if not table:
                    continue
                modes[(model, problem, method)] = cell_terminal_mode(
                    table, (model, problem, method))
                ranked = sorted(table.items(), key=lambda kv: kv[1]["lpips"])
                win_key, win = ranked[0]
                chosen = [(k, "top%d" % (i + 1)) for i, (k, _) in enumerate(ranked[:a.top])]
                notes = []
                if a.paper:
                    pkey = paper_key(model, method, problem)
                    if pkey is not None:
                        hit = [i for i, (k, _) in enumerate(ranked) if same(k, pkey)]
                        if hit:
                            notes.append("paper #%d (+%.4f)"
                                         % (hit[0] + 1, ranked[hit[0]][1]["lpips"] - win["lpips"]))
                            if hit[0] + 1 > a.top:
                                chosen.append((ranked[hit[0]][0], "paper"))
                        else:
                            notes.append("paper cfg NOT in sweep")
                for key, why in edge_pushes(win_key, method, grids[(model, method)],
                                            locks.get(method, ())):
                    if key is None:
                        notes.append(why)
                    elif not any(same(key, k) for k in table):
                        chosen.append((key, why))
                        notes.append(why)
                desc = "  ".join("%s=%s" % (k, fmt(v, k)) for k, v in zip(KNOBS[method], win_key))
                print("%-4s %-17s %-12s %6d  %-44s %8.4f %s"
                      % (model, problem, method, len(table), desc, win["lpips"], "; ".join(notes)))
                for key, why in chosen:
                    candidates[(problem, method)].append((model, key, why))
    if not candidates:
        print("\nno cell needs another push: every winner has a worse neighbour on both sides "
              "(or a hard bound), so the search is finished")
        return

    lines = [HEADER % (
        "CANDIDATE ROUND at %d images" % a.images,
        "#   python run.py --config %s --models <model> --run-id <id> --no-figures\n" % a.out,
        "Each entry is a candidate this round should settle: the best of its cell so far, the\n"
        "# manuscript's configuration where the cell has one, or one step past a grid edge the\n"
        "# winner sat on.  The trailing comment on each line says which.",
        "propose", ", ".join(getattr(a, "from")), "true", 4000, a.pool, model_block(models))]
    n_jobs = 0
    for problem in PROBLEMS:
        for method in SEARCHED:
            per_model = collections.defaultdict(list)
            for model, key, why in candidates.get((problem, method), []):
                if not any(same(key, k) for k, _ in per_model[model]):
                    per_model[model].append((key, why))
            if not per_model:
                continue
            for i in range(max(len(v) for v in per_model.values())):
                lines.append("\n  %s__%s__c%d:" % (EXP_NAME[problem], method, i + 1))
                lines.append("    enabled: true")
                lines.append("    problem: %s" % problem)
                lines.append("    num_images: %d" % a.images)
                lines.append("    degradation: %s" % DEGRADATION[problem])
                lines.append("    models:")
                for model in models:
                    if i >= len(per_model.get(model, [])):
                        continue
                    key, why = per_model[model][i]
                    lines.append("      %s:" % model)
                    lines.append("        batch_size: %d" % batch_for(model, method, key))
                    lines.append("        methods:")
                    fields = method_fields(method, model, key, modes[(model, problem, method)])
                    lines.append("          %-12s {%s}   # %s"
                                 % (method + ":", ", ".join(fields), why))
                    n_jobs += 1
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\nwrote %s: %d jobs at %d images" % (a.out, n_jobs, a.images))


# =====================================================================================
# finalize -- the winner of every cell, as the benchmark configuration
# =====================================================================================
GROUP = {"core": ["sdedit", "pnp", "dflow", "mpc_delta_t"], "rhc": ["mpc_rhc"], "rhso": ["rhso"]}
# batch_size per (model, memory class) at 100 images, measured on an 80 GB card
BATCH_FINAL = {"jit": {"core": 20, "rhc": 10, "rhso": 8}, "sit": {"core": 16, "rhc": 10, "rhso": 6},
               "pmf": {"core": 10, "rhc": 5, "rhso": 10}, "imf": {"core": 4, "rhc": 4, "rhso": 4}}


def restrict(cells, models, grids=None):
    """(model, problem, method) -> the candidate table the winner is chosen from.

    With `grids`, only configurations whose every knob was part of the searched grid survive,
    so a candidate produced by pushing an axis past its edge cannot win.  Rank and gap are
    then reported against the same table, which is what keeps the two comparable.
    """
    tables = {}
    for model in models:
        for problem in PROBLEMS:
            for method in SEARCHED:
                table = cells.get((model, problem, method), {})
                if grids is not None:
                    axes = grids.get((model, method))
                    table = {k: v for k, v in table.items()
                             if axes and all(kv is None or kv in axes.get(kn, set())
                                             for kn, kv in zip(KNOBS[method], k))}
                if table:
                    tables[(model, problem, method)] = table
    return tables


def winners(tables):
    """(model, problem, method) -> (key, entry, candidates it beat)."""
    return {cell: (lambda kv: (kv[0], kv[1], len(t)))(min(t.items(), key=lambda kv: kv[1]["lpips"]))
            for cell, t in tables.items()}


def finalize(a):
    models = [m for m in a.models.split(",") if m]
    batch = {m: dict(BATCH_FINAL[m]) for m in models}
    for spec in a.batch:
        model, _, kvs = spec.partition(":")
        batch[model].update({k: int(v) for k, v in (kv.split("=") for kv in kvs.split(","))})
    cells, _, _ = load(getattr(a, "from"))
    grids = load(a.grid)[1] if a.grid else None
    best = winners(restrict(cells, models, grids))
    missing = [(m, pr, me) for m in models for pr in PROBLEMS for me in METHODS
               if (m, pr, me) not in best]
    if missing:
        raise SystemExit("no winner for cells: %s" % missing)

    stems = sorted({Path(r).name.split("_r")[0] for r in getattr(a, "from")})
    source = ", ".join(stems) + (" (+ follow-up rounds)" if len(stems) < len(getattr(a, "from")) else "")
    out = [HEADER % (
        "FINAL BENCHMARK at t0 = 1.0 on the frozen ImageNet-100 (class seed 42 / image seed 43)",
        "#   python scripts/build_local_imagenet_pool.py --frozen-manifest "
        "benchmarks/imagenet100_c42_i43/manifest.csv\n"
        "#   python run.py --config %s --models <model> --run-id <id> --no-figures\n" % a.out,
        "All four priors in one file; run one per GPU with --models.  Every entry is the\n"
        "# lowest-LPIPS configuration of its cell%s, and the trailing\n"
        "# comment is that score.  Each problem is split into __core / __rhc / __rhso so that\n"
        "# batch_size, which resolves per (experiment, model), can follow each memory class."
        % (" among the candidates inside the searched grid" if grids else ""),
        "finalize", source, "true", 200, a.pool, model_block(models))]
    for problem in PROBLEMS:
        for group, methods in GROUP.items():
            out.append("\n  %s__%s:" % (EXP_NAME[problem], group))
            out.append("    enabled: true")
            out.append("    problem: %s" % problem)
            out.append("    num_images: %d" % a.num_images)
            out.append("    degradation: %s" % DEGRADATION[problem])
            out.append("    models:")
            for model in models:
                out.append("      %s:" % model)
                out.append("        batch_size: %d" % batch[model][group])
                out.append("        methods:")
                for method in methods:
                    key, cell, _ = best[(model, problem, method)]
                    fields = method_fields(method, model, key,
                                           cell["row"].get("rhso_terminal_mode") or None)
                    out.append("          %-12s {%s}   # LPIPS %.4f PSNR %.2f at %d tuning images"
                               % (method + ":", ", ".join(fields),
                                  cell["lpips"], cell["psnr"] or 0.0, cell["n"]))
    Path(a.out).write_text("\n".join(out) + "\n")
    print("wrote %s (%d models x %d cells)" % (a.out, len(models), len(PROBLEMS) * len(METHODS)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q):
        q.add_argument("--from", dest="from", action="append", required=True, metavar="RUN",
                       help="run directory whose results.csv is ranked (repeatable)")
        q.add_argument("--grid", action="append", metavar="RUN",
                       help="run directories that define the searched grid (default: --from)")
        q.add_argument("--models", default="jit,sit,pmf,imf")
        q.add_argument("--out", required=True)

    q = sub.add_parser("propose", help="write the next round of candidates")
    common(q)
    q.add_argument("--top", type=int, default=3, help="candidates kept per cell (0 = edges only)")
    q.add_argument("--paper", action=argparse.BooleanOptionalAction, default=True,
                   help="also re-run the manuscript's configuration where a cell has one")
    q.add_argument("--images", type=int, default=8)
    q.add_argument("--lock", action="append", default=[], metavar="METHOD=KNOB[,KNOB]",
                   help="never extend these knobs past a grid edge, e.g. "
                        "rhso=num_rhso_steps,num_opt_steps for a matched-budget search "
                        "(repeatable)")
    q.add_argument("--pool", default="cache/data/imagenet_val_100")
    q.set_defaults(run=propose)

    q = sub.add_parser("finalize", help="write the benchmark configuration from the winners")
    common(q)
    q.add_argument("--batch", action="append", default=[], metavar="MODEL:core=N,rhc=N,rhso=N",
                   help="override the built-in batch sizes for one model (repeatable)")
    q.add_argument("--pool", default="cache/data/imagenet100_c42_i43_mirror")
    q.add_argument("--num-images", type=int, default=100)
    q.set_defaults(run=finalize)

    a = p.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
