#!/usr/bin/env python
"""Cheap and mid budget levels on the pixel priors (rebuttal Task 6).

Task 5 gave every method RHSO's wall-clock.  This is the reverse: the backprop methods
(RHSO, MPC-delta_t, D-Flow, MPC-RHC) get the measured time of the cheap methods (CHEAP level)
and 4x that (MID level, no deblurring).  The step values below were computed from per-image
times measured on the cluster; they are fixed inputs here, not searched.  Each entry starts
from its prior's own entry

    RHSO          pMF  configs/experiments_final_frozen100.yaml    <task>__rhso
                  JiT  configs/experiments_matched_frozen100.yaml  <task>__rhso_matched
                       (direct planner, Heun: both kept)
    MPC-delta_t, D-Flow  configs/experiments_final_frozen100.yaml  <task>__core
    MPC-RHC              configs/experiments_final_frozen100.yaml  <task>__rhc

and changes ONLY the step knob: RHSO num_rhso_steps and num_opt_steps (N, M), MPC-delta_t
n_ctrl (num_mpc_steps N fixed), D-Flow num_opt_steps, MPC-RHC n_ctrl.  Every other field and
the benchmark batch size are kept.  "--" means no new run: 66 entries (36 cheap, 30 mid).

Files written:

    configs/experiments_budget_levels_pixel.yaml               100 images, benchmark batch sizes
    configs/experiments_budget_levels_pixel_smoke.yaml         the same entries at 2 images
    configs/experiments_rebuttal_timing_budget_levels.yaml     timing: 3 images, batch size 1
    configs/experiments_rebuttal_timing_budget_levels_smoke.yaml             (2 images)

Blocks.  Quality: `<task>__budget_cheap` and `<task>__budget_mid`.  batch_size resolves per
(experiment, prior), so an entry whose benchmark batch size differs from the D-Flow /
MPC-delta_t one on its prior goes to an extra block: `<task>__budget_<level>_rhc` for MPC-RHC
(pMF 5 vs 10, JiT 10 vs 20) and `<task>__budget_<level>_rhso` for JiT's RHSO (8 vs 20); pMF's
RHSO (10) stays in the main block.  Timing: one block per entry,
`<quality block>__<method>__<prior>`, the quality entry with only num_images and batch_size
changed, for scripts/run_timing.py under one run id (timing_budget_levels).

    python scripts/make_budget_level_config.py            # write the four files
    python scripts/make_budget_level_config.py --check    # exit 1 if a committed file is stale
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mmt = _load("make_matched_time_config", REPO_ROOT / "scripts" / "make_matched_time_config.py")
_flow = mmt.mtc._flow

FINAL = "configs/experiments_final_frozen100.yaml"
MATCHED = "configs/experiments_matched_frozen100.yaml"
QUALITY_OUT = "configs/experiments_budget_levels_pixel.yaml"
TIMING_OUT = "configs/experiments_rebuttal_timing_budget_levels.yaml"
TIMING_RUN_ID = "timing_budget_levels"
FULL_IMAGES, SMOKE_IMAGES, TIMING_IMAGES = 100, 2, 3
PRIORS = ["pmf", "jit"]
LEVELS = ["cheap", "mid"]
METHODS = ["rhso", "mpc_delta_t", "dflow", "mpc_rhc"]       # order inside a block
TASKS = mmt.TASKS
EXPERIMENT = mmt.EXPERIMENT
KNOBS = {"rhso": ("num_rhso_steps", "num_opt_steps"), "mpc_delta_t": ("n_ctrl",),
         "dflow": ("num_opt_steps",), "mpc_rhc": ("n_ctrl",)}
# (method, prior) -> (source file, suffix of the source experiment)
SOURCE = {("rhso", "pmf"): (FINAL, "__rhso"), ("rhso", "jit"): (MATCHED, "__rhso_matched"),
          ("mpc_delta_t", "pmf"): (FINAL, "__core"), ("mpc_delta_t", "jit"): (FINAL, "__core"),
          ("dflow", "pmf"): (FINAL, "__core"), ("dflow", "jit"): (FINAL, "__core"),
          ("mpc_rhc", "pmf"): (FINAL, "__rhc"), ("mpc_rhc", "jit"): (FINAL, "__rhc")}
# The block whose batch size is the reference; an entry with another batch size is split off.
EXTRA_SUFFIX = {"rhso": "_rhso", "mpc_rhc": "_rhc"}
LEVEL_NOTE = {"cheap": "cheap level (the cheap methods' measured time)",
              "mid": "mid level (4x the cheap level's time)"}

# (level, prior, task) -> {method: knob value(s)}.  The user's table; a missing method is "--".
# RHSO is (N, M) = (num_rhso_steps, num_opt_steps).
VALUES = {
    ("cheap", "pmf", "denoising"):        {"rhso": (4, 1), "mpc_delta_t": 1, "dflow": 6, "mpc_rhc": 1},
    ("cheap", "pmf", "deblur"):           {"rhso": (8, 8), "mpc_delta_t": 9, "dflow": 69},
    ("cheap", "pmf", "super_resolution"): {"rhso": (4, 1), "mpc_delta_t": 1, "dflow": 7, "mpc_rhc": 1},
    ("cheap", "pmf", "random_inpaint"):   {"rhso": (4, 1), "mpc_delta_t": 1, "dflow": 6, "mpc_rhc": 1},
    ("cheap", "pmf", "box_inpaint"):      {"rhso": (8, 1), "mpc_delta_t": 1, "dflow": 8},
    ("cheap", "jit", "denoising"):        {"rhso": (7, 2), "mpc_delta_t": 3, "dflow": 25, "mpc_rhc": 6},
    ("cheap", "jit", "deblur"):           {"rhso": (8, 12), "mpc_delta_t": 14, "dflow": 103},
    ("cheap", "jit", "super_resolution"): {"rhso": (4, 5), "mpc_delta_t": 3, "dflow": 24, "mpc_rhc": 5},
    ("cheap", "jit", "random_inpaint"):   {"rhso": (7, 2), "mpc_delta_t": 3, "dflow": 25, "mpc_rhc": 6},
    ("cheap", "jit", "box_inpaint"):      {"rhso": (4, 5), "mpc_delta_t": 3, "dflow": 25},
    ("mid", "pmf", "denoising"):          {"rhso": (4, 7), "mpc_delta_t": 4, "dflow": 27, "mpc_rhc": 6},
    ("mid", "pmf", "super_resolution"):   {"rhso": (4, 7), "mpc_delta_t": 4, "dflow": 29, "mpc_rhc": 3},
    ("mid", "pmf", "random_inpaint"):     {"rhso": (4, 7), "mpc_delta_t": 3, "dflow": 27, "mpc_rhc": 3},
    ("mid", "pmf", "box_inpaint"):        {"rhso": (8, 4), "mpc_delta_t": 4, "dflow": 34},
    ("mid", "jit", "denoising"):          {"rhso": (7, 13), "mpc_delta_t": 14, "dflow": 102, "mpc_rhc": 25},
    ("mid", "jit", "super_resolution"):   {"rhso": (4, 23), "mpc_delta_t": 13, "dflow": 99, "mpc_rhc": 24},
    ("mid", "jit", "random_inpaint"):     {"rhso": (7, 14), "mpc_delta_t": 14, "dflow": 102, "mpc_rhc": 25},
    ("mid", "jit", "box_inpaint"):        {"rhso": (4, 24), "mpc_delta_t": 14, "dflow": 101},
}


def smoke_path(path):
    return path.replace(".yaml", "_smoke.yaml")


class Entry(dict):
    """One budget-level entry: level, prior, task, method, block, source, batch_size, params."""


def _values(method, value):
    return tuple(int(v) for v in value) if method == "rhso" else (int(value),)


def entries():
    """The 66 entries in a fixed order (level, task, prior, method)."""
    sources = {FINAL: mmt.load_yaml(FINAL), MATCHED: mmt.load_yaml(MATCHED)}
    out = []
    for level in LEVELS:
        for task in TASKS:
            for prior in PRIORS:
                cell = VALUES.get((level, prior, task))
                if not cell:
                    continue
                core = sources[FINAL]["experiments"][EXPERIMENT[task] + "__core"]
                core_bs = int(core["models"][prior]["batch_size"])
                for method in METHODS:
                    if method not in cell:
                        continue
                    path, src_suffix = SOURCE[(method, prior)]
                    src_exp = EXPERIMENT[task] + src_suffix
                    block = sources[path]["experiments"][src_exp]
                    if block["problem"] != task:
                        raise SystemExit("%s :: %s is problem %s, expected %s"
                                         % (path, src_exp, block["problem"], task))
                    model = block["models"][prior]
                    params = dict(model["methods"][method])
                    changes = []
                    for knob, value in zip(KNOBS[method], _values(method, cell[method])):
                        if knob not in params:
                            raise SystemExit("%s :: %s / %s / %s has no %s to change"
                                             % (path, src_exp, prior, method, knob))
                        changes.append((knob, params[knob], value))
                        params[knob] = value
                    bs = int(model["batch_size"])
                    suffix = "" if bs == core_bs else EXTRA_SUFFIX[method]
                    out.append(Entry(level=level, prior=prior, task=task, method=method,
                                     block="%s__budget_%s%s" % (EXPERIMENT[task], level, suffix),
                                     source=path, source_experiment=src_exp, batch_size=bs,
                                     degradation=dict(block.get("degradation") or {}),
                                     params=params, changes=changes))
    return out


def quality_blocks(items):
    """{block: {prior: (batch_size, [entries])}} in first-seen order; one batch per prior."""
    blocks = {}
    for e in items:
        per_prior = blocks.setdefault(e["block"], {})
        bs, mine = per_prior.setdefault(e["prior"], (e["batch_size"], []))
        if bs != e["batch_size"]:
            raise SystemExit("block %s mixes benchmark batch sizes %d and %d on %s; it needs "
                             "its own block" % (e["block"], bs, e["batch_size"], e["prior"]))
        mine.append(e)
    return blocks


def source_line(e):
    return "%s :: %s / %s / %s" % (e["source"], e["source_experiment"], e["prior"], e["method"])


def change_text(e):
    return ", ".join("%s %s -> %d" % (k, old, new) for k, old, new in e["changes"])


def _head(title, usage, description):
    lines = ["# " + "=" * 85, "# " + title, "# " + "=" * 85, "#",
             "# Generated by scripts/make_budget_level_config.py; do not edit by hand.", "#"]
    lines += ["#   " + u for u in usage]
    lines += ["#"] + ["# " + d if d else "#" for d in description] + ["# " + "=" * 85, ""]
    return lines


def _cell_text(method, value):
    if value is None:
        return "--"
    return "(%d, %d)" % tuple(value) if method == "rhso" else str(value)


def knob_table_lines():
    lines = ["Step values (-- = no new run): RHSO (num_rhso_steps, num_opt_steps), MPC-delta_t",
             "n_ctrl, D-Flow num_opt_steps, MPC-RHC n_ctrl.",
             "  %-5s %-4s %-17s %9s %8s %8s %8s" % ("", "", "", "RHSO", "MPC-dt", "D-Flow",
                                                    "MPC-RHC")]
    for level in LEVELS:
        for prior in PRIORS:
            for task in TASKS:
                v = VALUES.get((level, prior, task))
                if v is None:
                    continue
                lines.append("  %-5s %-4s %-17s %9s %8s %8s %8s"
                             % (level, prior, task,
                                *[_cell_text(m, v.get(m)) for m in METHODS]))
    return lines


def render_quality(num_images):
    smoke = num_images != FULL_IMAGES
    items = entries()
    path = smoke_path(QUALITY_OUT) if smoke else QUALITY_OUT
    title = ("CHEAP AND MID BUDGET LEVELS on the pixel priors at t0 = 1.0%s"
             % (" -- 2-image SMOKE variant" if smoke else " on the frozen ImageNet-100 "
                                                        "(class seed 42 / image seed 43)"))
    usage = ["python run.py --config %s --run-id <id> --no-figures" % path,
             "bash submit.sh --array 4 --config %s --gpu h100:1 --time <HH:MM:SS>" % path]
    description = [
        "The backprop methods get the measured time of the cheap methods (cheap level) and 4x",
        "that (mid level, no deblurring).  Each entry starts from its prior's own entry (the",
        "comment names it) and changes ONLY the step knob; every other field and the benchmark",
        "batch size are kept.  The step values were computed from per-image times measured on",
        "the cluster (scripts/run_timing.py) and are fixed here.  %d entries (%d cheap, %d mid)."
        % (len(items), sum(e["level"] == "cheap" for e in items),
           sum(e["level"] == "mid" for e in items)),
        "Entries whose benchmark batch size differs from D-Flow's on that prior sit in",
        "<task>__budget_<level>_rhc (MPC-RHC) or <task>__budget_<level>_rhso (JiT RHSO).",
        ""] + knob_table_lines()
    lines = _head(title, usage, description)
    updates = {"max_atomic_jobs": 96}
    if smoke:
        updates["save_individual_images"] = False
    lines.append(yaml.safe_dump(mmt.shared(updates, PRIORS), sort_keys=False,
                                default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    for name, per_prior in quality_blocks(items).items():
        first = next(iter(per_prior.values()))[1][0]
        lines += ["", "  %s:" % name, "    enabled: true", "    problem: %s" % first["task"],
                  "    num_images: %d" % num_images,
                  "    degradation: %s" % _flow(first["degradation"]), "    models:"]
        for prior, (bs, mine) in per_prior.items():
            lines += ["      %s:" % prior, "        batch_size: %d" % bs, "        methods:"]
            for e in mine:
                if e["degradation"] != first["degradation"]:
                    raise SystemExit("%s: sources disagree on the degradation" % name)
                lines.append("          %-12s %s   # source: %s; %s; %s"
                             % (e["method"] + ":", _flow(e["params"]), source_line(e),
                                LEVEL_NOTE[e["level"]], change_text(e)))
    return path, "\n".join(lines) + "\n"


def timing_block_name(e):
    return "%s__%s__%s" % (e["block"], e["method"], e["prior"])


def render_timing(num_images):
    smoke = num_images != TIMING_IMAGES
    items = entries()
    path = smoke_path(TIMING_OUT) if smoke else TIMING_OUT
    run_id = TIMING_RUN_ID + ("_smoke" if smoke else "")
    title = ("REBUTTAL TIMING -- cheap and mid budget-level entries on pMF and JiT%s"
             % (", 2-image SMOKE variant" if smoke else ""))
    usage = ["python scripts/run_timing.py --config %s --run-id %s%s"
             % (path, run_id, " --with-checks" if smoke else ""),
             "bash submit_timing.sh budget_levels%s --gpu h100:1 --time <HH:MM:SS>"
             % (" --smoke" if smoke else "")]
    description = [
        "One block per entry of %s, so each runs in its own process" % QUALITY_OUT,
        "(scripts/run_timing.py) and its time can be checked against its level's budget.",
        "Each block's comment names the quality entry; the entry is copied field for field",
        "and only num_images (%d) and batch_size (1) differ." % num_images]
    lines = _head(title, usage, description)
    lines.append(yaml.safe_dump(mmt.shared({"max_atomic_jobs": 96, "gpu_memory_profiling": True,
                                            "timing_detail": True}, PRIORS),
                                sort_keys=False, default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    for e in items:
        lines += ["",
                  "  # source: %s :: %s / %s / %s; %s; %s"
                  % (QUALITY_OUT, e["block"], e["prior"], e["method"], source_line(e),
                     LEVEL_NOTE[e["level"]]),
                  "  %s:" % timing_block_name(e), "    enabled: true",
                  "    problem: %s" % e["task"], "    num_images: %d" % num_images,
                  "    degradation: %s" % _flow(e["degradation"]), "    models:",
                  "      %s:" % e["prior"], "        batch_size: 1", "        methods:",
                  "          %s: %s" % (e["method"], _flow(e["params"]))]
    return path, "\n".join(lines) + "\n"


def outputs():
    files = {}
    for path, text in (render_quality(FULL_IMAGES), render_quality(SMOKE_IMAGES),
                       render_timing(TIMING_IMAGES), render_timing(SMOKE_IMAGES)):
        files[path] = text
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed files instead of writing")
    args = parser.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        target = REPO_ROOT / path
        if args.check:
            if not target.exists() or target.read_text() != text:
                stale.append(path)
            continue
        target.write_text(text, newline="\n")
        print("wrote %s" % path)
    items = entries()
    print("  %d entries: %s" % (len(items), ", ".join(
        "%s %s %d" % (lv, p, sum(e["prior"] == p and e["level"] == lv for e in items))
        for lv in LEVELS for p in PRIORS)))
    if stale:
        print("STALE: %s" % ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
