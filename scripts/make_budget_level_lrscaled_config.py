#!/usr/bin/env python
"""Cheap and mid budget levels with scaled learning rates (rebuttal Task 7).

Task 6 (scripts/make_budget_level_config.py) gave the backprop methods far fewer steps than
their tuned settings but kept the tuned learning rates; Adam moves each coordinate by about
the learning rate per step, so with 1-30 steps they barely moved.  This reruns the SAME 66
entries with one rule applied to every method:

    lr_new = lr_tuned x (tuned knob / new knob)

where the knob is RHSO num_opt_steps (M), MPC-delta_t n_ctrl, D-Flow num_opt_steps and
MPC-RHC n_ctrl, and the tuned values (lr and knob) are those of the entry's source entry,
which Task 6's comments name.  The entries come from Task 6's generator, so every other
field -- every step count, the blocks, the batch sizes -- is Task 6's.  Only lr changes;
it is stored rounded to %d significant figures.

Block names are Task 6's plus "_lr": <task>__budget_<level>_lr, and
<task>__budget_<level>_rhc_lr / _rhso_lr where Task 6 split off an extra block.

No timing configuration: the learning rate does not change the time per image, so
scripts/budget_level_table.py --timing-ignore lr pairs these runs with Task 6's timing run
(timing_budget_levels).

Files written:

    configs/experiments_budget_levels_lrscaled_pixel.yaml         100 images
    configs/experiments_budget_levels_lrscaled_pixel_smoke.yaml   the same entries at 2 images

    python scripts/make_budget_level_lrscaled_config.py            # write the two files
    python scripts/make_budget_level_lrscaled_config.py --check    # exit 1 if one is stale
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SIG_FIGS = 10
__doc__ = __doc__ % SIG_FIGS


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task6 = _load("make_budget_level_config", REPO_ROOT / "scripts" / "make_budget_level_config.py")
mmt, _flow = task6.mmt, task6._flow

QUALITY_OUT = "configs/experiments_budget_levels_lrscaled_pixel.yaml"
FULL_IMAGES, SMOKE_IMAGES = task6.FULL_IMAGES, task6.SMOKE_IMAGES
PRIORS, LEVELS, METHODS, TASKS = task6.PRIORS, task6.LEVELS, task6.METHODS, task6.TASKS
BLOCK_SUFFIX = "_lr"
# The knob whose tuned/new ratio scales the learning rate (RHSO: M only, N is not an
# optimisation step count).
LR_KNOB = {"rhso": "num_opt_steps", "mpc_delta_t": "n_ctrl", "dflow": "num_opt_steps",
           "mpc_rhc": "n_ctrl"}
LR_NOTE = "learning rate scaled by tuned/new steps"


def scaled_lr(lr_tuned, tuned, new):
    """lr_tuned x tuned / new, rounded to SIG_FIGS significant figures."""
    return float("%.*g" % (SIG_FIGS, float(lr_tuned) * float(tuned) / float(new)))


def entries():
    """Task 6's 66 entries, in its order, with lr scaled and the block renamed."""
    out = []
    for e in task6.entries():
        knob = LR_KNOB[e["method"]]
        tuned = [old for k, old, _ in e["changes"] if k == knob]
        if len(tuned) != 1 or "lr" not in e["params"]:
            raise SystemExit("%s: Task 6 entry has no %s change or no lr"
                             % (task6.source_line(e), knob))
        tuned, new, lr_tuned = int(tuned[0]), int(e["params"][knob]), e["params"]["lr"]
        params = dict(e["params"])
        params["lr"] = scaled_lr(lr_tuned, tuned, new)
        out.append(task6.Entry(e, params=params, task6_block=e["block"],
                               block=e["block"] + BLOCK_SUFFIX, lr_knob=knob, lr_tuned=lr_tuned,
                               knob_tuned=tuned, knob_new=new))
    return out


def lr_text(e):
    return "%s: lr %s x %d/%d -> %s" % (LR_NOTE, e["lr_tuned"], e["knob_tuned"], e["knob_new"],
                                        e["params"]["lr"])


def lr_table_lines(items):
    cell = {(e["level"], e["prior"], e["task"], e["method"]): e["params"]["lr"] for e in items}
    lines = ["Learning rates (4 significant figures; -- = no entry):",
             "  %-5s %-4s %-17s %9s %9s %9s %9s" % ("", "", "", "RHSO", "MPC-dt", "D-Flow",
                                                    "MPC-RHC")]
    for level in LEVELS:
        for prior in PRIORS:
            for task in TASKS:
                if (level, prior, task, "rhso") not in cell:
                    continue
                values = [cell.get((level, prior, task, m)) for m in METHODS]
                lines.append("  %-5s %-4s %-17s %9s %9s %9s %9s"
                             % (level, prior, task,
                                *["--" if v is None else "%.4g" % v for v in values]))
    return lines


def render_quality(num_images):
    smoke = num_images != FULL_IMAGES
    items = entries()
    path = task6.smoke_path(QUALITY_OUT) if smoke else QUALITY_OUT
    title = ("CHEAP AND MID BUDGET LEVELS, LEARNING RATE SCALED, on the pixel priors at t0 = 1.0%s"
             % (" -- 2-image SMOKE variant" if smoke else " on the frozen ImageNet-100 "
                                                        "(class seed 42 / image seed 43)"))
    usage = ["python run.py --config %s --run-id <id> --no-figures" % path,
             "bash submit.sh --array 4 --config %s --gpu h100:1 --time <HH:MM:SS>" % path]
    description = [
        "Task 6's %d entries (%s), every field and batch size unchanged except lr:"
        % (len(items), task6.QUALITY_OUT),
        "",
        "    lr_new = lr_tuned x (tuned knob / new knob)",
        "",
        "knob: RHSO num_opt_steps (M), MPC-delta_t n_ctrl, D-Flow num_opt_steps, MPC-RHC n_ctrl;",
        "tuned lr and knob: the entry's source entry (named in its comment).  lr is rounded",
        "to %d significant figures.  Blocks are Task 6's names plus \"%s\"." % (SIG_FIGS,
                                                                               BLOCK_SUFFIX),
        "No timing run: lr does not change the time per image; pair these with",
        "timing_budget_levels via scripts/budget_level_table.py --timing-ignore lr.",
        ""] + lr_table_lines(items)
    lines = task6._head(title, usage, description)
    lines[4] = "# Generated by scripts/make_budget_level_lrscaled_config.py; do not edit by hand."
    updates = {"max_atomic_jobs": 96}
    if smoke:
        updates["save_individual_images"] = False
    lines.append(yaml.safe_dump(mmt.shared(updates, PRIORS), sort_keys=False,
                                default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    for name, per_prior in task6.quality_blocks(items).items():
        first = next(iter(per_prior.values()))[1][0]
        lines += ["", "  %s:" % name, "    enabled: true", "    problem: %s" % first["task"],
                  "    num_images: %d" % num_images,
                  "    degradation: %s" % _flow(first["degradation"]), "    models:"]
        for prior, (bs, mine) in per_prior.items():
            lines += ["      %s:" % prior, "        batch_size: %d" % bs, "        methods:"]
            for e in mine:
                if e["degradation"] != first["degradation"]:
                    raise SystemExit("%s: sources disagree on the degradation" % name)
                lines.append("          %-12s %s   # source: %s; %s; %s; %s; Task 6 entry %s"
                             % (e["method"] + ":", _flow(e["params"]), task6.source_line(e),
                                task6.LEVEL_NOTE[e["level"]], task6.change_text(e), lr_text(e),
                                e["task6_block"]))
    return path, "\n".join(lines) + "\n"


def outputs():
    return dict(render_quality(n) for n in (FULL_IMAGES, SMOKE_IMAGES))


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
