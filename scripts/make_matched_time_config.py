#!/usr/bin/env python
"""Matched-time configurations on the pixel priors (rebuttal Task 5).

Every method gets the wall-clock per image that RHSO takes on the same prior and task.  The
knob values below were computed from per-image times measured on the cluster with
scripts/run_timing.py; they are fixed inputs here, not searched.  Each entry starts from its
prior's own frozen benchmark entry

    PnP-Flow, D-Flow, MPC-RHC   configs/experiments_final_frozen100.yaml   (<task>__core, __rhc)
    Flower                      configs/experiments_flower_frozen100.yaml  (<task>__flower)

and changes ONE field, the method's budget knob; every other field (t0, step sizes, solver,
lam, K, Flower's CG cap, ...) and the benchmark batch size are kept.  "--" in the table means
the method gets no new run on that cell: 33 entries in all.

Files written:

    configs/experiments_matched_time_pixel.yaml            100 images, benchmark batch sizes
    configs/experiments_matched_time_pixel_smoke.yaml      the same entries at 2 images
    configs/experiments_rebuttal_timing_matched_time.yaml  timing: 3 images, batch size 1
    configs/experiments_rebuttal_timing_matched_time_smoke.yaml               (2 images)

Blocks.  Quality: one block per task, `<task>__matched_time`.  batch_size is resolved per
(experiment, prior), and the benchmark ran MPC-RHC at a smaller batch than PnP-Flow, D-Flow
and Flower (pMF 5 vs 10, JiT 10 vs 20), so the three cells with an MPC-RHC entry get a
second block `<task>__matched_time_rhc` that keeps MPC-RHC's own batch size.  Timing: one
block per entry, `<quality block>__<method>__<prior>`, the quality entry field for field
with only num_images and batch_size changed, as in scripts/make_timing_config.py, so
scripts/run_timing.py runs each in a fresh process under one run id (timing_matched_time).

    python scripts/make_matched_time_config.py            # write the four files
    python scripts/make_matched_time_config.py --check    # exit 1 if a committed file is stale
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


mtc = _load("make_timing_config", REPO_ROOT / "scripts" / "make_timing_config.py")

FINAL = "configs/experiments_final_frozen100.yaml"
FLOWER = "configs/experiments_flower_frozen100.yaml"
QUALITY_OUT = "configs/experiments_matched_time_pixel.yaml"
TIMING_OUT = "configs/experiments_rebuttal_timing_matched_time.yaml"
TIMING_RUN_ID = "timing_matched_time"
FULL_IMAGES, SMOKE_IMAGES, TIMING_IMAGES = 100, 2, 3
PRIORS = ["pmf", "jit"]
METHODS = ["pnp", "flower", "dflow", "mpc_rhc"]          # order inside a block
TASKS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
EXPERIMENT = {"denoising": "denoising", "deblur": "deblurring",
              "super_resolution": "super_resolution", "random_inpaint": "random_inpainting",
              "box_inpaint": "box_inpainting"}
KNOB = {"pnp": "num_pnp_steps", "flower": "num_flower_steps", "dflow": "num_opt_steps",
        "mpc_rhc": "n_ctrl"}
# method -> (source file, suffix of the source experiment, suffix of the quality block)
SOURCE = {"pnp": (FINAL, "__core", "__matched_time"),
          "dflow": (FINAL, "__core", "__matched_time"),
          "flower": (FLOWER, "__flower", "__matched_time"),
          "mpc_rhc": (FINAL, "__rhc", "__matched_time_rhc")}
MATCH_NOTE = "matched to RHSO's measured time"

# (prior, task) -> {method: knob value}.  The user's table; a missing method is "--".
VALUES = {
    ("pmf", "denoising"):        {"pnp": 978, "flower": 1218, "dflow": 155},
    ("pmf", "deblur"):           {"pnp": 513, "flower": 13, "dflow": 162},
    ("pmf", "super_resolution"): {"pnp": 962, "flower": 1240, "dflow": 168, "mpc_rhc": 20},
    ("pmf", "random_inpaint"):   {"pnp": 888, "flower": 1210, "dflow": 151, "mpc_rhc": 18},
    ("pmf", "box_inpaint"):      {"pnp": 1053, "flower": 1155, "dflow": 166},
    ("jit", "denoising"):        {"pnp": 312, "flower": 323, "dflow": 145},
    ("jit", "deblur"):           {"pnp": 606, "flower": 171, "dflow": 286},
    ("jit", "super_resolution"): {"pnp": 600, "flower": 636, "dflow": 287, "mpc_rhc": 71},
    ("jit", "random_inpaint"):   {"pnp": 309, "flower": 324, "dflow": 145},
    ("jit", "box_inpaint"):      {"pnp": 309, "flower": 321, "dflow": 143},
}


def smoke_path(path):
    return path.replace(".yaml", "_smoke.yaml")


def load_yaml(path):
    return yaml.safe_load((REPO_ROOT / path).read_text())


class Entry(dict):
    """One matched-time entry: prior, task, method, block, source, batch_size, params, ..."""


def entries():
    """The 33 entries in a fixed order (task, block, prior, method)."""
    sources = {FINAL: load_yaml(FINAL), FLOWER: load_yaml(FLOWER)}
    out = []
    for task in TASKS:
        for prior in PRIORS:
            for method in METHODS:
                value = VALUES[(prior, task)].get(method)
                if value is None:
                    continue
                path, src_suffix, block_suffix = SOURCE[method]
                src_exp = EXPERIMENT[task] + src_suffix
                block = sources[path]["experiments"][src_exp]
                if block["problem"] != task:
                    raise SystemExit("%s :: %s is problem %s, expected %s"
                                     % (path, src_exp, block["problem"], task))
                model = block["models"][prior]
                params = dict(model["methods"][method])
                knob = KNOB[method]
                if knob not in params:
                    raise SystemExit("%s :: %s / %s / %s has no %s to change"
                                     % (path, src_exp, prior, method, knob))
                old = params[knob]
                params[knob] = int(value)
                out.append(Entry(prior=prior, task=task, method=method,
                                 block=EXPERIMENT[task] + block_suffix, source=path,
                                 source_experiment=src_exp, batch_size=int(model["batch_size"]),
                                 degradation=dict(block.get("degradation") or {}),
                                 params=params, knob=knob, old=old, new=int(value)))
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


def _head(title, usage, description):
    lines = ["# " + "=" * 85, "# " + title, "# " + "=" * 85, "#",
             "# Generated by scripts/make_matched_time_config.py; do not edit by hand.", "#"]
    lines += ["#   " + u for u in usage]
    lines += ["#"] + ["# " + d if d else "#" for d in description] + ["# " + "=" * 85, ""]
    return lines


def shared(runtime_updates, models):
    base, all_models = mtc.shared_sections()
    runtime = dict(base["runtime"])
    runtime.update(runtime_updates)
    return {"runtime": runtime, "data": base["data"],
            "models": {m: all_models[m] for m in models}, "defaults": base["defaults"],
            "metrics": base["metrics"]}


def knob_table_lines():
    lines = ["Knob values (-- = no new run): PnP-Flow %s, Flower %s, D-Flow %s, MPC-RHC %s."
             % (KNOB["pnp"], KNOB["flower"], KNOB["dflow"], KNOB["mpc_rhc"]),
             "  %-4s %-17s %8s %8s %8s %8s" % ("", "", "PnP", "Flower", "D-Flow", "MPC-RHC")]
    for prior in PRIORS:
        for task in TASKS:
            v = VALUES[(prior, task)]
            lines.append("  %-4s %-17s %8s %8s %8s %8s"
                         % (prior, task, *[str(v.get(m, "--")) for m in METHODS]))
    return lines


def render_quality(num_images):
    smoke = num_images != FULL_IMAGES
    items = entries()
    path = smoke_path(QUALITY_OUT) if smoke else QUALITY_OUT
    title = ("MATCHED-TIME on the pixel priors at t0 = 1.0%s"
             % (" -- 2-image SMOKE variant" if smoke else " on the frozen ImageNet-100 "
                                                        "(class seed 42 / image seed 43)"))
    usage = ["python run.py --config %s --run-id <id> --no-figures" % path,
             "bash submit.sh --array 4 --config %s --gpu h100:1 --time <HH:MM:SS>" % path]
    description = [
        "Every method gets the wall-clock per image that RHSO takes on the same prior and",
        "task.  Each entry starts from its prior's own frozen entry (the comment names it) and",
        "changes ONLY the method's budget knob; every other field and the benchmark batch size",
        "are kept.  The knob values were computed from per-image times measured on the",
        "cluster (scripts/run_timing.py) and are fixed here.  %d entries." % len(items),
        "MPC-RHC ran at its own benchmark batch size, so it sits in <task>__matched_time_rhc.",
        ""] + knob_table_lines()
    lines = _head(title, usage, description)
    updates = {"max_atomic_jobs": 64}
    if smoke:
        updates["save_individual_images"] = False
    lines.append(yaml.safe_dump(shared(updates, PRIORS), sort_keys=False,
                                default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    for name, per_prior in quality_blocks(items).items():
        first = next(iter(per_prior.values()))[1][0]
        lines += ["", "  %s:" % name, "    enabled: true", "    problem: %s" % first["task"],
                  "    num_images: %d" % num_images,
                  "    degradation: %s" % mtc._flow(first["degradation"]), "    models:"]
        for prior, (bs, mine) in per_prior.items():
            lines += ["      %s:" % prior, "        batch_size: %d" % bs, "        methods:"]
            for e in mine:
                if e["degradation"] != first["degradation"]:
                    raise SystemExit("%s: sources disagree on the degradation" % name)
                lines.append("          %-12s %s   # source: %s; %s %s -> %d, %s"
                             % (e["method"] + ":", mtc._flow(e["params"]), source_line(e),
                                e["knob"], e["old"], e["new"], MATCH_NOTE))
    return path, "\n".join(lines) + "\n"


def timing_block_name(e):
    return "%s__%s__%s" % (e["block"], e["method"], e["prior"])


def render_timing(num_images):
    smoke = num_images != TIMING_IMAGES
    items = entries()
    path = smoke_path(TIMING_OUT) if smoke else TIMING_OUT
    run_id = TIMING_RUN_ID + ("_smoke" if smoke else "")
    title = ("REBUTTAL TIMING -- matched-time entries on pMF and JiT%s"
             % (", 2-image SMOKE variant" if smoke else ""))
    usage = ["python scripts/run_timing.py --config %s --run-id %s%s"
             % (path, run_id, " --with-checks" if smoke else ""),
             "bash submit_timing.sh matched_time%s --gpu h100:1 --time <HH:MM:SS>"
             % (" --smoke" if smoke else "")]
    description = [
        "One block per entry of %s, so each runs in its own process" % QUALITY_OUT,
        "(scripts/run_timing.py) and its time can be checked against RHSO's on the same GPU",
        "type.  Each block's comment names the quality entry; the entry is copied field for",
        "field and only num_images (%d) and batch_size (1) differ." % num_images]
    lines = _head(title, usage, description)
    lines.append(yaml.safe_dump(shared({"max_atomic_jobs": 64, "gpu_memory_profiling": True,
                                        "timing_detail": True}, PRIORS),
                                sort_keys=False, default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    for e in items:
        lines += ["",
                  "  # source: %s :: %s / %s / %s; %s, %s"
                  % (QUALITY_OUT, e["block"], e["prior"], e["method"], source_line(e),
                     MATCH_NOTE),
                  "  %s:" % timing_block_name(e), "    enabled: true",
                  "    problem: %s" % e["task"], "    num_images: %d" % num_images,
                  "    degradation: %s" % mtc._flow(e["degradation"]), "    models:",
                  "      %s:" % e["prior"], "        batch_size: 1", "        methods:",
                  "          %s: %s" % (e["method"], mtc._flow(e["params"]))]
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
        "%s %d" % (p, sum(e["prior"] == p for e in items)) for p in PRIORS)))
    if stale:
        print("STALE: %s" % ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
