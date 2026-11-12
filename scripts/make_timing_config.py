#!/usr/bin/env python
"""Timing and peak-memory configurations for rebuttal Task 1.

Reads the three frozen 100-image configurations

    configs/experiments_final_frozen100.yaml     every method on all four priors, SDEdit
                                                 included as the plain-sampling reference
    configs/experiments_flower_frozen100.yaml    Flower on the pixel priors
    configs/experiments_matched_frozen100.yaml   matched-budget RHSO and MPC-delta_t

and writes ONE timing configuration per prior, so each prior runs as its own single-GPU job:

    configs/experiments_rebuttal_timing_<prior>.yaml          3 images, batch size 1
    configs/experiments_rebuttal_timing_<prior>_smoke.yaml    2 images, batch size 1

Every (prior, task, method) entry of the sources is timed: nothing is merged across tasks,
so the summary fills a table with one column per task.  Each entry becomes its own
experiment block named `<source experiment>__<method>` (e.g. `deblurring__core__pnp`,
`deblurring__rhso_matched__rhso`), so `run.py --experiments <block>` selects exactly one job
and scripts/run_timing.py can give each one a fresh process.  A block is the source entry
copied field for field; only num_images (3) and batch_size (1) change, and
`runtime.timing_detail: true` switches on the per-image synchronised timer and the
allocator snapshots (src/timing.py).  Two source entries are merged only when they are
identical in every field (problem, degradation, method and every method parameter), and
any such merge is reported here and in the file header.

    python scripts/make_timing_config.py            # write the eight files
    python scripts/make_timing_config.py --check    # exit 1 if a committed file is stale

Then, per prior, on one exclusive GPU:

    python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf.yaml --run-id timing_pmf
    python scripts/summarize_timing.py outputs/timing_pmf outputs/timing_jit outputs/timing_imf outputs/timing_sit
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ("final", "configs/experiments_final_frozen100.yaml"),
    ("flower", "configs/experiments_flower_frozen100.yaml"),
    ("matched", "configs/experiments_matched_frozen100.yaml"),
]
PRIORS = ["pmf", "jit", "imf", "sit"]
TASKS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
TIMING_IMAGES, SMOKE_IMAGES = 3, 2
OUT = "configs/experiments_rebuttal_timing_%s.yaml"

Entry = collections.namedtuple(
    "Entry", "source path experiment model method problem degradation params position")


def out_path(prior, smoke=False):
    path = OUT % prior
    return path.replace(".yaml", "_smoke.yaml") if smoke else path


def load_source(path):
    return yaml.safe_load((REPO_ROOT / path).read_text())


def source_entries():
    """Every (source, experiment, prior, method) entry, in a fixed order."""
    entries = []
    position = 0
    for source, path in SOURCES:
        cfg = load_source(path)
        for exp, block in (cfg.get("experiments") or {}).items():
            if not block.get("enabled", True):
                continue
            for model, mblock in (block.get("models") or {}).items():
                for method, params in (mblock.get("methods") or {}).items():
                    entries.append(Entry(source, path, exp, model, method, block["problem"],
                                         dict(block.get("degradation") or {}),
                                         dict(params or {}), position))
                    position += 1
    return entries


def _identity(entry):
    """Every field of a source entry that defines the job, as a comparable key."""
    return (entry.model, entry.problem, json.dumps(entry.degradation, sort_keys=True),
            entry.method, json.dumps(entry.params, sort_keys=True))


def plan(prior):
    """([(kept entry, [merged duplicates])] for one prior, in task order)."""
    order = {name: i for i, (name, _) in enumerate(SOURCES)}
    mine = sorted((e for e in source_entries() if e.model == prior),
                  key=lambda e: (TASKS.index(e.problem), order[e.source], e.position))
    kept, by_key = [], {}
    for entry in mine:
        key = _identity(entry)
        if key in by_key:
            by_key[key][1].append(entry)
            continue
        by_key[key] = (entry, [])
        kept.append(by_key[key])
    return kept


def block_name(entry):
    return "%s__%s" % (entry.experiment, entry.method)


def shared_sections():
    """data / models / defaults / metrics, which must agree across the three sources."""
    cfgs = [load_source(path) for _, path in SOURCES]
    for key in ("data", "defaults", "metrics"):
        values = [json.dumps(c.get(key), sort_keys=True) for c in cfgs]
        if len(set(values)) != 1:
            raise SystemExit("The source configurations disagree on `%s`; a timing block "
                             "would not reproduce every source exactly." % key)
    models = {}
    for cfg in cfgs:
        for name, value in (cfg.get("models") or {}).items():
            if name in models and models[name] != (value or {}):
                raise SystemExit("The source configurations disagree on models.%s." % name)
            models[name] = value or {}
    return cfgs[0], models


def _flow(mapping):
    return yaml.safe_dump(mapping, default_flow_style=True, sort_keys=False,
                          width=10000).strip()


def render(prior, num_images):
    base, models = shared_sections()
    kept = plan(prior)
    smoke = num_images != TIMING_IMAGES
    merges = [(e, dups) for e, dups in kept if dups]
    lines = [
        "# " + "=" * 85,
        "# REBUTTAL TIMING AND PEAK MEMORY -- %s%s" % (prior, ", 2-image SMOKE variant" if smoke else ""),
        "# " + "=" * 85,
        "#",
        "# Generated by scripts/make_timing_config.py; do not edit by hand.",
        "#",
        "#   python scripts/run_timing.py --config %s --run-id <id>" % out_path(prior, smoke),
        "#",
        "# One block per (task, method) entry of the three frozen configurations:",
    ]
    lines += ["#   %s" % path for _, path in SOURCES]
    lines += [
        "# Each block's comment names the source configuration and entry it was copied",
        "# from.  The entry is copied field for field; only num_images (%d) and batch_size (1)"
        % num_images,
        "# differ.  Nothing is merged across tasks.",
    ]
    if merges:
        lines.append("# Entries merged because they are identical in every field:")
        for entry, dups in merges:
            for dup in dups:
                lines.append("#   %s :: %s / %s / %s  ==  %s :: %s / %s / %s"
                             % (dup.path, dup.experiment, dup.model, dup.method,
                                entry.path, entry.experiment, entry.model, entry.method))
    else:
        lines.append("# No two source entries are identical in every field, so none was merged.")
    lines += ["# " + "=" * 85, ""]

    runtime = dict(base["runtime"])
    runtime.update(max_atomic_jobs=64, gpu_memory_profiling=True, timing_detail=True)
    doc_head = {"runtime": runtime, "data": base["data"],
                "models": {prior: models[prior]}, "defaults": base["defaults"],
                "metrics": base["metrics"]}
    lines.append(yaml.safe_dump(doc_head, sort_keys=False, default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    for entry, dups in kept:
        lines.append("")
        lines.append("  # source: %s :: %s / %s / %s"
                     % (entry.path, entry.experiment, entry.model, entry.method))
        for dup in dups:
            lines.append("  # also covers (identical in every field): %s :: %s / %s / %s"
                         % (dup.path, dup.experiment, dup.model, dup.method))
        lines += [
            "  %s:" % block_name(entry),
            "    enabled: true",
            "    problem: %s" % entry.problem,
            "    num_images: %d" % num_images,
            "    degradation: %s" % _flow(entry.degradation),
            "    models:",
            "      %s:" % prior,
            "        batch_size: 1",
            "        methods:",
            "          %s: %s" % (entry.method, _flow(entry.params)),
        ]
    return "\n".join(lines) + "\n"


def outputs():
    """{path: text} for every file this script owns."""
    files = {}
    for prior in PRIORS:
        files[out_path(prior)] = render(prior, TIMING_IMAGES)
        files[out_path(prior, smoke=True)] = render(prior, SMOKE_IMAGES)
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
    for prior in PRIORS:
        kept = plan(prior)
        merged = sum(len(d) for _, d in kept)
        print("  %-4s %3d block(s), %d merged duplicate(s)" % (prior, len(kept), merged))
        for entry, dups in kept:
            for dup in dups:
                print("       merged %s :: %s / %s into %s :: %s"
                      % (dup.path, dup.experiment, dup.method, entry.path, entry.experiment))
    if stale:
        print("STALE: %s" % ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
