#!/usr/bin/env python
"""Timing configurations for the main-paper runtime / memory table (rebuttal Task 9).

Two capacity-matched pixel checkpoints, every method, all five tasks, batch size 1, 3 images,
the measurement of rebuttal Task 1 (src/timing.py, scripts/run_timing.py):

    configs/experiments_rebuttal_timing_pmf_exact.yaml        run id timing_pmf_exact
        pMF-L/16: the SAME 45 blocks as configs/experiments_rebuttal_timing_pmf.yaml (same
        names, same fields), plus `models.pmf.param_loading: host_once`, the opt-in loader
        that keeps model loading from setting JAX's lifetime allocator peak, so each
        configuration's peak is exact (src/models/pmf.py, src/timing.exact_jax_peak).
    configs/experiments_rebuttal_timing_pmf_exact_smoke.yaml
        two of those blocks (one RHSO, one MPC-delta_t) at the SAME 3 images, so their PSNR
        can be compared with timing_pmf's directly.
    configs/experiments_rebuttal_timing_jitl.yaml             run id timing_jitl
        JiT-L/16: 7 methods x 5 tasks = 35 blocks.  PnP-Flow, Flower, D-Flow, MPC-RHC and
        MPC-delta_t take their STEP COUNTS from pMF-L/16's frozen entries
        (configs/experiments_final_frozen100.yaml, configs/experiments_flower_frozen100.yaml;
        STEP_FIELDS below) and every other field from the same task's JiT frozen entry.
        RHSO is the capacity-matched JiT-L/16 direct-planner entry with pMF-L/16's (N, M)
        from configs/experiments_theory_N_fixedB_final100.yaml (its solver and terminal mode
        kept, its theory diagnostics dropped: they are extra measurements inside the timed
        region).  Plain sampling (SDEdit) keeps JiT's own sampler settings.
    configs/experiments_rebuttal_timing_jitl_smoke.yaml
        two blocks (one RHSO, one MPC-delta_t) at 2 images.

    python scripts/make_paper_timing_configs.py           # write the four files
    python scripts/make_paper_timing_configs.py --check   # exit 1 if a committed file is stale
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_timing_config as mtc  # noqa: E402

FINAL = "configs/experiments_final_frozen100.yaml"
FLOWER = "configs/experiments_flower_frozen100.yaml"
THEORY = "configs/experiments_theory_N_fixedB_final100.yaml"
TASKS = mtc.TASKS
EXP = {"denoising": "denoising", "deblur": "deblurring", "super_resolution": "super_resolution",
       "random_inpaint": "random_inpainting", "box_inpaint": "box_inpainting"}
JITL_METHODS = ["sdedit", "pnp", "flower", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]
# The fields copied from pMF-L/16 for the count-matched methods.
STEP_FIELDS = {"pnp": ("num_pnp_steps",), "flower": ("num_flower_steps", "flower_cg_max_iters"),
               "dflow": ("steps", "num_opt_steps"), "mpc_rhc": ("num_mpc_steps", "n_ctrl", "K"),
               "mpc_delta_t": ("num_mpc_steps", "n_ctrl")}
# RHSO: pMF-L/16's (N, M); the entry itself comes from the capacity-matched B = 160 study.
RHSO_FIELDS = ("t0", "num_rhso_steps", "num_opt_steps", "lr", "mu", "optimizer", "solver",
               "rhso_terminal_mode")
# The task's table (rebuttal Task 9).  The generator refuses to write if pMF-L/16's frozen
# entries ever disagree with it.
TABLE = {
    "pnp": {"denoising": (10,), "deblur": (200,), "super_resolution": (20,),
            "random_inpaint": (20,), "box_inpaint": (50,)},
    "flower": {"denoising": (50, None), "deblur": (50, 100), "super_resolution": (50, None),
               "random_inpaint": (50, None), "box_inpaint": (50, None)},
    "dflow": {t: (1, n) for t, n in (("denoising", 320), ("deblur", 640),
                                     ("super_resolution", 640), ("random_inpaint", 640),
                                     ("box_inpaint", 640))},
    "mpc_rhc": {"denoising": (4, 40, 2), "deblur": (4, 40, 1), "super_resolution": (4, 40, 3),
                "random_inpaint": (4, 40, 3), "box_inpaint": (4, 40, 1)},
    "mpc_delta_t": {"denoising": (8, 20), "deblur": (8, 40), "super_resolution": (8, 40),
                    "random_inpaint": (8, 40), "box_inpaint": (8, 40)},
    "rhso": {"denoising": (4, 40), "deblur": (8, 20), "super_resolution": (4, 40),
             "random_inpaint": (4, 40), "box_inpaint": (8, 20)},
}
COUNT_MATCHED_COMMENT = ("timing only: step counts equal to pMF-L/16's frozen configuration; "
                         "quality is not evaluated from this run")
SDEDIT_COMMENT = ("timing only: plain sampling with JiT's own sampler settings (the reference, "
                  "not count-matched); quality is not evaluated from this run")
RHSO_COMMENT = ("timing only: RHSO, direct planner, capacity-matched JiT-L/16 entry with "
                "N = %d, M = %d (B = 160); quality is not evaluated from this run")
PMF_EXACT_SMOKE = ["denoising__rhso__rhso", "denoising__core__mpc_delta_t"]
JITL_SMOKE = ["denoising__rhso", "denoising__mpc_delta_t"]
TIMING_IMAGES, JITL_SMOKE_IMAGES = mtc.TIMING_IMAGES, mtc.SMOKE_IMAGES


def out_path(name, smoke=False):
    return mtc.out_path(name, smoke)


def _load(path):
    return yaml.safe_load((REPO_ROOT / path).read_text())


def _entry(cfg, exp, model, method):
    try:
        return dict(cfg["experiments"][exp]["models"][model]["methods"][method])
    except (KeyError, TypeError):
        raise SystemExit("no %s / %s / %s entry" % (exp, model, method))


# =====================================================================================
# block lists: [(name, problem, degradation, method, params, [comment lines])]
# =====================================================================================
def pmf_exact_blocks():
    blocks = []
    for entry, dups in mtc.plan("pmf"):
        comments = ["source: %s :: %s / %s / %s" % (entry.path, entry.experiment, entry.model,
                                                    entry.method)]
        comments += ["also covers (identical in every field): %s :: %s / %s / %s"
                     % (d.path, d.experiment, d.model, d.method) for d in dups]
        blocks.append((mtc.block_name(entry), entry.problem, entry.degradation, entry.method,
                       entry.params, comments))
    return blocks


def pmf_step_values(final, flower, method, problem):
    exp = EXP[problem]
    if method == "flower":
        src = _entry(flower, exp + "__flower", "pmf", "flower")
    elif method == "mpc_rhc":
        src = _entry(final, exp + "__rhc", "pmf", "mpc_rhc")
    elif method == "rhso":
        src = _entry(final, exp + "__rhso", "pmf", "rhso")
        return (src["num_rhso_steps"], src["num_opt_steps"])
    else:
        src = _entry(final, exp + "__core", "pmf", method)
    return tuple(src.get(k) for k in STEP_FIELDS[method])


def jitl_entry(final, flower, theory, method, problem):
    """(params, comment) of one JiT-L/16 block."""
    exp = EXP[problem]
    if method == "sdedit":
        return _entry(final, exp + "__core", "jit", "sdedit"), SDEDIT_COMMENT
    steps = pmf_step_values(final, flower, method, problem)
    if steps != TABLE[method][problem]:
        raise SystemExit("pMF-L/16's frozen %s / %s step counts %s differ from the task's table %s"
                         % (problem, method, steps, TABLE[method][problem]))
    if method == "rhso":
        n, m = steps
        block = theory["experiments"].get("%s_B160_N%d" % (exp, n))
        if block is None:
            raise SystemExit("%s has no %s_B160_N%d block" % (THEORY, exp, n))
        src = dict(block["models"]["jit"]["methods"]["rhso"])
        defaults = block.get("defaults") or {}
        if float(defaults.get("beta", 1.0)) != 1.0 or "beta" in src:
            raise SystemExit("%s_B160_N%d uses beta != 1" % (exp, n))
        merged = dict(defaults, **src)
        if (merged.get("num_rhso_steps"), merged.get("num_opt_steps")) != (n, m) \
                or merged.get("rhso_terminal_mode") != "direct":
            raise SystemExit("%s_B160_N%d is not the direct-planner (N, M) = (%d, %d) entry"
                             % (exp, n, n, m))
        params = {k: merged[k] for k in RHSO_FIELDS if k in merged}
        return params, RHSO_COMMENT % (n, m)
    if method == "sdedit":
        return _entry(final, exp + "__core", "jit", "sdedit"), SDEDIT_COMMENT
    if method == "flower":
        params = _entry(flower, exp + "__flower", "jit", "flower")
    elif method == "mpc_rhc":
        params = _entry(final, exp + "__rhc", "jit", "mpc_rhc")
    else:
        params = _entry(final, exp + "__core", "jit", method)
    for key, value in zip(STEP_FIELDS[method], steps):
        if value is None:
            params.pop(key, None)              # pMF-L/16 runs this field at its default
        else:
            params[key] = value
    return params, COUNT_MATCHED_COMMENT


def jitl_blocks():
    final, flower, theory = _load(FINAL), _load(FLOWER), _load(THEORY)
    blocks = []
    for problem in TASKS:
        degradation = dict(final["experiments"][EXP[problem] + "__core"]["degradation"])
        for method in JITL_METHODS:
            params, comment = jitl_entry(final, flower, theory, method, problem)
            blocks.append(("%s__%s" % (EXP[problem], method), problem, degradation, method,
                           params, [comment]))
    return blocks


# =====================================================================================
# rendering
# =====================================================================================
def render(title, header, models, blocks, num_images):
    base, _ = mtc.shared_sections()
    lines = ["# " + "=" * 85, "# " + title, "# " + "=" * 85, "#",
             "# Generated by scripts/make_paper_timing_configs.py; do not edit by hand.", "#"]
    lines += [("# " + line) if line else "#" for line in header]
    lines += ["# " + "=" * 85, ""]
    runtime = dict(base["runtime"])
    runtime.update(max_atomic_jobs=64, gpu_memory_profiling=True, timing_detail=True)
    head = {"runtime": runtime, "data": base["data"], "models": models,
            "defaults": base["defaults"], "metrics": base["metrics"]}
    lines.append(yaml.safe_dump(head, sort_keys=False, default_flow_style=False).rstrip())
    lines += ["", "experiments:"]
    model = next(iter(models))
    for name, problem, degradation, method, params, comments in blocks:
        lines.append("")
        lines += ["  # %s" % c for c in comments]
        lines += ["  %s:" % name, "    enabled: true", "    problem: %s" % problem,
                  "    num_images: %d" % num_images,
                  "    degradation: %s" % mtc._flow(degradation), "    models:",
                  "      %s:" % model, "        batch_size: 1", "        methods:",
                  "          %s: %s" % (method, mtc._flow(params))]
    return "\n".join(lines) + "\n"


PMF_HEADER = [
    "pMF-L/16 with EXACT JAX peak memory: the same %d blocks as",
    "configs/experiments_rebuttal_timing_pmf.yaml (names and fields unchanged; batch size 1,",
    "%d images) with models.pmf.param_loading: host_once.  That opt-in loader builds pMF's",
    "training state and restores the checkpoint on the host and transfers the selected EMA",
    "parameters to the GPU once, so the model load never sets a JAX allocator peak above",
    "the resident parameters; each configuration's lifetime peak (one configuration per",
    "process) is then its exact allocator high-water mark (src/timing.exact_jax_peak).",
    "",
    "  python scripts/run_timing.py --config %s --run-id %s",
]
JITL_HEADER = [
    "JiT-L/16 (PyTorch, BF16 network): 7 methods x 5 tasks, batch size 1, %d images, measured",
    "as in rebuttal Task 1.  PnP-Flow, Flower, D-Flow, MPC-RHC and MPC-delta_t use the step",
    "counts of pMF-L/16's frozen configuration.  RHSO uses the capacity-matched JiT-L/16",
    "direct-planner entries (B = 160) with pMF-L/16's (N, M), from",
    "configs/experiments_theory_N_fixedB_final100.yaml, without their theory diagnostics.",
    "Plain sampling (SDEdit) keeps JiT's own sampler settings: it is the reference, not",
    "count-matched.  Timing only: quality is not evaluated from this run.",
    "",
    "  python scripts/run_timing.py --config %s --run-id %s",
]


def outputs():
    files = {}
    pmf = pmf_exact_blocks()
    if len(pmf) != 45:
        raise SystemExit("expected the 45 pMF timing blocks, got %d" % len(pmf))
    pmf_models = {"pmf": {"param_loading": "host_once"}}
    for smoke in (False, True):
        path = out_path("pmf_exact", smoke)
        blocks = [b for b in pmf if b[0] in PMF_EXACT_SMOKE] if smoke else pmf
        if smoke and sorted(b[0] for b in blocks) != sorted(PMF_EXACT_SMOKE):
            raise SystemExit("pmf_exact smoke blocks missing: %s" % PMF_EXACT_SMOKE)
        header = list(PMF_HEADER)
        header[0] = PMF_HEADER[0] % len(blocks)
        header[2] = PMF_HEADER[2] % TIMING_IMAGES
        header[-1] = PMF_HEADER[-1] % (path, "timing_pmf_exact" + ("_smoke" if smoke else ""))
        if smoke:
            header += ["", "SMOKE: one RHSO and one MPC-delta_t block at the same 3 images as",
                       "timing_pmf, so their PSNR can be compared with timing_pmf's (identical",
                       "parameters); run it WITHOUT run.py's checks (they would allocate inside",
                       "the measured process and void the exact peak)."]
        files[path] = render("REBUTTAL TIMING AND EXACT PEAK MEMORY -- pMF-L/16%s"
                             % (", SMOKE variant" if smoke else ""),
                             header, pmf_models, blocks, TIMING_IMAGES)
    jit = jitl_blocks()
    jit_models = {"jit": {"variant": "JiT-L/16"}}
    for smoke in (False, True):
        path = out_path("jitl", smoke)
        blocks = [b for b in jit if b[0] in JITL_SMOKE] if smoke else jit
        n = JITL_SMOKE_IMAGES if smoke else TIMING_IMAGES
        header = list(JITL_HEADER)
        header[0] = JITL_HEADER[0] % n
        header[-1] = JITL_HEADER[-1] % (path, "timing_jitl" + ("_smoke" if smoke else ""))
        if smoke:
            header += ["", "SMOKE: one RHSO and one MPC-delta_t block at 2 images."]
        files[path] = render("REBUTTAL TIMING AND PEAK MEMORY -- JiT-L/16%s"
                             % (", SMOKE variant" if smoke else ""), header, jit_models,
                             blocks, n)
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
        print("wrote %s (%d blocks)" % (path, text.count("    enabled: true")))
    if stale:
        print("STALE: %s" % ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
