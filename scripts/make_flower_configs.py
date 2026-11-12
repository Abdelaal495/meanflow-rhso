#!/usr/bin/env python
"""Flower on the pixel priors (pMF-L/16, JiT-B/16): the configurations for rebuttal Task 3.

Flower has one tuned knob, the iteration count N = `num_flower_steps`, screened over the
pre-registered values {50, 100, 200, 500} (the paper uses 100 to 500) at t0 = 1 with the
paper's reported gamma = 0 and its CG settings.  Each prior is tuned on its own screen and
frozen from its own results; nothing is carried from one prior to the other.

    # 1. the screening grid (4 images) and its 2-image smoke variant
    python scripts/make_flower_configs.py screen
    python run.py --config configs/experiments_flower_screen_smoke.yaml --models pmf --run-id flower_smoke_pmf
    python run.py --config configs/experiments_flower_screen.yaml --models pmf --run-id flower_pmf --no-figures

    # 2. a refine round at 8 images that cannot leave the screened grid
    python scripts/hpo.py propose --from outputs/flower_pmf --models pmf --images 8 --no-paper \
        --lock flower=num_flower_steps --out configs/flower_round2_pmf.yaml
    python run.py --config configs/flower_round2_pmf.yaml --run-id flower_pmf_r2 --no-figures

    # 3. freeze: lowest LPIPS at 8 images inside the screened grid -> the 100-image evaluation
    python scripts/make_flower_configs.py freeze --from outputs/flower_pmf_r2 \
        --grid outputs/flower_pmf --models pmf

The smoke variants also run each prior's frozen SDEdit configuration on the same images, so
the `shared_initial_state` check has two methods to compare at the same t0 and actually
tests Flower's pairing instead of passing for want of a partner.

`freeze` is the rule of `make_matched_configs.py freeze` (hpo.restrict + hpo.winners: the
lowest LPIPS inside the screened grid), used because `hpo.py finalize` expects all six
benchmark methods in every cell.  It refuses a winner outside the pre-registered N values or
run with a non-default gamma / CG setting.
"""
import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matched = _load("make_matched_configs")
hpo = matched.hpo

MODELS = ["pmf", "jit"]                 # pixel priors only: Flower refines the image
METHOD = "flower"
SCREEN_STEPS = [50, 100, 200, 500]
SMOKE_STEPS = SCREEN_STEPS[0]
# The paper's reported setting; written into no config (they are the builtin defaults), but
# a winner that ran with anything else was not screened under this protocol.
DEFAULT_GAMMA, DEFAULT_CG_MAX_ITERS, DEFAULT_CG_TOL = 0, 50, 1e-5

BENCHMARK = matched.BENCHMARK
SCREEN_OUT = "configs/experiments_flower_screen.yaml"
FROZEN_OUT = "configs/experiments_flower_frozen100.yaml"
SCREEN_IMAGES, SMOKE_IMAGES, FROZEN_IMAGES = 4, 2, 100


def sdedit_entry(benchmark, model, problem):
    """The prior's own frozen SDEdit configuration for this task (pairing partner)."""
    entry = matched._entry(benchmark, hpo.EXP_NAME[problem] + "__core", model, "sdedit")
    fields = [("t0", "1.0")]
    for key in ("solver", "steps"):
        if key in entry:
            fields.append((key, entry[key]))
    return fields


def flower_fields(steps):
    return [("t0", "1.0"), ("num_flower_steps", steps)]


def experiment_block_multi(name, problem, num_images, per_model):
    """Like matched.experiment_block, but with several methods under one model."""
    lines = ["", "  %s:" % name, "    enabled: true", "    problem: %s" % problem,
             "    num_images: %d" % num_images,
             "    degradation: %s" % hpo.DEGRADATION[problem], "    models:"]
    for model, batch, methods in per_model:
        lines += ["      %s:" % model, "        batch_size: %d" % batch, "        methods:"]
        lines += [matched.method_line(method, fields, comment)
                  for method, fields, comment in methods]
    return lines


def write(out, smoke, models, lines, n_jobs, title, usage, text, pool, save_images):
    body = matched.header(title, usage, text, matched.max_jobs_for(n_jobs), pool, models,
                          save_images)
    Path(REPO_ROOT / out).write_text(body + "\n".join(lines) + "\n")
    print("wrote %s: %d jobs" % (out, n_jobs))


# =====================================================================================
# screen
# =====================================================================================
SCREEN_TEXT = """Screening grid for Flower (Pourya, El Rawas and Unser, ICLR 2026) on the two pixel
priors.  One tuned knob, num_flower_steps over the pre-registered %s, at t0 = 1
with gamma = 0 (the paper's reported setting) and conjugate gradient for deblurring at the
paper's limits (50 iterations, relative residual 1e-5).  Flower never backpropagates
through the prior: N network evaluations and 0 backprops per image.  Each prior is
screened and selected on its own results.  Selection: lowest LPIPS on 8 tuning images
inside this grid (scripts/hpo.py propose --lock flower=num_flower_steps, then
scripts/make_flower_configs.py freeze).
Generated by scripts/make_flower_configs.py screen; %d atomic jobs over both priors."""


def screen_lines(benchmark, smoke):
    lines, n_jobs = [], 0
    n_images = SMOKE_IMAGES if smoke else SCREEN_IMAGES
    for problem in hpo.PROBLEMS:
        per_model = []
        for model in MODELS:
            steps = SMOKE_STEPS if smoke else list(SCREEN_STEPS)
            jobs = 1 if smoke else len(SCREEN_STEPS)
            methods = [(METHOD, flower_fields(steps),
                        "N evals, 0 BP per image; %d job%s" % (jobs, "" if jobs == 1 else "s"))]
            n_jobs += jobs
            if smoke:
                methods.insert(0, ("sdedit", sdedit_entry(benchmark, model, problem),
                                   "pairing partner: this prior's frozen SDEdit"))
                n_jobs += 1
            per_model.append((model, min(hpo.BATCH[(model, METHOD)], n_images), methods))
        lines += experiment_block_multi("%s__flower" % hpo.EXP_NAME[problem], problem,
                                        n_images, per_model)
    return lines, n_jobs


def screen(a):
    benchmark = matched.load_yaml(a.benchmark)
    for smoke in (False, True):
        out = matched.smoke_path(a.out) if smoke else Path(a.out).as_posix()
        lines, n_jobs = screen_lines(benchmark, smoke)
        title = ("SMOKE: Flower on pMF / JiT, every task at 2 images" if smoke else
                 "FLOWER SCREEN: pMF-L/16 and JiT-B/16 at %d images" % SCREEN_IMAGES)
        usage = ("#   python run.py --config %s --models <prior> --run-id flower_<prior> --no-figures\n"
                 "#   one prior per job: every block carries per-model entries, and --models drops the rest\n"
                 % out)
        text = SCREEN_TEXT % (SCREEN_STEPS, n_jobs)
        if smoke:
            text += ("\nSmoke variant: N = %d only, next to each prior's frozen SDEdit from %s,\n"
                     "so the shared_initial_state check compares two methods at t0 = 1.  It checks\n"
                     "that every task resolves, runs and pairs, not which N wins."
                     % (SMOKE_STEPS, a.benchmark))
        write(out, smoke, MODELS, lines, n_jobs, title, usage, text, matched.TUNING_POOL, False)


# =====================================================================================
# freeze
# =====================================================================================
def _as_int(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def frozen_winners(runs_from, runs_grid, models):
    """(model, problem) -> (N, entry, candidates beaten), refusing anything off-protocol."""
    ranked, _, _ = hpo.load(runs_from)
    grids = hpo.load(runs_grid)[1]
    best = hpo.winners(hpo.restrict(ranked, models, grids))
    out = {}
    for model in models:
        for problem in hpo.PROBLEMS:
            cell = (model, problem, METHOD)
            if cell not in best:
                raise SystemExit("no Flower winner for %s inside the screened grid" % "/".join(cell))
            key, entry, n_candidates = best[cell]
            steps = int(round(key[0]))
            if steps not in SCREEN_STEPS:
                raise SystemExit("%s: winner N=%d is not one of the pre-registered %s"
                                 % ("/".join(cell), steps, SCREEN_STEPS))
            row = entry["row"]
            gamma = _as_int(row.get("flower_gamma"))
            iters = _as_int(row.get("flower_cg_max_iters"))
            tol = _as_float(row.get("flower_cg_tol"))
            if gamma != DEFAULT_GAMMA or iters != DEFAULT_CG_MAX_ITERS or (
                    tol is None or abs(tol - DEFAULT_CG_TOL) > 1e-12):
                raise SystemExit("%s: winner ran with flower_gamma=%s, CG %s/%s; the screen uses "
                                 "gamma=%d and CG %d/%g"
                                 % ("/".join(cell), row.get("flower_gamma"),
                                    row.get("flower_cg_max_iters"), row.get("flower_cg_tol"),
                                    DEFAULT_GAMMA, DEFAULT_CG_MAX_ITERS, DEFAULT_CG_TOL))
            out[(model, problem)] = (steps, entry, n_candidates)
    return out


def freeze(a):
    models = [m for m in a.models.split(",") if m]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise SystemExit("Flower runs on the pixel priors %s only, not %s" % (MODELS, unknown))
    benchmark = matched.load_yaml(a.benchmark)
    best = frozen_winners(getattr(a, "from"), a.grid, models)
    stems = ", ".join(sorted(Path(r).name for r in getattr(a, "from")))

    for smoke in (False, True):
        out = matched.smoke_path(a.out) if smoke else Path(a.out).as_posix()
        n_images = SMOKE_IMAGES if smoke else a.num_images
        lines, n_jobs = [], 0
        for problem in hpo.PROBLEMS:
            per_model = []
            for model in models:
                steps, entry, n_candidates = best[(model, problem)]
                comment = ("LPIPS %.4f PSNR %.2f at %d tuning images, best of %d"
                           % (entry["lpips"], entry["psnr"] or 0.0, entry["n"], n_candidates))
                methods = [(METHOD, flower_fields(steps), comment)]
                if smoke:
                    methods.insert(0, ("sdedit", sdedit_entry(benchmark, model, problem),
                                       "pairing partner: this prior's frozen SDEdit"))
                n_jobs += len(methods)
                batch = hpo.BATCH_FINAL[model]["core"]
                per_model.append((model, min(batch, n_images), methods))
            lines += experiment_block_multi("%s__flower" % hpo.EXP_NAME[problem], problem,
                                            n_images, per_model)
        title = ("SMOKE: Flower frozen configuration at 2 images" if smoke else
                 "FLOWER FINAL at t0 = 1.0 on the frozen ImageNet-100 (class seed 42 / image seed 43)")
        usage = ("#   python scripts/build_local_imagenet_pool.py --frozen-manifest "
                 "benchmarks/imagenet100_c42_i43/manifest.csv\n"
                 "#   python run.py --config %s --models <prior> --run-id <id> --no-figures\n" % out)
        text = ("Each entry is the lowest-LPIPS num_flower_steps of its (prior, task) cell among\n"
                "the candidates inside that prior's own screened grid %s (the trailing\n"
                "comment), with gamma = 0 and the paper's CG settings.\n"
                "Generated by scripts/make_flower_configs.py freeze from %s."
                % (SCREEN_STEPS, stems))
        write(out, smoke, models, lines, n_jobs, title, usage, text, matched.FROZEN_POOL, not smoke)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, out):
        q.add_argument("--benchmark", default=BENCHMARK,
                       help="the frozen benchmark whose SDEdit entries pair the smoke runs "
                            "(default: %(default)s)")
        q.add_argument("--out", default=out, help="output path; the smoke variant is <out>_smoke.yaml")

    q = sub.add_parser("screen", help="write the screening grid and its smoke variant")
    common(q, SCREEN_OUT)
    q.set_defaults(run=screen)

    q = sub.add_parser("freeze", help="write the frozen 100-image configuration and its smoke variant")
    common(q, FROZEN_OUT)
    q.add_argument("--from", dest="from", action="append", required=True, metavar="RUN",
                   help="run directory whose results.csv is ranked (repeatable)")
    q.add_argument("--grid", action="append", required=True, metavar="RUN",
                   help="the screening run directories: a winner must lie inside their grid")
    q.add_argument("--models", default=",".join(MODELS))
    q.add_argument("--num-images", type=int, default=FROZEN_IMAGES)
    q.set_defaults(run=freeze)

    a = p.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()
