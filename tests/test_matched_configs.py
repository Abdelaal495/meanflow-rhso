"""Matched-budget configurations (rebuttal Task 2): every job costs exactly its target.

`scripts/make_matched_configs.py` writes one experiment block per budget-defining pair, so
the Cartesian expansion in `src/config.py` can never produce an off-budget (N, M) or (N, C).
This suite resolves the generated files through the REAL validator and planner -- the same
code `run.py --dry-run` runs -- and checks:

  1. regenerating gives the committed screen and smoke files byte for byte;
  2. the budgets read from configs/experiments_final_frozen100.yaml are the rebuttal plan's
     Table 6, written out independently here;
  3. every resolved job's backprops per image equal the BP in its block name, N*M for RHSO
     and (N-1)*C for MPC-delta_t, and the (prior, task, method, pair) set is EXACTLY the
     intended one -- no off-budget pair appears anywhere;
  4. each pair carries the full pre-registered lr x mu (RHSO) or lam x lr (MPC-delta_t)
     grid, and JiT/SiT RHSO plans with `direct`;
  5. `--models <prior>` (run.py's own filter) leaves only that prior's jobs, under the
     file's max_atomic_jobs, so no --max-jobs flag is needed;
  6. the smoke variant covers every pair at 2 images, one job each;
  7. --include-optional adds exactly iMF MPC-delta_t at 640 BP, and the default does not;
  8. `freeze` on synthetic tuning runs picks the planted winner inside the screened grid
     (an edge-extended candidate cannot win), resolves to the target BP at 100 images, and
     refuses a winner whose pair is off its cell's budget;
  9. no generated comment claims a value was reused from another checkpoint.

Runs on CPU with no checkpoint:

    python tests/test_matched_configs.py
"""

import argparse
import collections
import csv
import importlib.util
import itertools
import re
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import resolve_run_plan, validate_config         # noqa: E402

SCREEN = REPO_ROOT / "configs" / "experiments_matched_screen.yaml"
SMOKE = REPO_ROOT / "configs" / "experiments_matched_screen_smoke.yaml"

PASSED = [0]
FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mmc = load_script("make_matched_configs")

# ---- the rebuttal plan, Task 2, written out independently of the generator -----------
TASKS = ["denoising", "deblur", "super_resolution", "random_inpaint", "box_inpaint"]
MPC_BP = {"pmf": [140, 280, 280, 280, 280], "jit": [140, 280, 280, 140, 140],
          "imf": [280] * 5, "sit": [280, 280, 120, 280, 280]}
RHSO_BP = {"pmf": 160, "imf": 640}
PAIRS = {("rhso", 140): [(4, 35), (7, 20)], ("rhso", 280): [(4, 70), (8, 35)],
         ("rhso", 120): [(4, 30), (8, 15)],
         ("mpc_delta_t", 160): [(5, 40), (9, 20)], ("mpc_delta_t", 640): [(9, 80), (17, 40)]}
RHSO_LR = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
RHSO_MU = [0.0, 0.2, 1.0]
MPC_LAM = [10.0, 30.0, 180.0, 360.0, 540.0, 1000.0]
MPC_LR = [0.05, 0.1, 0.15, 0.3]


def expected_cells(include_optional=False):
    """{(model, task, method, pair): BP} the plan asks for."""
    out = {}
    for model, bps in MPC_BP.items():
        for task, bp in zip(TASKS, bps):
            for p in PAIRS[("rhso", bp)]:
                out[(model, task, "rhso", p)] = bp
    for model in ["pmf"] + (["imf"] if include_optional else []):
        for task in TASKS:
            for p in PAIRS[("mpc_delta_t", RHSO_BP[model])]:
                out[(model, task, "mpc_delta_t", p)] = RHSO_BP[model]
    return out


def pair_of(s):
    return (s.num_rhso_steps, s.num_opt_steps) if s.method == "rhso" else (s.num_mpc_steps, s.n_ctrl)


def formula_bp(s):
    n, x = pair_of(s)
    return n * x if s.method == "rhso" else (n - 1) * x


def resolve(path_or_cfg, run_id="matched_test"):
    cfg = yaml.safe_load(open(path_or_cfg)) if isinstance(path_or_cfg, Path) else path_or_cfg
    return cfg, resolve_run_plan(cfg, validate_config(cfg), run_id=run_id)


NAME = re.compile(r"^(?P<task>\w+?)__(?P<method>rhso|mpc_delta_t)_bp(?P<bp>\d+)_n(?P<n>\d+)$")


def budget_violations(specs):
    """Every job whose resolved cost disagrees with its block name or the formula."""
    bad = []
    for s in specs:
        m = NAME.match(s.experiment)
        if (m is None or int(m["bp"]) != s.expected_backprops or int(m["n"]) != pair_of(s)[0]
                or formula_bp(s) != s.expected_backprops or m["method"] != s.method):
            bad.append((s.experiment, s.model, pair_of(s), s.expected_backprops))
    return bad


def run_generator(argv):
    mmc.main(argv)


# =====================================================================================
def test_screen(tmp):
    print("\n1. the committed files are what the generator writes")
    out = tmp / "experiments_matched_screen.yaml"
    run_generator(["screen", "--out", str(out)])
    smoke_out = Path(mmc.smoke_path(str(out)))
    # the usage line in each header names the file it was written to
    here = out.parent.as_posix() + "/"
    for label, fresh, committed in (("screen", out, SCREEN), ("smoke", smoke_out, SMOKE)):
        text = fresh.read_text().replace(here, "configs/")
        check("%s regenerates byte for byte" % label, text == committed.read_text(),
              committed.relative_to(REPO_ROOT).as_posix())

    print("\n2. budgets come from the frozen benchmark and match the plan's Table 6")
    targets = mmc.budget_targets(mmc.load_yaml(mmc.BENCHMARK), include_optional=True)
    got = {(m, p, me, pr): bp for m, p, me, bp, prs in targets for pr in prs}
    check("frozen benchmark budgets == Table 6", got == expected_cells(include_optional=True),
          "%d (prior, task, method, pair) cells" % len(got))

    print("\n3-4. the screen, resolved by src/config.py")
    cfg, plan = resolve(SCREEN)
    specs = plan.specs
    cells = collections.Counter((s.model, s.problem, s.method, pair_of(s)) for s in specs)
    bp_of = {(s.model, s.problem, s.method, pair_of(s)): s.expected_backprops for s in specs}
    check("job count", len(specs) == 960, "%d atomic jobs (720 RHSO + 240 pMF MPC-delta_t)" % len(specs))
    bad = budget_violations(specs)
    check("every job costs its block's BP", not bad,
          "%d jobs checked; violations: %s" % (len(specs), bad[:3]))
    check("exactly the intended pairs, no off-budget pair", bp_of == expected_cells(),
          "%d pairs; extra: %s missing: %s" % (
              len(bp_of), sorted(set(bp_of) - set(expected_cells()))[:3],
              sorted(set(expected_cells()) - set(bp_of))[:3]))
    grids_ok = True
    for (model, task, method, p), n in cells.items():
        sel = [s for s in specs if (s.model, s.problem, s.method, pair_of(s)) == (model, task, method, p)]
        if method == "rhso":
            want = set(itertools.product(RHSO_LR, RHSO_MU))
            have = {(s.lr, s.mu) for s in sel}
        else:
            want = set(itertools.product(MPC_LAM, MPC_LR))
            have = {(s.lam, s.lr) for s in sel}
        grids_ok &= (have == want and n == len(want))
    check("full pre-registered tuning grid per pair", grids_ok,
          "RHSO 6 lr x 3 mu = 18, MPC-delta_t 6 lam x 4 lr = 24, per pair")
    planners = {(s.model, s.rhso_terminal_mode, s.rhso_terminal_planner) for s in specs
                if s.method == "rhso"}
    check("RHSO plans directly on every prior", planners == {
        ("jit", "direct", "direct_clean_endpoint_prediction"),
        ("sit", "direct", "velocity_extrapolation"),
        ("pmf", "direct", "learned_finite_interval_map"),
        ("imf", "direct", "learned_finite_interval_map")}, str(sorted(planners)))

    print("\n5. one prior at a time with run.py's --models filter")
    import run as runpy
    limit = int(cfg["runtime"]["max_atomic_jobs"])
    want_counts = {"jit": 180, "sit": 180, "pmf": 420, "imf": 180}
    counts = {}
    for prior in want_counts:
        args = argparse.Namespace(max_jobs=None, output_root=None, cache_root=None,
                                  replicate=None, resume=None, no_gpu_memory=False,
                                  experiments=None, models=prior, num_images=None)
        sub = runpy.apply_cli_overrides(cfg, args)
        _, sub_plan = resolve(sub, "matched_%s" % prior)
        counts[prior] = (len(sub_plan.specs), {s.model for s in sub_plan.specs},
                         not budget_violations(sub_plan.specs))
    check("--models <prior> keeps only that prior",
          all(c[0] == want_counts[p] and c[1] == {p} and c[2] for p, c in counts.items()),
          ", ".join("%s=%d" % (p, c[0]) for p, c in counts.items()))
    check("max_atomic_jobs covers the whole file",
          len(specs) <= limit and all(c[0] <= limit for c in counts.values()),
          "limit %d >= %d jobs (all priors); no --max-jobs needed" % (limit, len(specs)))

    print("\n6. the 2-image smoke variant")
    _, smoke = resolve(SMOKE, "matched_smoke")
    smoke_cells = {(s.model, s.problem, s.method, pair_of(s)): s.expected_backprops for s in smoke.specs}
    per_block = collections.Counter((s.experiment, s.model) for s in smoke.specs)
    check("smoke covers every pair once at 2 images",
          smoke_cells == bp_of and set(per_block.values()) == {1}
          and {s.num_images for s in smoke.specs} == {2} and not budget_violations(smoke.specs),
          "%d jobs, same %d pairs and BP as the screen" % (len(smoke.specs), len(smoke_cells)))

    print("\n7. --include-optional")
    opt = tmp / "screen_optional.yaml"
    run_generator(["screen", "--include-optional", "--out", str(opt)])
    _, opt_plan = resolve(opt, "matched_optional")
    extra = {(s.model, s.problem, s.method, pair_of(s)): s.expected_backprops for s in opt_plan.specs}
    added = {k: v for k, v in extra.items() if k not in bp_of}
    check("adds exactly iMF MPC-delta_t at 640 BP",
          extra == expected_cells(include_optional=True) and len(opt_plan.specs) == 960 + 240
          and set(added.values()) == {640} and {k[0] for k in added} == {"imf"}
          and not budget_violations(opt_plan.specs),
          "%d added pairs, %d jobs" % (len(added), len(opt_plan.specs)))
    check("the default screen has no iMF MPC-delta_t",
          not any(s.model == "imf" and s.method == "mpc_delta_t" for s in specs))
    return specs


# =====================================================================================
FREEZE_COLUMNS = ["job_id", "model", "problem", "method", "status", "lpips", "psnr", "num_images",
                  "num_rhso_steps", "num_opt_steps", "lr", "mu", "num_mpc_steps", "n_ctrl",
                  "lam", "rhso_terminal_mode"]


def as_row(s, lpips, n_images, **override):
    row = dict(job_id=s.job_id, model=s.model, problem=s.problem, method=s.method, status="ok",
               lpips=lpips, psnr=25.0, num_images=n_images)
    if s.method == "rhso":
        row.update(num_rhso_steps=s.num_rhso_steps, num_opt_steps=s.num_opt_steps, lr=s.lr,
                   mu=s.mu, rhso_terminal_mode=s.rhso_terminal_mode)
    else:
        row.update(num_mpc_steps=s.num_mpc_steps, n_ctrl=s.n_ctrl, lam=s.lam, lr=s.lr)
    row.update(override)
    return row


def write_results(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FREEZE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FREEZE_COLUMNS})
    return directory


def test_freeze(tmp, screen_specs):
    print("\n8. freeze on synthetic tuning runs")
    # The planted winner of each cell is the job with the highest index in the screen;
    # round 2 re-runs it and two others at 8 images, plus an lr pushed past the grid edge
    # that scores best of all and must still lose (it lies outside the screened grid).
    by_cell = collections.defaultdict(list)
    for s in screen_specs:
        by_cell[(s.model, s.problem, s.method)].append(s)
    screen_rows, round2_rows, planted = [], [], {}
    for cell, specs in sorted(by_cell.items()):
        for i, s in enumerate(specs):
            screen_rows.append(as_row(s, 0.5 - 0.001 * i, 4))
        win = specs[-1]
        planted[cell] = win
        for i, s in enumerate([win, specs[0], specs[len(specs) // 2]]):
            round2_rows.append(as_row(s, 0.2 + 0.01 * i, 8))
        round2_rows.append(as_row(win, 0.1, 8, job_id=win.job_id + "_edge", lr=win.lr * 1.8))
    screen_run = write_results(tmp / "matched_screen", screen_rows)
    round2_run = write_results(tmp / "matched_r2", round2_rows)

    out = tmp / "frozen.yaml"
    run_generator(["freeze", "--from", str(round2_run), "--grid", str(screen_run), "--out", str(out)])
    _, plan = resolve(out, "matched_frozen")
    _, smoke = resolve(Path(mmc.smoke_path(str(out))), "matched_frozen_smoke")
    target = {(m, p, me): bp for (m, p, me, _), bp in expected_cells().items()}
    got = {(s.model, s.problem, s.method): s for s in plan.specs}
    check("one job per (prior, task, method) at 100 images",
          set(got) == set(target) and len(plan.specs) == len(target)
          and {s.num_images for s in plan.specs} == {100}, "%d jobs" % len(plan.specs))
    check("every frozen job costs its cell's target BP",
          all(got[c].expected_backprops == bp and formula_bp(got[c]) == bp for c, bp in target.items()))
    same = all(pair_of(got[c]) == pair_of(w) and got[c].lr == w.lr
               and (got[c].mu == w.mu if c[2] == "rhso" else got[c].lam == w.lam)
               for c, w in planted.items())
    check("planted winner chosen; edge-extended lr cannot win", same,
          "every cell froze its planted (pair, lr, mu|lam)")
    check("JiT/SiT RHSO stays direct",
          all(s.rhso_terminal_mode == "direct" for s in plan.specs if s.method == "rhso"))
    smoke_bp = {(s.model, s.problem, s.method): s.expected_backprops for s in smoke.specs}
    check("frozen smoke: same jobs and BP at 2 images",
          smoke_bp == target and {s.num_images for s in smoke.specs} == {2})

    # an off-budget winner: pMF deblurring (280 BP) given (7, 35) = 245 BP, whose N and M
    # both exist elsewhere in pMF's screened grid, so the grid restriction alone admits it
    win = planted[("pmf", "deblur", "rhso")]
    rows = round2_rows + [as_row(win, 0.01, 8, job_id="offbudget", num_rhso_steps=7,
                                 num_opt_steps=35)]
    bad_run = write_results(tmp / "matched_r2_bad", rows)
    try:
        run_generator(["freeze", "--from", str(bad_run), "--grid", str(screen_run),
                       "--out", str(tmp / "bad.yaml")])
        refused = ""
    except SystemExit as e:
        refused = str(e)
    check("refuses a winner off its cell's budget", "not one of the 280-BP pairs" in refused,
          refused[:90])
    return [out, Path(mmc.smoke_path(str(out)))]


def test_comments(paths):
    print("\n9. config comments")
    offending = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if "#" in line and re.search(r"reus|carried over|copied from|same as (the )?(jit|sit|pmf|imf)",
                                         line.split("#", 1)[1], re.I):
                offending.append((Path(p).name, line.strip()))
    check("no comment says a value was reused across checkpoints", not offending,
          "%d files scanned; %s" % (len(paths), offending[:2]))


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        specs = test_screen(tmp)
        frozen = test_freeze(tmp, specs)
        test_comments([SCREEN, SMOKE] + frozen)
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
