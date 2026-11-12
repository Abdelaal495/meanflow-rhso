"""`scripts/hpo.py propose --lock`: a matched-budget search can never leave its budget.

The matched-budget comparison (rebuttal Task 2) screens RHSO at fixed (N, M) pairs and
MPC-delta_t at fixed (N, C) pairs.  Without `--lock`, `propose` pushes a winner that sits on a
grid edge one step further, and on N, M or C that changes the backprop count.  This suite
runs the real script on synthetic `results.csv` files whose winners sit on EVERY edge, and
checks:

  1. without --lock the push really happens (so the rest of the suite has teeth);
  2. with --lock no candidate has a budget-defining pair that was not screened, the locked
     edges are reported, and the unlocked knobs (lr, mu, lam) are still extended;
  3. the round-2 configuration, resolved by the REAL config system, costs exactly the
     screened backprops on every job;
  4. JiT RHSO candidates keep `rhso_terminal_mode: direct` (otherwise `auto` would silently
     switch them to the suffix planner and a much larger cost);
  5. on an input that predates both features, the output is byte-identical to hpo.py as it
     stood before this change (commit 4d63970);
  6. malformed --lock values and a cell that mixes planners are refused.

Runs on CPU with no checkpoint:

    python tests/test_hpo_lock.py
"""

import csv
import importlib.util
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import resolve_run_plan, validate_config         # noqa: E402

HPO = REPO_ROOT / "scripts" / "hpo.py"
BASE_COMMIT = "4d63970"                     # hpo.py before --lock and terminal-mode carry-over

PASSED = [0]
FAILED = []


def check(name, ok, detail=""):
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


def load_hpo():
    spec = importlib.util.spec_from_file_location("hpo_under_test", HPO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hpo = load_hpo()

RHSO_PAIRS = [(4, 70), (8, 35)]             # 280 BP
MPC_PAIRS = [(5, 40), (9, 20)]              # 160 BP
RHSO_LR = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
RHSO_MU = [0.0, 0.2, 1.0]
MPC_LAM = [10.0, 30.0, 180.0, 360.0, 540.0, 1000.0]
MPC_LR = [0.05, 0.1, 0.15, 0.3]
COLUMNS = ["job_id", "model", "problem", "method", "status", "lpips", "psnr", "num_images",
           "num_rhso_steps", "num_opt_steps", "lr", "mu", "num_mpc_steps", "n_ctrl", "lam",
           "rhso_terminal_mode"]


def screen_rows(jit_mode="direct", with_mode_column=True):
    """A screen whose winners sit on every edge: N high, M/C low, lr/mu/lam high."""
    rows = []
    for model in ("pmf", "jit"):
        for (n, m), lr, mu in itertools.product(RHSO_PAIRS, RHSO_LR, RHSO_MU):
            lp = 0.5 - 0.02 * (n == 8) - 0.5 * lr - 0.01 * mu
            rows.append(dict(model=model, problem="deblur", method="rhso", num_rhso_steps=n,
                             num_opt_steps=m, lr=lr, mu=mu, lpips=lp,
                             rhso_terminal_mode=(jit_mode if model == "jit" else "direct")))
    for (n, c), lam, lr in itertools.product(MPC_PAIRS, MPC_LAM, MPC_LR):
        lp = 0.5 - 0.02 * (n == 9) - 1e-5 * lam - 0.1 * lr
        rows.append(dict(model="pmf", problem="deblur", method="mpc_delta_t", num_mpc_steps=n,
                         n_ctrl=c, lam=lam, lr=lr, lpips=lp))
    for i, r in enumerate(rows):
        r.update(job_id="job%04d" % i, status="ok", psnr=25.0, num_images=4)
        if not with_mode_column:
            r.pop("rhso_terminal_mode", None)
    return rows


def write_run(directory, rows, with_mode_column=True):
    directory.mkdir(parents=True, exist_ok=True)
    cols = COLUMNS if with_mode_column else [c for c in COLUMNS if c != "rhso_terminal_mode"]
    with open(directory / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return directory


def propose(script, run, out, *extra):
    cmd = [sys.executable, str(script), "propose", "--from", str(run), "--models", "pmf,jit",
           "--images", "8", "--no-paper", "--out", str(out)] + list(extra)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    return res


def candidates(path):
    """[(model, method, fields)] from a proposed configuration."""
    cfg = yaml.safe_load(open(path))
    out = []
    for block in cfg["experiments"].values():
        for model, entry in block["models"].items():
            for method, fields in entry["methods"].items():
                out.append((model, method, fields))
    return cfg, out


def pair(method, f):
    return ((f["num_rhso_steps"], f["num_opt_steps"]) if method == "rhso"
            else (f["num_mpc_steps"], f["n_ctrl"]))


LOCKS = ["--lock", "rhso=num_rhso_steps,num_opt_steps", "--lock", "mpc_delta_t=num_mpc_steps,n_ctrl"]


def test_lock(tmp):
    print("\n1-4. propose with and without --lock on a screen whose winners sit on every edge")
    run = write_run(tmp / "screen", screen_rows())

    res = propose(HPO, run, tmp / "unlocked.yaml")
    _, unlocked = candidates(tmp / "unlocked.yaml")
    off = sorted({(model, method, pair(method, f)) for model, method, f in unlocked
                  if pair(method, f) not in (RHSO_PAIRS if method == "rhso" else MPC_PAIRS)})
    check("without --lock the budget knobs ARE pushed", res.returncode == 0 and len(off) > 0,
          "%d off-budget candidates, e.g. %s" % (len(off), off[:2]))

    res = propose(HPO, run, tmp / "locked.yaml", *LOCKS)
    cfg, locked = candidates(tmp / "locked.yaml")
    pairs = {(model, method, pair(method, f)) for model, method, f in locked}
    allowed = all(p in (RHSO_PAIRS if method == "rhso" else MPC_PAIRS) for _, method, p in pairs)
    check("with --lock every pair was screened", res.returncode == 0 and allowed,
          "pairs proposed: %s" % sorted(pairs))
    check("locked edges are reported, not pushed",
          "num_rhso_steps=8 locked" in res.stdout and "num_opt_steps=35 locked" in res.stdout
          and "num_mpc_steps=9 locked" in res.stdout and "n_ctrl=20 locked" in res.stdout,
          "stdout names all four locked edges")

    lrs = {f["lr"] for _, method, f in locked if method == "rhso"}
    mus = {f["mu"] for _, method, f in locked if method == "rhso"}
    lams = {f["lam"] for _, method, f in locked if method == "mpc_delta_t"}
    check("unlocked knobs are still extended",
          max(lrs) > max(RHSO_LR) and max(mus) > max(RHSO_MU) and max(lams) > max(MPC_LAM),
          "lr up to %g, mu up to %g, lam up to %g" % (max(lrs), max(mus), max(lams)))

    jit_modes = {f.get("rhso_terminal_mode") for model, method, f in locked
                 if model == "jit" and method == "rhso"}
    pmf_modes = {f.get("rhso_terminal_mode") for model, method, f in locked
                 if model == "pmf" and method == "rhso"}
    check("JiT keeps rhso_terminal_mode: direct", jit_modes == {"direct"} and pmf_modes == {None},
          "JiT %s; pMF writes nothing (auto is already direct)" % sorted(map(str, jit_modes)))

    plan = resolve_run_plan(cfg, validate_config(cfg), run_id="lock_test")
    costs = {(s.model, s.method, s.expected_backprops) for s in plan.specs}
    want = {("pmf", "rhso", 280), ("jit", "rhso", 280), ("pmf", "mpc_delta_t", 160)}
    check("round 2 resolves to the screened budget", costs == want and len(plan.specs) == len(locked),
          "%d jobs; (model, method, BP per image) = %s" % (len(plan.specs), sorted(costs)))


def test_backward_compatible(tmp):
    print("\n5. inputs that predate --lock and the terminal-mode column")
    old = tmp / "hpo_base.py"
    res = subprocess.run(["git", "show", "%s:scripts/hpo.py" % BASE_COMMIT], cwd=str(REPO_ROOT),
                         capture_output=True, text=True)
    if res.returncode != 0:
        check("byte-identical to the base hpo.py", False, "git show failed: %s" % res.stderr.strip())
        return
    old.write_text(res.stdout)
    for label, rows, with_col in (("no terminal-mode column", screen_rows(with_mode_column=False), False),
                                  ("JiT on its default suffix planner", screen_rows("suffix"), True)):
        run = write_run(tmp / label.replace(" ", "_"), rows, with_col)
        # one output path for both: the configuration and stdout both name it
        out = tmp / "round2.yaml"
        a = propose(HPO, run, out)
        new_bytes = out.read_bytes() if out.exists() else b""
        out.unlink(missing_ok=True)
        b = propose(old, run, out)
        old_bytes = out.read_bytes() if out.exists() else None
        same = (a.returncode == b.returncode == 0 and a.stdout == b.stdout
                and new_bytes == old_bytes)
        check("identical to %s: %s" % (BASE_COMMIT, label), same,
              "stdout and the written configuration match byte for byte")


def test_refusals(tmp):
    print("\n6. malformed --lock and mixed planners are refused")
    for bad in (["rhso=num_rhso_steps,K"], ["dps=steps"], ["rhso"]):
        try:
            hpo.parse_locks(bad)
            ok = False
        except SystemExit as e:
            ok = "--lock" in str(e)
        check("rejects --lock %s" % bad[0], ok)
    pushes = hpo.edge_pushes((8.0, 35.0, 0.1, 1.0), "rhso",
                             {"num_rhso_steps": {4.0, 8.0}, "num_opt_steps": {35.0, 70.0},
                              "lr": set(RHSO_LR), "mu": set(RHSO_MU)},
                             {"num_rhso_steps", "num_opt_steps"})
    moved = [k for k, _ in pushes if k is not None]
    check("edge_pushes never moves a locked knob",
          all(k[0] == 8.0 and k[1] == 35.0 for k in moved) and len(moved) == 2,
          "%d pushes, all on lr / mu" % len(moved))
    rows = screen_rows()
    for r in rows:
        if r["model"] == "jit" and r["lr"] == 0.005:
            r["rhso_terminal_mode"] = "suffix"
    run = write_run(tmp / "mixed", rows)
    res = propose(HPO, run, tmp / "mixed.yaml", *LOCKS)
    check("a cell mixing direct and suffix is refused",
          res.returncode != 0 and "mixes rhso_terminal_mode" in (res.stderr + res.stdout))


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_lock(tmp)
        test_backward_compatible(tmp)
        test_refusals(tmp)
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
