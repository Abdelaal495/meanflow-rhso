"""Rebuttal Task 9: exact JAX peak memory (opt-in pMF loader) and the JiT-L/16 timing table.

Run with:  python tests/test_paper_timing.py          (needs jax + pyyaml; imports run.py)

No checkpoint, no GPU.  What is checked:

  * the opt-in loader is OFF by default: the registry default is "device", every committed
    configuration except the two pmf_exact files resolves to it, bad values and other models
    are rejected, iMF has no such key;
  * on a SYNTHETIC pMF repository and checkpoint (same call sequence as pMF's own
    create_train_state -> restore_checkpoint, restore onto the target's sharding), the
    `host_once` loader gives parameters BITWISE identical to the default loader, runs the
    train-state code with the CPU as JAX's default device, transfers once, and records the
    mode in the provenance only when it is not the default;
  * src.timing.exact_jax_peak: exact when the load peak is close to the resident bytes, the
    peak did not rise before the warm-up and rose afterwards; each failure is reported;
    TimingDetail uses it only for host_once and keeps Task 1's logic otherwise;
  * the four generated configurations: regenerate byte for byte; pmf_exact = the 45 blocks of
    timing_pmf plus param_loading host_once (smoke: one RHSO + one MPC-delta_t, 3 images);
    JiT-L/16 = 7 methods x 5 tasks at batch size 1, 3 images, with the step counts of the
    task's table, which equal pMF-L/16's frozen values; RHSO from the capacity-matched
    entries with solver and terminal mode kept; the exact comment on count-matched entries;
  * the driver refuses --with-checks for an exact-peak config; the Slurm scripts accept
    pmf_exact and jitl and run the pmf_exact smoke without checks;
  * summarize_timing.py: --compare-to flags > 5%; --main-table writes the table, and refuses
    (writing nothing) on mixed GPU types or a non-exact JAX cell.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib
import copy
import importlib.util
import io
import json
import re
import tempfile

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
TABLE = {   # the task's table, written out independently of the generator
    "pnp": {"denoising": [10], "deblur": [200], "super_resolution": [20], "random_inpaint": [20],
            "box_inpaint": [50]},
    "flower": {"denoising": [50], "deblur": [50, 100], "super_resolution": [50],
               "random_inpaint": [50], "box_inpaint": [50]},
    "dflow": {"denoising": [1, 320], "deblur": [1, 640], "super_resolution": [1, 640],
              "random_inpaint": [1, 640], "box_inpaint": [1, 640]},
    "mpc_rhc": {"denoising": [4, 40, 2], "deblur": [4, 40, 1], "super_resolution": [4, 40, 3],
                "random_inpaint": [4, 40, 3], "box_inpaint": [4, 40, 1]},
    "mpc_delta_t": {"denoising": [8, 20], "deblur": [8, 40], "super_resolution": [8, 40],
                    "random_inpaint": [8, 40], "box_inpaint": [8, 40]},
    "rhso": {"denoising": [4, 40], "deblur": [8, 20], "super_resolution": [4, 40],
             "random_inpaint": [4, 40], "box_inpaint": [8, 20]},
}
FIELDS = {"pnp": ["num_pnp_steps"], "flower": ["num_flower_steps", "flower_cg_max_iters"],
          "dflow": ["steps", "num_opt_steps"], "mpc_rhc": ["num_mpc_steps", "n_ctrl", "K"],
          "mpc_delta_t": ["num_mpc_steps", "n_ctrl"], "rhso": ["num_rhso_steps", "num_opt_steps"]}
EXP = {"denoising": "denoising", "deblur": "deblurring", "super_resolution": "super_resolution",
       "random_inpaint": "random_inpainting", "box_inpaint": "box_inpainting"}
EXACT_COMMENT = ("timing only: step counts equal to pMF-L/16's frozen configuration; "
                 "quality is not evaluated from this run")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("  [%s] %-66s %s" % ("PASS" if ok else "FAIL", name, detail))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / ("%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(module)
    return module


# =====================================================================================
# 1. opt-in, off by default
# =====================================================================================
def check_default_off():
    print("\n[1] the exact-peak loader is opt-in")
    from src.config import (ConfigError, MODEL_REGISTRY_DEFAULTS, load_config,
                            resolve_model_registry, validate_config)
    check("registry default is the original loader",
          MODEL_REGISTRY_DEFAULTS["pmf"]["param_loading"] == "device")
    on = []
    for path in sorted((REPO / "configs").glob("*.yaml")):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(str(path))
        if resolve_model_registry(cfg, "pmf").get("param_loading") != "device":
            on.append(path.name)
    check("only the pmf_exact configs switch it on", sorted(on) == [
        "experiments_rebuttal_timing_pmf_exact.yaml",
        "experiments_rebuttal_timing_pmf_exact_smoke.yaml"], ", ".join(on))
    base = yaml.safe_load((REPO / "configs" / "experiments_rebuttal_timing_pmf_smoke.yaml").read_text())
    for bad, model in (("lazy", "pmf"), ("host_once", "imf"), ("host_once", "jit")):
        cfg = copy.deepcopy(base)
        cfg["models"] = {model: {"param_loading": bad}}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                validate_config(cfg)
            ok = False
        except ConfigError:
            ok = True
        check("param_loading=%s on %s is rejected" % (bad, model), ok)
    imf_src = (REPO / "src" / "models" / "imf.py").read_text()
    check("iMF untouched: no param_loading key, no reference in imf.py",
          "param_loading" not in MODEL_REGISTRY_DEFAULTS["imf"] and "param_loading" not in imf_src)


# =====================================================================================
# 2. bitwise identical parameters on a synthetic pMF repository
# =====================================================================================
FAKE_DEFAULT = '''
class C(dict):
    def __getattr__(self, k):
        return self[k]
    def __setattr__(self, k, v):
        self[k] = v
def get_config():
    return C(dataset=C(image_size=16, image_channels=3, num_classes=10),
             model=C(model_str="tiny", num_classes=10, P_mean=0.8, P_std=0.8, noise_scale=1.0,
                     data_proportion=0.5, cfg_beta=2.0, cfg_max=7.0, class_dropout_prob=0.1,
                     norm_p=1.0, norm_eps=1e-3, lpips=True, convnext=True, lpips_lambda=0.4,
                     convnext_lambda=0.1),
             training=C(ema_val=[500, 1000, 2000], ema_type="edm"),
             logging=C(), sampling=C(), eval_only=False, load_from="")
'''
FAKE_PMF = '''
class pixelMeanFlow:
    def __init__(self, **kw):
        self.kw = kw
'''
FAKE_TRAINSTATE = '''
from copy import deepcopy
import jax
SEEN = []
class State:
    def __init__(self, step, params, ema_params):
        self.step, self.params, self.ema_params = step, params, ema_params
def create_train_state(rng, config, model, image_size, lr_fn):
    SEEN.append(("create", jax.config.jax_default_device))
    @jax.jit
    def init(key):
        k1, k2 = jax.random.split(key)
        return {"net": {"w": jax.random.normal(k1, (64, 32)), "b": jax.random.normal(k2, (32,))}}
    params = init(rng)
    ema = {}
    for v in config.training.ema_val:              # deepcopy + update_ema(alpha=0), as pMF
        ema[v] = deepcopy(params)
        ema[v] = jax.tree_util.tree_map(lambda e, p: 0.0 * e + 1.0 * p, ema[v], params)
    return State(0, params, ema)
'''
FAKE_CKPT = '''
import os
import numpy as np
import jax
from utils.trainstate_util import SEEN, State
def restore_checkpoint(state, workdir):
    SEEN.append(("restore", jax.config.jax_default_device))
    target = state.ema_params[1000]
    with np.load(os.path.join(workdir, "ckpt.npz")) as data:
        # orbax semantics: each leaf restored onto the TARGET leaf's sharding
        restored = {"net": {k: jax.device_put(np.array(data[k]), target["net"][k].sharding)
                            for k in ("w", "b")}}
        step = int(data["step"])
    ema = dict(state.ema_params)
    ema[1000] = restored
    return State(step, {}, ema)
'''
FAKE_LR = "def lr_schedules(cfg, n):\n    return None\n"


def build_fake_repo(root):
    files = {"configs/__init__.py": "", "configs/default.py": FAKE_DEFAULT,
             "configs/tiny.yml": "model:\n  model_str: tiny\n", "pmf.py": FAKE_PMF,
             "utils/__init__.py": "", "utils/trainstate_util.py": FAKE_TRAINSTATE,
             "utils/ckpt_util.py": FAKE_CKPT, "utils/lr_utils.py": FAKE_LR}
    for rel, text in files.items():
        path = root / "repo" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    ckpt = root / "ckpt" / "pMF-test"
    ckpt.mkdir(parents=True)
    (ckpt / "checkpoint").write_text("")
    rng = np.random.default_rng(3)
    np.savez(ckpt / "ckpt.npz", w=rng.standard_normal((64, 32)).astype(np.float32),
             b=rng.standard_normal(32).astype(np.float32), step=np.int64(1234))
    return root / "repo", root / "ckpt"


def check_bitwise_loader():
    print("\n[2] host_once loader on a synthetic pMF checkpoint")
    import jax
    import src.models.pmf as pmf_module
    from src.config import resolve_model_registry
    with tempfile.TemporaryDirectory() as tmp:
        repo, ckpt = build_fake_repo(Path(tmp))
        original_download = pmf_module.download_and_extract_zip
        original_transfer = pmf_module.host_to_device_once
        transfers = []

        def counting_transfer(jax_, tree):
            transfers.append(1)
            return original_transfer(jax_, tree)

        pmf_module.download_and_extract_zip = lambda *a, **k: ckpt
        pmf_module.host_to_device_once = counting_transfer
        adapters, seen = {}, {}
        try:
            for mode in ("device", "host_once"):
                registry = resolve_model_registry(
                    {"models": {"pmf": {"config_yml": "configs/tiny.yml",
                                        "param_loading": mode}}}, "pmf")
                with contextlib.redirect_stdout(io.StringIO()):
                    adapters[mode] = pmf_module.PMFAdapter(registry, repo, ckpt)
                trainstate = sys.modules["utils.trainstate_util"]
                seen[mode] = list(trainstate.SEEN)
                trainstate.SEEN.clear()
                if mode == "device":
                    transfers_device = len(transfers)
        finally:
            pmf_module.download_and_extract_zip = original_download
            pmf_module.host_to_device_once = original_transfer
        with np.load(next(ckpt.rglob("ckpt.npz"))) as npz:
            data = {k: np.array(npz[k]) for k in npz.files}
        a, b = adapters["device"]._params, adapters["host_once"]._params
        la, lb = jax.tree_util.tree_leaves_with_path(a), jax.tree_util.tree_leaves_with_path(b)
        same = (jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
                and all(pa == pb and np.asarray(x).dtype == np.asarray(y).dtype
                        and np.asarray(x).tobytes() == np.asarray(y).tobytes()
                        for (pa, x), (pb, y) in zip(la, lb)))
        check("parameters bitwise identical (structure, dtype, bytes)", same,
              "%d leaves" % len(la))
        check("they are the checkpoint's values, not the random init",
              np.asarray(b["params"]["net"]["w"]).tobytes() == data["w"].tobytes()
              and adapters["host_once"].step == 1234 == adapters["device"].step)
        cpu = jax.devices("cpu")[0]
        check("host_once runs create_train_state and restore with the CPU as default device",
              [d for _, d in seen["host_once"]] == [cpu, cpu]
              and [d for _, d in seen["device"]] == [None, None],
              "device: %s | host_once: %s" % (seen["device"], seen["host_once"]))
        check("one transfer in host_once, none in the default loader",
              transfers_device == 0 and len(transfers) == 1)
        dev = jax.devices()[0]
        check("the parameters land on the default device",
              all(leaf.devices() == {dev} for leaf in jax.tree_util.tree_leaves(b)))
        pa = adapters["device"].spec.checkpoint
        pb = adapters["host_once"].spec.checkpoint
        check("provenance names the mode only off the default",
              "param_loading" not in pa and pb.get("param_loading") == "host_once"
              and adapters["device"].param_loading == "device")


# =====================================================================================
# 3. the exact-peak rule
# =====================================================================================
def snap(peak, in_use):
    return {"jax": {"peak_bytes_in_use": int(peak), "bytes_in_use": int(in_use)}}


def check_exact_rule():
    print("\n[3] src.timing.exact_jax_peak and TimingDetail")
    from src.timing import TimingDetail, exact_jax_peak
    load = snap(1.56 * GIB, 1.55 * GIB)
    good = {"after_model_load": load, "before_warm_up": snap(1.56 * GIB, 1.55 * GIB),
            "before_timed_region": snap(4.2 * GIB, 1.6 * GIB),
            "after_timed_region": snap(4.2 * GIB, 1.6 * GIB)}
    r = exact_jax_peak(good)
    check("exact when load peak ~ in use and the peak rose after the load",
          r["exact"] and r["peak_bytes"] == int(4.2 * GIB) and r["scope"].startswith("warm_up"))
    rose = dict(good, after_timed_region=snap(4.5 * GIB, 1.6 * GIB))
    check("scope is the timed region when the peak rose there",
          exact_jax_peak(rose)["scope"] == "timed_region")
    for name, change, word in (
            ("the default loader's 7.85 GiB load peak", {"after_model_load": snap(7.85 * GIB, 1.55 * GIB),
                                                          "before_warm_up": snap(7.85 * GIB, 1.55 * GIB)},
             "load peak"),
            ("a peak raised between load and warm-up (e.g. checks)",
             {"before_warm_up": snap(3.0 * GIB, 1.55 * GIB)}, "between the model load"),
            ("no rise above the pre-warm-up peak", {"after_timed_region": snap(1.56 * GIB, 1.55 * GIB)},
             "did not rise")):
        r = exact_jax_peak(dict(good, **change))
        check("not exact: %s" % name, not r["exact"] and any(word in x for x in r["reasons"]),
              "; ".join(r["reasons"])[:90])
    td = TimingDetail("jax", after_load=good["after_model_load"],
                      before_warm_up=good["before_warm_up"], param_loading="host_once")
    td.snapshots.update(before_timed_region=good["before_timed_region"],
                        after_timed_region=good["after_timed_region"])
    out = td.allocator_peak({})
    check("host_once: one exact figure", out["allocator_peak_exact"]
          and out["allocator_peak_gib"] == round(4.2, 4)
          and out["allocator_peak_mode"] == "exact_jax_host_once")
    td2 = TimingDetail("jax", after_load=good["after_model_load"],
                       before_warm_up=good["before_warm_up"], param_loading="device")
    td2.snapshots.update(before_timed_region=good["before_timed_region"],
                         after_timed_region=good["after_timed_region"])
    out2 = td2.allocator_peak({})
    check("default loader keeps Task 1's bracket logic", "allocator_peak_mode" not in out2
          and out2["allocator_peak_scope"].startswith("warm_up_of_this_configuration"))


# =====================================================================================
# 4. configurations
# =====================================================================================
def check_configs():
    print("\n[4] generated configurations")
    gen = load_script("make_paper_timing_configs")
    stale = [p for p, text in gen.outputs().items() if (REPO / p).read_text() != text]
    check("the four files regenerate byte for byte", not stale, ", ".join(stale))
    load = lambda name: yaml.safe_load((REPO / "configs" / name).read_text())
    pmf, exact = load("experiments_rebuttal_timing_pmf.yaml"), load("experiments_rebuttal_timing_pmf_exact.yaml")
    check("pmf_exact = timing_pmf's 45 blocks + param_loading host_once",
          exact["experiments"] == pmf["experiments"] and len(exact["experiments"]) == 45
          and exact["models"] == {"pmf": {"param_loading": "host_once"}}
          and exact["runtime"] == pmf["runtime"] and exact["data"] == pmf["data"])
    smoke = load("experiments_rebuttal_timing_pmf_exact_smoke.yaml")
    check("pmf_exact smoke: one RHSO and one MPC-delta_t block, timing_pmf's 3 images",
          sorted(smoke["experiments"]) == ["denoising__core__mpc_delta_t", "denoising__rhso__rhso"]
          and all(smoke["experiments"][k] == pmf["experiments"][k] for k in smoke["experiments"]))

    jitl = load("experiments_rebuttal_timing_jitl.yaml")
    final, flower = load("experiments_final_frozen100.yaml"), load("experiments_flower_frozen100.yaml")
    theory = load("experiments_theory_N_fixedB_final100.yaml")
    exps = jitl["experiments"]
    shape_ok = (len(exps) == 35 and jitl["models"] == {"jit": {"variant": "JiT-L/16"}}
                and all(e["num_images"] == 3 and e["models"]["jit"]["batch_size"] == 1
                        for e in exps.values()))
    methods = sorted({m for e in exps.values() for m in e["models"]["jit"]["methods"]})
    check("JiT-L/16: 7 methods x 5 tasks, batch size 1, 3 images", shape_ok and methods == sorted(
        ["sdedit", "pnp", "flower", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]))
    table_ok, pmf_ok, detail = True, True, []
    for problem, exp in EXP.items():
        for method, fields in FIELDS.items():
            entry = exps["%s__%s" % (exp, method)]["models"]["jit"]["methods"][method]
            got = [entry.get(f) for f in fields]
            want = list(TABLE[method][problem])
            if method == "flower" and len(want) == 1:
                got = got[:1]
                table_ok &= "flower_cg_max_iters" not in entry
            table_ok &= got == want
            if method == "rhso":
                src = final["experiments"][exp + "__rhso"]["models"]["pmf"]["methods"]["rhso"]
            elif method == "flower":
                src = flower["experiments"][exp + "__flower"]["models"]["pmf"]["methods"]["flower"]
            elif method == "mpc_rhc":
                src = final["experiments"][exp + "__rhc"]["models"]["pmf"]["methods"]["mpc_rhc"]
            else:
                src = final["experiments"][exp + "__core"]["models"]["pmf"]["methods"][method]
            pmf_vals = [src.get(f) for f in fields]
            if method == "flower" and len(want) == 1:
                pmf_vals = pmf_vals[:1]
            pmf_ok &= pmf_vals == got
            if got != want:
                detail.append("%s/%s %s != %s" % (problem, method, got, want))
    check("step counts equal the task's table", table_ok, "; ".join(detail))
    check("... and pMF-L/16's frozen values", pmf_ok)
    rhso_ok = True
    for problem, exp in EXP.items():
        mine = exps["%s__rhso" % exp]["models"]["jit"]["methods"]["rhso"]
        n = TABLE["rhso"][problem][0]
        src = theory["experiments"]["%s_B160_N%d" % (exp, n)]["models"]["jit"]["methods"]["rhso"]
        rhso_ok &= all(mine[k] == src[k] for k in ("num_rhso_steps", "num_opt_steps", "lr", "mu",
                                                   "optimizer", "solver", "rhso_terminal_mode"))
        rhso_ok &= mine["rhso_terminal_mode"] == "direct" and not any(
            k.endswith("_diagnostics") for k in mine)
    check("RHSO = capacity-matched JiT-L/16 entries, solver and terminal mode kept", rhso_ok)
    other_ok = True
    for problem, exp in EXP.items():
        for method in ("pnp", "dflow", "mpc_delta_t"):
            mine = exps["%s__%s" % (exp, method)]["models"]["jit"]["methods"][method]
            jb = final["experiments"][exp + "__core"]["models"]["jit"]["methods"][method]
            other_ok &= all(mine[k] == v for k, v in jb.items() if k not in FIELDS[method])
        sd = exps["%s__sdedit" % exp]["models"]["jit"]["methods"]["sdedit"]
        other_ok &= sd == final["experiments"][exp + "__core"]["models"]["jit"]["methods"]["sdedit"]
    check("other fields from the same task's JiT entry; plain sampling unchanged", other_ok)
    text = (REPO / "configs" / "experiments_rebuttal_timing_jitl.yaml").read_text()
    comments = {}
    for comment, name in re.findall(r"^  # (.*)\n  ([a-z_]+):$", text, re.MULTILINE):
        comments[name] = comment
    counted = [n for n in comments if n.rsplit("__", 1)[1] in ("pnp", "flower", "dflow",
                                                                  "mpc_rhc", "mpc_delta_t")]
    check("the exact comment on every count-matched entry",
          len(counted) == 25 and all(comments[n] == EXACT_COMMENT for n in counted)
          and "JiT-B" not in text)


# =====================================================================================
# 5. driver and Slurm scripts
# =====================================================================================
def check_scripts():
    print("\n[5] driver and submit scripts")
    rt = load_script("run_timing")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rt.main(["--config", "configs/experiments_rebuttal_timing_pmf_exact_smoke.yaml",
                     "--run-id", "x", "--output-root", tempfile.gettempdir(), "--dry-run",
                     "--with-checks"])
        ok = False
    except SystemExit as exc:
        ok = "refused" in str(exc)
    check("the driver refuses --with-checks for an exact-peak config", ok)
    submit = (REPO / "submit_timing.sh").read_text()
    run_sh = (REPO / "slurm" / "run_timing.sh").read_text()
    check("submit_timing.sh and slurm/run_timing.sh accept pmf_exact and jitl",
          "|pmf_exact|jitl)" in submit and "pmf_exact|jitl}" in run_sh)
    check("the pmf_exact smoke runs without run.py's checks",
          'if [ "$PRIOR" = "pmf_exact" ]; then EXTRA=(); fi' in run_sh)


# =====================================================================================
# 6. summarize_timing.py: --compare-to and --main-table
# =====================================================================================
def payload(model, problem, method, experiment, seconds, peak, exact, gpu, framework,
            evals=10, bps=4, mode=None):
    detail = {"chunk_seconds_outer_synced": list(seconds), "chunk_sizes": [1] * len(seconds),
              "chunk_seconds_method": list(seconds), "allocator_peak_gib": peak if exact else None,
              "allocator_peak_exact": exact,
              "allocator_peak_scope": "timed_region" if exact else None,
              "environment": {"hardware": {"gpu_name": gpu}}}
    if mode:
        detail.update(allocator_peak_mode=mode, allocator_peak_upper_gib=peak,
                      allocator_peak_not_exact_reasons=[] if exact else ["did not rise"])
    return {"job_id": "%s_%s_%s" % (model, problem, method), "model": model, "problem": problem,
            "method": method, "experiment": experiment, "status": "ok",
            "num_images": len(seconds), "batch_size": 1, "framework": framework,
            "model_evaluations": evals * len(seconds),
            "backprops_through_model": bps * len(seconds), "psnr": 25.0, "ssim": 0.8,
            "lpips": 0.2, "timing_detail": detail}


def write_run(root, model, rows, provenance, baseline_gib):
    root.mkdir(parents=True)
    for i, p in enumerate(rows):
        d = root / ("job%02d" % i)
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps(p))
    (root / ("memory_baseline_%s.json" % model)).write_text(json.dumps(
        {"model": model, "allocator_in_use_gib": baseline_gib}))
    (root / "run_metadata.json").write_text(json.dumps({"model_provenance": {model: provenance}}))


def check_summary():
    print("\n[6] summarize_timing.py --compare-to and --main-table")
    st = load_script("summarize_timing")
    tasks = list(EXP)
    methods = ["sdedit", "pnp", "flower", "dflow", "mpc_rhc", "mpc_delta_t", "rhso"]

    def make(tmp, name, model, gpu_pmf="NVIDIA H100 80GB HBM3", gpu_jit="NVIDIA H100 80GB HBM3",
             broken=None):
        rows = []
        for t in tasks:
            for m in methods:
                if model == "pmf":
                    exact = (t, m) != broken
                    rows.append(payload("pmf", t, m, "%s__core__%s" % (EXP[t], m), [2.0, 2.1, 2.2],
                                        4.25, exact, gpu_pmf, "jax", mode="exact_jax_host_once"))
                else:
                    rows.append(payload("jit", t, m, "%s__%s" % (EXP[t], m), [1.0, 1.1, 1.2],
                                        3.5, True, gpu_jit, "torch"))
        prov = ({"model_str": "pmfDiT_L_16", "param_loading": "host_once"} if model == "pmf"
                else {"variant": "JiT-L/16", "network_dtype": "torch.bfloat16"})
        write_run(tmp / name, model, rows, prov, 1.55 if model == "pmf" else 0.86)
        return tmp / name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pmf_run, jit_run = make(tmp, "pmf_ok", "pmf"), make(tmp, "jit_ok", "jit")
        out = tmp / "table"
        with contextlib.redirect_stdout(io.StringIO()):
            code = st.main(["--main-table", str(pmf_run), str(jit_run), "--out", str(out)])
        tex = (out / "main_table.tex").read_text() if (out / "main_table.tex").exists() else ""
        rows = list(__import__("csv").DictReader(open(out / "main_table.csv"))) if code == 0 else []
        check("writes the table: 70 cells, mean +- sd, exact GiB, eval / BP, per-checkpoint line",
              code == 0 and len(rows) == 70
              and "\\makecell{2.10 $\\pm$ 0.10 s \\\\ 4.25 GiB \\\\ 10 / 4}" in tex
              and "pMF-L/16: model-only resident memory 1.55 GiB; JAX; FP32" in tex
              and "JiT-L/16: model-only resident memory 0.86 GiB; PyTorch; BF16 network" in tex
              and "H100 80GB HBM3" in tex)
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            st.main([str(jit_run), "--out", str(tmp / "jit_summary")])
        tables = (tmp / "jit_summary" / "timing_tables.tex").read_text()
        check("the per-run summary labels the run with its recorded variant (JiT-L/16)",
              "JiT-L/16  --  GPU" in text.getvalue() and "JiT-B/16" not in text.getvalue()
              and "{JiT-L/16}" in tables and "JiT-B/16" not in tables)
        mixed = make(tmp, "jit_a100", "jit", gpu_jit="NVIDIA A100-SXM4-40GB")
        out2 = tmp / "refused1"
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = st.main(["--main-table", str(pmf_run), str(mixed), "--out", str(out2)])
        check("refuses on different GPU types (nothing written)",
              code == 2 and not out2.exists() and "different GPU types" in err.getvalue())
        broken = make(tmp, "pmf_broken", "pmf", broken=("deblur", "rhso"))
        out3 = tmp / "refused2"
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = st.main(["--main-table", str(broken), str(jit_run), "--out", str(out3)])
        check("refuses when a JAX cell is not exact (nothing written)",
              code == 2 and not out3.exists() and "deblur: JAX peak not exact" in err.getvalue())

        slow = []
        for t in tasks:
            for m in methods:
                s = [2.4, 2.5, 2.6] if (t, m) == ("denoising", "rhso") else [2.0, 2.1, 2.2]
                slow.append(payload("pmf", t, m, "%s__core__%s" % (EXP[t], m), s, 4.25, True,
                                    "NVIDIA H100 80GB HBM3", "jax", mode="exact_jax_host_once"))
        write_run(tmp / "pmf_slow", "pmf", slow, {}, 1.55)
        out4 = tmp / "cmp"
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            st.main([str(tmp / "pmf_slow"), "--compare-to", str(pmf_run), "--out", str(out4)])
        cmp_rows = list(__import__("csv").DictReader(open(out4 / "timing_compare.csv")))
        flagged = [r["experiment"] for r in cmp_rows if r["flag"]]
        check("--compare-to flags only the configuration more than 5% slower",
              flagged == ["denoising__core__rhso"] and len(cmp_rows) == 35, ", ".join(flagged))


def main():
    print("Rebuttal Task 9: exact JAX peak and the JiT-L/16 timing table")
    check_default_off()
    check_bitwise_loader()
    check_exact_rule()
    check_configs()
    check_scripts()
    check_summary()
    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
