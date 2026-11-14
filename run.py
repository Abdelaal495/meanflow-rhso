#!/usr/bin/env python
"""SDEdit vs. MPC-Flow vs. PnP-Flow vs. D-Flow vs. RHSO -- the experiment orchestrator.

    python run.py --config configs/experiments.yaml --dry-run     # plan only, no models
    python run.py --config configs/experiments.yaml               # full run

In Colab, `%run run.py --config configs/experiments.yaml` behaves like a notebook: the
resolved plan, progress, summary tables and matplotlib figures all appear inline.

Execution is MODEL-MAJOR: each checkpoint is loaded once, every job that needs it runs, and
it is released before the next model loads -- which matters because JiT is Torch and pMF is
JAX and they should not be resident simultaneously.

Timing policy (brief section 32).  The reported reconstruction runtime covers generative
model calls, integration, MPC planning, control optimisation, PnP's initial prior
projection and its correction cycles, D-Flow's forward trajectories AND backward passes,
and the controlled execution.  It EXCLUDES checkpoint download, model loading, dataset
download, metrics, visualisation and image saving.  Model loading is recorded separately,
and every (model, method, shape) is warmed up untimed first so JAX compilation never lands
inside a measured run.

Memory policy (see src/memory.py).  Each atomic job reports the GPU memory it was sitting
at before its reconstruction (baseline), the peak during it, and the difference.  The
measured region is exactly the reconstruction -- never model loading, metrics or warm-up --
and the peak is a JOB peak at the configured batch size, never divided by the batch.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import checks as checks_module
from src.config import (COMPARED_METHODS, ConfigError, MODEL_REGISTRY_DEFAULTS,
                        METHOD_DECLARATIONS, check_accelerator_compatibility, load_config,
                        print_run_plan, resolve_model_registry, resolve_run_plan,
                        validate_config)
from src.data import DataManager
from src.memory import GpuMemoryProfiler
from src.metrics import configure_lpips, degraded_baseline, evaluate_reconstruction
from src.models import ModelManager, build_initial_state, load_adapters
from src.models.base import PREFETCH_HOOKS
from src.models.base import Conditioning
from src.reconstruction import select_reconstructor
from src.problems import ProblemStore
from src.rhso import rhso_time_grid
from src.rhso_diagnostics import CONSISTENCY_MEANING, stage_arrays, stage_summary
from src.sdedit import ReconstructionStats
from src.utils import (RULE, THIN, apply_offline_mode, detect_accelerator,
                       detect_environment, env_default, free_memory, in_ipython, jsonable,
                       load_dotenv, now_iso, pixel_fingerprint, progress, save_json,
                       save_yaml, set_global_seed, set_progress_enabled, timing_summary,
                       to_uint8)
import src.visualization as viz


# =====================================================================================
# CLI
# =====================================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiments.yaml",
                   help="experiment configuration (default: configs/experiments.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate, resolve and print the plan, then exit without loading "
                        "any model")
    p.add_argument("--check", dest="check", action="store_true", default=True,
                   help="run the structural and per-model sanity checks (default)")
    p.add_argument("--no-check", dest="check", action="store_false")
    p.add_argument("--checks-only", action="store_true",
                   help="run the structural checks against the built problems, then exit")
    p.add_argument("--max-jobs", type=int, default=None,
                   help="override runtime.max_atomic_jobs (the sweep-explosion guard)")
    p.add_argument("--output-root", default=None, help="override runtime.output_root")
    p.add_argument("--cache-root", default=None, help="override runtime.cache_root")
    p.add_argument("--models", default=None,
                   help="comma-separated subset of models to run (must appear in the config)")
    p.add_argument("--experiments", default=None,
                   help="comma-separated subset of experiments to enable")
    p.add_argument("--num-images", type=int, default=None,
                   help="override num_images for every enabled experiment (quick smoke test)")
    p.add_argument("--replicate", type=int, default=None,
                   help="stochastic replicate index; changes epsilon deliberately")
    p.add_argument("--resume", dest="resume", action="store_true", default=None,
                   help="reuse finished jobs whose resolved metadata matches (default: from "
                        "runtime.resume)")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-figures", action="store_true", help="skip figure generation")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the untimed warm-up (timings will then include compilation)")
    p.add_argument("--no-gpu-memory", action="store_true",
                   help="disable GPU peak-memory instrumentation (it is cheap and "
                        "non-invasive, but this is the escape hatch)")
    p.add_argument("--run-id", default=None, help="reuse an existing run directory name")
    p.add_argument("--memory-baseline-only", action="store_true",
                   help="load each planned model, record what it alone costs in GPU memory "
                        "(memory_baseline_<model>.json in the run directory) and exit "
                        "without running a job; used by scripts/run_timing.py")

    cluster = p.add_argument_group(
        "cluster / offline execution",
        "Alliance (Compute Canada) compute nodes have NO internet access, so every asset "
        "must be on disk before a job starts.")
    cluster.add_argument("--prefetch", action="store_true",
                         help="download every asset the config needs (model repositories, "
                              "checkpoints, ImageNet images, LPIPS weights) and exit. Run "
                              "this ONCE on a login node; no GPU and no model build.")
    cluster.add_argument("--offline", dest="offline", action="store_true", default=None,
                         help="forbid all network access (auto-enabled under SLURM)")
    cluster.add_argument("--online", dest="offline", action="store_false",
                         help="allow network access even under SLURM")
    cluster.add_argument("--shard", default=None, metavar="K/N",
                         help="run only shard K of N (0-based) for SLURM job arrays. Shards "
                              "are contiguous over a model-major ordering, so a shard "
                              "usually loads a single checkpoint.")
    cluster.add_argument("--aggregate", action="store_true",
                         help="merge the per-job results already on disk in --run-id into "
                              "results.csv plus figures, then exit. Use after a job array.")
    return p.parse_args(argv)


def parse_shard(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    try:
        k, n = text.split("/")
        shard, total = int(k), int(n)
    except Exception:
        raise ConfigError("--shard must look like K/N, e.g. 0/4 (K is 0-based). Got %r"
                          % text)
    if total < 1 or not 0 <= shard < total:
        raise ConfigError("--shard K/N needs N >= 1 and 0 <= K < N. Got %r" % text)
    return shard, total


def shard_suffix(shard: Optional[Tuple[int, int]]) -> str:
    """The filename suffix that makes a top-level artefact PRIVATE to one array task.

    Every mutable file an array task writes into the SHARED run directory must carry this
    suffix.  Two tasks writing `checks.json` (or `run_metadata.json`) is a last-writer-wins
    race: the file that survives contains one shard's information and the other shard's is
    silently lost -- which is exactly what happened in the two-shard H100 smoke test, where
    the downloaded `checks.json` held only one model family even though both families'
    checks had clearly run.  `--aggregate` merges the shard-private files into the
    canonical unsuffixed ones afterwards.
    """
    return "" if shard is None else "_shard%02d" % shard[0]


SHARD_FILE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # (canonical name, glob for the shard-private copies) -- everything an array task may
    # write at the TOP level of the run directory.  Audited against the whole of run.py:
    # per-JOB artefacts live in unique per-spec directories and shards are disjoint, so
    # they need no suffix; config.yaml / resolved_config.yaml are identical for every task
    # and are written once (see `_write_shared_plan_files`).
    ("results.csv", "results_shard*.csv"),
    ("results_per_image.csv", "results_per_image_shard*.csv"),
    ("results.jsonl", "results_shard*.jsonl"),
    ("experiment_log.jsonl", "experiment_log_shard*.jsonl"),
    ("checks.json", "checks_shard*.json"),
    ("run_metadata.json", "run_metadata_shard*.json"),
)


def select_shard(specs: Sequence, shard: int, total: int) -> List:
    """Contiguous slice of a MODEL-MAJOR ordering.

    Contiguous rather than round-robin so a shard usually needs one checkpoint: loading pMF
    costs ~2 minutes, and interleaving would make most shards pay for both models.  The
    ordering is derived from stable job ids, so every array task computes the same split
    without communicating.
    """
    from src.config import MODEL_NAMES
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    ordered = sorted(specs, key=lambda s: (order.get(s.model, 99), s.experiment, s.method,
                                           s.job_id))
    n = len(ordered)
    lo = (n * shard) // total
    hi = (n * (shard + 1)) // total
    return ordered[lo:hi]


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    rt = config.setdefault("runtime", {})
    # Job scripts set these once and every array task inherits them.
    if os.environ.get("MPCFLOW_OUTPUT_ROOT"):
        rt["output_root"] = os.environ["MPCFLOW_OUTPUT_ROOT"]
    if os.environ.get("MPCFLOW_CACHE_ROOT"):
        rt["cache_root"] = os.environ["MPCFLOW_CACHE_ROOT"]
    if args.max_jobs is not None:
        rt["max_atomic_jobs"] = int(args.max_jobs)
    if args.output_root:
        rt["output_root"] = args.output_root
    if args.cache_root:
        rt["cache_root"] = args.cache_root
    if args.replicate is not None:
        rt["replicate"] = int(args.replicate)
    if args.resume is not None:
        rt["resume"] = bool(args.resume)
    if args.no_gpu_memory:
        rt["gpu_memory_profiling"] = False
    if args.experiments:
        wanted = {name.strip() for name in args.experiments.split(",") if name.strip()}
        unknown = wanted - set(config.get("experiments", {}))
        if unknown:
            raise ConfigError("--experiments names unknown experiments: %s" % sorted(unknown))
        for name, block in config["experiments"].items():
            block["enabled"] = name in wanted
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        for name, block in config["experiments"].items():
            if not block.get("enabled"):
                continue
            models = block.get("models") or {}
            kept = {k: v for k, v in models.items() if k in wanted}
            if not kept:
                block["enabled"] = False
            else:
                block["models"] = kept
    if args.num_images is not None:
        for block in config["experiments"].values():
            if block.get("enabled"):
                block["num_images"] = int(args.num_images)
    return config


# =====================================================================================
# Environment
# =====================================================================================
def ensure_repositories(plan, cache_root: Path, verbose: bool = True) -> Dict[str, Any]:
    """Clone the model repositories at their pinned revisions if they are missing.

    setup_colab.sh normally does this; doing it here too keeps a fresh checkout working
    without an extra step, and it is idempotent.
    """
    # Absolute: adapters chdir into these directories (RepoSandbox / pushd).
    repo_root = Path(cache_root).resolve() / "repos"
    repo_root.mkdir(parents=True, exist_ok=True)
    paths, heads = {}, {}
    for model in plan.resources.models:
        spec = plan.resources.repositories[model]
        dest = repo_root / spec["dirname"]
        paths[model] = dest
        if not (dest / ".git").is_dir():
            if verbose:
                print("Cloning %s -> %s" % (spec["url"], dest))
            subprocess.run('git clone --quiet "%s" "%s"' % (spec["url"], dest), shell=True,
                           check=True)
            if spec.get("rev"):
                subprocess.run('cd "%s" && git fetch --quiet --depth 1 origin "%s" && '
                               'git checkout --quiet "%s"' % (dest, spec["rev"], spec["rev"]),
                               shell=True, check=False)
        heads[model] = subprocess.run('cd "%s" && git rev-parse HEAD' % dest, shell=True,
                                      capture_output=True, text=True).stdout.strip()
        if verbose:
            print("   %-4s %s @ %s" % (model, dest, heads[model][:12]))
    return {"repo_paths": paths, "repo_heads": heads}


def configure_jax_environment(accel: Dict[str, Any], cache_root: Path) -> Path:
    """Set JAX's allocator policy before anything imports jax.

    Read once when the backend initialises: set later and JAX has already preallocated
    ~75% of the device, and Torch-side D-Flow then dies with CUDA OOM.
    """
    if accel["kind"] == "gpu":
        # Do not grab the whole device: PyTorch may share this process.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    jax_cache = Path(cache_root) / "jax_compilation_cache"
    jax_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(jax_cache))
    return jax_cache


def init_frameworks(plan, config: Dict[str, Any], accel: Dict[str, Any],
                    cache_root: Path, verbose: bool = True) -> Dict[str, Any]:
    """Import and configure only the frameworks this plan needs."""
    from src.utils import DEEP_MEMORY_HOOKS, MEMORY_HOOKS
    context: Dict[str, Any] = {"torch_dtypes": {}, "local_device_count": 1}
    frameworks = plan.resources.frameworks

    if "jax" in frameworks:
        jax_cache = configure_jax_environment(accel, cache_root)
        import jax
        # Persist compiled executables to disk so compilation is paid ONCE EVER rather
        # than once per session. The defaults skip small/fast entries; pMF's model step is
        # neither, but being explicit means a rerun of the same benchmark recompiles nothing.
        for option, value in (("jax_compilation_cache_dir", str(jax_cache)),
                              ("jax_persistent_cache_min_entry_size_bytes", -1),
                              ("jax_persistent_cache_min_compile_time_secs", 0.5)):
            try:
                jax.config.update(option, value)
            except Exception:
                pass                      # older JAX: the env var above still applies
        # Compatibility shims: aliases recent JAX removed but the repositories still use.
        for alias, target in (("tree_map", "tree_map"), ("tree_leaves", "tree_leaves"),
                              ("tree_flatten", "tree_flatten")):
            if not hasattr(jax, alias):
                setattr(jax, alias, getattr(jax.tree_util, target))
        jax.distributed.initialize = lambda *a, **k: None      # single-host Colab
        context["local_device_count"] = jax.local_device_count()
        # DEEP, not per-job: jax.clear_caches() discards every compiled executable. Running
        # it between jobs made pMF recompile the whole model 24 times (~470 s of compilation
        # for ~26 s of actual work). It now runs only when the model is released.
        DEEP_MEMORY_HOOKS.append(lambda: jax.clear_caches())
        backend = jax.default_backend()
        if verbose:
            print("JAX             : %s | backend: %s | devices: %s"
                  % (jax.__version__, backend, jax.devices()))
        if accel["kind"] == "gpu" and backend != "gpu":
            # Silent CPU fallback is the worst outcome: pMF/iMF still produce correct
            # numbers, just 50-100x slower, so a job hits its wall clock instead of failing.
            print("!" * 94)
            print("! JAX IS RUNNING ON THE CPU although this node has a GPU.")
            print("! pMF and iMF would be 50-100x slower and will likely exceed your time")
            print("! limit. Modern JAX keeps CUDA support in a SEPARATE plugin package that")
            print("! `pip install jax` does not pull in. On an Alliance cluster, from a")
            print("! LOGIN node, with the same modules loaded:")
            print("!     avail_wheels jax jaxlib jax_cuda12_plugin jax_cuda12_pjrt")
            print("!     pip install --no-index jax_cuda12_plugin jax_cuda12_pjrt")
            print("! Verify with:  python -c \"import jax; print(jax.devices())\"")
            print("! Meanwhile, run PyTorch models only:  --models jit")
            print("!" * 94)

    if "torch" in frameworks:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before cuBLAS starts
        import torch
        # AUTOGRAD STAYS ENABLED: MPC-delta_t, RHC K>1 and D-Flow differentiate the terminal
        # loss through the generative model, and PnP differentiates the data-fidelity term
        # through the decoder.  Model PARAMETERS keep requires_grad_(False), so no parameter
        # gradients are ever allocated.  SDEdit and PnP's denoiser wrap their own loops in
        # no_grad().
        torch.set_grad_enabled(True)
        want_cuda = accel["kind"] == "gpu"
        if want_cuda and not torch.cuda.is_available():
            print("A GPU was detected but torch.cuda is unavailable; falling back to CPU.")
            want_cuda = False
        device = torch.device("cuda" if want_cuda else "cpu")
        context["torch_device"] = device

        import random as _random
        seed = int(plan.seed)
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for flag, value in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False),
                                ("enable_cudnn_sdp", False), ("enable_math_sdp", True)):
                if hasattr(torch.backends.cuda, flag):
                    getattr(torch.backends.cuda, flag)(value)

        def resolve_torch_dtype(name: str):
            table = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
            if name in table:
                return table[name]
            if name == "safe_auto":
                # JiT's official evaluation uses BF16 autocast.  FP16 has a much narrower
                # exponent range and destabilises pixel-space generation, so this prefers
                # BF16 and otherwise falls back to FP32 -- never FP16.
                if device.type == "cuda" and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float32
            if name != "auto":
                raise ValueError("Unsupported dtype setting: %r" % name)
            if device.type != "cuda":
                return torch.float32
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for model in plan.resources.models:
            reg = resolve_model_registry(config, model)
            if reg["framework"] == "torch":
                context["torch_dtypes"][model] = resolve_torch_dtype(reg["dtype"])
        MEMORY_HOOKS.append(lambda: torch.cuda.empty_cache()
                            if torch.cuda.is_available() else None)
        if verbose:
            print("PyTorch         : %s | device: %s | dtypes: %s"
                  % (torch.__version__, device,
                     {k: str(v) for k, v in context["torch_dtypes"].items()}))
            print("Autograd enabled: %s (required by MPC and D-Flow)" % torch.is_grad_enabled())
    return context


# =====================================================================================
# One atomic job
# =====================================================================================
def run_job(adapter, spec, problem, manager, profiler=None, timing=None) -> Dict[str, Any]:
    """Execute one atomic job: one model, one problem, one method, one hyperparameter set.

    Batching is the only decision made here; every scientific setting arrived in the spec.
    Fixed-batch MeanFlow models repeat-pad a short tail chunk to their compiled shape.  Both
    terms of the MPC objective are sums over the batch and a padded row carries its own
    duplicated measurement and its own control, so padding cannot influence a real image's
    gradient or its metrics.  The same holds for PnP and D-Flow: their per-measurement
    fidelity is summed over the batch, so a padded row contributes only to its own
    duplicate's gradient.  Padded rows are counted and reported, never treated as real.

    `profiler`, when given, measures GPU memory around the reconstruction ONLY -- the
    baseline is taken immediately before the first chunk's reconstruction, so building the
    initial state and loading the model are outside the incremental figure.  It is None
    during warm-up, which must never contribute to a reported number.

    `timing` (a src.timing.TimingDetail, OPT-IN via runtime.timing_detail) adds an outer,
    device-synchronised wall clock around each chunk's reconstruction and memory snapshots
    just before the first chunk and after the last.  With timing=None nothing changes.
    """
    reconstruct = select_reconstructor(spec.dynamics_family, spec.method)
    bs = max(1, int(spec.batch_size))
    indices = list(range(int(spec.num_images)))
    total = ReconstructionStats()
    outputs: List[np.ndarray] = []
    fingerprints: List[str] = []
    padded_items = 0

    for start in range(0, len(indices), bs):
        chunk = indices[start:start + bs]
        size = len(chunk)
        pad = 0
        if adapter.spec.fixed_batch_shape and size < bs:
            pad = bs - size
            chunk = chunk + [chunk[-1]] * pad
            padded_items += pad

        # `pad` is recorded on the subset so per-image diagnostics can drop the repeated
        # rows. It changes no measurement, no objective and no gradient: the fidelity is
        # summed over the batch, so a padded row only ever feeds its own duplicate.
        sub_problem = problem.subset(chunk, padded_rows=pad)
        labels = (problem.labels[np.asarray(chunk, np.int64)] if problem.labels is not None
                  else np.zeros((len(chunk),), np.int32))
        # Every strategy receives the SAME conditioning object for an image.
        cond = Conditioning(labels=np.asarray(labels, np.int32), guidance=dict(spec.guidance))
        x0 = build_initial_state(adapter, problem, spec, chunk, manager)
        fingerprints.append(pixel_fingerprint(adapter.to_numpy(x0)))

        if profiler is not None and profiler.baseline_gib is None:
            profiler.establish_baseline()
        if timing is not None and timing.snapshots.get("before_timed_region") is None:
            timing.snapshot("before_timed_region")
        region = profiler.measure() if profiler is not None else contextlib.nullcontext()
        with region:
            if timing is None:
                x_final, stats = reconstruct(adapter, cond, x0, sub_problem, spec)
            else:
                timing.timer.start()
                x_final, stats = reconstruct(adapter, cond, x0, sub_problem, spec)
                timing.timer.stop(x_final, stats.seconds, size)
        if timing is not None:
            timing.snapshot("after_timed_region")
        total.merge(stats)
        pixels = adapter.to_pixels(x_final)
        outputs.append(pixels[:size] if pad else pixels)

    result = np.concatenate(outputs, axis=0)[:len(indices)]
    return {"pixels": np.asarray(result, np.float32), "stats": total,
            "padded_items": padded_items, "initial_state_fingerprints": fingerprints,
            "status": "ok" if total.finite else "non_finite_state"}


def warm_up(adapter, spec, problem, manager, verbose: bool = True) -> float:
    """One untimed reduced run per (model, method, required shape).

    pMF/iMF use JAX and pay a compilation cost on the first call for each shape.  Comparing
    a cold first run against already-compiled later jobs would make the runtime column
    meaningless, so the benchmark reports steady-state reconstruction runtime and records
    the cold/compile time separately.
    """
    # The warm-up must trace EXACTLY the trajectory the timed job will run.  Reducing
    # steps / num_mpc_steps / num_pnp_steps here would change the time grid, so the real
    # job would meet uncompiled intervals and pay for their compilation inside the measured
    # region -- which is precisely what "steady-state runtime" is supposed to exclude.
    #
    #   PnP     each correction time s_k -> 1 is a DISTINCT transition for a MeanFlow model,
    #           so all N of them must be traced.  noise_samples is kept as well: it changes
    #           no shape, but keeping it makes the warm-up a faithful rehearsal.
    #   D-Flow  the trajectory shape is fixed by `steps` and the solver, and every optimiser
    #           iteration re-runs the SAME traced computation, so a single iteration is
    #           enough to compile everything the timed run needs -- including the backward
    #           pass and the Adam update.
    #   RHSO    num_rhso_steps and beta fix the outer grid, and EVERY outer stage traces a
    #           different computation (a different s_k, and for a standard flow a shorter
    #           remaining suffix), so all N stages must run.  Only the inner Adam budget is
    #           reduced: each of its iterations re-runs the same traced objective.
    #
    # Only n_ctrl / num_opt_steps and the image count are reduced: none of them changes the
    # traced computation.  `beta` is NEVER reduced -- it defines the time grid itself.
    reductions: Dict[str, Any] = {
        "num_images": min(int(spec.batch_size), int(spec.num_images)),
        "record_loss_history": False,
    }
    if spec.method in ("mpc_rhc", "mpc_delta_t"):
        reductions["n_ctrl"] = 1
    if spec.method in ("dflow", "rhso"):
        reductions["num_opt_steps"] = 1
    if spec.method == "rhso" and spec.rhso_jacobian_diagnostics:
        # The probes and power iterations re-run the SAME traced computation, so one of
        # each compiles everything the timed job needs. Their number is a cost, not a
        # shape -- and the diagnostics are excluded from the timed number anyway.
        reductions["rhso_jacobian_probes"] = 1
        reductions["rhso_jacobian_power_iters"] = 1
    tiny = dataclasses.replace(spec, **reductions)
    started = time.perf_counter()
    run_job(adapter, tiny, problem, manager, profiler=None)
    seconds = time.perf_counter() - started
    if verbose:
        print("        warm-up (untimed, excluded from results): %.2fs" % seconds)
    return seconds


def warmup_key(spec, problem) -> Tuple:
    """Everything that changes the traced computation or the compiled shapes.

    Two jobs sharing this key can share one warm-up; two jobs differing in any of it cannot,
    or the second would compile inside its measured region.
    """
    return (spec.method, spec.batch_size, spec.problem,
            tuple(problem.measurement.shape[1:]), spec.K, spec.t0,
            # beta changes the RESOLVED time grid, so two jobs differing only in it must not
            # share a warm-up: the second would compile inside its measured region.
            spec.beta,
            spec.steps, spec.num_mpc_steps, spec.solver,
            spec.phi_normalization, spec.control_cost_normalization,
            # PnP: N fixes the whole s_k grid, M fixes how many denoiser calls are traced.
            spec.num_pnp_steps, spec.noise_samples,
            # D-Flow: the trajectory graph, not the number of Adam updates.
            spec.optimizer,
            # RHSO: N fixes the outer grid AND, for a standard flow, every planning suffix;
            # mu changes the objective that is traced and differentiated; the terminal mode
            # decides WHICH computation the objective is (one endpoint prediction versus an
            # integrated suffix), and the diagnostics add traced computations of their own.
            spec.num_rhso_steps, spec.mu, spec.rhso_terminal_mode,
            # The anchor schedule selects which objective is traced (constant: mu closed
            # over; increasing / decreasing: mu_k passed as an argument).
            getattr(spec, "mu_schedule", None),
            spec.rhso_stage_diagnostics, spec.rhso_consistency_diagnostics,
            spec.rhso_jacobian_diagnostics, spec.rhso_gradient_authority_diagnostics,
            # Flower: N fixes the time grid; gamma adds the posterior-sample computation;
            # the CG limits bound the traced solve.
            getattr(spec, "num_flower_steps", None), getattr(spec, "flower_gamma", None),
            getattr(spec, "flower_cg_max_iters", None), getattr(spec, "flower_cg_tol", None))


# =====================================================================================
# Result records
# =====================================================================================
RESULT_COLUMNS = [
    "run_id", "job_id", "task", "experiment", "problem", "problem_key", "image_id",
    "model", "model_family", "framework", "state_space", "method", "method_title",
    "t0", "beta", "canonical_start_time", "native_start_time", "native_end_time",
    "native_time_mapping", "initialization_kind", "initialization_guide_mode",
    "steps", "solver", "num_mpc_steps", "delta", "delta_nominal_uniform", "delta_min",
    "delta_max", "K", "lam", "n_ctrl", "lr",
    "optimizer", "warm_start", "grad_clip", "phi_normalization",
    "control_cost_normalization", "delta_t_lambda_scaling",
    # PnP-Flow / D-Flow / RHSO hyperparameters
    "num_pnp_steps", "gamma0", "alpha", "noise_samples", "num_opt_steps", "num_rhso_steps",
    "mu",
    # RHSO anchor schedule ("constant" | "increasing" | "decreasing"; None off RHSO)
    "mu_schedule",
    # Flower hyperparameters
    "num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol",
    "hyperparameter_source_lam", "hyperparameter_source_n_ctrl", "hyperparameter_source_lr",
    "hyperparameter_source_gamma0", "hyperparameter_source_alpha",
    "hyperparameter_source_mu",
    "psnr", "ssim", "lpips", "measurement_rmse", "missing_rmse", "missing_psnr",
    "missing_ssim", "observed_rmse", "observed_psnr",
    "degraded_psnr", "degraded_ssim", "degraded_lpips", "guide_psnr",
    "runtime", "runtime_per_image", "reconstruction_steps", "network_forwards",
    "model_evaluations", "planning_model_evaluations", "control_iterations",
    "backprops_through_model", "expected_model_evaluations", "expected_backprops",
    # compute accounting added with PnP-Flow / D-Flow
    "objective_evaluations", "data_gradient_evaluations", "optimizer_iterations",
    "denoiser_samples", "expected_objective_evaluations", "expected_data_gradients",
    # inner linear-solver iterations (Flower's CG); 0 for every other method
    "linear_solve_iterations",
    # RHSO terminal planner and theory diagnostics
    "rhso_terminal_mode", "rhso_terminal_planner", "rhso_consistency_kind",
    "rhso_stage_diagnostics", "rhso_consistency_diagnostics", "rhso_jacobian_diagnostics",
    "rhso_jacobian_probes", "rhso_jacobian_power_iters", "rhso_jacobian_seed",
    "rhso_gradient_authority_diagnostics",
    # Which common-random-number pairing group produced this row's measurement, if any.
    "measurement_noise_group",
    # DIAGNOSTIC cost, kept strictly apart from the algorithmic counters above
    "diagnostic_model_evaluations", "diagnostic_network_forwards", "diagnostic_jvps",
    "diagnostic_vjps", "diagnostic_seconds",
    # job-level summaries of the per-(image, stage) arrays in results.npz
    "stage_count", "stage_delta_first", "stage_delta_last", "stage_delta_per_step_first",
    "stage_delta_per_step_last", "stage_theta_first", "stage_theta_last",
    "stage_v_pre_first", "stage_v_post_last",
    "endpoint_shift_native_relative_mean", "endpoint_shift_pixel_rmse_mean",
    "fidelity_shift_execution_mean", "next_stage_recovery_mean",
    "jacobian_sigma_max_first", "jacobian_sigma_max_last",
    "jacobian_log_anisotropy_first", "jacobian_log_anisotropy_last",
    "gradient_aligned_authority_first", "gradient_aligned_authority_last",
    "gradient_aligned_authority_mean", "endpoint_fidelity_grad_norm_mean",
    "state_fidelity_grad_norm_mean",
    # GPU memory (job peak at the configured batch size -- never per image)
    "gpu_baseline_gib", "gpu_peak_gib", "gpu_incremental_peak_gib", "gpu_memory_source",
    # The ONE memory metric that is comparable between PyTorch and JAX (see src/memory.py)
    "gpu_process_baseline_gib", "gpu_process_peak_gib", "gpu_process_incremental_peak_gib",
    "gpu_process_memory_source",
    "batch_size", "num_images", "padded_items", "warmup_seconds", "model_load_seconds",
    "seed", "replicate", "generative_noise_id", "measurement_id", "guide_id",
    "initial_state_fingerprint", "conditioning_label", "cfg_scale",
    "status", "failure", "output_dir", "timestamp",
]


def build_record(spec, plan, problem, run: Optional[Dict[str, Any]],
                 metrics: Optional[Dict[str, Any]], degraded: Dict[str, Any],
                 status: str, failure: Optional[str], warmup_seconds: float,
                 load_seconds: float,
                 memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stats = run["stats"] if run else ReconstructionStats()
    metrics = metrics or {}
    memory = memory or {}
    guidance = spec.guidance or {}
    record = {
        "run_id": plan.run_id, "job_id": spec.job_id, "task": spec.experiment,
        "experiment": spec.experiment, "problem": spec.problem,
        "problem_key": spec.problem_key, "image_id": "ALL",
        "model": spec.model, "model_family": spec.dynamics_family,
        "framework": spec.framework, "state_space": spec.state_space,
        "method": spec.method, "method_title": spec.method_title,
        "t0": spec.t0, "beta": spec.beta,
        "canonical_start_time": spec.canonical_start_time,
        "native_start_time": spec.native_start_time, "native_end_time": spec.native_end_time,
        "native_time_mapping": spec.native_time_mapping,
        "initialization_kind": spec.initialization_kind,
        "initialization_guide_mode": spec.initialization_guide_mode,
        "steps": spec.steps, "solver": spec.solver, "num_mpc_steps": spec.num_mpc_steps,
        # `delta` is NOMINAL uniform spacing and is null when beta != 1; delta_min/max are
        # the step sizes actually executed.
        "delta": spec.delta, "delta_nominal_uniform": spec.delta_nominal_uniform,
        "delta_min": spec.delta_min, "delta_max": spec.delta_max,
        "K": spec.K, "lam": spec.lam, "n_ctrl": spec.n_ctrl,
        "lr": spec.lr, "optimizer": spec.optimizer, "warm_start": spec.warm_start,
        "grad_clip": spec.grad_clip, "phi_normalization": spec.phi_normalization,
        "control_cost_normalization": spec.control_cost_normalization,
        "delta_t_lambda_scaling": spec.delta_t_lambda_scaling,
        "num_pnp_steps": spec.num_pnp_steps, "gamma0": spec.gamma0, "alpha": spec.alpha,
        "noise_samples": spec.noise_samples, "num_opt_steps": spec.num_opt_steps,
        "num_rhso_steps": spec.num_rhso_steps, "mu": spec.mu,
        "mu_schedule": getattr(spec, "mu_schedule", None),
        "num_flower_steps": spec.num_flower_steps, "flower_gamma": spec.flower_gamma,
        "flower_cg_max_iters": spec.flower_cg_max_iters, "flower_cg_tol": spec.flower_cg_tol,
        "hyperparameter_source_lam": spec.hyperparameter_sources.get("lam"),
        "hyperparameter_source_n_ctrl": spec.hyperparameter_sources.get("n_ctrl"),
        "hyperparameter_source_lr": spec.hyperparameter_sources.get("lr"),
        "hyperparameter_source_gamma0": spec.hyperparameter_sources.get("gamma0"),
        "hyperparameter_source_alpha": spec.hyperparameter_sources.get("alpha"),
        "hyperparameter_source_mu": spec.hyperparameter_sources.get("mu"),
        "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"), "measurement_rmse": metrics.get("measurement_rmse"),
        "missing_rmse": metrics.get("missing_rmse"), "missing_psnr": metrics.get("missing_psnr"),
        "missing_ssim": metrics.get("missing_ssim"),
        "observed_rmse": metrics.get("observed_rmse"),
        "observed_psnr": metrics.get("observed_psnr"),
        "degraded_psnr": degraded.get("psnr"), "degraded_ssim": degraded.get("ssim"),
        "degraded_lpips": degraded.get("lpips"), "guide_psnr": degraded.get("guide_psnr"),
        "runtime": stats.seconds,
        "runtime_per_image": stats.seconds / max(1, int(spec.num_images)),
        "reconstruction_steps": spec.reconstruction_steps,
        "network_forwards": stats.network_forwards,
        "model_evaluations": stats.model_evals_total,
        "planning_model_evaluations": stats.model_evals_planning,
        "control_iterations": stats.control_iterations,
        "backprops_through_model": stats.backprops_through_model,
        "expected_model_evaluations": spec.expected_model_evals * spec.num_chunks,
        "expected_backprops": spec.expected_backprops * spec.num_chunks,
        "objective_evaluations": stats.objective_evals,
        "data_gradient_evaluations": stats.data_gradient_evals,
        "optimizer_iterations": stats.optimizer_iterations,
        "denoiser_samples": stats.denoiser_samples,
        "expected_objective_evaluations": spec.expected_objective_evals * spec.num_chunks,
        "expected_data_gradients": spec.expected_data_gradients * spec.num_chunks,
        "linear_solve_iterations": stats.linear_solve_iterations,
        "rhso_terminal_mode": spec.rhso_terminal_mode,
        "rhso_terminal_planner": spec.rhso_terminal_planner,
        # What ||p_k - r_k|| means for this job. A pMF semigroup defect and a JiT endpoint
        # predictor's shift are different claims; the column keeps them apart.
        "rhso_consistency_kind": (spec.rhso_consistency_kind
                                  if spec.rhso_consistency_diagnostics else None),
        "rhso_stage_diagnostics": spec.rhso_stage_diagnostics,
        "rhso_consistency_diagnostics": spec.rhso_consistency_diagnostics,
        "rhso_jacobian_diagnostics": spec.rhso_jacobian_diagnostics,
        "rhso_jacobian_probes": (spec.rhso_jacobian_probes
                                 if spec.rhso_jacobian_diagnostics else None),
        "rhso_jacobian_power_iters": (spec.rhso_jacobian_power_iters
                                      if spec.rhso_jacobian_diagnostics else None),
        "rhso_jacobian_seed": (spec.rhso_jacobian_seed
                               if spec.rhso_jacobian_diagnostics else None),
        "rhso_gradient_authority_diagnostics": spec.rhso_gradient_authority_diagnostics,
        "measurement_noise_group": spec.measurement_noise_group,
        # Diagnostic compute, never added to model_evaluations / network_forwards /
        # runtime: those keep their existing meaning as the ALGORITHM's cost.
        "diagnostic_model_evaluations": stats.diagnostic_model_evals,
        "diagnostic_network_forwards": stats.diagnostic_network_forwards,
        "diagnostic_jvps": stats.diagnostic_jvps,
        "diagnostic_vjps": stats.diagnostic_vjps,
        "diagnostic_seconds": round(float(stats.diagnostic_seconds), 4),
        "gpu_baseline_gib": memory.get("gpu_baseline_gib"),
        "gpu_peak_gib": memory.get("gpu_peak_gib"),
        "gpu_incremental_peak_gib": memory.get("gpu_incremental_peak_gib"),
        "gpu_memory_source": memory.get("gpu_memory_source"),
        "gpu_process_baseline_gib": memory.get("gpu_process_baseline_gib"),
        "gpu_process_peak_gib": memory.get("gpu_process_peak_gib"),
        "gpu_process_incremental_peak_gib": memory.get("gpu_process_incremental_peak_gib"),
        "gpu_process_memory_source": memory.get("gpu_process_memory_source"),
        "batch_size": spec.batch_size, "num_images": spec.num_images,
        "padded_items": (run or {}).get("padded_items", 0),
        "warmup_seconds": round(float(warmup_seconds), 4),
        "model_load_seconds": round(float(load_seconds), 2),
        "seed": spec.seed, "replicate": spec.replicate,
        # Identity columns: two rows that agree on these three share the same setup exactly.
        "generative_noise_id": "seed=%d|model=%s|rep=%d" % (spec.seed, spec.model,
                                                            spec.replicate),
        "measurement_id": problem.to_metadata()["measurement_fingerprint"],
        "guide_id": problem.to_metadata()["guide_fingerprint"],
        "initial_state_fingerprint": ((run or {}).get("initial_state_fingerprints") or [""])[0],
        "conditioning_label": "source_imagenet_class",
        "cfg_scale": guidance.get("scale"),
        "status": status, "failure": failure, "output_dir": None, "timestamp": now_iso(),
    }
    # Job-level summaries of the per-(image, stage) diagnostics. The RAW [image, stage]
    # arrays stay in the per-job results.npz -- results.csv gets scalars only.
    record.update(stage_summary(stats.stage_records))
    return record


def per_image_rows(record: Dict[str, Any], metrics: Dict[str, Any],
                   image_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    ps = metrics.get("psnr_per_image") or []
    ss = metrics.get("ssim_per_image") or []
    lp = metrics.get("lpips_per_image") or []
    for i, image_id in enumerate(image_ids):
        row = dict(record)
        row["image_id"] = image_id
        row["psnr"] = ps[i] if i < len(ps) else None
        row["ssim"] = ss[i] if i < len(ss) else None
        row["lpips"] = lp[i] if i < len(lp) else None
        # Memory is a JOB peak over a computational batch; splitting it across the images
        # in that batch would invent a per-image number that was never measured.
        for key in ("gpu_baseline_gib", "gpu_peak_gib", "gpu_incremental_peak_gib"):
            row[key] = record.get(key)
        rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """One CSV writer for the shard files AND for the merged files.

    Shared deliberately: `results_per_image.csv` produced by `--aggregate` must have the
    same columns, in the same order, as `results_per_image_shardNN.csv` written by the
    array tasks, or a downstream analysis cannot read both.  The header is always written,
    even for an empty row list.
    """
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in RESULT_COLUMNS})
    return path


class ResultWriter:
    """Append-only, crash-safe persistence.

    Every atomic job is written the moment it finishes: a Colab crash halfway through a
    sweep never erases earlier reconstructions.
    """

    def __init__(self, run_dir: Path, shard: Optional[Tuple[int, int]] = None):
        self.run_dir = Path(run_dir)
        self.records: List[Dict[str, Any]] = []
        self.image_rows: List[Dict[str, Any]] = []
        # Parallel array tasks must never write the same file: each shard owns its own,
        # and `--aggregate` merges them from the per-job metadata afterwards.
        self.suffix = shard_suffix(shard)
        self.jsonl = self.run_dir / ("results%s.jsonl" % self.suffix)
        self.log = self.run_dir / ("experiment_log%s.jsonl" % self.suffix)

    def add(self, record: Dict[str, Any], image_rows: Sequence[Dict[str, Any]] = ()) -> None:
        self.records.append(record)
        self.image_rows.extend(image_rows)
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(jsonable(record), default=str) + "\n")
        self.write_csv()

    def event(self, status: str, **fields: Any) -> None:
        entry = {"status": status, "time": now_iso()}
        entry.update(fields)
        with open(self.log, "a") as fh:
            fh.write(json.dumps(jsonable(entry), default=str) + "\n")

    def write_csv(self) -> Path:
        path = self.run_dir / ("results%s.csv" % self.suffix)
        write_rows_csv(path, self.records)
        # Written unconditionally, header included: downstream theory analysis reads
        # `results_per_image.csv` by name, and a file that only appears when some job
        # happened to produce per-image metrics is a schema that cannot be relied on.
        write_rows_csv(self.run_dir / ("results_per_image%s.csv" % self.suffix),
                       self.image_rows)
        return path


def job_dir_for(run_dir: Path, spec) -> Path:
    return (run_dir / spec.experiment / spec.model / spec.method / spec.leaf_dir)


def persist_job(run_dir: Path, spec, run: Dict[str, Any], record: Dict[str, Any],
                metrics: Dict[str, Any], problem, save_images: bool) -> Path:
    """Write one job's artefacts immediately."""
    out = job_dir_for(run_dir, spec)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["resolved_spec"] = spec.to_dict()
    if spec.method == "rhso" and getattr(spec, "mu_schedule", None) not in (None, "constant"):
        # The per-stage anchor weights actually used, mu_k = mu * m(s_k). Written only for a
        # non-constant schedule, so a default job's metadata keeps its previous keys.
        from src.rhso import rhso_stage_mus
        grid = rhso_time_grid(spec)
        payload["rhso_mu_schedule"] = {"schedule": spec.mu_schedule, "mu": spec.mu,
                                       "stage_times": grid[:-1],
                                       "stage_mu": rhso_stage_mus(spec)}
    payload["problem_metadata"] = problem.to_metadata()
    payload["per_image"] = {k: metrics.get(k) for k in
                            ("psnr_per_image", "ssim_per_image", "lpips_per_image")}
    stats = run["stats"]
    if stats.stage_records:
        # Everything a later analysis needs to interpret the arrays in results.npz without
        # rerunning a single model.
        payload["rhso_diagnostics"] = {
            "model": spec.model,
            "dynamics_family": spec.dynamics_family,
            "terminal_mode": spec.rhso_terminal_mode,
            "terminal_planner": stats.terminal_planner_kind or spec.rhso_terminal_planner,
            "consistency_kind": stats.consistency_kind,
            "consistency_meaning": CONSISTENCY_MEANING.get(stats.consistency_kind or ""),
            "settings": stats.diagnostic_settings,
            "batch_size": spec.batch_size,
            "num_images": spec.num_images,
            "padded_items": run.get("padded_items", 0),
            "num_rhso_steps": spec.num_rhso_steps,
            "num_opt_steps": spec.num_opt_steps,
            "t0": spec.t0, "canonical_start_time": spec.canonical_start_time,
            "beta": spec.beta, "mu": spec.mu, "lr": spec.lr,
            "phi_normalization": spec.phi_normalization,
            "outer_times": rhso_time_grid(spec),
            "diagnostic_cost": {
                "model_evals": stats.diagnostic_model_evals,
                "network_forwards": stats.diagnostic_network_forwards,
                "jvps": stats.diagnostic_jvps, "vjps": stats.diagnostic_vjps,
                "seconds": round(float(stats.diagnostic_seconds), 4)},
            "gpu_memory_source": record.get("gpu_memory_source"),
            "gpu_process_memory_source": record.get("gpu_process_memory_source"),
            "conventions": (
                "Arrays are [image, stage] (gains are [image, stage, probe]); rows follow "
                "problem.image_ids and padded batch rows are excluded. V_pre / V_post are "
                "PER-IMAGE terminal fidelities from problems.make_phi_per_sample, and "
                "V_post is evaluated AFTER the final Adam update, never read off the loss "
                "history. The final stage has no successor, so its consistency, "
                "fidelity_shift_execution and next_stage_recovery entries are NaN by "
                "convention rather than fabricated. jacobian_sigma_max is a power-iteration "
                "estimate of the DOMINANT singular scale only; the gain statistics are "
                "empirical sample-based anisotropy, and neither is an exact condition "
                "number. gradient_aligned_authority is "
                "||J^T grad Phi|| / ||grad Phi|| of the FIDELITY ONLY -- the state-anchor "
                "mu penalty is excluded -- measured at the state entering the stage, and "
                "is NaN by convention whenever the endpoint gradient norm falls below "
                "the documented threshold rather than being divided by an epsilon."),
        }
    save_json(out / "metadata.json", payload)
    arrays: Dict[str, Any] = {"reconstruction": to_uint8(run["pixels"])}
    if spec.record_loss_history and run["stats"].loss_history:
        arrays["loss_history"] = np.asarray(run["stats"].loss_history, np.float32)
        # RHSO with a state-anchor penalty splits that same scalar into its two parts, on
        # the same per-image scale: loss = fidelity + mu * state_penalty. Absent for every
        # method that has no such split.
        if run["stats"].fidelity_history:
            arrays["fidelity_history"] = np.asarray(run["stats"].fidelity_history,
                                                    np.float32)
            arrays["state_penalty_history"] = np.asarray(
                run["stats"].state_penalty_history, np.float32)
    # [image, stage] structure is preserved deliberately: the whole point of these
    # diagnostics is to compare stage k with stage k+1 for the SAME image.
    arrays.update(stage_arrays(run["stats"].stage_records, problem.image_ids))
    np.savez_compressed(out / "results.npz", **arrays)
    if save_images:
        img_dir = out / "images"
        img_dir.mkdir(exist_ok=True)
        from PIL import Image
        for i, pix in enumerate(run["pixels"]):
            Image.fromarray(to_uint8(pix)).save(img_dir / ("%s.png" % problem.image_ids[i]))
    return out


# Fields added to JobSpec AFTER runs may already have been written to disk.  A stored job
# that predates them can only be reused when the new job would run the SAME computation.
NEW_SPEC_FIELDS: Tuple[str, ...] = (
    "rhso_terminal_mode", "rhso_terminal_planner", "rhso_consistency_kind",
    "rhso_stage_diagnostics", "rhso_consistency_diagnostics", "rhso_jacobian_diagnostics",
    "rhso_jacobian_probes", "rhso_jacobian_power_iters", "rhso_jacobian_seed",
    "rhso_gradient_authority_diagnostics", "measurement_noise_group",
    "num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol",
    "mu_schedule")


def _describes_legacy_behaviour(name: str, current: Dict[str, Any]) -> bool:
    """Would this NEW field's value reproduce what a run predating it did?

    * `rhso_terminal_mode` must resolve to the planner its family already used -- a job
      that now asks a standard flow for DIRECT planning is a different experiment and must
      re-run.
    * `rhso_terminal_planner` and `rhso_consistency_kind` are derived LABELS, not settings:
      they are functions of the mode and the model, so they cannot change what is computed.
    * every diagnostic flag must be off; diagnostics add measurements a stored run does not
      contain, so a job that wants them cannot be satisfied by an old artefact.
    * the probe budget and seed only matter when the Jacobian diagnostic is on, which the
      previous rule has already excluded.
    """
    from src.rhso import legacy_terminal_mode
    value = current.get(name)
    if name == "rhso_terminal_mode":
        return value is None or value == legacy_terminal_mode(current.get("dynamics_family"))
    if name in ("rhso_terminal_planner", "rhso_consistency_kind"):
        return True
    if name in ("rhso_stage_diagnostics", "rhso_consistency_diagnostics",
                "rhso_jacobian_diagnostics", "rhso_gradient_authority_diagnostics"):
        return not value
    if name == "measurement_noise_group":
        # A stored artefact was produced with the original, unpaired seeding. A job that
        # now asks for a pairing group wants a DIFFERENT y and must re-run.
        return value is None
    if name in ("num_flower_steps", "flower_gamma", "flower_cg_max_iters", "flower_cg_tol"):
        # Only a job of another method can predate Flower; for it these are all None.
        return value is None
    if name == "mu_schedule":
        # Every RHSO run before the field existed used a time-independent mu.
        return value in (None, "constant")
    return True


def _resolved_spec_matches(stored: Any, current: Dict[str, Any]) -> bool:
    """Is a stored resolved spec the same experiment as the one about to run?

    Every field they share must agree exactly -- that is the existing rule, and it is what
    stops a job differing only in gamma0 or lr from reusing another's reconstruction.  The
    one concession is to fields that did not EXIST when the stored job ran: an artefact
    written before the RHSO terminal-planner and diagnostic fields were added is still
    resumable, but only when the new job's values for them describe the same computation
    (see `_describes_legacy_behaviour`).
    """
    if not isinstance(stored, dict):
        return False
    if stored == current:
        return True
    if set(stored) - set(current):
        return False                              # the stored run knew fields we do not
    added = set(current) - set(stored)
    if any(name not in NEW_SPEC_FIELDS for name in added):
        return False
    if not all(_describes_legacy_behaviour(name, current) for name in added):
        return False
    return all(current[name] == value for name, value in stored.items())


def load_finished_job(run_dir: Path, spec) -> Optional[Dict[str, Any]]:
    """Resume support: reuse a job only when its RESOLVED spec matches exactly.

    The comparison is over the whole resolved spec, so a job that differs only in gamma0,
    alpha, noise_samples, num_opt_steps or the fidelity normalisation can never reuse
    another configuration's reconstruction.
    """
    out = job_dir_for(run_dir, spec)
    meta_path, array_path = out / "metadata.json", out / "results.npz"
    if not (meta_path.exists() and array_path.exists()):
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return None
    if payload.get("status") != "ok" or payload.get("job_id") != spec.job_id:
        return None
    if not _resolved_spec_matches(payload.get("resolved_spec"), jsonable(spec.to_dict())):
        return None
    with np.load(array_path) as z:
        pixels = z["reconstruction"]
    return {"record": payload, "pixels": pixels, "output_dir": str(out)}


# =====================================================================================
# Reporting
# =====================================================================================
METHOD_ORDER = {"sdedit": 0, "mpc_rhc": 1, "mpc_delta_t": 2, "pnp": 3, "dflow": 4,
                "rhso": 5, "flower": 6}


def print_summary_tables(records: Sequence[Dict[str, Any]]) -> None:
    """Per-configuration tables.

    Nothing is averaged across models, methods or hyperparameters: each row below is one
    atomic job, because a mean over a 4-step and a 25-step SDEdit (or over two lambdas)
    describes a run that was never executed.  The only averaging is over the images inside
    a single job, which is what `num_images` means.
    """
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        print("No successful job to summarise.")
        return

    print("\n" + RULE)
    print("RESULTS BY RESOLVED CONFIGURATION   (one row = one atomic job, averaged over its "
          "images only)")
    print(RULE)
    print("%-18s %-5s %-26s %7s %7s %8s %10s %9s"
          % ("task", "model", "configuration", "PSNR", "SSIM", "LPIPS", "meas-RMSE",
             "s/image"))
    print(THIN)
    last_task = None
    for row in sorted(ok, key=lambda r: (r["task"], r["model"],
                                         METHOD_ORDER.get(r["method"], 9),
                                         viz.config_label(r))):
        if last_task is not None and row["task"] != last_task:
            print()
        last_task = row["task"]
        degraded = row.get("degraded_psnr")
        flag = " " if (row.get("psnr") or 0) >= (degraded or -1e9) else "!"
        print("%-18s %-5s %-26s %7.2f %7.4f %8s %10.4f %9.2f %s"
              % (row["task"][:18], row["model"], viz.config_label(row)[:26], row["psnr"],
                 row["ssim"],
                 ("%.4f" % row["lpips"]) if row.get("lpips") is not None else "n/a",
                 row["measurement_rmse"], row["runtime_per_image"], flag))
    print(THIN)
    print("!  marks a reconstruction whose PSNR is BELOW the degraded observation itself,")
    print("   i.e. one that did worse than leaving the measurement untouched.")

    # ---------------------------------------------------------------- compute and memory
    print("\n" + RULE)
    print("COMPUTE AND MEMORY BY RESOLVED CONFIGURATION")
    print("memory is a JOB peak at the stated batch size -- never divided by the batch")
    print(RULE)
    print("%-18s %-5s %-24s %5s %8s %8s %8s %9s %9s"
          % ("task", "model", "configuration", "batch", "modelev", "backprop", "datagrad",
             "peakGiB", "deltaGiB"))
    print(THIN)
    any_memory = False
    for row in sorted(ok, key=lambda r: (r["task"], r["model"],
                                         METHOD_ORDER.get(r["method"], 9),
                                         viz.config_label(r))):
        peak, delta = row.get("gpu_peak_gib"), row.get("gpu_incremental_peak_gib")
        any_memory = any_memory or peak is not None
        print("%-18s %-5s %-24s %5s %8s %8s %8s %9s %9s"
              % (row["task"][:18], row["model"], viz.config_label(row)[:24],
                 row.get("batch_size"), row.get("model_evaluations"),
                 row.get("backprops_through_model"),
                 row.get("data_gradient_evaluations"),
                 ("%.3f" % peak) if peak is not None else "n/a",
                 ("%.3f" % delta) if delta is not None else "n/a"))
    print(THIN)
    if any_memory:
        sources = sorted({r.get("gpu_memory_source") for r in ok
                          if r.get("gpu_memory_source")})
        for source in sources:
            print("   measurement source: %s" % source)
        print("   deltaGiB is the peak MINUS the memory already resident before the")
        print("   reconstruction, i.e. what the method itself added on top of the model.")
    else:
        print("   No GPU memory was recorded (CPU run, instrumentation unavailable, or")
        print("   --no-gpu-memory).")

    # ---------------------------------------------------------------- paired deltas
    print("\n" + RULE)
    print("PAIRED COMPARISON vs STEP-MATCHED SDEdit")
    print("each non-baseline job against the SDEdit job at the SAME task, model and t0 whose")
    print("step count is closest to its own -- never against an average of SDEdit runs")
    print(RULE)
    print("%-18s %-5s %-24s %-14s %8s %8s %9s %8s"
          % ("task", "model", "configuration", "baseline", "dPSNR", "dSSIM", "dLPIPS",
             "runtime"))
    print(THIN)
    compared = [r for r in ok if r["method"] in COMPARED_METHODS]
    printed = 0
    last_task = None
    for row in sorted(compared, key=lambda r: (r["task"], r["model"],
                                               METHOD_ORDER.get(r["method"], 9),
                                               viz.config_label(r))):
        base = viz.step_matched_baseline(row, ok)
        if base is None:
            continue
        if last_task is not None and row["task"] != last_task:
            print()
        last_task = row["task"]
        printed += 1
        d_lpips = ("%+9.4f" % (row["lpips"] - base["lpips"])
                   if None not in (row.get("lpips"), base.get("lpips")) else "      n/a")
        ratio = (row["runtime_per_image"] / base["runtime_per_image"]
                 if base.get("runtime_per_image") else float("nan"))
        print("%-18s %-5s %-24s %-14s %+8.2f %+8.4f %s %7.1fx"
              % (row["task"][:18], row["model"], viz.config_label(row)[:24],
                 viz.config_label(base)[:14], row["psnr"] - base["psnr"],
                 row["ssim"] - base["ssim"], d_lpips, ratio))
    if not printed:
        print("  (no job has a paired SDEdit baseline at the same t0)")
    print(THIN)
    print("dLPIPS < 0 is an improvement. 'runtime' is the multiple of the named baseline.")

    failed = [r for r in records if r["status"] != "ok"]
    if failed:
        print("\n%d job(s) did not complete:" % len(failed))
        for r in failed[:12]:
            print("  %-10s %-18s %-5s %-12s %s"
                  % (r["status"], r["task"][:18], r["model"], r["method"],
                     (r.get("failure") or "")[:70]))


def make_figures(run_dir: Path, plan, store, records, reconstructions,
                 show: bool = True) -> List[Path]:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    problems = {key: store.get(key) for key in
                dict.fromkeys(s.problem_key for s in plan.specs)}
    paths.append(viz.plot_problem_overview(problems, figures_dir / "problem_instances.png",
                                           show))
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for spec in plan.specs:
        images = reconstructions.get(spec.job_id)
        if images is None:
            continue
        by_group.setdefault((spec.experiment, spec.model), []).append(
            {"title": spec.figure_title(), "images": images, "method": spec.method,
             "t0": spec.t0, "sort_key": spec.leaf_dir})
    for (experiment, model), entries in sorted(by_group.items()):
        spec = next(s for s in plan.specs if s.experiment == experiment and s.model == model)
        paths += viz.plot_comparison_grid(experiment, model, entries,
                                          store.get(spec.problem_key), figures_dir, show)
    for model in dict.fromkeys(s.model for s in plan.specs):
        breakdown = viz.plot_configuration_breakdown(
            records, model, figures_dir / ("configurations_%s.png" % model), show)
        if breakdown:
            paths.append(breakdown)
    deltas = viz.plot_paired_deltas(records, figures_dir / "paired_deltas.png", show)
    if deltas:
        paths.append(deltas)
    cost = viz.plot_quality_vs_cost(records, figures_dir / "quality_vs_cost.png", show)
    if cost:
        paths.append(cost)
    memory = viz.plot_quality_vs_memory(records, figures_dir / "quality_vs_memory.png", show)
    if memory:
        paths.append(memory)
    return paths


# =====================================================================================
# Prefetch (login node) and aggregate (after a job array)
# =====================================================================================
def prefetch_assets(plan, config: Dict[str, Any], cache_root: Path) -> Dict[str, Any]:
    """Download everything a run needs, so compute nodes never touch the network.

    Alliance/DRAC compute nodes have no route to the internet.  Anything not on disk when
    the job starts is a failure, not a slow download, so this runs once on a login node.
    Deliberately download-only: no model is built and no GPU is required.
    """
    report: Dict[str, Any] = {"cache_root": str(cache_root), "models": {}, "started": now_iso()}
    print(RULE)
    print("PREFETCH -- staging every asset for offline execution")
    print("  cache root: %s" % cache_root)
    print(RULE)

    print("\n[1/4] Model repositories")
    context = ensure_repositories(plan, cache_root)
    context["ckpt_cache"] = (cache_root / "checkpoints").resolve()
    context["ckpt_cache"].mkdir(parents=True, exist_ok=True)
    report["repositories"] = context.get("repo_heads", {})

    print("\n[2/4] Model checkpoints")
    load_adapters(plan.resources.models)
    for model in plan.resources.models:
        hook = PREFETCH_HOOKS.get(model)
        if hook is None:
            print("   %-5s no prefetch hook; it will download at first use." % model)
            report["models"][model] = {"status": "no_hook"}
            continue
        try:
            registry = resolve_model_registry(config, model)
            info = hook(registry, context)
            report["models"][model] = dict(info, status="ok")
            print("   %-5s ok  %s" % (model, info))
        except Exception as exc:
            report["models"][model] = {"status": "failed", "error": str(exc)}
            print("   %-5s FAILED: %s" % (model, exc))
            traceback.print_exc(limit=4)

    print("\n[3/4] Source images")
    try:
        data = DataManager(config, cache_root / "data")
        pool = data.pool(plan.resources.source_pool_size)
        report["data"] = {"count": len(pool), "source": pool.source,
                          "image_ids": pool.image_ids}
        print("   cached %d image(s) from %s" % (len(pool), pool.source))
    except Exception as exc:
        report["data"] = {"status": "failed", "error": str(exc)}
        print("   FAILED: %s" % exc)
        print("   (a gated-dataset failure here is almost always a missing HF_TOKEN)")

    print("\n[4/4] LPIPS weights")
    if (config.get("metrics") or {}).get("lpips", True):
        try:
            from src.metrics import lpips_per_image, repair_lpips_weights
            # Some redistributed wheels omit the bundled calibration head; fix it here,
            # where there is still a network, rather than failing inside a job.
            repair_lpips_weights((config.get("metrics") or {}).get("lpips_net", "alex"))
            probe = np.zeros((1, 64, 64, 3), np.float32)
            ok = lpips_per_image(probe, probe) is not None
            report["lpips"] = "ok" if ok else "unavailable"
            print("   %s" % ("AlexNet weights cached" if ok
                             else "LPIPS unavailable -- results will have empty LPIPS"))
        except Exception as exc:
            report["lpips"] = "failed: %s" % exc
            print("   FAILED: %s" % exc)
    else:
        report["lpips"] = "disabled"

    report["finished"] = now_iso()
    save_json(cache_root / "prefetch_report.json", report)
    print("\n%s\nPREFETCH COMPLETE -- report: %s"
          % (RULE, cache_root / "prefetch_report.json"))
    failures = [m for m, v in report["models"].items() if v.get("status") == "failed"]
    if failures or report.get("data", {}).get("status") == "failed":
        print("SOME ASSETS FAILED (%s). Fix them here: a compute node cannot download."
              % (failures or "data"))
        return report
    print("Compute nodes can now run fully offline. Point runs at the SAME cache root:")
    print("    python run.py --config %s --cache-root %s" % ("configs/experiments.yaml",
                                                             cache_root))
    print(RULE)
    return report


def image_rows_from_metadata(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild one job's per-image rows from its persisted metadata.json.

    `persist_job` stores the per-image metric vectors under "per_image" and the image ids
    inside "problem_metadata", so the rows can be reconstructed with the SAME function that
    produced them during the run -- no separate schema, and nothing to drift.  Returns []
    when a job predates those fields rather than inventing rows.
    """
    per_image = payload.get("per_image") or {}
    image_ids = ((payload.get("problem_metadata") or {}).get("image_ids") or [])
    if not image_ids:
        # An older artefact without problem metadata: fall back to the length of whichever
        # metric vector is present, so a partial run still contributes its rows.
        longest = max((len(per_image.get(k) or [])
                       for k in ("psnr_per_image", "ssim_per_image", "lpips_per_image")),
                      default=0)
        if not longest:
            return []
        image_ids = ["%s#%d" % (payload.get("job_id", "job"), i) for i in range(longest)]
    record = {k: payload.get(k) for k in RESULT_COLUMNS}
    return per_image_rows(record, per_image, image_ids)


def collect_finished_jobs(run_dir: Path, plan) -> Tuple[List[Dict[str, Any]],
                                                        List[Dict[str, Any]],
                                                        Dict[str, np.ndarray]]:
    """Read every per-job artefact under a run directory.

    Per-job metadata.json is the source of truth, so merging the output of parallel array
    tasks needs no coordination between them and no shared append-only file.

    Three things this guarantees for `--aggregate`:

      * NO DUPLICATES -- one job_id contributes once, even if a stale directory from an
        earlier layout still holds a copy;
      * NO DROPPED ROWS -- every finished job contributes both its summary record and its
        per-image rows, so a complete 4-job x 4-image smoke run merges to exactly 16
        per-image rows;
      * DETERMINISTIC ORDER -- sorted by the job's position in the resolved plan (job_id
        breaks ties and orders anything the plan does not contain), so two aggregations of
        the same directory produce byte-identical CSVs regardless of filesystem order.
    """
    order = {spec.job_id: i for i, spec in enumerate(plan.specs)}
    by_id = set(order)
    found: Dict[str, Dict[str, Any]] = {}
    reconstructions = {}
    for meta_path in sorted(run_dir.rglob("metadata.json")):
        try:
            payload = json.loads(meta_path.read_text())
        except Exception:
            continue
        job_id = payload.get("job_id")
        if not job_id or payload.get("status") != "ok" or job_id in found:
            continue
        found[job_id] = payload
        array_path = meta_path.parent / "results.npz"
        if job_id in by_id and array_path.exists():
            try:
                with np.load(array_path) as z:
                    reconstructions[job_id] = (
                        np.asarray(z["reconstruction"], np.float32) / 127.5) - 1.0
            except Exception:
                pass

    ranked = sorted(found.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
    records = [{k: payload.get(k) for k in RESULT_COLUMNS} for _, payload in ranked]
    rows: List[Dict[str, Any]] = []
    for _, payload in ranked:
        rows.extend(image_rows_from_metadata(payload))
    return records, rows, reconstructions


def _shard_index(path: Path) -> int:
    """The NN of a `..._shardNN.ext` filename, for deterministic ordering."""
    stem = Path(path).stem
    marker = stem.rfind("_shard")
    if marker < 0:
        return -1
    try:
        return int(stem[marker + len("_shard"):])
    except ValueError:
        return -1


def _shard_files(run_dir: Path, pattern: str) -> List[Tuple[int, Path]]:
    return sorted(((_shard_index(p), p) for p in run_dir.glob(pattern)),
                  key=lambda pair: (pair[0], pair[1].name))


def merge_shard_checks(run_dir: Path) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Merge `checks_shardNN.json` into ONE canonical check list.

    Model checks live under a per-model scope, so a two-shard run that put pMF in shard 00
    and JiT in shard 01 contributes both families here -- which the last-writer-wins
    `checks.json` of the first smoke test did not.

    Structural checks run in EVERY shard and therefore appear repeatedly.  They are
    de-duplicated on (scope, name), and when two shards disagree the FAILING result is the
    one kept: a merge must never be able to hide a failure that some shard observed.
    """
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen: List[int] = []
    for index, path in _shard_files(run_dir, "checks_shard*.json"):
        try:
            entries = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        seen.append(index)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = (str(entry.get("scope")), str(entry.get("name")))
            record = dict(entry)
            record["shard"] = index
            previous = merged.get(key)
            if previous is None:
                record["shards"] = [index]
                merged[key] = record
            else:
                shards = list(previous.get("shards") or [])
                if index not in shards:
                    shards.append(index)
                # A failure anywhere wins over a pass elsewhere.
                keep = previous if previous.get("passed") is False else record
                keep = dict(keep)
                keep["shards"] = shards
                merged[key] = keep
    ordered = sorted(merged.values(),
                     key=lambda e: (min(e.get("shards") or [0]), str(e.get("scope")),
                                    str(e.get("name"))))
    return ordered, seen


def merge_shard_metadata(run_dir: Path, checks: List[Dict[str, Any]],
                         figures: Sequence[Path]) -> Optional[Dict[str, Any]]:
    """Merge `run_metadata_shardNN.json` into one canonical provenance document.

    Nothing is dropped.  Per-model dictionaries (provenance, load seconds, repositories,
    problems, degraded metrics) are unioned across shards; scalars that describe the whole
    run are taken from the lowest-numbered shard; anything that is genuinely per-shard --
    the accelerator each task landed on, its elapsed time, its timing summary -- is kept
    under `shards`, and each shard's COMPLETE original payload is preserved verbatim under
    `shard_metadata` so no field can be lost by an omission here.
    """
    payloads: List[Tuple[int, Dict[str, Any]]] = []
    for index, path in _shard_files(run_dir, "run_metadata_shard*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append((index, payload))
    if not payloads:
        return None

    first = payloads[0][1]
    merged: Dict[str, Any] = {k: v for k, v in first.items()
                              if k not in ("checks", "timing", "accelerator", "figures",
                                           "elapsed_seconds", "warnings",
                                           "model_provenance", "model_load_seconds",
                                           "repositories", "problems",
                                           "degraded_metrics")}
    for key in ("model_provenance", "model_load_seconds", "repositories", "problems",
                "degraded_metrics"):
        union: Dict[str, Any] = {}
        for _, payload in payloads:
            value = payload.get(key)
            if isinstance(value, dict):
                union.update(value)
        merged[key] = union

    warnings_: List[str] = []
    for _, payload in payloads:
        for note in (payload.get("warnings") or []):
            if note not in warnings_:
                warnings_.append(note)
    merged["warnings"] = warnings_

    merged["checks"] = checks
    merged["figures"] = [str(p) for p in figures]
    merged["accelerator"] = first.get("accelerator")
    merged["finished"] = now_iso()
    merged["elapsed_seconds"] = round(
        sum(float(p.get("elapsed_seconds") or 0.0) for _, p in payloads), 2)
    merged["shards"] = [
        {"shard": index,
         "run_id": payload.get("run_id"),
         "finished": payload.get("finished"),
         "elapsed_seconds": payload.get("elapsed_seconds"),
         "accelerator": payload.get("accelerator"),
         "models": sorted((payload.get("model_provenance") or {}).keys()),
         "timing": payload.get("timing")}
        for index, payload in payloads]
    merged["shard_metadata"] = {"%02d" % index: payload for index, payload in payloads}
    merged["aggregation_policy"] = (
        "Written by `run.py --aggregate`. Array tasks write shard-private "
        "checks_shardNN.json and run_metadata_shardNN.json; this file is their merge. "
        "elapsed_seconds is the SUM over shards (total GPU wall time), not the makespan; "
        "per-shard values, the node each shard ran on and each shard's complete original "
        "metadata are preserved under `shards` and `shard_metadata`.")
    return merged


def aggregate_only(plan, config: Dict[str, Any], run_dir: Path, args) -> int:
    """Merge the per-job artefacts written by parallel array tasks into one result set."""
    print("\n%s\nAGGREGATE -- merging finished jobs under %s\n%s" % (RULE, run_dir, RULE))
    records, image_rows, reconstructions = collect_finished_jobs(run_dir, plan)
    if not records:
        print("No finished job found. Did the array tasks write to this --run-id?")
        return 1

    done = {r["job_id"] for r in records}
    missing = [s for s in plan.specs if s.job_id not in done]
    print("  found %d finished job(s) of %d planned" % (len(records), len(plan.specs)))
    if missing:
        print("  %d job(s) are MISSING -- the merge below is partial:" % len(missing))
        for spec in missing[:10]:
            print("     %s" % spec.label)
        if len(missing) > 10:
            print("     ... and %d more" % (len(missing) - 10))

    writer = ResultWriter(run_dir)
    writer.records = records
    writer.image_rows = image_rows
    csv_path = writer.write_csv()
    print("  wrote %s  (%d job row(s))" % (csv_path, len(records)))
    print("  wrote %s  (%d job-image row(s))"
          % (run_dir / "results_per_image.csv", len(image_rows)))

    print_summary_tables(records)

    figures: List[Path] = []
    if not args.no_figures:
        print("\nRebuilding problem instances for the figures ...")
        cache_root = Path(plan.cache_root).resolve()
        data = DataManager(config, cache_root / "data")
        store = ProblemStore()
        store.build_all(plan.problems, data, verbose=False)
        try:
            figures = make_figures(run_dir, plan, store, records, reconstructions,
                                   show=in_ipython())
            print("  %d figure(s) written to %s" % (len(figures), run_dir / "figures"))
        except Exception as exc:
            print("  figure generation failed (%s); results.csv is unaffected." % exc)

    # ---------------------------------------------------------------- shard-private files
    checks, check_shards = merge_shard_checks(run_dir)
    if checks:
        save_json(run_dir / "checks.json", checks)
        scopes = sorted({str(c.get("scope")) for c in checks})
        failed = [c for c in checks if c.get("passed") is False]
        print("  merged %d check(s) from shard(s) %s into checks.json"
              % (len(checks), ", ".join("%02d" % s for s in check_shards) or "(none)"))
        print("     scopes: %s" % ", ".join(scopes))
        if failed:
            print("     %d FAILING check(s) preserved by the merge:" % len(failed))
            for entry in failed[:10]:
                print("       %s / %s" % (entry.get("scope"), entry.get("name")))
    else:
        print("  no checks_shardNN.json found; checks.json left untouched "
              "(run the array with --check to produce them)")

    metadata = merge_shard_metadata(run_dir, checks, figures)
    if metadata is not None:
        save_json(run_dir / "run_metadata.json", metadata)
        print("  merged run_metadata.json from shard(s) %s (models: %s)"
              % (", ".join("%02d" % s["shard"] for s in metadata["shards"]),
                 ", ".join(sorted(metadata.get("model_provenance", {}))) or "(none)"))
    else:
        print("  no run_metadata_shardNN.json found; run_metadata.json left untouched")

    save_json(run_dir / "aggregate_metadata.json",
              {"merged": now_iso(), "found": len(records), "planned": len(plan.specs),
               "per_image_rows": len(image_rows),
               "missing_job_ids": [s.job_id for s in missing],
               "merged_checks": len(checks),
               "check_shards": check_shards,
               "metadata_shards": ([s["shard"] for s in metadata["shards"]]
                                   if metadata else []),
               "outputs": ["results.csv", "results_per_image.csv"]
                          + (["checks.json"] if checks else [])
                          + (["run_metadata.json"] if metadata else [])})
    return 0


def memory_baseline_only(plan, manager, run_dir: Path) -> int:
    """`--memory-baseline-only`: what each resident model alone costs, then exit.

    Run in its own process by scripts/run_timing.py, through the same steps as a timing
    process up to the model load (data, problems, frameworks), so the difference between
    a configuration's peak and this baseline is that configuration's own memory.
    """
    from src.timing import model_only_baseline
    for model in plan.scheduled_models():
        adapter = manager.acquire(model)
        payload = model_only_baseline(adapter, model)
        payload["model_load_seconds"] = round(float(manager.load_seconds.get(model, 0.0)), 2)
        path = run_dir / ("memory_baseline_%s.json" % model)
        save_json(path, payload)
        print("Model-only baseline %s: allocator %s GiB in use, NVML reserved %s GiB -> %s"
              % (model.upper(), payload.get("allocator_in_use_gib"),
                 payload.get("nvml_process_reserved_gib"), path))
        manager.release(model)
    return 0


# =====================================================================================
# Main
# =====================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    shard = parse_shard(args.shard)
    env = load_dotenv(".env")
    if env:
        print("Loaded .env keys: %s" % ", ".join(sorted(env)))

    where = detect_environment()
    offline = where["offline"] if args.offline is None else bool(args.offline)
    if args.prefetch:
        offline = False                     # prefetch exists precisely to use the network
    apply_offline_mode(offline)
    print("Environment      : %s%s | offline=%s"
          % (where["kind"], (" [%s]" % where["cluster"]) if where["cluster"] else "", offline))
    if where["job_id"]:
        print("SLURM            : job %s%s on %s | SLURM_TMPDIR=%s"
              % (where["job_id"],
                 (" task %s" % where["array_task"]) if where["array_task"] else "",
                 where["node"], where["tmpdir"]))
    if offline and not args.prefetch:
        print("Offline mode: no download will be attempted. Anything missing from the cache")
        print("  is a hard failure -- run `python run.py --prefetch` on a login node first.")
    # Progress bars are useless in a SLURM log and produce megabytes of carriage returns.
    if not sys.stdout.isatty() and where["kind"] != "colab":
        set_progress_enabled(False)

    config = apply_cli_overrides(load_config(args.config), args)
    warnings_ = validate_config(config)
    plan = resolve_run_plan(config, warnings_, run_id=args.run_id)

    rt = config["runtime"]
    set_global_seed(plan.seed)
    set_progress_enabled(bool(rt.get("progress_bars", True)))
    configure_lpips(bool((config.get("metrics") or {}).get("lpips", True)),
                    str((config.get("metrics") or {}).get("lpips_net", "alex")))

    accel = detect_accelerator(rt.get("accelerator", "auto"))
    if args.prefetch:
        # Prefetch only downloads files. Login nodes have no GPU, so demanding a usable
        # accelerator here would block the one step that must run on a login node.
        accel_notes = ["--prefetch: downloading only, so the accelerator is not checked."]
    else:
        accel_notes = check_accelerator_compatibility(plan, accel)
    print_run_plan(plan, accel, accel_notes)

    cache_root_early = Path(rt.get("cache_root", "cache")).resolve()
    if args.prefetch:
        prefetch_assets(plan, config, cache_root_early)
        return 0

    run_dir = Path(plan.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        if not args.run_id:
            raise ConfigError("--aggregate needs --run-id naming the directory to merge.")
        return aggregate_only(plan, config, run_dir, args)

    # config.yaml / resolved_config.yaml describe the WHOLE plan and are byte-identical for
    # every array task (they are written before the shard slice is taken), so the shards do
    # not need private copies -- but they must not be rewritten concurrently either: the
    # `created` timestamp differs per task, which would make the surviving file depend on
    # scheduling.  First writer wins; save_yaml itself is atomic.
    suffix = shard_suffix(shard)
    for name, payload in (("config.yaml", plan.raw_config),
                          ("resolved_config.yaml", plan.to_dict())):
        if shard is None or not (run_dir / name).exists():
            save_yaml(run_dir / name, payload)
    print("\nSaved %s and %s" % (run_dir / "config.yaml", run_dir / "resolved_config.yaml"))

    if args.dry_run:
        print("\n--dry-run: the plan above was validated and resolved. No model was loaded, "
              "no checkpoint downloaded and no job executed.")
        return 0

    # ---------------------------------------------------------------- shard selection
    all_specs = plan.specs
    if shard is not None:
        selected = select_shard(all_specs, shard[0], shard[1])
        plan = dataclasses.replace(plan, specs=tuple(selected))
        models = sorted({s.model for s in selected})
        print("\n%s\nSHARD %d of %d -- %d of %d atomic jobs, model(s): %s\n%s"
              % (RULE, shard[0], shard[1], len(selected), len(all_specs),
                 ", ".join(m.upper() for m in models) or "(none)", RULE))
        if not selected:
            print("This shard is empty (more shards than jobs). Nothing to do.")
            return 0
        print("Shard-private outputs (no array task overwrites another's):")
        for canonical, _pattern in SHARD_FILE_PATTERNS:
            stem, dot, ext = canonical.rpartition(".")
            print("    %s%s%s%s" % (stem, suffix, dot, ext))
        print("Merge the shards afterwards with:")
        print("    python run.py --config %s --run-id %s --aggregate"
              % (args.config, plan.run_id))
        if not args.no_figures:
            # figures/ has fixed filenames, so concurrent tasks would overwrite each
            # other's PNGs with partial views of the run. --aggregate builds them once,
            # from the complete merged record set.
            print("Figures are DISABLED for array tasks; --aggregate builds them once.")
            args.no_figures = True

    # ---------------------------------------------------------------- data and problems
    cache_root = Path(plan.cache_root).resolve()
    data = DataManager(config, cache_root / "data")
    source = data.pool(plan.resources.source_pool_size)
    print("\nShared image pool (largest enabled experiment needs %d):"
          % plan.resources.source_pool_size)
    for i, image_id in enumerate(source.image_ids):
        print("   [%d] %-28s class=%d" % (i, image_id, source.labels[i]))

    print("\nBuilding inverse-problem instances (once each, shared by every model/method):")
    store = ProblemStore()
    problems = store.build_all(plan.problems, data)
    degraded_metrics = {key: degraded_baseline(p) for key, p in problems.items()}
    print("\nDegraded baseline (observation vs ground truth):")
    print("   %-17s %8s %8s %8s %10s" % ("problem", "PSNR", "SSIM", "LPIPS", "guidePSNR"))
    for key, p in problems.items():
        d = degraded_metrics[key]
        print("   %-17s %8.2f %8.4f %8s %10.2f"
              % (p.name, d["psnr"], d["ssim"],
                 ("%.4f" % d["lpips"]) if d["lpips"] is not None else "n/a",
                 d["guide_psnr"]))

    report = checks_module.CheckReport()
    if args.check or args.checks_only:
        print("\n" + RULE)
        # src/checks.py imports JAX even on a Torch-only plan, so the allocator policy
        # must be set before the checks run, not in init_frameworks after them.
        configure_jax_environment(accel, cache_root)
        checks_module.run_structural_checks(plan, problems, data, report)
        failures = report.failures()
        if failures:
            print("\n%d structural check(s) FAILED. Inspect them before trusting any result."
                  % len(failures))
            if not rt.get("continue_on_experiment_error", True):
                save_json(run_dir / ("checks%s.json" % suffix), report.to_dict())
                raise RuntimeError("Structural checks failed: %s"
                                   % [f.name for f in failures])
        save_json(run_dir / ("checks%s.json" % suffix), report.to_dict())

    if args.checks_only:
        print("\n--checks-only: stopping before any model is loaded.")
        return 0 if not report.failures() else 1

    # ---------------------------------------------------------------- environment
    print("\n" + RULE)
    print("ENVIRONMENT")
    print(RULE)
    context = ensure_repositories(plan, cache_root)
    context.update(init_frameworks(plan, config, accel, cache_root))
    context["ckpt_cache"] = cache_root / "checkpoints"
    context["ckpt_cache"].mkdir(parents=True, exist_ok=True)
    load_adapters(plan.resources.models)

    release = rt.get("release_model_after_use", "auto")
    if release == "auto":
        vram = accel.get("gpu_memory_mb") or 0
        release = bool(len(plan.resources.models) > 1
                       and (accel["kind"] != "gpu" or vram < 24_000
                            or len(plan.resources.frameworks) > 1))
    manager = ModelManager(config, plan, bool(release), context)

    if args.memory_baseline_only:
        return memory_baseline_only(plan, manager, run_dir)

    writer = ResultWriter(run_dir, shard=shard)
    reconstructions: Dict[str, np.ndarray] = {}
    resume = bool(rt.get("resume", True))
    save_images = bool(rt.get("save_individual_images", True))
    keep_going = bool(rt.get("continue_on_experiment_error", True))
    do_warmup = not args.no_warmup
    profile_memory = bool(rt.get("gpu_memory_profiling", True))
    nvml_interval = float(rt.get("nvml_sample_interval", 0.02))
    # OPT-IN (rebuttal Task 1, src/timing.py); off unless the config sets it.
    timing_detail = bool(rt.get("timing_detail", False))

    # ---------------------------------------------------------------- model-major execution
    started = time.perf_counter()
    scheduled = plan.scheduled_models()
    print("\n" + RULE)
    print("EXECUTION -- model-major over %d atomic job(s), %d model(s)"
          % (len(plan.specs), len(scheduled)))
    print("  physical order : %s" % " -> ".join(m.upper() for m in scheduled))
    print("  logical order  : %s" % ", ".join(plan.experiment_names()))
    print("  release models : %s | resume finished jobs: %s" % (bool(release), resume))
    print("  gpu memory     : %s" % ("measured per job" if profile_memory else "disabled"))
    print(RULE)

    for position, model in enumerate(scheduled, 1):
        specs = sorted(plan.specs_for_model(model),
                       key=lambda s: (s.experiment, s.problem, s.method, s.t0,
                                      s.K or 0, s.reconstruction_steps, s.lam or 0.0))
        print("\n%s\n[%d/%d] MODEL %s -- %d job(s)\n%s"
              % (RULE, position, len(scheduled), model.upper(), len(specs), RULE))
        writer.event("model_start", model=model, jobs=len(specs))

        try:
            adapter = manager.acquire(model)
        except Exception as exc:
            message = "%s: %s" % (type(exc).__name__, exc)
            print("  MODEL LOAD FAILED: %s" % message)
            traceback.print_exc(limit=6)
            writer.event("model_load_failed", model=model, error=message)
            for spec in specs:                       # nothing is silently dropped
                writer.add(build_record(spec, plan, store.get(spec.problem_key), None, None,
                                        degraded_metrics[spec.problem_key], "skipped",
                                        "model load failed: %s" % message, 0.0, 0.0))
            if not keep_going:
                raise
            continue

        load_seconds = manager.load_seconds.get(model, 0.0)
        after_load = None
        if timing_detail:
            from src.timing import memory_snapshot
            after_load = memory_snapshot(adapter.spec.framework)

        if args.check:
            probe = specs[0]
            methods_here = sorted({s.method for s in specs})
            checks_module.run_model_checks(adapter, store.get(probe.problem_key), probe,
                                           manager, report, methods=methods_here,
                                           specs=specs)
            # Fairness, measured rather than assumed: jobs that differ only by method or by
            # hyperparameters must start from a bit-identical z_t0.
            groups: Dict[Tuple, List] = {}
            for s in specs:
                groups.setdefault((s.problem_key, s.t0), []).append(s)
            for key, group in list(groups.items())[:3]:
                report.run(model, "shared_initial_state",
                           lambda g=group, k=key: checks_module.check_shared_initial_state(
                               adapter, store.get(k[0]), g, manager))
            save_json(run_dir / ("checks%s.json" % suffix), report.to_dict())

        warmed: Dict[Tuple, float] = {}
        for index, spec in enumerate(specs, 1):
            problem = store.get(spec.problem_key)
            print("\n  [%d/%d] %s" % (index, len(specs), spec.label))
            print("        job=%s guide=%s init=%s images=%d batch=%d"
                  % (spec.job_id, spec.initialization_guide_mode, spec.initialization_kind,
                     spec.num_images, spec.batch_size))
            writer.event("job_start", job_id=spec.job_id, experiment=spec.experiment,
                         model=model, method=spec.method, t0=spec.t0)

            if resume:
                finished = load_finished_job(run_dir, spec)
                if finished is not None:
                    record = finished["record"]
                    record["status"] = "ok"
                    reconstructions[spec.job_id] = (
                        np.asarray(finished["pixels"], np.float32) / 127.5) - 1.0
                    # The per-image rows are rebuilt from the stored metadata rather than
                    # skipped: a resumed job used to contribute to results.csv but not to
                    # results_per_image.csv, so resuming silently shrank the per-image file.
                    writer.add({k: record.get(k) for k in RESULT_COLUMNS},
                               image_rows_from_metadata(record))
                    print("        -> reused a finished job from %s" % finished["output_dir"])
                    continue

            key = warmup_key(spec, problem)
            warm_seconds = warmed.get(key, 0.0)
            before_warm_up = None
            if timing_detail:
                # Task 9: what the allocator had seen before THIS configuration's warm-up, so
                # an exact JAX peak can prove nothing else in the process set it.
                from src.timing import memory_snapshot
                before_warm_up = memory_snapshot(adapter.spec.framework)
            try:
                if do_warmup and key not in warmed:
                    warm_seconds = warm_up(adapter, spec, problem, manager)
                    warmed[key] = warm_seconds

                # Instrumentation is created AFTER warm-up, so compilation buffers and
                # first-call allocations are part of the baseline, not of the measurement.
                profiler = GpuMemoryProfiler(adapter, enabled=profile_memory,
                                             nvml_interval=nvml_interval)
                timing = None
                if timing_detail:
                    from src.timing import TimingDetail
                    timing = TimingDetail(adapter.spec.framework, after_load=after_load,
                                          before_warm_up=before_warm_up,
                                          param_loading=getattr(adapter, "param_loading",
                                                                None))
                    from src.timing import attach_jax_sampler
                    timing.jax_sampler = attach_jax_sampler(profiler)
                run = run_job(adapter, spec, problem, manager, profiler=profiler,
                              timing=timing)
                memory = profiler.report()
                metrics = evaluate_reconstruction(run["pixels"], problem, spec)
                record = build_record(spec, plan, problem, run, metrics,
                                      degraded_metrics[spec.problem_key], run["status"],
                                      None if run["status"] == "ok" else run["status"],
                                      warm_seconds, load_seconds, memory=memory)
                if timing is not None:
                    # metadata.json and results.jsonl only; results.csv keeps its columns.
                    record["timing_detail"] = timing.report(memory)
                out = persist_job(run_dir, spec, run, record, metrics, problem, save_images)
                record["output_dir"] = str(out)
                reconstructions[spec.job_id] = run["pixels"]
                writer.add(record, per_image_rows(record, metrics, problem.image_ids))
                print("        -> PSNR %.2f | SSIM %.4f | LPIPS %s | meas-RMSE %.4f"
                      % (record["psnr"], record["ssim"],
                         ("%.4f" % record["lpips"]) if record["lpips"] is not None else "n/a",
                         record["measurement_rmse"]))
                print("        -> %.1fs total, %.2fs/image | %d model eval(s) (expected %d) | "
                      "%d backprop(s)%s"
                      % (record["runtime"], record["runtime_per_image"],
                         record["model_evaluations"], record["expected_model_evaluations"],
                         record["backprops_through_model"],
                         " | padded %d" % record["padded_items"]
                         if record["padded_items"] else ""))
                if record["gpu_peak_gib"] is not None:
                    print("        -> GPU %.3f GiB peak (%.3f GiB above the %.3f GiB "
                          "baseline), batch=%d"
                          % (record["gpu_peak_gib"], record["gpu_incremental_peak_gib"],
                             record["gpu_baseline_gib"], record["batch_size"]))
                writer.event("job_done", job_id=spec.job_id, psnr=record["psnr"],
                             lpips=record["lpips"], seconds=record["runtime"],
                             gpu_peak_gib=record["gpu_peak_gib"])
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                print("        FAILED: %s" % message)
                traceback.print_exc(limit=6)
                writer.add(build_record(spec, plan, problem, None, None,
                                        degraded_metrics[spec.problem_key], "failed",
                                        message, warm_seconds, load_seconds))
                writer.event("job_failed", job_id=spec.job_id, error=message)
                if not keep_going:
                    manager.release(model)
                    raise
            finally:
                free_memory()

        manager.clear_encode_cache(model=model)
        if release:
            manager.release(model)
        manager.memory_report("  memory after %s:" % model.upper())
        writer.event("model_done", model=model)

    manager.release_all()
    elapsed = time.perf_counter() - started

    # ---------------------------------------------------------------- reporting
    csv_path = writer.write_csv()
    ok = len([r for r in writer.records if r["status"] == "ok"])
    print("\n%s\nEXECUTION COMPLETE in %.1f s (%.1f min)" % (RULE, elapsed, elapsed / 60))
    print("  ok=%d  failed=%d  skipped=%d  (of %d atomic jobs)"
          % (ok, len([r for r in writer.records if r["status"] == "failed"]),
             len([r for r in writer.records if r["status"] == "skipped"]), len(plan.specs)))
    # Warm-up and loading are excluded from every reported runtime, so show them here --
    # otherwise a compilation problem hides inside the wall clock and looks like slow compute.
    print("  wall-clock breakdown (all EXCLUDED from the runtime column):")
    for model in scheduled:
        rows = [r for r in writer.records if r["model"] == model and r["status"] == "ok"]
        warm = sum(r.get("warmup_seconds") or 0.0 for r in rows)
        work = sum(r.get("runtime") or 0.0 for r in rows)
        print("    %-5s load %6.1f s | warm-up/compile %7.1f s | measured work %7.1f s%s"
              % (model.upper(), manager.load_seconds.get(model, 0.0), warm, work,
                 "   <-- compilation dominates; see README" if warm > 3 * max(work, 1e-9)
                 else ""))
    print(RULE)

    print_summary_tables(writer.records)

    figures: List[Path] = []
    if not args.no_figures:
        print("\nGenerating figures ...")
        try:
            figures = make_figures(run_dir, plan, store, writer.records, reconstructions,
                                   show=in_ipython())
        except Exception as exc:
            print("Figure generation failed (%s); results are unaffected." % exc)

    save_json(run_dir / ("run_metadata%s.json" % suffix), {
        "run_id": plan.run_id, "created": plan.created, "finished": now_iso(),
        "seed": plan.seed, "replicate": plan.replicate,
        "accelerator": accel, "elapsed_seconds": round(elapsed, 2),
        "model_provenance": manager.provenance,
        "model_load_seconds": manager.load_seconds,
        "repositories": context.get("repo_heads", {}),
        "problems": {k: p.to_metadata() for k, p in problems.items()},
        "degraded_metrics": degraded_metrics,
        "checks": report.to_dict(),
        "timing": timing_summary(),
        "figures": [str(p) for p in figures],
        "gpu_memory_policy_process": (
            "gpu_process_peak_gib is the peak GPU memory of THIS PROCESS during the "
            "measured reconstruction, sampled through NVML with a FRESH sampler per atomic "
            "job. It is the only memory number that is comparable between the PyTorch and "
            "JAX models, and it is the one a batch-size-4 memory table should use: it is a "
            "JOB peak at the configured batch size and is never divided by the batch, so it "
            "is not single-image latency or per-image memory. When NVML process accounting "
            "is unavailable the field reports 'unavailable'; JAX's peak_bytes_in_use is a "
            "LIFETIME high-water mark with no reset API and is never substituted for it."),
        "rhso_diagnostic_cost_policy": (
            "diagnostic_model_evaluations, diagnostic_network_forwards, diagnostic_jvps, "
            "diagnostic_vjps and diagnostic_seconds cover the theory-validation "
            "measurements only. They are EXCLUDED from model_evaluations, network_forwards, "
            "backprops_through_model and runtime, so those columns keep their meaning as "
            "the algorithm's own cost."),
        "gpu_memory_policy": (
            "gpu_peak_gib is the peak GPU memory during the measured reconstruction of one "
            "atomic job at its configured batch size; gpu_baseline_gib is the steady state "
            "after model load and warm-up, immediately before it; gpu_incremental_peak_gib "
            "is the difference. Model loading, dataset preparation, metrics, visualisation "
            "and warm-up are outside the measured region. The numbers are NOT per image and "
            "are never divided by the batch size. gpu_memory_source records how each number "
            "was obtained -- a Torch allocator high-water mark and a sampled NVML process "
            "peak are not the same kind of measurement."),
        "class_conditioning": (
            "Every reconstruction is conditioned on the TRUE ImageNet class of the source "
            "image, identically for SDEdit, both MPC methods, PnP-Flow, D-Flow and RHSO. The "
            "benchmark therefore measures reconstruction given a known class, not blind "
            "restoration."),
        "time_schedule_policy": (
            "Every method places its times on s_k = s0 + (1 - s0)(k/N)^beta. beta = 1 is the "
            "uniform grid this repository used before the exponent existed, reproduced "
            "bitwise. The `delta` column is NOMINAL uniform spacing and is null whenever "
            "beta != 1, because a non-uniform trajectory has no single step size; the "
            "executed step sizes are bounded by delta_min and delta_max, and MPC-delta_t's "
            "inverse-delta lambda scaling uses the LOCAL dt_k of each step."),
        "warnings": list(plan.warnings) + list(accel_notes),
    })
    print("\nResults : %s" % csv_path)
    print("Run dir : %s" % run_dir)
    if figures:
        print("Figures : %d written to %s" % (len(figures), run_dir / "figures"))
    return 0


def _exit(code: int, force: Optional[bool] = None) -> int:
    """Terminate without running interpreter finalization -- but NEVER inside a kernel.

    This process ends up holding ~100 independently-built native extensions (pyarrow,
    torch, jax, aiohttp via `datasets`, PIL, scipy).  At shutdown they tear down in an
    order none of them agreed on, and a background thread left by aiohttp can touch the GIL
    after torch has released its thread state:

        Fatal Python error: PyGILState_Release ... Python runtime state: finalizing

    It happens strictly AFTER main() returns, so no result is affected -- but it turns a
    successful run into a non-zero exit code, which makes `set -e` job scripts report a
    failure that did not occur.  Every artefact this program produces is written and closed
    as it goes (results are persisted per job, not buffered to the end), so skipping
    finalization costs nothing.  The streams are flushed explicitly first.

    THE EXCEPTION: `%run run.py` in Colab or Jupyter executes in the NOTEBOOK'S OWN
    process, where os._exit() would kill the kernel the moment the run finished -- which
    looks exactly like a crash even though the run succeeded and every result was already
    on disk.  Under IPython this therefore returns the status code instead of terminating,
    and the messy-teardown problem does not arise there anyway: the interpreter is not
    finalizing, the kernel simply carries on.

    `force` overrides the detection in both directions, for tests.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if force is None:
        force = not in_ipython()
    if not force:
        return int(code)
    os._exit(int(code))


if __name__ == "__main__":
    _status = main()
    if in_ipython():
        # A notebook: return the status rather than taking the kernel down with us.
        if _status:
            print("\nrun.py finished with status %d (see the failure column in results.csv)."
                  % _status)
    else:
        _exit(_status)
