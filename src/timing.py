"""Opt-in timing and memory detail for the rebuttal timing run (rebuttal Task 1).

Nothing here runs unless a configuration sets ``runtime.timing_detail: true`` (the
``configs/experiments_rebuttal_timing_<prior>.yaml`` files do) or ``run.py`` is started with
``--memory-baseline-only``.  Every other run behaves exactly as before: the results.csv
columns, the timed region and the memory columns are untouched.  What it adds lands in each
job's ``metadata.json`` under ``timing_detail``.

Per-image wall clock
    Each method already ends its timed region with ``adapter.block(x)``: a device-wide
    ``torch.cuda.synchronize()`` on PyTorch, but ``jax.block_until_ready(x)`` on the OUTPUT
    only on JAX.  ``ChunkTimer`` adds an OUTER timer around each batch chunk (one image at
    batch size 1) that waits for the whole device before it starts and after the method
    returns, so asynchronous dispatch can never make a method look faster.  The method's
    own ``stats.seconds`` is recorded next to it for comparison.

Memory
    The timing run gives every configuration its own process (scripts/run_timing.py), so a
    lifetime statistic of that process belongs to that configuration alone.  Recorded at
    three points -- after the model load, immediately before the timed region and after
    it -- are this process's NVML usage (the reserved pool plus CUDA context, NOT live
    tensors), PyTorch's allocator counters, and JAX's ``memory_stats()`` in full
    (``bytes_in_use``, ``peak_bytes_in_use``, ``bytes_limit``, ``pool_bytes``, ...).

    The primary memory figure is the ALLOCATOR peak:
      * PyTorch: ``torch.cuda.max_memory_allocated`` reset after the warm-up, so it is the
        peak of the timed region itself (src/memory.py already reports it).
      * JAX: ``peak_bytes_in_use``, a lifetime high-water mark with no reset API.  In a
        fresh process it covers model load + warm-up + timed run of ONE configuration.
        ``allocator_peak_scope`` says which of the three set it: if the counter rose during
        the timed region the value is that region's peak EXACTLY.  Otherwise it is only an
        UPPER bound, and the LOWER bound is the largest ``bytes_in_use`` sampled during the
        timed region -- read in the same thread and loop as the NVML samples
        (``make_jax_sampler``), plus the snapshots on either side of the region.  When the
        upper bound equals the model-only baseline's load peak it was set by loading the
        model, and scripts/summarize_timing.py flags it as uninformative.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

GIB = float(2 ** 30)

# Allocator settings that decide what the NVML number means.  Recorded verbatim (None when
# unset) together with the default each one falls back to.
ALLOCATOR_ENV = (
    "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "XLA_FLAGS",
    "TF_GPU_ALLOCATOR", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
)
ALLOCATOR_DEFAULTS = {
    "XLA_PYTHON_CLIENT_ALLOCATOR": "default (XLA's BFC pool allocator)",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75 of total device memory (XLA default)",
}

# Python distributions whose versions describe the software stack.
DISTRIBUTIONS = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
                 "jax-cuda13-plugin", "jax-cuda13-pjrt", "torch", "torchvision",
                 "nvidia-ml-py", "pynvml", "numpy")


# =====================================================================================
# Device synchronisation
# =====================================================================================
def device_sync(framework: Optional[str]) -> str:
    """Wait for ALL queued device work of this framework; return how it was done.

    PyTorch: ``torch.cuda.synchronize()`` (device-wide).  JAX: every local device's
    ``synchronize_all_activity()`` where the runtime provides it; otherwise a tiny compiled
    barrier op enqueued behind everything already dispatched and waited for (XLA runs a
    device's executables in dispatch order on its compute stream, so it completes last);
    and only if that fails, every live array's ``block_until_ready()``.  The barrier costs
    microseconds, where looping over a model's parameter arrays would cost milliseconds
    per image -- a visible bias for a two-evaluation method.
    """
    if framework == "torch":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                return "torch.cuda.synchronize"
            return "none (torch on CPU: synchronous)"
        except Exception as exc:                                         # pragma: no cover
            return "failed: %s" % exc
    if framework == "jax":
        try:
            import jax
        except Exception as exc:                                         # pragma: no cover
            return "failed: %s" % exc
        devices = jax.local_devices()
        if devices and all(hasattr(d, "synchronize_all_activity") for d in devices):
            try:
                for device in devices:
                    device.synchronize_all_activity()
                return "jax.Device.synchronize_all_activity"
            except Exception:
                pass
        try:
            for device in devices:
                _jax_barrier(device).block_until_ready()
            return "jax compiled barrier op per device"
        except Exception:
            pass
        for array in jax.live_arrays():
            try:
                array.block_until_ready()
            except Exception:
                pass
        return "jax.live_arrays().block_until_ready"
    return "none (no framework)"


_BARRIER: Dict[Any, Any] = {}


def _jax_barrier(device: Any) -> Any:
    """Dispatch a one-element compiled op on `device` (compiled once, then cached)."""
    import jax
    import jax.numpy as jnp
    if "fn" not in _BARRIER:
        _BARRIER["fn"] = jax.jit(lambda v: v + 1.0)
    if device not in _BARRIER:
        _BARRIER[device] = jax.device_put(jnp.zeros((), jnp.float32), device)
    return _BARRIER["fn"](_BARRIER[device])


def _block_output(x: Any) -> None:
    """Wait for the method's returned state itself (cheap; the device sync follows)."""
    try:
        import jax
        jax.block_until_ready(x)
    except Exception:
        pass


class ChunkTimer:
    """Outer, device-synchronised wall clock around each reconstructed batch chunk."""

    def __init__(self, framework: Optional[str]):
        self.framework = framework
        self.outer_seconds: List[float] = []
        self.method_seconds: List[float] = []
        self.chunk_sizes: List[int] = []
        self.sync_method: Optional[str] = None
        self._started: Optional[float] = None

    def start(self) -> None:
        self.sync_method = device_sync(self.framework)
        self._started = time.perf_counter()

    def stop(self, x_final: Any, method_seconds: float, size: int) -> None:
        if self._started is None:
            raise RuntimeError("ChunkTimer.stop() without start()")
        if self.framework == "jax":
            _block_output(x_final)
        device_sync(self.framework)
        self.outer_seconds.append(time.perf_counter() - self._started)
        self.method_seconds.append(float(method_seconds))
        self.chunk_sizes.append(int(size))
        self._started = None

    def to_dict(self) -> Dict[str, Any]:
        return {"chunk_seconds_outer_synced": [round(v, 6) for v in self.outer_seconds],
                "chunk_seconds_method": [round(v, 6) for v in self.method_seconds],
                "chunk_sizes": list(self.chunk_sizes),
                "sync_method": self.sync_method}


# =====================================================================================
# Environment, versions and hardware
# =====================================================================================
def allocator_settings() -> Dict[str, Any]:
    values = {name: os.environ.get(name) for name in ALLOCATOR_ENV}
    effective = {}
    for name, default in ALLOCATOR_DEFAULTS.items():
        effective[name] = values.get(name) if values.get(name) is not None else (
            "unset -> %s" % default)
    fraction = values.get("XLA_CLIENT_MEM_FRACTION") or values.get(
        "XLA_PYTHON_CLIENT_MEM_FRACTION")
    effective["memory_fraction"] = (fraction if fraction is not None
                                    else "unset -> 0.75 (XLA default)")
    return {"environment": values, "effective": effective}


def software_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        from importlib import metadata
        for dist in DISTRIBUTIONS:
            try:
                out[dist] = metadata.version(dist)
            except Exception:
                out[dist] = None
    except Exception:                                                    # pragma: no cover
        pass
    if "torch" in sys.modules:
        try:
            import torch
            out["torch_cuda"] = torch.version.cuda
            out["torch_cudnn"] = (torch.backends.cudnn.version()
                                  if torch.backends.cudnn.is_available() else None)
        except Exception:
            pass
    if "jax" in sys.modules:
        try:
            import jax
            out["jax_backend"] = jax.default_backend()
            device = jax.local_devices()[0]
            out["jax_device_kind"] = getattr(device, "device_kind", None)
            out["jax_platform_version"] = getattr(device.client, "platform_version", None)
        except Exception:
            pass
    return out


def _decode(value: Any) -> Any:
    return value.decode() if isinstance(value, bytes) else value


def gpu_hardware() -> Dict[str, Any]:
    """GPU name, total memory, driver and CUDA driver version: NVML first, then nvidia-smi."""
    from src.memory import _nvml, _nvml_device_handle, _visible_device_token
    out: Dict[str, Any] = {"source": None, "gpu_name": None, "gpu_memory_total_mib": None,
                           "driver_version": None, "cuda_driver_version": None,
                           "visible_device": _visible_device_token()}
    nvml = _nvml()
    handle = _nvml_device_handle(nvml) if nvml is not None else None
    if handle is not None:
        try:
            out["gpu_name"] = _decode(nvml.nvmlDeviceGetName(handle))
            out["gpu_memory_total_mib"] = int(nvml.nvmlDeviceGetMemoryInfo(handle).total
                                              // 2 ** 20)
            out["driver_version"] = _decode(nvml.nvmlSystemGetDriverVersion())
            cuda = int(nvml.nvmlSystemGetCudaDriverVersion())
            out["cuda_driver_version"] = "%d.%d" % (cuda // 1000, (cuda % 1000) // 10)
            out["source"] = "nvml"
            return out
        except Exception as exc:
            out["nvml_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits", "-i", out["visible_device"]],
            capture_output=True, text=True, timeout=20)
        line = query.stdout.strip().splitlines()[0]
        name, total, driver = [part.strip() for part in line.split(",")[:3]]
        out.update(gpu_name=name, gpu_memory_total_mib=int(float(total)),
                   driver_version=driver, source="nvidia-smi")
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        marker = "CUDA Version:"
        if marker in header.stdout:
            out["cuda_driver_version"] = header.stdout.split(marker)[1].split()[0]
    except Exception as exc:
        out["nvidia_smi_error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


_PROCESS_ENVIRONMENT: Dict[str, Any] = {}


def process_environment() -> Dict[str, Any]:
    """Hardware, versions and allocator settings, collected once per process."""
    if not _PROCESS_ENVIRONMENT:
        _PROCESS_ENVIRONMENT.update(hardware=gpu_hardware(), versions=software_versions(),
                                    allocator=allocator_settings(), pid=os.getpid())
    return dict(_PROCESS_ENVIRONMENT)


# =====================================================================================
# Memory snapshots
# =====================================================================================
def jax_memory_stats() -> Optional[Dict[str, int]]:
    """``memory_stats()`` of the first local device, integers only; None if unavailable.

    The pooling (BFC) allocator reports bytes_in_use, peak_bytes_in_use, bytes_limit and,
    in recent versions, pool_bytes / peak_pool_bytes.  The platform allocator reports None.
    """
    if "jax" not in sys.modules:
        return None
    try:
        import jax
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {str(k): int(v) for k, v in stats.items() if isinstance(v, (int, float))}


def torch_memory_stats() -> Optional[Dict[str, int]]:
    if "torch" not in sys.modules:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return {"memory_allocated": int(torch.cuda.memory_allocated()),
                "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                "memory_reserved": int(torch.cuda.memory_reserved()),
                "max_memory_reserved": int(torch.cuda.max_memory_reserved())}
    except Exception:
        return None


def nvml_process_gib() -> Optional[float]:
    """This process's GPU memory as NVML sees it (reserved pools + CUDA context)."""
    try:
        from src.memory import _NvmlSampler
        return _NvmlSampler().sample_once()
    except Exception:
        return None


def _jax_bytes_in_use() -> Optional[int]:
    try:
        import jax
        stats = jax.local_devices()[0].memory_stats() or {}
        value = stats.get("bytes_in_use")
        return None if value is None else int(value)
    except Exception:
        return None


def make_jax_sampler(interval: float):
    """An NVML process sampler that ALSO reads JAX's bytes_in_use in the same loop.

    One thread samples both, so the timing run adds no sampling thread of its own.  The
    sampled bytes_in_use maximum is a LOWER bound on the timed region's allocator peak
    (a sampler can miss a short spike); `peak_bytes_in_use` is the matching upper bound.
    Defined lazily so importing this module never imports src.memory's NVML binding.
    """
    from src.memory import _NvmlSampler

    class JaxStatsNvmlSampler(_NvmlSampler):
        def __init__(self, interval_: float):
            super().__init__(interval_)
            self.jax_peak_bytes: Optional[int] = None
            self.jax_samples = 0

        def _record_jax(self) -> None:
            value = _jax_bytes_in_use()
            if value is not None:
                self.jax_samples += 1
                self.jax_peak_bytes = (value if self.jax_peak_bytes is None
                                       else max(self.jax_peak_bytes, value))

        def _loop(self) -> None:                       # same loop as the NVML samples
            while not self._stop.is_set():
                used = self._used_bytes()
                if used is not None:
                    self.peak_bytes = used if self.peak_bytes is None else max(
                        self.peak_bytes, used)
                self._record_jax()
                self._stop.wait(self.interval)

        def start(self) -> None:
            self._record_jax()
            super().start()

        def stop(self) -> Optional[float]:
            result = super().stop()
            self._record_jax()
            return result

    return JaxStatsNvmlSampler(interval)


def attach_jax_sampler(profiler: Any) -> Optional[Any]:
    """Swap a JAX job's NVML process sampler for one that also samples bytes_in_use.

    Only on GpuMemoryProfiler's JAX NVML path, where the one sampler serves both memory
    metrics; it is built per job, so nothing carries over between jobs.  Returns the
    sampler, or None when the profiler is on any other path (it is then left untouched).
    """
    if getattr(profiler, "framework", None) != "jax" or getattr(profiler, "_mode", None) != "nvml":
        return None
    current = getattr(profiler, "_sampler", None)
    if current is None or getattr(profiler, "_process_sampler", None) is not current:
        return None
    sampler = make_jax_sampler(getattr(profiler, "nvml_interval", 0.02))
    if not sampler.available:
        return None
    profiler._sampler = sampler
    profiler._process_sampler = sampler
    return sampler


def memory_snapshot(framework: Optional[str]) -> Dict[str, Any]:
    device_sync(framework)
    return {"time": time.time(), "nvml_process_gib": nvml_process_gib(),
            "jax": jax_memory_stats(), "torch": torch_memory_stats()}


def _gib(value: Optional[int]) -> Optional[float]:
    return None if value is None else round(float(value) / GIB, 4)


def _jax_field(snapshot: Optional[Dict[str, Any]], key: str) -> Optional[int]:
    stats = (snapshot or {}).get("jax") or {}
    return stats.get(key)


class TimingDetail:
    """Everything the timing run records for ONE atomic job."""

    def __init__(self, framework: Optional[str],
                 after_load: Optional[Dict[str, Any]] = None):
        self.framework = framework
        self.timer = ChunkTimer(framework)
        # Set by run.py from attach_jax_sampler(profiler) on the JAX NVML path.
        self.jax_sampler: Optional[Any] = None
        self.snapshots: Dict[str, Optional[Dict[str, Any]]] = {
            "after_model_load": after_load, "before_timed_region": None,
            "after_timed_region": None}

    def snapshot(self, label: str) -> None:
        self.snapshots[label] = memory_snapshot(self.framework)

    def allocator_peak(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """The PRIMARY memory figure, with the scope it is valid for."""
        if self.framework == "torch":
            exact = memory.get("gpu_memory_source") == "torch.cuda.max_memory_allocated"
            value = memory.get("gpu_peak_gib") if exact else None
            return {"allocator_peak_gib": value,
                    "allocator_baseline_gib": memory.get("gpu_baseline_gib") if exact else None,
                    "allocator_peak_source": ("torch.cuda.max_memory_allocated" if exact
                                              else "unavailable"),
                    "allocator_peak_scope": "timed_region" if exact else None,
                    "allocator_peak_exact": bool(exact),
                    "allocator_peak_lower_gib": value, "allocator_peak_upper_gib": value,
                    "allocator_peak_upper_bytes": None, "jax_bytes_in_use_samples": 0}
        if self.framework == "jax":
            post = _jax_field(self.snapshots["after_timed_region"], "peak_bytes_in_use")
            pre = _jax_field(self.snapshots["before_timed_region"], "peak_bytes_in_use")
            load = _jax_field(self.snapshots["after_model_load"], "peak_bytes_in_use")
            if post is None:
                return {"allocator_peak_gib": None, "allocator_baseline_gib": None,
                        "allocator_peak_source": "unavailable (jax memory_stats() is None)",
                        "allocator_peak_scope": None, "allocator_peak_exact": False,
                        "allocator_peak_lower_gib": None, "allocator_peak_upper_gib": None,
                        "allocator_peak_upper_bytes": None, "jax_bytes_in_use_samples": 0}
            # LOWER bound: the largest bytes_in_use seen during the timed region -- sampled
            # in the NVML loop, plus the two snapshots that bracket it.  A sampler can miss
            # a short spike, so the true peak can only be higher.
            seen = [v for v in (
                getattr(self.jax_sampler, "jax_peak_bytes", None),
                _jax_field(self.snapshots["before_timed_region"], "bytes_in_use"),
                _jax_field(self.snapshots["after_timed_region"], "bytes_in_use"))
                if v is not None]
            lower = max(seen) if seen else None
            # UPPER bound: the lifetime peak of this one-configuration process.  If it ROSE
            # during the timed region, the new maximum was reached there: exact.
            if pre is None or post > pre:
                scope, exact = "timed_region", pre is not None
            elif load is None or pre > load:
                scope, exact = "warm_up_of_this_configuration (timed region <= this)", False
            else:
                scope, exact = "model_load (timed region <= this)", False
            if exact:
                lower = post
            return {"allocator_peak_gib": _gib(post),
                    "allocator_baseline_gib": _gib(_jax_field(
                        self.snapshots["before_timed_region"], "bytes_in_use")),
                    "allocator_peak_source": "jax memory_stats(): lower = max sampled "
                                             "bytes_in_use in the timed region, upper = "
                                             "peak_bytes_in_use (lifetime of this "
                                             "one-configuration process)",
                    "allocator_peak_scope": scope, "allocator_peak_exact": exact,
                    "allocator_peak_lower_gib": _gib(lower),
                    "allocator_peak_upper_gib": _gib(post),
                    "allocator_peak_upper_bytes": int(post),
                    "jax_bytes_in_use_samples": int(getattr(self.jax_sampler,
                                                            "jax_samples", 0) or 0)}
        return {"allocator_peak_gib": None, "allocator_baseline_gib": None,
                "allocator_peak_source": "unavailable", "allocator_peak_scope": None}

    def report(self, memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        memory = memory or {}
        out: Dict[str, Any] = {"timing_detail_version": 1}
        out.update(self.timer.to_dict())
        out.update(self.allocator_peak(memory))
        out["nvml_process_reserved_peak_gib"] = memory.get("gpu_process_peak_gib")
        out["nvml_process_reserved_baseline_gib"] = memory.get("gpu_process_baseline_gib")
        out["nvml_meaning"] = ("this process's reserved GPU memory as NVML reports it: the "
                               "framework pools plus CUDA context, not live tensors")
        for label in ("before_timed_region", "after_timed_region"):
            snap = self.snapshots.get(label)
            out["jax_pool_bytes_%s" % label] = _jax_field(snap, "pool_bytes")
            out["jax_bytes_limit_%s" % label] = _jax_field(snap, "bytes_limit")
        out["memory_snapshots"] = self.snapshots
        out["environment"] = process_environment()
        return out


def model_only_baseline(adapter: Any, model: str) -> Dict[str, Any]:
    """What a resident model alone costs, measured in a process that ran nothing else."""
    framework = getattr(getattr(adapter, "spec", None), "framework", None)
    snap = memory_snapshot(framework)
    jax_stats = snap.get("jax") or {}
    torch_stats = snap.get("torch") or {}
    if framework == "torch":
        allocated = torch_stats.get("memory_allocated")
    else:
        allocated = jax_stats.get("bytes_in_use")
    return {"model": model, "framework": framework,
            "allocator_in_use_gib": _gib(allocated),
            "allocator_peak_during_load_gib": _gib(
                torch_stats.get("max_memory_allocated") if framework == "torch"
                else jax_stats.get("peak_bytes_in_use")),
            "allocator_peak_during_load_bytes": (
                torch_stats.get("max_memory_allocated") if framework == "torch"
                else jax_stats.get("peak_bytes_in_use")),
            "nvml_process_reserved_gib": snap.get("nvml_process_gib"),
            "jax_pool_bytes": jax_stats.get("pool_bytes"),
            "jax_bytes_limit": jax_stats.get("bytes_limit"),
            "snapshot": snap, "environment": process_environment()}
