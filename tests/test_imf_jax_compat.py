"""iMF adapter: compatibility shim for the removed `jax.device_put_replicated`.

Newer JAX dropped `jax.device_put_replicated`; the name survives only as a deprecation entry
whose access raises AttributeError, so `flax.jax_utils.replicate` (called by iMF's
`LatentManager`) fails at model load. `src/models/imf.py` installs `_device_put_replicated`
before the LatentManager is built, only when JAX has no working original. This suite checks:

  1. the missing function is detected both when ACCESS raises AttributeError (newer JAX) and
     when a reachable stub raises AttributeError when CALLED; the shim is installed in both;
  2. the shim's semantics: every leaf gets a leading axis equal to the device count, each
     slice equals the input, dtypes are kept, one shard per device in order, and a pmap over
     the result sees the same shapes as before; a device subset and an empty list behave;
  3. when the original works it is left untouched (same object), and a non-AttributeError
     from the probe propagates without installing anything;
  4. `IMFAdapter._load` installs the shim before it builds the LatentManager.

JAX is forced onto 4 CPU devices so the per-device checks are meaningful. Runs on CPU with
no checkpoint:

    python tests/test_imf_jax_compat.py
"""

import contextlib
import inspect
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_force_host_platform_device_count=4").strip()

import numpy as np                                               # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax                                                       # noqa: E402

from src.models import imf                                       # noqa: E402

PASSED = [0]
FAILED = []
NAME = "device_put_replicated"
REMOVED_MSG = "jax.device_put_replicated is deprecated; use jax.device_put instead."


def check(name, ok, detail=""):
    print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    if ok:
        PASSED[0] += 1
    else:
        FAILED.append(name)


@contextlib.contextmanager
def jax_module_state():
    """Restore jax's `device_put_replicated` and `__getattr__` entries on exit."""
    keys = (NAME, "__getattr__")
    saved = {k: vars(jax)[k] for k in keys if k in vars(jax)}
    try:
        yield
    finally:
        for k in keys:
            vars(jax).pop(k, None)
        vars(jax).update(saved)


def access_raises():
    """Make `jax.device_put_replicated` behave as in newer JAX: access raises."""
    vars(jax).pop(NAME, None)
    inner = vars(jax).get("__getattr__")

    def __getattr__(name):
        if name == NAME:
            raise AttributeError(REMOVED_MSG)
        if inner is not None:
            return inner(name)
        raise AttributeError("module 'jax' has no attribute %r" % name)

    jax.__getattr__ = __getattr__


def call_raises(*args, **kwargs):
    raise AttributeError(REMOVED_MSG)


def sample_tree():
    return {"params": {"w": np.arange(6, dtype=np.float32).reshape(2, 3),
                       "b": np.float32(0.5),
                       "idx": np.arange(4, dtype=np.int32)}}


# ---------------------------------------------------------------------- 1. detection
def check_detection():
    print("1. missing function detected, shim installed")
    with jax_module_state():
        access_raises()
        check("simulated access-raising entry: hasattr is False",
              not hasattr(jax, NAME))
        check("access-raising entry reported as not working",
              imf._device_put_replicated_works(jax) is False)
        check("shim installed when access raises",
              imf._ensure_device_put_replicated(jax) is True
              and vars(jax).get(NAME) is imf._device_put_replicated)
    with jax_module_state():
        jax.device_put_replicated = call_raises
        check("call-raising stub: hasattr is True", hasattr(jax, NAME))
        check("call-raising stub reported as not working",
              imf._device_put_replicated_works(jax) is False)
        check("shim installed over a call-raising stub",
              imf._ensure_device_put_replicated(jax) is True
              and jax.device_put_replicated is imf._device_put_replicated)
    check("module state restored after the simulation",
          vars(jax).get(NAME) is not call_raises
          and vars(jax).get(NAME) is not imf._device_put_replicated)


# ---------------------------------------------------------------------- 2. semantics
def check_semantics():
    print("2. shim semantics")
    devices = jax.local_devices()
    n = len(devices)
    check("JAX sees 4 CPU devices", n == 4, "devices=%d" % n)
    tree = sample_tree()
    with jax_module_state():
        jax.device_put_replicated = call_raises
        imf._ensure_device_put_replicated(jax)
        # Called the way flax.jax_utils.replicate calls it: through the module attribute.
        rep = jax.device_put_replicated(tree, devices)

    leaves = jax.tree_util.tree_leaves_with_path(tree)
    rep_leaves = dict(jax.tree_util.tree_leaves_with_path(rep))
    check("same tree structure",
          jax.tree_util.tree_structure(rep) == jax.tree_util.tree_structure(tree))
    for path, leaf in leaves:
        key = jax.tree_util.keystr(path)
        out = rep_leaves[path]
        leaf = np.asarray(leaf)
        check("%s: leading axis = device count" % key,
              out.shape == (n,) + leaf.shape, "shape=%s" % (out.shape,))
        check("%s: dtype kept" % key, out.dtype == leaf.dtype, str(out.dtype))
        host = np.asarray(out)
        check("%s: every slice equals the input" % key,
              all(np.array_equal(host[i], leaf) for i in range(n)))
        shards = sorted(out.addressable_shards, key=lambda s: s.index[0].start or 0)
        check("%s: one shard per device, in device order" % key,
              len(shards) == n
              and [s.device for s in shards] == devices
              and all(s.data.shape == (1,) + leaf.shape for s in shards)
              and all(np.array_equal(np.asarray(s.data)[0], leaf) for s in shards))

    out = jax.pmap(lambda p: p["params"]["w"] * 2.0 + p["params"]["b"])(rep)
    expected = tree["params"]["w"] * 2.0 + tree["params"]["b"]
    check("pmap over the replicated tree sees per-device shapes",
          out.shape == (n, 2, 3)
          and all(np.allclose(np.asarray(out)[i], expected) for i in range(n)),
          "shape=%s" % (out.shape,))

    sub = imf._device_put_replicated(tree, devices[:2])
    check("device subset gives a leading axis of that size",
          sub["params"]["w"].shape == (2, 2, 3)
          and {s.device for s in sub["params"]["w"].addressable_shards} == set(devices[:2]))
    try:
        imf._device_put_replicated(tree, [])
        raised = False
    except ValueError:
        raised = True
    check("empty device list raises ValueError", raised)


# ---------------------------------------------------------------------- 3. no-op when working
def check_untouched():
    print("3. original left untouched when it works")
    calls = []

    def working(x, devices):
        calls.append(len(list(devices)))
        return imf._device_put_replicated(x, devices)

    with jax_module_state():
        jax.device_put_replicated = working
        check("working original reported as working",
              imf._device_put_replicated_works(jax) is True)
        installed = imf._ensure_device_put_replicated(jax)
        check("shim not installed when the original works", installed is False)
        check("jax.device_put_replicated is still the original object",
              jax.device_put_replicated is working)
        check("the original was probed (called)", len(calls) >= 1, "calls=%d" % len(calls))

    def broken(x, devices):
        raise RuntimeError("unrelated failure")

    with jax_module_state():
        jax.device_put_replicated = broken
        try:
            imf._ensure_device_put_replicated(jax)
            propagated = False
        except RuntimeError:
            propagated = True
        check("non-AttributeError from the probe propagates", propagated)
        check("nothing installed after a propagated error",
              jax.device_put_replicated is broken)


# ---------------------------------------------------------------------- 4. call site
def check_call_site():
    print("4. IMFAdapter._load installs the shim before the LatentManager")
    src = inspect.getsource(imf.IMFAdapter._load)
    shim_at = src.find("_ensure_device_put_replicated(jax)")
    lm_at = src.find("LatentManager(")
    check("_load calls _ensure_device_put_replicated", shim_at >= 0)
    check("... before LatentManager(...) is built", 0 <= shim_at < lm_at,
          "shim@%d LatentManager@%d" % (shim_at, lm_at))


def main():
    print("jax %s, real jax.%s %s" % (
        jax.__version__, NAME,
        "works" if imf._device_put_replicated_works(jax) else "is missing (shim needed)"))
    check_detection()
    check_semantics()
    check_untouched()
    check_call_site()
    total = PASSED[0] + len(FAILED)
    print("\n%d/%d checks passed" % (PASSED[0], total))
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
