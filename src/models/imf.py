"""iMeanFlow (iMF) adapter -- class-conditional MeanFlows in SD-VAE latent space (32x32x4).

Preserved from the notebooks and fully usable, but NOT enabled in the default configuration
(see configs/experiments.yaml for how to switch it on).

Loading, config handling, EMA selection, latent normalisation and guidance are unchanged.
Two details are easy to get wrong and are therefore called out:

  * the latent normalisation is the repository's PER-CHANNEL AFFINE transform from
    `utils/vae_util.LatentManager` -- NOT the Stable Diffusion 0.18215 scale.  Using the
    latter would place the guide in the wrong latent scale entirely.
  * `to_pixels(differentiable=True)` runs a jitted Flax decode so gradients from the
    pixel-space measurement loss reach the latent state; the display path uses the
    repository's own compiled pmap decoder.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..utils import (MEANFLOW, assert_pixel_batch, gaussian_noise, native_time,
                     pixel_fingerprint, record_time, timed)
from .base import (AdapterSpec, Conditioning, MeanFlowAdapter, RepoSandbox,
                   download_and_extract_zip, find_checkpoint_dir, register_adapter,
                   register_prefetch, stub_missing_module)


@contextmanager
def _jax_cost_analysis_as_list():
    """iMF's `LatentManager` logs `cost_analysis()[0]["flops"]`; current JAX returns the
    dict itself, so without this shim the indexing raises KeyError and iMF never loads.
    """
    import jax.stages
    compiled = jax.stages.Compiled
    original = compiled.cost_analysis

    def as_list(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return [result] if isinstance(result, dict) else result

    compiled.cost_analysis = as_list
    try:
        yield
    finally:
        compiled.cost_analysis = original


def _device_put_replicated(x, devices):
    """Drop-in for the removed `jax.device_put_replicated`, with the old semantics.

    Every leaf becomes an array of shape (len(devices),) + leaf.shape whose i-th slice is a
    copy of the leaf living on devices[i] (one shard per device), so pmap-style code that
    consumes `flax.jax_utils.replicate` output sees the same shapes and placement as before.
    """
    import jax
    devices = list(devices)
    if not devices:
        raise ValueError("`devices` argument to `device_put_replicated` must be a "
                         "non-empty sequence.")
    mesh = jax.sharding.Mesh(np.array(devices, dtype=object), ("replica",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("replica"))

    def put(leaf):
        shards = [jax.numpy.expand_dims(jax.device_put(leaf, d), 0) for d in devices]
        shape = (len(devices),) + tuple(shards[0].shape[1:])
        return jax.make_array_from_single_device_arrays(shape, sharding, shards)

    return jax.tree_util.tree_map(put, x)


def _device_put_replicated_works(jax) -> bool:
    """True when `jax.device_put_replicated` exists AND can be called.

    Newer JAX keeps the name only as a deprecation entry: `hasattr` is False because
    attribute access raises AttributeError. A stub that is reachable but raises
    AttributeError when called counts as missing too, so the function is probed once on a
    tiny array. Any other exception from the probe propagates.
    """
    if not hasattr(jax, "device_put_replicated"):
        return False
    try:
        fn = jax.device_put_replicated
        fn(np.zeros((1,), np.float32), jax.local_devices()[:1])
    except AttributeError:
        return False
    return True


def _ensure_device_put_replicated(jax) -> bool:
    """Install `_device_put_replicated` as `jax.device_put_replicated` only when JAX has no
    working one (iMF's `LatentManager` calls `flax.jax_utils.replicate`, which needs it).

    Installed for the rest of the process, since the repository's decoder may replicate
    again after construction. Returns True when the shim was installed; when the original
    works, nothing is changed.
    """
    if _device_put_replicated_works(jax):
        return False
    jax.device_put_replicated = _device_put_replicated
    return True


class IMFAdapter(MeanFlowAdapter):

    def __init__(self, registry: Dict[str, Any], repo_dir: Path, ckpt_cache: Path,
                 local_device_count: int = 1):
        import jax
        import jax.numpy as jnp
        super().__init__("imf", registry)
        self.jax, self.jnp = jax, jnp
        # Resolved BEFORE the sandbox chdirs into the repository (see pmf.py).
        self.repo_dir = Path(repo_dir).resolve()
        self.ckpt_cache = Path(ckpt_cache).resolve()
        self.local_device_count = int(local_device_count)
        self.sandbox = RepoSandbox("imf", self.repo_dir, extra_roots=("imf",))
        self.latent_mean = np.asarray(registry["latent_mean"], np.float32)
        self.latent_std = np.asarray(registry["latent_std"], np.float32)
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        import time
        jax = self.jax
        t_start = time.perf_counter()
        cfg = self.registry
        # The repository's logging utilities import wandb at module scope; this
        # run never logs, so an absent wandb must not block a checkpoint load.
        stub_missing_module("wandb")
        with self.sandbox:
            import yaml
            from configs.default import get_config as get_default_config
            from imf import iMeanFlow
            from utils.ckpt_util import restore_checkpoint
            from utils.trainstate_util import create_train_state
            from utils.lr_utils import lr_schedules

            with open(self.repo_dir / cfg["config_yml"]) as fh:
                overrides = yaml.safe_load(fh)
            config = get_default_config()
            for k, v in overrides.items():
                config[k].update(v) if isinstance(v, dict) else config.__setitem__(k, v)
            config.eval_only = True
            config.model.model_str = cfg["model_str"]     # eval_config.yml names iMF-B/2 only

            extracted = download_and_extract_zip(cfg["hf_repo"], cfg["ckpt_file"],
                                                 self.ckpt_cache)
            config.load_from = find_checkpoint_dir(extracted)

            self.config = config
            self.latent_size = int(config.dataset.image_size)          # 32
            self.latent_channels = int(config.dataset.image_channels)  # 4
            self.num_classes = int(config.dataset.num_classes)         # 1000
            self.vae_type = cfg["vae_type"] or config.dataset.vae      # "mse"
            self.pixel_res = self.latent_size * 8                      # 256

            self._model = iMeanFlow(**config.model.to_dict(), eval=True)
            state = create_train_state(jax.random.key(0), config, self._model,
                                       self.latent_size, lr_schedules(config, 1000))
            state = restore_checkpoint(state, config.load_from)
            self.step = int(state.step)

            # iMF stores a SINGLE EMA pytree (contrast: pMF stores a dict keyed by EMA value).
            params = state.ema_params if cfg["use_ema"] else state.params
            self._params = {"params": jax.device_get(params)}
            del state

            from utils.vae_util import LatentManager
            from diffusers.models import FlaxAutoencoderKL
            self._FlaxAutoencoderKL = FlaxAutoencoderKL
            _ensure_device_put_replicated(jax)
            with _jax_cost_analysis_as_list():
                self._latent_manager = LatentManager(
                    vae_type=self.vae_type,
                    decode_batch_size=cfg["vae_decode_batch"],
                    input_size=self.latent_size)
            self._vae = self._latent_manager.vae
            self._vae_params = self._latent_manager.vae_params

        # As loaded, the model params are host numpy and the VAE params are committed to
        # the CPU device; a jitted function follows its committed inputs, so without this
        # the differentiable decode runs the VAE on the CPU in every gradient step (~4 s
        # per iteration, which made iMF 20-280x slower than pMF).
        device = jax.devices()[0]
        self._params = jax.device_put(self._params, device)
        self._vae_params = jax.device_put(self._vae_params, device)

        self._build_jitted()
        load_s = time.perf_counter() - t_start
        record_time("load/imf", load_s)

        self.spec = AdapterSpec(
            name="imf", display_name=cfg["display_name"], dynamics_family=MEANFLOW,
            framework="jax", state_space="latent",
            native_shape=(self.latent_size, self.latent_size, self.latent_channels),
            layout="BHWC", pixel_resolution=self.pixel_res,
            prediction_kind="mean_velocity", native_time_mapping=cfg["native_time_mapping"],
            batch_size=int(cfg["batch_size"]), fixed_batch_shape=True,
            num_classes=self.num_classes, null_label=int(cfg["null_label"]),
            guidance=dict(cfg["guidance"]),
            checkpoint={"hf_repo": cfg["hf_repo"], "file": cfg["ckpt_file"],
                        "ckpt_dir": self.config.load_from, "step": self.step,
                        "model_str": self.config.model.model_str, "ema": cfg["use_ema"],
                        "repo_rev": cfg.get("repo_rev"),
                        "vae": "pcuenq/sd-vae-ft-%s-flax" % self.vae_type,
                        "vae_encode_mode": cfg["vae_encode_mode"],
                        "latent_normalization": "per_channel_affine (NOT 0.18215)",
                        "load_seconds": round(load_s, 2)})

    def _build_jitted(self) -> None:
        jax, jnp = self.jax, self.jnp
        model, vae, FlaxAutoencoderKL = self._model, self._vae, self._FlaxAutoencoderKL
        mean_bchw = jnp.asarray(self.latent_mean, jnp.float32).reshape(1, -1, 1, 1)
        std_bchw = jnp.asarray(self.latent_std, jnp.float32).reshape(1, -1, 1, 1)

        def _step(params, z, labels, t_steps, omega, t_min, t_max):
            # Repository single-interval API: i = 0, t_steps = [t, r].
            return model.apply(params, z, labels, jnp.int32(0), t_steps,
                               omega, t_min, t_max, method=model.sample_one_step)

        def _encode(vae_params, x_bhwc):
            # Exactly utils/data_util.py::compute_latent_dataset
            dist = vae.apply({"params": vae_params},
                             jnp.transpose(x_bhwc, (0, 3, 1, 2)),      # BHWC -> BCHW
                             method=FlaxAutoencoderKL.encode).latent_dist
            return dist.mean, dist.std                                 # both NHWC, 4 channels

        def _decode_diff(vae_params, z_bhwc):
            """Differentiable decode: normalised latent BHWC -> canonical pixels BHWC.

            Mirrors utils/vae_util.LatentManager.decode exactly
                latents = latents * std + mean          (BCHW per-channel affine)
                vae.apply(..., method=decode).sample
            but without the pmap/device_get round trip, so the graph survives.
            """
            z = jnp.transpose(z_bhwc, (0, 3, 1, 2))                    # BHWC -> BCHW
            z = z * std_bchw + mean_bchw                               # LatentManager denorm
            out = vae.apply({"params": vae_params}, z,
                            method=FlaxAutoencoderKL.decode).sample     # BCHW, ~[-1,1]
            return jnp.transpose(out, (0, 2, 3, 1))                    # -> canonical BHWC

        self._jit_step = jax.jit(_step)
        self._jit_encode = jax.jit(_encode)
        self._decode_diff = _decode_diff                # un-jitted: usable inside jax.grad
        self._jit_decode_diff = jax.jit(_decode_diff)   # jitted: forward-only and grad-through

    # ------------------------------------------------------------------ latent normalisation
    def normalize_latent(self, z):
        return (z - self.latent_mean.reshape(1, 1, 1, -1)) / self.latent_std.reshape(1, 1, 1, -1)

    def denormalize_latent(self, z):
        return z * self.latent_std.reshape(1, 1, 1, -1) + self.latent_mean.reshape(1, 1, 1, -1)

    # ------------------------------------------------------------------ canonical interface
    def to_native_noise(self, noise: np.ndarray):
        return self.jnp.asarray(np.asarray(noise, np.float32))

    def encode_pixels(self, pixels: np.ndarray):
        """Canonical pixels -> NORMALISED iMF latent, via the repository's own conventions."""
        jnp = self.jnp
        assert_pixel_batch(pixels, "imf guide", self.pixel_res)
        x = jnp.asarray(np.asarray(pixels, np.float32))
        with timed("encode/imf"):
            mean, std = self._jit_encode(self._vae_params, x)
            mode = self.registry["vae_encode_mode"]
            if mode == "mean":
                z = mean
            elif mode == "sample":
                eps = jnp.asarray(gaussian_noise(
                    tuple(np.asarray(mean).shape), "imf", "vae_encode",
                    pixel_fingerprint(pixels)))
                z = mean + std * eps
            else:
                raise ValueError("imf.vae_encode_mode must be 'sample' or 'mean'")
            return self.block(self.normalize_latent(z))

    def _lerp(self, guide, noise, keep: float, add: float):
        return self.jnp.float32(keep) * guide + self.jnp.float32(add) * noise

    def to_pixels(self, state, differentiable: bool = False):
        jax = self.jax
        if differentiable:
            # Jitted: jax.grad differentiates through one compiled unit rather than tracing
            # the whole VAE eagerly, and the backward pass is compiled once and reused.
            return self._jit_decode_diff(self._vae_params, state)      # graph preserved
        with timed("decode/imf"):
            latents = np.asarray(jax.device_get(state), np.float32).transpose(0, 3, 1, 2)
            n = len(latents)
            full = self.local_device_count * int(self.registry["vae_decode_batch"])
            out = []
            for start in range(0, n, full):
                chunk = latents[start:start + full]
                actual = len(chunk)
                if actual < full:
                    pad = np.zeros((full - actual,) + chunk.shape[1:], chunk.dtype)
                    chunk = np.concatenate([chunk, pad], axis=0)
                with self.sandbox:
                    imgs_bchw = self._latent_manager.decode(chunk)     # applies z*std + mean
                imgs = np.asarray(jax.device_get(imgs_bchw)).transpose(0, 2, 3, 1)
                out.append(imgs[:actual])
            pixels = np.concatenate(out, axis=0).astype(np.float32)
        return np.clip(pixels, -1.0, 1.0)

    # ------------------------------------------------------------------ MeanFlow dynamics
    def transition(self, state, s_from: float, s_to: float, conditioning: Conditioning):
        """T_theta(x; s_from -> s_to) on the canonical clock."""
        jnp = self.jnp
        g = {**self.spec.guidance, **(conditioning.guidance or {})}
        interval = g.get("interval") or self.spec.guidance["interval"]
        labels = jnp.asarray(np.asarray(conditioning.labels, np.int32))
        t_steps = jnp.asarray([native_time(s_from, self.spec.native_time_mapping),
                               native_time(s_to, self.spec.native_time_mapping)], jnp.float32)
        self.count_forwards(1)
        return self._jit_step(self._params, state, labels, t_steps,
                              jnp.float32(g["scale"]),
                              jnp.float32(interval[0]), jnp.float32(interval[1]))

    # ------------------------------------------------------------------ sanity checks
    def sanity_checks(self) -> Dict[str, Any]:
        return {"latent_normalization": self._check_latent_normalization,
                "vae_roundtrip": self._check_encode_decode,
                "decode_matches_repository": self._check_decode_agreement}

    def _check_latent_normalization(self, ctx) -> Tuple[bool, str]:
        jax = self.jax
        m = np.asarray(jax.device_get(self._latent_manager.mean)).reshape(-1)
        s = np.asarray(jax.device_get(self._latent_manager.std)).reshape(-1)
        ok = (np.allclose(m, self.latent_mean, atol=1e-6)
              and np.allclose(s, self.latent_std, atol=1e-6))
        return ok, ("per-channel affine mean=%s std=%s (0.18215 is NOT used)"
                    % (np.round(m, 4).tolist(), np.round(s, 3).tolist()))

    def _check_encode_decode(self, ctx) -> Tuple[bool, str]:
        from ..metrics import psnr
        orig = ctx["pixels"]
        recon = self.to_pixels(self.encode_pixels(orig))
        mae = float(np.mean(np.abs(recon - orig)))
        ctx["reconstruction"] = recon
        return (mae < 0.15 and psnr(recon, orig) > 18.0), \
            "encode_pixels -> decode: MAE=%.4f PSNR=%.2f dB" % (mae, psnr(recon, orig))

    def _check_decode_agreement(self, ctx) -> Tuple[bool, str]:
        jax = self.jax
        z = self.prior_sample(range(self.spec.batch_size)) * 0.5
        a = np.asarray(jax.device_get(self._jit_decode_diff(self._vae_params, z)), np.float32)
        b = self.to_pixels(z)                                    # repository pmap decoder
        err = float(np.max(np.abs(np.clip(a, -1, 1) - b)))
        return err < 1e-3, "max|differentiable - repository decode| = %.2e" % err


@register_adapter("imf")
def _make_imf(registry: Dict[str, Any], context: Dict[str, Any]) -> IMFAdapter:
    return IMFAdapter(registry, context["repo_paths"]["imf"], context["ckpt_cache"],
                      context.get("local_device_count", 1))


@register_prefetch("imf")
def _prefetch_imf(registry: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Download the iMF checkpoint AND the Flax SD-VAE the LatentManager will need."""
    target = download_and_extract_zip(registry["hf_repo"], registry["ckpt_file"],
                                      Path(context["ckpt_cache"]).resolve())
    info = {"path": str(target), "checkpoint_dir": find_checkpoint_dir(target)}
    vae_type = registry.get("vae_type") or "mse"
    try:
        from huggingface_hub import snapshot_download
        repo = "pcuenq/sd-vae-ft-%s-flax" % vae_type
        snapshot_download(repo_id=repo)
        info["vae"] = repo
    except Exception as exc:                                            # pragma: no cover
        info["vae_error"] = str(exc)
    return info
