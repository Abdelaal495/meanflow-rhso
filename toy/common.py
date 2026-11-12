"""Shared pieces of the toy studies: time features, the two MLPs, and their trainers.

Moved here from toy/rhso_two_moons.py so the Gaussian-mixture study (toy/gmm_study.py) can
use the same models in any dimension.  At `dim = 2` with the default width and depth, every
module has exactly the layout, parameter names and initialisation order it had in
rhso_two_moons.py, and `train_meanflow` consumes random numbers in exactly the order the
old `train` did, so two-moons training is bit-for-bit what it was (tests/test_gmm_study.py
checks this against the pre-refactor file in git) and its cached models still load.

Time convention: t = 0 is noise, t = 1 is data, x_t = (1 - t) x0 + t x1, v = x1 - x0.

Two trainers with ONE recipe -- the same data stream, seed, optimiser (Adam), learning
rate, warm-up + cosine schedule, gradient clipping, EMA, batch size and step count.  Only
the objective differs, as each family requires:

    train_flow_matching   v_theta(x_t, t)      -> v                       (plain regression)
    train_meanflow        u_theta(x_t, t, s)   -> v + (s - t) du/dt       (MeanFlow identity,
                          one JVP, adaptive weighting, interval curriculum)
"""
import math
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

N_FREQ = 8


class TimeFeatures(nn.Module):
    def __init__(self, n_freq: int = N_FREQ):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n_freq) * math.pi)

    def forward(self, t):                          # t: (B,) in [0, 1]
        a = t[:, None] * self.freq[None, :]
        return torch.cat([t[:, None], torch.sin(a), torch.cos(a)], 1)


def _mlp(d_in: int, d_out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


class MeanFlowMLP(nn.Module):
    """u_theta(x, t, s): average velocity from t to s, for x at time t."""

    def __init__(self, dim: int = 2, width: int = 256, depth: int = 4):
        super().__init__()
        self.dim, self.width, self.depth = dim, width, depth
        self.tf = TimeFeatures()
        d_in = dim + 3 * (1 + 2 * N_FREQ)          # x, and features of t, s, s - t
        self.net = _mlp(d_in, dim, width, depth)

    def forward(self, x, t, s):
        h = torch.cat([x, self.tf(t), self.tf(s), self.tf(s - t)], 1)
        return self.net(h)


class VelocityMLP(nn.Module):
    """v_theta(x, t): the instantaneous velocity field, trained by plain flow matching."""

    def __init__(self, dim: int = 2, width: int = 256, depth: int = 4):
        super().__init__()
        self.dim, self.width, self.depth = dim, width, depth
        self.tf = TimeFeatures()
        self.net = _mlp(dim + (1 + 2 * N_FREQ), dim, width, depth)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.tf(t)], 1))


def _same_architecture(model: nn.Module) -> nn.Module:
    """A fresh module of the same class and shape (consumes the torch RNG like the original)."""
    return type(model)(dim=model.dim, width=model.width, depth=model.depth).to(
        next(model.parameters()).device)


def _schedule(opt, steps: int, warmup: int):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, k / steps))))


def _ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    # 1 - 0.999 is 0.0010000000000000009 in floating point; the rounded complement is the
    # literal 0.001 the two-moons trainer always used, so its EMA stays bit-identical.
    alpha = float("%.12g" % (1.0 - decay))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(pm, alpha=alpha)


DataFn = Callable[[np.random.Generator], np.ndarray]


def train_meanflow(model: MeanFlowMLP, data_fn: DataFn, steps: int, batch: int, seed: int,
                   long_share: float = 0.5, lr: float = 3e-4, warmup: int = 500,
                   ema_decay: float = 0.999, clip: float = 1.0, verbose: bool = True,
                   log_every: int = 1000, log: Optional[List[dict]] = None,
                   adaptive_p: float = 0.5) -> MeanFlowMLP:
    """MeanFlow objective (Geng et al.), in this file's time convention.

    With x_t = (1 - t) x0 + t x1 and v = x1 - x0, the average velocity obeys
        u(x_t, t, s) = v + (s - t) * du/dt,     du/dt = d/dt u(x_t, t, s) along the path,
    and the total derivative is one JVP with tangent (v, 1, 0).  A pair with s = t is plain
    flow matching.  `data_fn(rng)` returns the training pool (numpy, float32).  Training runs
    on the device the model is on.  `adaptive_p` is the exponent of the adaptive weight
    1 / (err + 1e-3)^p.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    data = torch.from_numpy(data_fn(rng)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _schedule(opt, steps, warmup)
    ema = _same_architecture(model)
    ema.load_state_dict(model.state_dict())
    started = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (batch,), device=device)
        x1 = data[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, device=device)
        # Curriculum on the interval: the JVP target is only as good as u already is, so the
        # early model must not be asked about long jumps.  The share of s != t pairs and the
        # longest allowed interval both grow linearly over the first third of training.
        ramp = min(1.0, step / (steps / 3))
        long = torch.rand(batch, device=device) < long_share * ramp
        s = torch.where(long, t + (1 - t) * torch.rand(batch, device=device) * ramp, t)
        z = (1 - t)[:, None] * x0 + t[:, None] * x1
        v = x1 - x0
        u, dudt = torch.func.jvp(lambda z_, t_, s_: model(z_, t_, s_), (z, t, s),
                                 (v, torch.ones_like(t), torch.zeros_like(s)))
        target = (v + (s - t)[:, None] * dudt).detach()
        err = ((u - target) ** 2).sum(1)
        # adaptive weighting from the MeanFlow paper (default p = 0.5): tames the large-|s - t|
        # terms; p = 1 makes every sample's weighted error err / (err + 1e-3) < 1
        w = 1.0 / (err.detach() + 1e-3) ** adaptive_p
        loss = (w * err).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        _ema_update(ema, model, ema_decay)
        if step % log_every == 0 or step == 1:
            elapsed = time.perf_counter() - started
            if log is not None:
                log.append(dict(step=step, loss=loss.item(), raw_mse=err.mean().item(),
                                seconds=elapsed))
            if verbose:
                print("  step %5d  loss %.4f  raw mse %.4f  (%.0fs)"
                      % (step, loss.item(), err.mean().item(), elapsed))
    return ema


def train_flow_matching(model: VelocityMLP, data_fn: DataFn, steps: int, batch: int,
                        seed: int, lr: float = 3e-4, warmup: int = 500,
                        ema_decay: float = 0.999, clip: float = 1.0, verbose: bool = True,
                        log_every: int = 1000, log: Optional[List[dict]] = None) -> VelocityMLP:
    """Flow matching with train_meanflow's recipe: v_theta(x_t, t) regressed on v = x1 - x0.
    Runs on the device the model is on."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    data = torch.from_numpy(data_fn(rng)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _schedule(opt, steps, warmup)
    ema = _same_architecture(model)
    ema.load_state_dict(model.state_dict())
    started = time.perf_counter()
    for step in range(1, steps + 1):
        x1 = data[torch.randint(0, len(data), (batch,), device=device)]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, device=device)
        z = (1 - t)[:, None] * x0 + t[:, None] * x1
        err = ((model(z, t) - (x1 - x0)) ** 2).sum(1)
        loss = err.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        _ema_update(ema, model, ema_decay)
        if step % log_every == 0 or step == 1:
            elapsed = time.perf_counter() - started
            if log is not None:
                log.append(dict(step=step, loss=loss.item(), raw_mse=loss.item(),
                                seconds=elapsed))
            if verbose:
                print("  step %5d  flow-matching mse %.4f  (%.0fs)" % (step, loss.item(), elapsed))
    return ema


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
