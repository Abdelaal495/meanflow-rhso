# Toy example: RHSO on two moons

A 2-D illustration of Receding-Horizon State Optimization with a MeanFlow-style prior
(one learned jump from any time `t` to any time `s`), independent of the image benchmark.

```bash
python toy/rhso_two_moons.py                      # trains a small MLP on the CPU, then draws
python toy/rhso_two_moons.py --N 1 2 4 8 --M 40   # stages N and inner steps M per stage
python toy/rhso_two_moons.py --budget 160         # fixed total budget N * M
```

Outputs `figures/toy_rhso_two_moons.pdf` (trajectories and per-stage terminal error) and a
sanity plot of the learned transport.  The trained model is cached under `cache/`.
Only `numpy`, `torch` and `matplotlib` are required.

## Shared models: `toy/common.py`

`TimeFeatures`, `MeanFlowMLP(dim, width, depth)`, `VelocityMLP(dim, width, depth)` and two
trainers with one recipe (Adam 3e-4, warm-up + cosine, grad clip 1, EMA 0.999):
`train_meanflow` (the MeanFlow identity with one JVP, adaptive weighting and an interval
curriculum) and `train_flow_matching`. At `dim = 2` they reproduce the two-moons models and
trainers bit for bit (checked by `tests/test_gmm_study.py`), so cached two-moons models load.

## Gaussian-mixture study: `toy/gmm_study.py` (rebuttal Task 4)

A 16-D mixture of 8 anisotropic Gaussians (scales 0.5 down to 0.02, so some directions carry
almost no signal) with a closed-form posterior under y = Hx + sigma eps, and the exact
velocity field of the mixture.

```bash
python toy/gmm_study.py stage1             # train flow matching and MeanFlow (cached), sanity checks
python toy/gmm_study.py stage1 --retrain
python toy/gmm_study.py stage1 --lr 1e-4 --long-share 0.25 --adaptive-p 1.0 --s-min 0.05
python toy/gmm_study.py sweep              # MeanFlow stability sweep, 10k steps per setting
python toy/gmm_study.py steptime           # ms per training step of both priors
```

Every command takes `--device` (default `cuda` when available, else `cpu`).

Both priors use the same MLP (width 512, depth 4) and the same recipe; MeanFlow is trained
with the MeanFlow objective, not distilled. `--lr` and `--s-min` (the thinnest component
std before scaling, i.e. the data) apply to both priors, so a MeanFlow model is always
compared with a flow-matching model trained on the same data with the same learning rate,
steps and batch; `--long-share` and `--adaptive-p` exist only in the MeanFlow objective.
Non-default settings get their own cache files and output folder (`stage1_<tag>`).

Stage 1 writes `outputs/gmm_study/stage1/sanity.json` and the training curves: sliced W2
to true samples for every sampler, the MeanFlow jumps T(x; t -> 1) from t = 0, 0.25, 0.5
and 0.75 against a fine RK4 solve of the exact flow (x_t is the exact flow's state at t),
flow-matching Heun from the same states for reference, training times, and the pass
criteria (MeanFlow one-step SW2 <= 0.05, its low-signal variance ratio in [0.7, 1.5], and
MeanFlow 8-step SW2 within 1.5x of flow-matching Heun-100).

The sweep trains MeanFlow only, for every combination of long-jump share {0.25, 0.5},
adaptive-weight exponent {0.5, 1}, learning rate {3e-4, 1e-4} and thinnest std {0.02, 0.05},
logs the raw MeanFlow error every 100 steps and writes `outputs/gmm_study/sweep/` (one JSON
per setting, `table.csv`, `curves.png`). A setting blows up when the running median of the
raw error rises above 10x the lowest value it reached before.

### Oracle-prior stage 2 (paused)

Work on the Gaussian-mixture study was paused before the full oracle run finished; Task 4 now
uses the hexagon study below. What exists:

- `toy/gmm_oracle.py`: exact priors with no training. The flow-matching interface is the
  exact mixture velocity, and its direct endpoint `x + (1 - t) v` is exactly `E[x1 | x_t]`.
  The MeanFlow interface is the exact map `Psi(x; t -> s)`, a 16-step RK4 solve of that
  velocity, chosen by `python toy/gmm_oracle.py` as the smallest step count within 1e-3
  relative RMSE of 1000 steps from t = 0, 0.25, 0.5, 0.75 (`outputs/gmm_study/oracle_rk4_check.csv`).
  One map evaluation counts as one model evaluation.
- `toy/gmm_solvers.py`: RHSO (direct, and the flow-matching Heun suffix), one-step D-Flow,
  PnP-Flow, MPC-Δt, MPC-RHC and Flower ported from `src/` onto the toy interfaces, batched
  over independent rows with a per-row Adam, with the image code's cost accounting.
- `toy/gmm_stage2.py`: tuning on 64 problems x 4 seeds over the pMF grids, test on
  512 x 16, metrics against the exact posterior, CSV / LaTeX / figures
  (`python toy/gmm_stage2.py --block oracle [--smoke]`).

## Hexagon study: `toy/hexagon_study.py` (rebuttal Task 4)

Modelled on the toy of MPC-Flow (Webber et al., 2026, Appendix C): data uniform on the
boundary of a regular hexagon with side 2, centred, flat lower edge, with N(0, 0.02^2 I)
jitter.

```bash
python -u toy/hexagon_study.py train --model flow_matching   # one prior per run, on the GPU (cached)
python -u toy/hexagon_study.py train --model meanflow_p0
python -u toy/hexagon_study.py train --model meanflow_p0.5
python -u toy/hexagon_study.py stage1                         # sanity table from the cached priors
```

The study runs on CUDA only (`--device cuda`, the default): it stops at once when CUDA is
missing or a test forward + backward fails, and exits with the traceback on any error rather
than hanging. Training prints a progress line every 1,000 steps (step, loss, elapsed time)
and writes the same lines to `outputs/hexagon_study/train_<model>.log`.

A flow-matching and a MeanFlow prior share the 2-D MLP of `toy/common.py` (width 256,
depth 4) and one recipe (30k steps, batch 1024, Adam 3e-4, warm-up + cosine, EMA 0.999).
MeanFlow is trained with the MeanFlow objective at adaptive-weight exponents p = 0 and 0.5.
Stage 1 writes `outputs/hexagon_study/stage1/` (`sanity.csv`, `sanity.json`, `samples.png`,
`training_curves.png`). The table reports sliced W2 to the data against the two-sample floor,
and the distribution of distances to the hexagon for flow-matching Heun-100 and MeanFlow
with 1, 2 and 8 steps. It also covers the jumps T(x_t; t -> 1) from the true marginal at
t = 0.25, 0.5, 0.75, next to flow-matching Heun from the same x_t.
