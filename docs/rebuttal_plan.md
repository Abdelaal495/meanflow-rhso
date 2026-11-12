# RHSO Rebuttal: Code Changes Plan

Nov 11, 2026 · Ahmed Abdelaal

## Purpose and ground rules

This plan adds four things to `Abdelaal495/meanflow-rhso` for the ICLR 2027 rebuttal (submission 58448): a timing and peak-memory run, a matched-budget RHSO vs MPC-Δt comparison, Flower as a new solver, and a Gaussian-mixture study with known posteriors. It is written to be handed to Claude Code on a local clone; everything is additive.

Rules for the implementation:

1. **Additive only.** Do not change the behaviour of any existing strategy, model adapter, operator or config. New code goes in new modules, new configs and new scripts. The existing test suite must still pass.
2. **Follow `docs/extending.md`** for the new strategy: all seven steps (function, declaration in `src/config.py`, registration in `src/reconstruction.py`, cost estimate, figure labels, warm-up key, pairing check).
3. **Keep the pairing invariant.** Never redraw prior noise or rebuild `x0`; use `make_phi`, `adapter.to_pixels(..., differentiable=True)` and the schedule helpers in `src/schedule.py`.
4. **Count cost honestly** in `ReconstructionStats` (`model_evals_total`, `backprops_through_model`, `data_gradient_evals`, `network_forwards`). The rebuttal's matched-budget argument rests on these counts.
5. **Config comments must not say that hyperparameters selected on one checkpoint were reused on another.** Each config records its own selection.
6. **Every new config gets a 2-image smoke variant** and must pass `python run.py --config <file> --dry-run` before any cluster submission.
7. **Anonymity.** The rebuttal must not link this named repository. Code for reviewers goes in an anonymized zip, as with the original supplementary material.

## Task 1: Timing and peak memory

No new solver code is needed: `run.py` already times steady-state reconstruction after a warm-up (`runtime_per_image`) and `src/memory.py` records peak GPU memory per job. This task is a config generator plus a summary script. It answers Reviewer rSAb (latent wall-clock and peak memory) and gives WhU5's matched-budget question a wall-clock axis.

**What to build**

1. `scripts/make_timing_config.py`: reads `configs/experiments_final_frozen100.yaml` (plus the Task 2 and Task 3 configs once they exist) and writes `configs/experiments_rebuttal_timing.yaml`.
   - Every method on all four priors, at its frozen per-task configuration.
   - `num_images: 3`, `batch_size: 1` for every job, `gpu_memory_profiling: true`.
   - Deduplicate: a (prior, method, configuration) whose cost-relevant fields are identical across tasks is timed once (for example RHSO on SiT-XL/2, identical on all five tasks). Operator cost differs slightly by task, so keep one task per distinct configuration and record which task was used.
2. `scripts/summarize_timing.py`: reads the run's CSV and writes one table per prior with columns method, task, Eval./BP per image, seconds per image (mean of 3), `gpu_process_peak_gib`, `gpu_process_incremental_peak_gib`.
   - Use the `gpu_process_*` columns for any comparison across priors; the Torch and JAX native columns measure different things (see the docstring of `src/memory.py`).
   - Also emit LaTeX rows matching the style of Table 29.

**How it must be run** (note for the cluster, not for Claude Code)

- One prior per job (`--models <prior>`), one job per GPU, nothing else on that GPU. The original benchmark shared GPUs across shards, which is why its wall-clock times were not reported.
- Same GPU type for all four priors (A100-SXM4-40GB, as in Table 29).

**Expected cost:** about 1 GPU-hour, dominated by SiT-XL/2 RHSO at 5776 evaluations per image.

## Task 2: Matched-budget RHSO vs MPC-Δt

The goal is a table showing RHSO and the original MPC-Δt at equal backpropagations through the prior, tuned by the same protocol as Appendix C.4. No new solver code is needed: RHSO's direct planner for flow-matching priors already exists (`rhso_terminal_mode: direct`), and `scripts/hpo.py` already runs the screen, refine and finalize loop. The work is configs, one small `hpo.py` option, and a paired-statistics script.

**Budgets to match** (backprops per image, from Table 6)

| Prior | MPC-Δt BP per task (den / deb / SR / rand / box) | Benchmark RHSO BP | Runs to add |
| --- | --- | --- | --- |
| pMF-L/16 | 140 / 280 / 280 / 280 / 280 | 160 | A: RHSO at MPC-Δt's BP. B: MPC-Δt at 160 BP |
| JiT-B/16 | 140 / 280 / 280 / 140 / 140 | 640–1280 (suffix) | A: RHSO, direct planner, at MPC-Δt's BP |
| iMF-B/2 | 280 on all tasks | 640 | A: RHSO at 280 BP. B (optional): MPC-Δt at 640 BP |
| SiT-XL/2 | 280 / 280 / 120 / 280 / 280 | 5760 (suffix) | A: RHSO, direct planner, at MPC-Δt's BP |

Direction A is the main result. Direction B checks that MPC-Δt does not simply need more budget.

**Exact (N, M) and (N, C) pairs**

- RHSO backprops = N · M with the MeanFlow map or the direct planner. Candidate pairs: 140 BP → (4, 35), (7, 20); 280 BP → (4, 70), (8, 35); 120 BP → (4, 30), (8, 15).
- MPC-Δt backprops = (N − 1) · C, since its last stage does not differentiate the prior. Candidate pairs: 160 BP → (5, 40), (9, 20); 640 BP → (9, 80), (17, 40).
- Tuned per task: RHSO learning rate and anchor μ over the values in `configs/experiments_hpo_grid.yaml`; MPC-Δt λ and learning rate over its grid there.

**What to build**

1. `scripts/make_matched_configs.py`, writing three configs: a screening grid (4 tuning images), and after tuning, the frozen 100-image evaluation and its 2-image smoke variant.
   - `src/config.py` expands list-valued fields as a Cartesian product, so `num_rhso_steps: [4, 8]` with `num_opt_steps: [70, 35]` would also produce off-budget pairs. Emit one experiment block per (N, M) or (N, C) pair instead, with names like `deblurring__rhso_bp280_n4`.
   - Set `rhso_terminal_mode: direct` on JiT and SiT. Verify every block's cost with `--dry-run` against the target BP before submitting.
2. `scripts/hpo.py`: add a `--lock` option (for example `--lock rhso=num_rhso_steps,num_opt_steps` and `--lock mpc_delta_t=num_mpc_steps,n_ctrl`) so `propose` never pushes a budget-defining knob past a grid edge. Without it, the edge-extension rule would raise M or C and break the budget match. Use `--no-paper` so manuscript values are not added as candidates.
3. `scripts/paired_stats.py`: from `results_per_image.csv` of the frozen run plus the original benchmark run, compute per task and prior the paired mean differences in PSNR, SSIM, LPIPS and measurement RMSE, with nonparametric bootstrap 95% intervals over the 100 images (10,000 resamples, seed 20260903 as in Appendix C.5). Output a CSV and LaTeX rows.

**Selection rule:** lowest LPIPS on the 8 tuning images, inside the pre-registered grid, frozen before the 100-image evaluation, exactly as in Appendix C.4.

**Expected cost:** 6–10 GPU-hours including tuning. The benchmark ran batched, so this should be well below the batch-1 runtimes of Table 29.

## Task 3: Flower as a new solver

Add Flower (Pourya, El Rawas and Unser, ICLR 2026, [arXiv 2509.26287](https://arxiv.org/abs/2509.26287)) as a new strategy `flower` in `src/flower.py`, for pixel priors only (pMF-L/16, JiT-B/16). `src/pnp.py` is the template: same time grid helpers, same no-backprop structure, same deterministic reprojection noise, same MeanFlow substitution. Only the refinement step is new.

**The algorithm** (canonical clock, t = 0 noise, t = 1 data; grid t_k from `canonical_time_grid`, so t_0 = 0 at t0 = 1)

For k = 0, …, N − 1, starting from the shared `x0`:

1. **Destination.** Flow matching: x̂ = x_t + (1 − t) v(x_t, t). MeanFlow: x̂ = T(x_t; t → 1). No gradient.
2. **Refinement.** Solve the proximal step below, with σ the task's true measurement noise (`problem.sigma`, same scale as y) and ν_t the paper's schedule. Default γ = 0, as in all of the paper's reported results; implement γ = 1 (adds κ_t, a sample with covariance Σ_t) behind a flag.
3. **Time progression.** x_{t_{k+1}} = (1 − t_{k+1}) ε + t_{k+1} x̃, with fresh ε every iteration. At the last step t_N = 1, so the output is x̃.

```latex
\mu_t = \left(\nu_t^{-2} I + \sigma^{-2} H^\top H\right)^{-1}\left(\nu_t^{-2}\,\hat{x} + \sigma^{-2} H^\top y\right), \qquad \nu_t = \frac{1-t}{\sqrt{t^2 + (1-t)^2}}
```

**Solving the refinement per task**

| Task | HᵀH | Solve |
| --- | --- | --- |
| Denoising | I | Closed form |
| Random and box inpainting | diag(mask) | Closed form, elementwise |
| 2× SR (strided `x[:, ::f, ::f]`) | diag(sampling mask) | Closed form; Hᵀy = zero-upsampled y |
| Deblurring (separable Gaussian, reflect padding) | Not diagonal | Conjugate gradient, max 50 iterations, relative residual 1e-5 (the paper's settings) |

- Get Hᵀ by autodiff (vjp) of `problem.apply`, which is linear, so this is exact. For the diagonal cases diag(HᵀH) = Hᵀ(H(1)).
- Reject `stroke_painting` and any nonlinear operator with a clear error.
- Reject latent adapters (`spec.state_space == "latent"`) with a clear error, and leave `flower` out of the SiT and iMF capability entries. Flower's refinement acts on the image, and a latent version needs an encoder or an inner optimization, which would be a new method.

**Integration** (the seven steps of `docs/extending.md`)

- `flow_flower` (PyTorch, JiT) and `meanflow_flower` (JAX, pMF), registered for both families.
- `METHOD_DECLARATIONS["flower"]` with fields `num_flower_steps` (sweepable), `flower_gamma` (0 or 1), `flower_cg_max_iters`, `flower_cg_tol`, plus defaults and validation.
- Deterministic noise: add `flower_noise_parts(model, image_id, replicate, iteration)` to `src/utils.py`, mirroring `pnp_reprojection_parts`, and inject it with `adapter.corrupt` as PnP does.
- Cost accounting: N model evaluations, 0 backprops through the model. Count H and Hᵀ applications in `data_gradient_evals`, and add an optional `linear_solve_iterations` field (default 0) to `ReconstructionStats` for CG iterations.
- `hpo.py`: add `"flower": ["num_flower_steps"]` to `KNOBS`.
- Docstring in the style of `pnp.py`: published (the three steps, ν_t, γ = 0) vs adapted (start from the shared `x0`; the MeanFlow destination is a research extension; σ is the oracle noise level, as in the paper).

**Tuning:** `num_flower_steps` in {50, 100, 200, 500} (the paper used 100 to 500), same screen-refine-freeze protocol, then the 100-image evaluation. About 2 GPU-hours.

## Task 4: Gaussian-mixture study with known posteriors

This answers WhU5's request for a controlled low-dimensional setting with matched data, architecture and training recipe. A flow-matching model and a MeanFlow model share one MLP and one recipe, every solver runs on both, and every result is scored against the exact posterior. It lives in `toy/gmm_study.py` and runs on CPU in a few hours or on one GPU in under an hour.

**Reuse from `toy/rhso_two_moons.py`:** `TimeFeatures`, `MeanFlowMLP`, `VelocityMLP` and the MeanFlow `train` function (JVP target, adaptive weighting, interval curriculum). Move them to `toy/common.py` with a `dim` argument (both are hard-coded to 2-D today) and keep the two-moons script working unchanged.

**Two traps**

- **Train MeanFlow with the MeanFlow objective, not `train_distilled`.** The toy's default distils from a flow-matching teacher, which would undercut the matched-recipe claim. The two-moons docstring notes the MeanFlow objective is less stable there; check training on the mixture first, before writing any solver.
- **Use d = 16, not 2.** In 2-D every direction carries signal, so nothing shows noise-fitting in low-signal directions.

**Data and models**

- d = 16, 8 components with equal weights, means of norm about 2, and anisotropic covariances: each a random rotation of diag(s_i²) with s_i decaying as a power law from 0.5 to 0.02, so some directions carry almost no signal. Fixed seed; scale the data to unit overall variance.
- Same MLP for both families (width 512, depth 4; MeanFlow takes the extra s input as in `MeanFlowMLP`), same optimizer, steps, batch size, learning-rate schedule and EMA.
- Optional third prior: the exact velocity of the mixture, which has a closed form. It gives a reference transport to check the learned maps against.
- Sanity checks after training, logged and plotted: sliced Wasserstein distance between generated and true samples for both models; MeanFlow one-step T(x; 0 → 1) against a fine ODE solve of the exact flow.

**Inverse problems**

| Operator | Definition | Noise σ |
| --- | --- | --- |
| Denoising | H = I | 0.1, 0.2, 0.4 |
| Masking | keep a fixed random 50% of coordinates | 0.05 |
| Projection | fixed random Gaussian H with 8 rows | 0.05 |

The posterior is exactly a Gaussian mixture. Per component, use the standard linear-Gaussian update for the weights, means and covariances; implement it in closed form and unit-test it against importance sampling.

**Solvers**

Reimplement each solver on the toy interfaces (`transition` and `velocity` helpers), following Appendix A: RHSO, D-Flow (1-step), PnP-Flow, MPC-Δt, MPC-RHC and Flower. Run all of them on both priors at the paper's budgets (N · M = 160 for RHSO) and record evaluations and backprops the same way as the image code. Optionally add the order-flipped RHSO (optimize after each jump) behind a flag.

**Tuning and evaluation**

- Tuning set: 64 problems (separate seed). Select each solver's hyperparameters by lowest mean squared error to x⋆, over a small grid per solver and prior.
- Test set: 512 problems, 16 solver seeds each.
- Metrics: MSE to x⋆; distance to the exact posterior mean; log density under the exact posterior; measurement RMSE; sliced Wasserstein-2 between each solver's 16 samples per problem and exact posterior samples; and the error split across the eigendirections of the data covariance (high- vs low-signal directions).
- Outputs in `outputs/gmm_study/`: a CSV of all metrics, LaTeX table rows, and two figures (error against σ per solver and prior; per-direction error).

## Testing and cluster run order

Nothing goes to the cluster until the existing tests pass, the new tests pass, and every new config passes `--dry-run` with the expected cost.

**Tests to add** (Claude Code, locally; standalone scripts like the existing ones, each printing N/N checks passed and exiting non-zero on failure)

- `tests/test_flower.py`: ν_t values at t = 0, 0.5 and near 1; the closed-form refinement equals the CG solution on denoising, masking and strided subsampling; CG converges on the blur operator; Eval./BP counts equal N and 0; the noise is deterministic per (image, iteration) and independent of `num_flower_steps`; latent adapters and `stroke_painting` are rejected.
- `tests/test_matched_configs.py`: every generated block's dry-run cost equals its target backprops, and no off-budget (N, M) pair appears.
- `tests/test_hpo_lock.py`: `propose --lock` never extends a locked knob.
- `toy/`: the closed-form mixture posterior matches importance sampling to within Monte Carlo error.
- Run every existing test script directly (`python tests/test_<name>.py`) and confirm each still passes. They are scripts, so `pytest tests/` collects nothing.

**Cluster order**

1. Push a branch `rebuttal-experiments`; pull it on Narval or Rorqual. No new dependencies are expected.
2. Run the 2-image smoke configs for Tasks 2 and 3, and confirm the `shared_initial_state` pairing check passes for `flower`.
3. Submit the Task 2 screening grid and the Flower tuning grid at the same time; they need the queue the longest.
4. Meanwhile, run Task 4 on a CPU node or locally.
5. `hpo.py propose --lock … --no-paper --images 8`, run the refine round, then `hpo.py finalize`, then the frozen 100-image runs.
6. Run Task 1 last, so it also times Flower and the matched-budget configurations, with one exclusive A100-40GB per prior.
7. Aggregate (`run.py --aggregate`), then run `summarize_timing.py` and `paired_stats.py`.

Total compute is about 10–15 GPU-hours. Steps 3 and 5 depend most on queue times.

## Deliverables and open questions

**Code (Claude Code)**

- [ ] `scripts/make_timing_config.py` and `scripts/summarize_timing.py`
- [ ] `scripts/make_matched_configs.py`, the `--lock` option in `scripts/hpo.py`, and `scripts/paired_stats.py`
- [ ] `src/flower.py`, registered and declared per `docs/extending.md`, with `flower_noise_parts` and the `linear_solve_iterations` stats field
- [ ] `toy/common.py` (refactor) and `toy/gmm_study.py`
- [ ] New tests passing, existing suite unchanged
- [ ] Smoke configs and dry-run output for every new config

**Runs (cluster)**

- [ ] Task 2: screening, refinement, frozen 100-image evaluation
- [ ] Task 3: Flower tuning and 100-image evaluation on pMF-L/16 and JiT-B/16
- [ ] Task 4: mixture study outputs
- [ ] Task 1: timing and memory on all four priors

**Open questions**

- The rebuttal deadline decides whether the optional runs fit: MPC-Δt at 640 backprops on iMF-B/2, and the order-flipped variant in the mixture study.
- Confirm an A100-SXM4-40GB is available for all four timing jobs; if not, time all four on the same alternative GPU.
