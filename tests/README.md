# Executable tests for PnP-Flow and D-Flow

Two scripts that run the reconstruction code against **synthetic analytic "models"** — no
checkpoint, no GPU, no download. They exist because the structural checks in `src/checks.py`
run inside a real experiment, and these can run in seconds on a laptop or a login node.

```bash
python tests/test_meanflow_pnp_dflow.py     # needs jax + optax
python tests/test_flow_trajectory.py        # needs jax
python tests/test_beta_schedule.py          # pure Python (+ pyyaml for the config checks)
python tests/test_rhso.py                   # needs jax + optax
python tests/test_rhso_theory.py            # needs jax + optax + pyyaml
python tests/test_cluster_and_aggregation.py  # needs pyyaml + bash (no jax/torch)
python tests/test_theory_extensions.py      # needs jax + pyyaml
python tests/test_matched_configs.py        # needs pyyaml (imports run.py)
python tests/test_hpo_lock.py               # needs pyyaml + git
python tests/test_paired_stats.py           # needs numpy
python tests/test_flower.py                 # needs jax + pyyaml (+ torch for the JiT path)
python tests/test_fingerprints.py           # needs jax + pyyaml (imports run.py)
python tests/test_gmm_study.py             # needs torch + git (no trained model)
python tests/test_gmm_solvers.py           # needs torch (no trained model)
python tests/test_hexagon_study.py         # needs torch (no trained model)
python tests/test_timing_configs.py         # needs jax + pyyaml (imports run.py)
python tests/test_matched_time.py           # needs jax + pyyaml (imports run.py)
python tests/test_budget_levels.py          # needs jax + pyyaml (imports run.py), bash
python tests/test_budget_levels_lrscaled.py # needs jax + pyyaml (imports run.py)
```

> These are **scripts**, not pytest test functions: `pytest tests/` collects nothing. Run
> each file directly; each prints `N/N checks passed` and exits non-zero on failure.

`test_meanflow_pnp_dflow.py` executes `pnp.meanflow_pnp` and `dflow.meanflow_dflow`
end to end against a toy MeanFlow adapter and checks: the `1 + N·M` denoiser accounting and
zero generative backprops; the correction schedule and step sizes; bitwise determinism;
that the reprojection noise does **not** move when `gamma0` does; that the fidelity actually
falls; that Adam reduces the D-Flow objective; that the returned reconstruction is the
trajectory of the **final** `q` rather than a stale iterate; that the gradient with respect
to `q` is finite and non-zero; and that the planner's cost model matches the measured
counters.

`test_flow_trajectory.py` executes the framework-neutral parts of the standard-flow paths:
`dflow._integrate_flow` for euler / heun / rk4 with and without JiT's final-Euler policy,
compared **bitwise** against an independent transcription of the reference integrator in
`sdedit.py`; the evaluation counts; the terminal gradient through a multi-step trajectory
and that a descent step reduces the loss; and the PnP schedule and per-image reprojection
noise.

`test_beta_schedule.py` needs nothing but the standard library and checks the universal
power-law grid: `beta` validation, exact endpoints, strict monotonicity, the worked
`beta = 0.5` / `beta = 2` examples, interval direction, PnP's times staying strictly inside
`(s0, 1)` for every `beta`, that `MPC-Δt`'s inverse-delta scaling uses the **local** `dt_k`,
and — most importantly — that `beta = 1` reproduces the pre-`beta` grid **bitwise**, against
an independent transcription of the original expressions.

`test_rhso.py` executes `rhso.meanflow_rhso` against toy MeanFlow adapters (one pixel-space,
one latent with a differentiable decoder) and asserts the *structure* rather than the
output: the recorded transition trace shows `M` direct `s_k → 1` planning calls and exactly
one `s_k → s_{k+1}` execution per stage; `lr = 0` degenerates the run to plain successive
execution, which is what proves `q` starts from `x_k`; the Adam state is verified fresh at
every stage against a reference (with the carried-moments variant shown to differ); the
counters match `N·M` and `N·(M+1)` and the planner's cost model; the result is deterministic
and distinct from D-Flow. The standard-flow parts that are not torch-specific — the planning
suffix, the evaluation counts, `flow_step`, and `integrate_flow` over a non-uniform suffix
compared bitwise against a reference integrator — are executed too.

It also covers RHSO's optional state-anchor regularisation `mu`: that `R` is zero at the
anchor and positive after a displacement, that its gradient equals `(q − anchor)/d` and a
descent step strictly shortens the distance to the anchor, that `R` is summed over the batch
and normalised by the state dimensionality, that the anchor stays bitwise fixed across a
stage's inner iterations while the displacement grows from exactly zero, that stage `k+1`
anchors bitwise to stage `k`'s **executed** state (not `x0`, not the optimised `q`), that a
larger `mu` really does hold `q` nearer the anchor, that every compute counter is identical
at `mu = 0` and `mu > 0`, and that a spec carrying `mu = 0.0` and one with no `mu` attribute
at all produce a **bitwise identical** result. Torch/JAX parity is asserted structurally —
both loops are shown (by scanning their bytecode, nested closures included) to call the one
backend-generic `rhso.state_anchor_penalty`, whose NumPy and JAX evaluations agree — because
`flow_rhso` itself cannot run here.

`test_rhso_theory.py` covers the direct terminal planner and the theory-validation
diagnostics. It executes a JiT-shaped toy standard flow (native clean prediction, guidance
applied to it, velocity **derived** from it) and the toy MeanFlow, and checks: that
`x + (1-s)·v` recovers the direct prediction at equal guidance cost; that direct planning
performs exactly one endpoint prediction with `integrate_flow` replaced by a tripwire, while
the legacy suffix planner still integrates; that `auto` resolves to each family's previous
planner and that a MeanFlow refuses `suffix`; that the one-interval execution rule and the
executed trajectory are bitwise unchanged with diagnostics on; that direct JiT planning costs
one evaluation per objective in the planner's cost model; that every algorithmic counter is
identical with diagnostics on and off while the JVP/VJP/eval/second counters fill separately;
that the matched-budget config yields exactly `(1,160) (2,80) (4,40) (8,20)` and 70 jobs with
the frozen learning rates; that `sum(per-sample fidelity) == batch fidelity` for all four
normalisations; that a batch of 4 with one padded row gives three distinct per-image rows per
stage; that `V_pre`/`V_post` match an independent re-run and `V_post` differs from the last
recorded inner loss; that the final stage's replanning fields are NaN rather than invented;
that the pMF consistency metric equals an independently computed predict/execute/re-predict
discrepancy and that the JiT metric is labelled `terminal_prediction_inconsistency` rather
than a semigroup defect; that the Jacobian probes are matrix-free, per-image isolated,
seed-deterministic and claim no condition number; that the stage and spectral arrays survive
`results.npz` round-tripping; and that the per-job process memory sampler resets while JAX's
lifetime high-water mark is never reported as a process peak.

`tests/spec_support.py` builds every spec through the **real** validator and planner: it
assembles a small in-memory configuration, runs `validate_config` and `resolve_run_plan`,
and splits overrides automatically between the configuration (where they are validated) and
`dataclasses.replace`. A test that runs at all has therefore already proved its method and
fields are properly declared. Both scripts also exercise the configuration directly — `beta`
sweeping into distinct job ids and directories, `delta` becoming null at `beta != 1`, RHSO
rejecting `lam`/`K`/`control_cost_normalization`/non-Adam optimisers, `mu` sweeping into
distinct job ids and output paths while leaving the cost estimate untouched, `mu` being
refused for every non-RHSO method, and the warm-up key separating beta- and mu-distinct
jobs. A documented stand-in remains as a fallback if
`src/config.py` cannot be imported at all; each script prints which path it used.

`test_cluster_and_aggregation.py` is the regression suite for the bugs found by the first
real Rorqual H100 smoke run. It needs neither JAX nor PyTorch and checks: that the
`_probe_spec` helper lets an explicit override win instead of raising
`dataclasses.replace() got multiple values ...` (and that the *old* form really did raise,
so the test proves the bug rather than assuming it); that the direct-terminal-planning
check's baseline probe disables all three theory diagnostics explicitly rather than
inheriting them from a theory config; that every mutable top-level file an array task writes
is shard-private, that `collect_finished_jobs` merges a 4-job x 4-image run into exactly 16
per-image rows with no duplicates, deterministic ordering and correct partial-completion
behaviour; that merged `checks.json` contains **both** model families and that a failure in
any one shard survives the merge; that merged `run_metadata.json` unions per-model
provenance and preserves each shard's complete payload; that the "REMAINING suffix" warning
appears for `suffix` jobs (including `auto` on JiT) and never for JiT-`direct`; that Rorqual
resolves to `cuda/12.9` + `cudnn/9.13.1.26` while **Narval's module line is byte-identical
to the previous behaviour** and never contains `12.9`; that `setup_cluster.sh` contains no
hard-coded `module load cuda` and installs `nvidia-ml-py` with a wheelhouse-first fallback;
that `src/memory.py` never calls `nvmlDeviceGetMemoryInfo` (checked by parsing its AST, not
by substring search, since the docstring forbidding it names it) and never fills
`gpu_process_peak_gib` from a framework counter; and that the four theory configs still
resolve to 4 / 70 / 10 / 10 atomic jobs with the matched-budget pairs and the two distinct
terminal-planner identities intact.

`test_theory_extensions.py` covers the two opt-in capabilities added for the six theory
experiments, and the six configurations themselves.

For the **paired measurement noise** (`measurement_noise_group`) it checks: that with no
group the built `y` is **bitwise** what an independent transcription of the original seeding
expression produces; that `(y − Ax)/σ` is identical across `σ ∈ {0.1, 0.2, 0.4}` inside one
group and that the shared draw really is standard normal; that a different group, and a
different image id, each re-draw independently while the other images keep their own `ε`;
that a different *structural* parameter (`blur_sigma` 1.0 vs 2.0) stays independent while
two `σ` at one blur width stay paired — i.e. only the noise **amplitude** is excluded from
the seed; that rebuilding a request reproduces `y` bitwise; that the mask is bitwise
unchanged with and without a group, so the feature touches the noise seed and never the
operator; and that the group and the recipe actually used are recorded in the problem
metadata.

For the **gradient-aligned authority** diagnostic it checks against a closed form — a linear
terminal map `P(x) = Dx` with a quadratic fidelity, where `A = ‖Dᵀg‖/‖g‖` is known exactly —
that the two norms and the ratio match analytically (max relative error ~3e-8), that the log
field is the log of the ratio, that an image scores the same alone as inside a batch of
three (per-image isolation), that `real_rows` truncation keeps padded rows out, that the
probe is deterministic, that its cost lands only in `diagnostic_model_evals` /
`diagnostic_vjps`, and that a vanishing endpoint gradient yields the documented **NaN**
while both norms stay on the row. It then runs `rhso.meanflow_rhso` end to end with the flag
on and off and asserts the executed states are **bitwise identical**, that every algorithmic
counter matches, that the diagnostic counters and clock fill only when the flag is on, that
there is exactly one populated row per (image, stage), and that the default run records
nothing at all. Serialisation is checked by round-tripping the new `[image, stage]` arrays
through a real `.npz` **without** `allow_pickle`, and by showing the summary averages skip a
`NaN` row instead of propagating it.

For **backward compatibility** it asserts both fields default to off, that a spec predating
either field resolves to the old behaviour, that
`configs/experiments_theory_validation_final100.yaml` still resolves to 70 jobs with the
four matched pairs and unchanged job ids and leaf directories, and that turning the new flag
on in one experiment block moves **exactly** that block's 2 job ids while the other 68 are
untouched — so a partially-completed sweep keeps its results.

For the **six new configurations** it pins each one's job count, uniqueness, models, methods
and image count, and then the scientific content of each: that Experiment 1 is D-Flow with
`steps=4`, 160 iterations, Heun and the validated JiT learning rate; that 2A's six pairs all
satisfy `N·M = 160` and are not a 36-job Cartesian product; that 2B holds `M = 40` while `B`
runs 40 → 640; that the noise sweep carries one pairing group across four `σ` that remain
four distinct problem instances; that the `mu` sweep is log-spaced including 0 at both
`(4,40)` and `(8,20)` with the loss-history split enabled; that `beta` moves the stage times
in the documented direction (`s₁ = 0.5000` at `β=0.5`, `0.2500` uniform, `0.0625` at `β=2`)
and that `delta` is null off the uniform grid; and that Experiment 6 enables both
diagnostics on exactly the jacobian config's settings, images and probe seed while keeping
the old jobs' identities disjoint.

`test_timing_configs.py` covers the timing and peak-memory run (rebuttal Task 1): the
generated per-prior timing configurations regenerate byte for byte, every block resolves to
one job at batch size 1 that equals its frozen source entry in every non-identity field, and
every source entry is covered exactly once; `runtime.timing_detail` is opt-in; the
synchronised per-image timer, the JAX allocator-peak scope rule and the hardware records work;
`run_job` returns bit-identical pixels with and without the timing detail; the driver runs a
model-only baseline, one process per block and an aggregate, skips blocks already complete in
the run directory (and only those), and estimates the wall-clock from logged runs; the JAX
memory bracket (sampled `bytes_in_use` lower bound, lifetime peak upper bound) is computed and
sampled in the NVML loop; and `summarize_timing.py` builds the three Table 29-style tables
from synthetic runs, with the bracket and the uninformative flag in the memory table.

`test_matched_time.py` covers the matched-time comparison on the pixel priors (rebuttal
Task 5): the quality and timing configurations regenerate byte for byte; the quality config is
exactly the 33 entries of the task's table (written out independently in the test) at the
benchmark batch sizes, each equal to its frozen source job except for its budget knob and the
cost fields the planner derives from it; D-Flow's backprops equal `num_opt_steps` and
MPC-RHC's `num_mpc_steps x n_ctrl x (K-1)`; every entry's comment names its source and says it
is matched to RHSO's measured time; the timing config has one one-job block per entry at batch
size 1; the driver runs one model-only baseline process per prior on that two-prior config;
and `matched_time_table.py` picks RHSO's and MPC-Δt's entries per prior, reproduces
`paired_stats.py`'s bootstrap, applies the 10% time rule at its boundary, and refuses
duplicate cells, mismatched timing jobs and unpaired image sets.

`test_budget_levels.py` covers the cheap and mid budget levels on the pixel priors (rebuttal
Task 6): the four configurations regenerate byte for byte from the task's table (written out
independently in the test); the quality config is exactly the 66 entries (36 cheap, 30 mid)
at the source entries' batch sizes, MPC-RHC and JiT's RHSO in their own `_rhc` / `_rhso`
blocks, each job equal to its source except for its step knob(s) and the derived cost fields,
JiT's RHSO keeping the direct planner and Heun; backprops follow RHSO `N·M`, MPC-Δt
`(N − 1)·n_ctrl`, D-Flow `num_opt_steps`, MPC-RHC `num_mpc_steps·n_ctrl·(K − 1)`; comments
name the source and the level; the timing config has one one-job block per entry at batch
size 1, `run_timing.py --dry-run` gives a baseline per prior and 66 blocks, and an LF copy of
`submit_timing.sh` accepts `budget_levels`. On synthetic runs, `budget_level_table.py` uses
the task's budgets, picks the lowest-LPIPS run within 1.1× (boundary included) or else the
cheapest run marked over budget, never lets an over-budget run be the best, drops untimed
jobs and non-direct JiT RHSO, takes the RHSO-time budget from the reference's timing,
reproduces `paired_stats.py`'s bootstrap for method − best, counts wins and ties, writes its
CSVs and LaTeX, and refuses duplicate jobs, ambiguous timing and unpaired image sets.

`test_budget_levels_lrscaled.py` covers the lr-scaled rerun of the budget levels (rebuttal
Task 7): the two configurations regenerate byte for byte and their learning rates equal the
task's table (written out in the test) to 4 significant figures; every lr follows
`lr_tuned × tuned knob / new knob`, recomputed from the source configs; every job equals its
Task 6 job in every field but lr (same batch sizes, steps and backprops); comments name the
source, the level and the lr scaling. On synthetic runs `budget_level_table.py` labels the
`_lr` variants, leaves them unpaired by default, pairs them with Task 6's timing jobs under
`--timing-ignore lr` (preferring Task 6's over another variant's with the same knob), and
rejects any other field.

`test_matched_configs.py`, `test_hpo_lock.py` and `test_paired_stats.py` cover the
matched-budget comparison of RHSO and MPC-Δt (rebuttal Task 2). The first resolves the
generated screen, smoke and frozen configurations through the real validator and planner and
checks that every job's backprops per image equal the budget in its block name (`N·M` for
RHSO, `(N − 1)·C` for MPC-Δt), that the pair set is exactly the plan's with no off-budget
pair, that the budgets read from the frozen benchmark are the plan's Table 6, that
`--models <prior>` keeps one prior under the file's own `max_atomic_jobs`, and that `freeze`
picks the planted winner inside the screened grid and refuses an off-budget one. The second
runs `hpo.py propose` on a screen whose winners sit on every edge: without `--lock` the budget
knobs are pushed (so the test has teeth), with it they never are, while lr, mu and lam still
are; JiT keeps `rhso_terminal_mode: direct`; and on inputs that predate both features the
output is byte-identical to `hpo.py` at commit `4d63970`. The third checks the paired
bootstrap against an independent transcription of the protocol (10,000 resamples, seed
20260903), that pairing is by `image_id`, and that measurement RMSE is a job-level
difference with no interval.

`test_flower.py` covers Flower (rebuttal Task 3) against two synthetic pixel priors that share
one destination map: a JAX MeanFlow (BHWC) and a **torch** standard flow with JiT's BCHW layout,
so `flow_flower` is executed here too and must agree with `meanflow_flower` (1.6e-5). It checks
`nu_t` at 0, 0.5 and near 1 and the time grid; that the closed-form refinement equals
conjugate gradient on denoising, both inpaintings and strided SR, and equals the plan's
unscaled textbook formula; that the VJP adjoint is exact on the reflect-padded blur (whose H
is not symmetric) and CG matches a dense float64 solve; that at the paper's cap of 50
iterations the early, ill-conditioned blur solves stop at the cap while late ones converge,
that CG starts at `x_hat` (zero iterations return it) and that at t = 0 a cap of 50 stops at
the cap while 100 converges;
that the `gamma = 1` sample has covariance `Sigma_t`; that Eval./BP are N and 0 with the H/H^T
and CG counters as documented and equal to the planner's estimate; determinism, noise that
follows (image, iteration) and not `num_flower_steps`, and batch independence; that the last
iteration returns `x_tilde`; the rejections (latent adapter, `stroke_painting`, `sigma = 0`, SiT
and iMF, bad `flower_gamma`); that the job ids, directories and costs of all 14,967 jobs in the
24 configurations committed before Flower are unchanged and pre-Flower artefacts still
resume; that the committed screen/smoke configs are the generator's output (48 screen jobs:
the CG cap {50, 100} on deblurring only) and that `shared_initial_state` passes for Flower
next to SDEdit; that `hpo.py` keys the CG cap on deblurring only, that `propose --lock
flower=num_flower_steps,flower_cg_max_iters` keeps both knobs inside the screened grid
(locking N alone does not hold the cap), that `make_flower_configs.py freeze` picks the planted
winner and writes its deblurring cap into the frozen configs, refusing off-protocol winners;
and that `finalize` still demands the six benchmark methods.

`test_fingerprints.py` covers `scripts/check_fingerprints.py` on realistic job sets: the
benchmark is `experiments_final_frozen100.yaml` resolved by the planner, and the Flower and
matched-budget runs are the frozen configs their `freeze` commands write from planted tuning
results, written with run.py's own CSV writer. Initial-state fingerprints are computed as
`run.run_job` computes them (first batch of `prior_sample` at t0 = 1, on toy adapters). It
checks that correct runs pass with every one of their 35 jobs compared (so the frozen batch
sizes line up with the benchmark's), that the first-batch z_t0 depends on the batch size and
not the method, and that a changed measurement, guide or initial state, a batch size or image
count without a benchmark counterpart, an inconsistent benchmark, a missing fingerprint and an
unaggregated run all fail and name the job.

`test_gmm_study.py` covers the exact quantities of the Gaussian-mixture study (rebuttal
Task 4), with no trained model: the mixture's structure and sample moments; the closed-form
posterior against importance sampling (log evidence, mean, covariance and component
probabilities within Monte Carlo error for denoising, masking and the random projection,
with an equal-weight proposal so the closed-form weights are tested, and a sigma-doubled
posterior shown to fail); the exact velocity against the posterior of x_t = t x1 + (1 - t) x0
and its values at t = 0 and 1; the exact flow carrying N(0, I) to the mixture at the sliced-W2
floor; that `toy/common.py` reproduces the pre-refactor two-moons trainers bit for bit
(against `toy/rhso_two_moons.py` at commit 0d1eaed); and the raw-error floor of the MeanFlow
target: the per-row exact velocity equals the scalar one, the batch replay reproduces
`train_meanflow`'s own batches (at learning rate 0 the logged raw error is recomputed
exactly), and the closed form |dPhi/dz (v - v*)|^2 matches the exact average velocity's
error against a finite-difference target built the way the trainer builds it. It also checks
the sign of iMF's velocity prediction (`common.imeanflow_prediction`, used by
`train_imeanflow`). With the exact average velocity (a differentiable RK4 solve of the exact
velocity) in place of the network, and the exact marginal velocity as the JVP tangent,
V = u - (s - t) du/dt equals v up to RK4 error, including the jump from t = 0 to s = 1. The
flipped sign fails. iMF's time sampling, in this repo's convention, gives s >= t, s != t on
half the batch, and logit-normal(0.4, 1) times.

`test_gmm_solvers.py` covers the paused oracle stage 2 of the mixture study, on the CPU in
float64: the torch exact velocity against the numpy one, the direct endpoint against the
closed-form E[x1 | x_t], the 16-step RK4 map against 1000 steps and the numpy exact flow,
backprop through the map against finite differences; the per-row Adam against
torch.optim.Adam; that no solver lets rows interact; the cost counts (RHSO against
`src.rhso.rhso_cost_estimate`); Flower's exact refinement; and that one-step Euler D-Flow
on the exact velocity collapses to E[x1].

`test_hexagon_study.py` covers the hexagon study's data and metrics: the hexagon is regular
with side 2, centred, with the lower-right corner at (1, -sqrt 3); the distance to the
hexagon is exact on known points and matches a dense polyline; the data is uniform in arc
length, and its distance to the hexagon follows the 0.02 jitter; sliced W2 is correct on a
known Gaussian pair. For stage 2, the exact posterior of x given y = x_1 + sigma eps,
computed by 1-D quadrature along the hexagon, matches importance sampling of the data model
for sigma 0.1 and 0.3, with both branches present at sigma 0.1. The solvers' optional
`trace` leaves every output unchanged and records x0 plus one state per outer step.

## What these tests do NOT cover

The toy adapters are not the real models, so nothing here says anything about
reconstruction *quality*. More specifically:

* **JiT-direct is not executed end to end here.** `rhso.flow_rhso` and the real
  `JiTAdapter` both need PyTorch. What is covered for that path is structural: the shared
  `make_terminal_planner` factory is executed against a JiT-shaped toy adapter, and
  `JiTAdapter.velocity` / `JiTAdapter.clean_prediction` are shown by bytecode scan to be
  wrappers around one `_guided_clean` with no duplicated guidance. The torch glue itself is
  checked on a GPU by `run.py --check`, which now runs `rhso_direct_terminal_planning`
  against the loaded model.
* `pnp.flow_pnp`, `dflow.flow_dflow` and `rhso.flow_rhso` are **not** executed — they call
  `torch.autograd.grad`, `torch.optim.Adam` and `torch.no_grad`, which need PyTorch. Their
  trajectory arithmetic and counters are covered indirectly (the integrator above and the
  MeanFlow twins), but the torch glue itself is not.
* `src/memory.py`'s Torch and NVML paths need a CUDA device; on a CPU box the profiler
  reports `cpu_no_gpu_memory` and the columns stay empty.
* The VAE-decoder gradient path for latent models (SiT, iMF) needs a real checkpoint.

Run the real per-model checks for those: `python run.py --config ... --check` executes
`pnp_initial_projection`, `pnp_determinism`, `dflow_gradient` and `dflow_optimisation`
against the actual loaded model, and they are recorded in `checks.json`.



