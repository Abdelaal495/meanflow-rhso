# scripts/

| script | what it does |
|---|---|
| `build_local_imagenet_pool.py` | builds the image folder `data.source: local_folder` reads. `--frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv` rebuilds the paper's 100-image benchmark from an ungated mirror |
| `hpo.py` | the hyperparameter search: `propose` the next round of candidates from a finished one, `finalize` the winners into the benchmark configuration |
| `make_matched_configs.py` | matched-budget RHSO vs MPC-Δt: `screen` writes the screening grid (one block per budget-defining pair), `freeze` writes the 100-image evaluation from the tuning runs; each with a 2-image smoke variant |
| `paired_stats.py` | paired RHSO − MPC-Δt differences per prior and task, with bootstrap 95% intervals, as CSV and LaTeX rows |
| `make_flower_configs.py` | Flower on pMF / JiT: `screen` writes the screen (`num_flower_steps` 50/100/200/500, plus `flower_cg_max_iters` 50/100 on deblurring; 4 images), `freeze` writes the 100-image evaluation from the tuning runs; each with a 2-image smoke variant |
| `make_timing_config.py`, `run_timing.py`, `summarize_timing.py` | rebuttal Task 1: one timing configuration per prior (every task and method of the three frozen configs, 3 images, batch size 1, plus a 2-image smoke variant); a driver that runs each configuration in a fresh process with one shared run id after a model-only memory baseline; and the per-prior summary (seconds per image, allocator peak memory, seconds per forward-equivalent) with LaTeX tables |
| `make_matched_time_config.py`, `matched_time_table.py` | rebuttal Task 5: the matched-time configurations on pMF and JiT (each method's budget knob set so it takes RHSO's measured time; 100-image quality config, timing config at batch size 1, smoke variants), and the table of seconds per image, quality and paired differences against RHSO |
| `make_budget_level_config.py`, `budget_level_table.py` | rebuttal Task 6: the cheap and mid budget levels on pMF and JiT (RHSO, MPC-Δt, D-Flow and MPC-RHC with their step knob set to the cheap methods' time and 4× that; 100-image quality config, timing config at batch size 1, smoke variants), and the per-level table: each method's lowest-LPIPS run within 1.1× the budget, paired differences against the level's best, wins and ties |
| `make_budget_level_lrscaled_config.py` | rebuttal Task 7: Task 6's 66 budget-level entries with only the learning rate changed, `lr × (tuned knob / new knob)`, 100-image config and smoke variant; `budget_level_table.py --timing-ignore lr` pairs them with Task 6's timing run |
| `make_anchor_schedule_tuning.py`, `freeze_anchor_schedules.py`, `anchor_schedule_table.py` | rebuttal Task 8: RHSO with a time-dependent state anchor (`mu_schedule`) on pMF; the mu0 tuning grid (22 per task on 8 tuning images, fields from the frozen entries), the freeze step (lowest LPIPS per task and schedule, 100-image config, tuning curves) and the 100-image table with paired differences against the retuned constant schedule |
| `check_fingerprints.py` | fails loudly unless every job of a Flower or matched-budget run has the benchmark run's measurement and initial-state fingerprints |
| `extract_imagenet100_from_tar.py` | the same pool from a local ImageNet tar instead of the mirror |
| `export_recon_bundle_only.py`, `add_corrupted_images_to_bundle.py` | package a run's reconstructions for sharing |
| `cluster_modules.sh` | cluster-conditional CUDA/cuDNN module selection, sourced by `setup_cluster.sh` |

Nothing here trains or reconstructs: `run.py` is the only entry point that needs a GPU.

## The search loop

There are no numbered stages. One round is run, `propose` reads it and writes the next, and
the loop ends when `propose` reports that no cell wants to move.

```bash
python run.py --config configs/experiments_hpo_grid.yaml --models pmf --run-id hpo_pmf --no-figures
python scripts/hpo.py propose --from outputs/hpo_pmf --models pmf --out configs/round2.yaml
python run.py --config configs/round2.yaml --run-id hpo_pmf_r2 --no-figures
python scripts/hpo.py propose --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf outputs/hpo_pmf_r2 \
    --top 0 --no-paper --models pmf --out configs/round3.yaml        # only cells still on an edge
python scripts/hpo.py finalize --from outputs/hpo_pmf_r2 --grid outputs/hpo_pmf \
    --models pmf --out configs/experiments_final_frozen100.yaml
```

`--from` is what the ranking reads; `--grid` is the union of everything tried so far and
decides what counts as a grid edge. Passing the original grid to `finalize` keeps the winner
inside the pre-registered ranges, which matters because on several axes (D-Flow and PnP step
counts, RHSO `N` and `M`) LPIPS keeps improving with compute far past the grid — a
compute-budget effect rather than a hyperparameter optimum.

Only the first round's configuration is committed. Later rounds are a function of results, so
they are generated rather than stored.

## Matched-budget RHSO vs MPC-Δt

The same loop, at equal backprops through the prior per image: RHSO at `N·M` (the MeanFlow
map, or `rhso_terminal_mode: direct` on JiT/SiT), MPC-Δt at `(N − 1)·C`. Each cell's target is
its own prior's frozen budget in `configs/experiments_final_frozen100.yaml`. Every `(N, M)`
or `(N, C)` pair has its own experiment block, so the Cartesian expansion never produces an
off-budget pair, and `--lock` stops `propose` from pushing a budget knob past an edge.

```bash
python scripts/make_matched_configs.py screen                # [--include-optional]: + iMF MPC-Δt at 640 BP
python run.py --config configs/experiments_matched_screen_smoke.yaml --models pmf --run-id matched_smoke_pmf
python run.py --config configs/experiments_matched_screen.yaml --models pmf --run-id matched_pmf --no-figures
python scripts/hpo.py propose --from outputs/matched_pmf --models pmf --images 8 --no-paper \
    --lock rhso=num_rhso_steps,num_opt_steps --lock mpc_delta_t=num_mpc_steps,n_ctrl \
    --out configs/matched_round2_pmf.yaml
python run.py --config configs/matched_round2_pmf.yaml --run-id matched_pmf_r2 --no-figures
python scripts/make_matched_configs.py freeze --from outputs/matched_pmf_r2 --grid outputs/matched_pmf \
    --models pmf --out configs/experiments_matched_frozen100_pmf.yaml
python run.py --config configs/experiments_matched_frozen100_pmf.yaml --models pmf --run-id matched_final_pmf --no-figures
python scripts/paired_stats.py --matched outputs/matched_final_pmf --benchmark outputs/<benchmark run>
```

`freeze` replaces `hpo.py finalize` here (which expects all six methods per cell) with the
same rule, the lowest LPIPS inside the screened grid, and refuses a winner whose pair or
terminal planner is not what its cell was screened with. `paired_stats.py` reports
measurement RMSE as a job-level difference with no interval: `run.py` records it once per
job, so it cannot be resampled over images.

## Flower

Flower (`src/flower.py`) runs on the two pixel priors only, at the paper's `gamma = 0` and CG
tolerance (relative residual 1e-5). Two knobs are tuned:

* `num_flower_steps` over {50, 100, 200, 500}, on every task;
* `flower_cg_max_iters` over {50, 100}, on **deblurring only**. 50 is the paper's cap; at the
  benchmark's sigma = 0.05 the early blur solves need about 60-80 conjugate-gradient
  iterations to reach 1e-5. Every other task's refinement is closed form, so the cap does not
  exist there: `hpo.py` keeps it out of those cells' keys (`KNOB_PROBLEMS`) and never writes
  it into their configurations.

Each prior is screened and frozen from its own runs. `hpo.py` ranks and proposes Flower like
any method (`KNOBS["flower"]`), but `finalize` still requires exactly the six benchmark
methods, so the winners are frozen by `make_flower_configs.py freeze` with the rule Task 2
uses. Lock **both** knobs in `propose`: locking `num_flower_steps` alone still lets an edge
push move the CG cap to 200.

```bash
python scripts/make_flower_configs.py screen
python run.py --config configs/experiments_flower_screen_smoke.yaml --models pmf --run-id flower_smoke_pmf
python run.py --config configs/experiments_flower_screen.yaml --models pmf --run-id flower_pmf --no-figures
python scripts/hpo.py propose --from outputs/flower_pmf --models pmf --images 8 --no-paper \
    --lock flower=num_flower_steps,flower_cg_max_iters --out configs/flower_round2_pmf.yaml
python run.py --config configs/flower_round2_pmf.yaml --run-id flower_pmf_r2 --no-figures
python scripts/make_flower_configs.py freeze --from outputs/flower_pmf_r2 --grid outputs/flower_pmf \
    --models pmf --out configs/experiments_flower_frozen100_pmf.yaml
python run.py --config configs/experiments_flower_frozen100_pmf.yaml --models pmf --run-id flower_final_pmf --no-figures
```

The smoke variants put each prior's frozen SDEdit next to Flower, so the
`shared_initial_state` check compares two methods. `freeze` refuses a winner outside
{50, 100, 200, 500}, a deblurring cap outside {50, 100}, a cap other than 50 off deblurring,
or a non-default `flower_gamma` or CG tolerance, and it writes the selected deblurring cap
into the frozen configuration.

## Fingerprint check against the benchmark

Before any comparison with the original benchmark, check that the Flower and matched-budget
runs saw the benchmark's measurements and started from its initial states:

```bash
python scripts/check_fingerprints.py --benchmark outputs/<benchmark run> \
    --run outputs/flower_final_pmf --run outputs/matched_final_pmf
```

It compares `measurement_id` and `guide_id` (fingerprints of the whole y and g(y)) with the
benchmark jobs of the same problem instance and image count, and `initial_state_fingerprint`
(the first batch's z_t0) with the benchmark jobs of the same prior, t0, seed, replicate,
batch size and image count. The frozen configs use the benchmark's batch sizes, so every job
has a counterpart. It exits non-zero, naming each job, on any mismatch, on a job with no
benchmark counterpart (a 2-image smoke run, say), on a job that recorded no fingerprint, and
on an internally inconsistent benchmark. Aggregate sharded runs (`run.py --aggregate`) first.

## Timing and peak memory (rebuttal Task 1)

```bash
python scripts/make_timing_config.py            # writes configs/experiments_rebuttal_timing_<prior>[_smoke].yaml
python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf.yaml \
    --run-id timing_pmf --estimate                # wall-clock estimate and suggested --time
python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf_smoke.yaml \
    --run-id timing_pmf_smoke --with-checks       # smoke, same GPU type
python scripts/run_timing.py --config configs/experiments_rebuttal_timing_pmf.yaml --run-id timing_pmf
python scripts/summarize_timing.py outputs/timing_pmf outputs/timing_jit outputs/timing_imf outputs/timing_sit
```

On the cluster, `bash submit_timing.sh <prior> --time HH:MM:SS [--smoke]` submits
`slurm/run_timing.sh` for one prior on one GPU. Like `submit.sh` it takes the account, memory
and CPUs from `cluster.env`; the GPU defaults to `a100:1` (Table 29's A100-SXM4-40GB),
overridable with `MPCFLOW_TIMING_GPU` or `--gpu`. The run id is stable (`timing_<prior>`), and
`run_timing.py` skips every configuration already complete in the run directory, so
resubmitting the same command after a timeout runs only what is left.

Each configuration gets its own process, because a framework never returns reserved GPU
memory while its process lives: in a long multi-job process the NVML number of a job is the
pool left by every job before it. The driver sets `XLA_PYTHON_CLIENT_ALLOCATOR=bfc` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` for its child processes only. The primary memory
figure is the allocator peak: PyTorch `max_memory_allocated` over the timed region, exact;
JAX exact when `peak_bytes_in_use` rose during the timed region, otherwise a bracket from the
largest `bytes_in_use` sampled during the region (sampled in the NVML loop) to the process's
lifetime `peak_bytes_in_use`, flagged uninformative when that upper bound equals the
model-only baseline's load peak. The NVML figure is the process's reserved pool plus CUDA
context. `runtime.timing_detail: true` (src/timing.py) adds a device-synchronised per-image
timer and allocator, pool, hardware and version records to each job's `metadata.json`;
results.csv is unchanged.

## Matched-time comparison on the pixel priors (rebuttal Task 5)

Every method gets RHSO's measured wall-clock per image on the same prior and task. The knob
values (PnP-Flow `num_pnp_steps`, Flower `num_flower_steps`, D-Flow `num_opt_steps`,
MPC-RHC `n_ctrl`) are fixed in `make_matched_time_config.py`; each entry is its prior's
frozen entry with only that knob changed, at the benchmark batch size. MPC-RHC ran at a
smaller batch than the other three, so it sits in `<task>__matched_time_rhc`.

```bash
python scripts/make_matched_time_config.py     # quality + timing configs and smoke variants
python run.py --config configs/experiments_matched_time_pixel_smoke.yaml --run-id matched_time_smoke
python run.py --config configs/experiments_matched_time_pixel.yaml --run-id <id> --no-figures
python scripts/run_timing.py --config configs/experiments_rebuttal_timing_matched_time.yaml \
    --run-id timing_matched_time               # or: bash submit_timing.sh matched_time ...
python scripts/matched_time_table.py \
    --quality outputs/<matched-time run> --quality outputs/run_9187465 --quality outputs/run_9191132 \
    --quality outputs/run_9187499 --quality outputs/run_9187476 \
    --timing outputs/timing_pmf --timing outputs/timing_jit --timing outputs/timing_matched_time
```

The timing config holds both priors, so `run_timing.py` runs one model-only baseline process
per prior (`--models`). In the table, RHSO is the frozen RHSO on pMF and the matched-BP
(direct planner) RHSO on JiT; MPC-Δt is Direction B on pMF and the frozen MPC-Δt on JiT.
Every other method shows its frozen and matched-time entries, and the one with the lower
LPIPS among those within 10% of RHSO's time (`--tolerance`) is marked. Differences are
RHSO − method with `paired_stats.py`'s bootstrap. A cell provided by two runs, a timing job
whose configuration differs from its quality job, or unpaired image sets stop the script.

## Cheap and mid budget levels on the pixel priors (rebuttal Task 6)

The reverse of Task 5: the backprop methods get the measured time of the cheap methods
(cheap level) and 4× that (mid level, no deblurring). The step values (RHSO `num_rhso_steps`,
`num_opt_steps`; MPC-Δt `n_ctrl` with N fixed; D-Flow `num_opt_steps`; MPC-RHC `n_ctrl`) are
fixed in `make_budget_level_config.py`. Each entry is its prior's own entry (pMF RHSO:
`<task>__rhso` of the final benchmark; JiT RHSO: `<task>__rhso_matched`, direct planner and
Heun; MPC-Δt and D-Flow: `<task>__core`; MPC-RHC: `<task>__rhc`) with only that knob
changed, at the source's batch size: `<task>__budget_<level>`, plus `_rhc` for MPC-RHC and
`_rhso` for JiT's RHSO, whose batch sizes differ. 66 entries.

```bash
python scripts/make_budget_level_config.py     # quality + timing configs and smoke variants
python run.py --config configs/experiments_budget_levels_pixel_smoke.yaml --run-id budget_levels_smoke
bash submit.sh --array 4 --config configs/experiments_budget_levels_pixel.yaml --gpu h100:1 --time <HH:MM:SS>
bash submit_timing.sh budget_levels --gpu h100:1 --time <HH:MM:SS>   # run id timing_budget_levels
python scripts/budget_level_table.py \
    --quality outputs/<budget-level run> --quality outputs/run_26548623 --quality outputs/run_26548625 \
    --quality outputs/run_9187465 --quality outputs/run_9191132 --quality outputs/run_9187499 \
    --quality outputs/run_9187476 \
    --timing outputs/timing_pmf --timing outputs/timing_jit --timing outputs/timing_matched_time \
    --timing outputs/timing_budget_levels
```

The budgets (seconds per image, per prior, level and task) are hard-coded in
`budget_level_table.py`; a third level is RHSO's own Task 5 time (pMF frozen RHSO, JiT
matched-BP direct-planner RHSO), read from the timing runs. Per (prior, task, level), every
method gets its lowest-LPIPS run among ALL its runs within 1.1× the budget (`--tolerance`),
else its cheapest run marked "over budget (×ratio)"; on JiT only direct-planner RHSO runs
count. A quality job is paired with the timing job whose computation fields agree with it.
Differences are method − (the level's best in-budget method) with `paired_stats.py`'s
bootstrap; an in-budget method whose ΔLPIPS interval contains 0 ties with the best, and the
best wins significantly when nobody ties. Outputs: `budget_levels.csv`,
`budget_levels_summary.csv`, `budget_levels_wins.csv` and `budget_levels.tex`.

## Budget levels with scaled learning rates (rebuttal Task 7)

Task 6 cut the steps but kept the tuned learning rates. Task 7 reruns the same 66 entries with

    lr_new = lr_tuned × (tuned knob / new knob)

(knob: RHSO `num_opt_steps` M, MPC-Δt `n_ctrl`, D-Flow `num_opt_steps`, MPC-RHC `n_ctrl`;
tuned values from the entry's source entry). Everything else, every step count and batch size
included, is Task 6's; `make_budget_level_lrscaled_config.py` builds its entries from Task 6's
generator. Blocks are Task 6's names plus `_lr`. There is no timing run: lr does not change the
time per image, so the table reuses `timing_budget_levels`.

```bash
python scripts/make_budget_level_lrscaled_config.py      # quality config + smoke variant
bash submit.sh --array 4 --config configs/experiments_budget_levels_lrscaled_pixel.yaml --gpu h100:1 --time <HH:MM:SS>
python scripts/budget_level_table.py --timing-ignore lr --quality outputs/<lr-scaled run> ...   # as Task 6
```

`--timing-ignore lr` leaves lr out of the quality-to-timing pairing (when several timing jobs
then agree it prefers the same variant, then the Task 6 variant). The `_lr` runs are labelled
"cheap level, lr scaled" and "mid level, lr scaled". Without the option nothing changes.

## RHSO anchor schedules (rebuttal Task 8)

RHSO's stage-k weight becomes `mu_k = mu0 * m(t_k)` (`mu_schedule`: `constant` m = 1, the
default; `increasing` m = (t² + (1−t)²)/(1−t)²; `decreasing` its reciprocal; t_k = grid[k]).
On pMF every task retunes mu0 per schedule over {0.01, 0.03, 0.1, 0.3, 1, 3, 10} plus one
shared mu0 = 0 run, with every other field copied from the task's pMF `__rhso` entry of
`experiments_final_frozen100.yaml`, on the first 8 images of the tuning pool (batch 4).

```bash
python scripts/make_anchor_schedule_tuning.py           # tuning config (110 jobs) + 2-image smoke
python scripts/freeze_anchor_schedules.py --from outputs/<tuning run> --out-dir outputs/anchor_schedule_tuning_curves
python scripts/anchor_schedule_table.py --final outputs/<final run> --reference outputs/run_9187465     --out-dir outputs/anchor_schedule_table
```

`freeze` picks, per (task, schedule), the lowest-LPIPS mu0 (ties to the smaller mu0; mu0 = 0
is a candidate of every schedule) and writes `configs/experiments_anchor_schedule_frozen100.yaml`
(100 images, benchmark pool and pMF batch sizes). Schedules of one task that all select
mu0 = 0 share one run, in a block named for each of them
(`<task>__anchor_constant_increasing`). It also writes the LPIPS / PSNR tuning curves as CSV
and LaTeX. The table reports, per task and schedule, mu0, PSNR / SSIM / LPIPS, the job-level
measurement RMSE next to σ and `schedule − constant` with `paired_stats.py`'s bootstrap
(10,000 resamples, seed 20260903), plus the paper's frozen RHSO from run_9187465 as unpaired
means (it ran on a different GPU type).
