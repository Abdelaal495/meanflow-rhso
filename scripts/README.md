# scripts/

| script | what it does |
|---|---|
| `build_local_imagenet_pool.py` | builds the image folder `data.source: local_folder` reads. `--frozen-manifest benchmarks/imagenet100_c42_i43/manifest.csv` rebuilds the paper's 100-image benchmark from an ungated mirror |
| `hpo.py` | the hyperparameter search: `propose` the next round of candidates from a finished one, `finalize` the winners into the benchmark configuration |
| `make_matched_configs.py` | matched-budget RHSO vs MPC-Δt: `screen` writes the screening grid (one block per budget-defining pair), `freeze` writes the 100-image evaluation from the tuning runs; each with a 2-image smoke variant |
| `paired_stats.py` | paired RHSO − MPC-Δt differences per prior and task, with bootstrap 95% intervals, as CSV and LaTeX rows |
| `make_flower_configs.py` | Flower on pMF / JiT: `screen` writes the screen (`num_flower_steps` 50/100/200/500, plus `flower_cg_max_iters` 50/100 on deblurring; 4 images), `freeze` writes the 100-image evaluation from the tuning runs; each with a 2-image smoke variant |
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
