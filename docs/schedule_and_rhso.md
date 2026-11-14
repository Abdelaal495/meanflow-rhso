# The `beta` time schedule, and RHSO

Two additions, documented together because RHSO uses the schedule and the schedule is
available to every method including RHSO.

---

## A. The universal power-law time schedule

Every strategy discretises the canonical interval `[s0, 1]`, where `s0 = 1 - t0`
(`s = 0` noise, `s = 1` data). That discretisation used to be uniform. It is now

```
s_k = s0 + (1 - s0) · (k / N)^beta,        k = 0 … N
```

with one new shared hyperparameter:

| `beta` | effect |
|---|---|
| `< 1` | large steps early, progressively **smaller** steps near clean space — more resolution where the image is formed |
| `= 1` | **exactly** the legacy uniform schedule |
| `> 1` | small steps early, progressively **larger** steps near clean space — more resolution near the noisy start |

Worked examples at `s0 = 0`, `N = 4`:

```
beta = 0.5    0, 0.5,    0.7071, 0.8660, 1
beta = 1      0, 0.25,   0.5,    0.75,   1
beta = 2      0, 0.0625, 0.25,   0.5625, 1
```

`beta` controls **time discretisation only**. It is not a step count, it never changes how
many model evaluations a method performs, and it is **not** PnP's `alpha` — that scales the
data-consistency step size `γ_k = γ0·(1 − s_k)^α` and keeps its meaning unchanged. The two
are deliberately never coupled: `beta` moves *where* the corrections happen, `alpha` shapes
*how large* they are at whatever times result.

A power-law family of this shape was studied in the Flower paper's time-discretisation
ablation, where an exponent equivalent to `beta = 0.5` behaved favourably on inverse
problems at low step counts. Applying it uniformly to SDEdit, MPC, PnP, D-Flow and RHSO is
this repository's generalisation — Flower proposes neither that generalisation nor RHSO.

### One implementation

`src/schedule.py` owns the schedule and nothing else:

```python
canonical_time_grid(s_start, steps, beta)   # steps + 1 points, s_start … 1
interior_time_grid(s_start, n, beta)        # n points strictly inside (s_start, 1)  [PnP]
grid_intervals(grid)                        # the ACTUAL dt_k = s_{k+1} - s_k
grid_metadata(grid, beta)                   # what a result row should record
spec_beta(spec)                             # spec.beta, or 1.0 if the field is absent
```

`(k/N)^beta` appears nowhere else. `canonical_time_grid` validates `steps ≥ 1` and
`beta` finite and `> 0`, returns exactly `steps + 1` points, starts exactly at `s_start`,
ends exactly at `1.0`, and is strictly increasing (an exponent so extreme that two times
collapse is an error, not a silently degenerate grid).

**`beta == 1.0` takes the original linear code path verbatim**, so old experiments
reproduce the same floating-point grid *bitwise*, not merely to within a rounding error.
`sdedit.canonical_time_grid` remains importable and now resolves to this implementation.

### Per method

| method | what `beta` shapes | what it does **not** shape |
|---|---|---|
| `sdedit` | the `steps` execution intervals; each solver step uses its own `dt_k` | solver arithmetic (Euler / Heun / RK4 and the adapter's final-step policy) |
| `dflow` | the differentiable trajectory grid; gradients still flow through all of it | anything else; at `steps = 1` `beta` is inert |
| `mpc_rhc` | **only** the outer receding-horizon execution grid (`N = num_mpc_steps`) | the internal `K`-step planning discretisation, still uniform over the remaining horizon `1 − s_k` |
| `mpc_delta_t` | the outer execution grid; every `dt` is the local `s_{k+1} − s_k` | — |
| `pnp` | the `N` correction times, still strictly inside `(s0, 1)` | `alpha`, `gamma0`, and the reprojection noise seeding |
| `rhso` | the outer stage grid, and hence the standard-flow planning suffix | — |

`beta` and `K` are not confounded on purpose: **`beta` says how often the controller
replans and how much it commits; `K` says how finely it plans.**

### The end of the constant `delta`

A non-uniform trajectory has no single physical step size. Concretely:

* nothing may use `spec.delta` (nominal `t0/N`) as an actual `dt` when `beta ≠ 1`;
* `MPC-Δt`'s `delta_t_lambda_scaling: inverse_delta` is now **step-dependent**,
  `λ_eff,k = λ / dt_k` (`src.mpc.inverse_delta_lambda`). With `beta = 1` and a planner-
  supplied `spec.delta`, that scalar is used verbatim, so legacy runs are bitwise
  unchanged. With scaling disabled, `λ` stays `λ`;
* `checks.check_time_grid` no longer asserts `dt = t0/N`. It asserts the endpoints,
  strict monotonicity, uniformity **where `beta = 1`**, and the correct interval direction
  otherwise;
* `schedule.grid_metadata` reports `delta: null` for `beta ≠ 1` and adds
  `delta_nominal_uniform`, `delta_min`, `delta_max`.

The job-defining quantities are now **`t0`, `N`, `beta`**.

If `beta` is swept where the relevant trajectory has a single interval (`steps: 1`,
`num_mpc_steps: 1`, `num_rhso_steps: 1`), it has no mathematical effect — a warning, not an
error.

---

## B. RHSO — Receding-Horizon State Optimization

Method key `rhso`, implemented for **both** dynamics families in `src/rhso.py`.

```
at s_k, holding the current state x_k:

    anchor = stop_gradient(x_k)              ← FIXED for the whole stage
    q⁽⁰⁾   = x_k
    for j = 0 … M-1:
        fidelity = Φ( to_pixels( G_{s_k → 1}( q⁽ʲ⁾ ), differentiable=True ) )
        R        = Σ_b  ‖q⁽ʲ⁾_b − anchor_b‖²  /  (2 d_b)
        loss     = fidelity + μ · R
        q⁽ʲ⁺¹⁾  = AdamUpdate( q⁽ʲ⁾, ∂loss/∂q )
    q* = q⁽ᴹ⁾

    x_{k+1} = G_{s_k → s_{k+1}}( q* )        ← only ONE interval is executed

then discard the problem and start again at s_{k+1}, with a FRESH anchor = x_{k+1}
```

with `N = num_rhso_steps`, `M = num_opt_steps`, `μ = mu` and the outer times from the
`beta` schedule. In one line: **optimise the current state using its terminal prediction →
execute one scheduled interval → replan.**

The variable being optimised is the **current generative state itself**. There is no
control `u`, no control penalty, no `λ`, no `K`, no `τ` and no adaptive step size.
`μ = 0` (the default) is the plain terminal-fidelity objective and is exactly what RHSO did
before the penalty existed.

### State-anchor regularisation (`mu`) — optional, off by default

Pure terminal fidelity places no cost on *where* `q` goes: it may drift far from the state
the generative trajectory actually reached and still score well on the measurement. The
failure mode this addresses is empirical — pushing the inner optimisation harder keeps
improving measurement consistency while PSNR / SSIM / LPIPS start to degrade. `mu > 0` adds
a trust-region-like penalty on that displacement:

```
R(q, x_k) = Σ_b  1/(2 d_b) · ‖q_b − x_{k,b}‖²

    squared difference  →  MEAN over the non-batch state dimensions  →  × ½  →  SUM over the batch

L_k(q) = Φ( G_{s_k → 1}(q) )  +  μ · R(q, x_k)
```

* **The anchor is `x_k`**: the state at the *start of that outer stage*, detached. It is
  **not** the trajectory origin `x_0`, and it is **not** a moving average — it does not
  follow `q`. Stage `k+1` takes a fresh anchor from the state actually executed.
* **Batch sum, not batch mean**, matching the repository's per-measurement fidelity: one
  sample's gradient never depends on the batch size.
* **Divided by `d_b`**, the number of scalar state dimensions per sample, so a useful `μ`
  does not scale with image or latent dimensionality — a pixel model and a latent model are
  on comparable footing. An unnormalised `‖q − x_k‖²` would not be.
* **No `σ²`, `dt` or `beta`.** By default (`mu_schedule: constant`) `μ` is also the same at
  every stage; the optional schedules below make it time dependent.
* The penalty contains **no generative-model evaluation**, so every compute counter and
  every dry-run cost estimate is unchanged (verified by a check, not asserted).

### Anchor schedules (`mu_schedule`) — optional, `constant` by default

Stage `k` uses `μ_k = μ₀ · m(s_k)`, with `μ₀ = mu` and `s_k = grid[k]` the stage's time on
the canonical clock (0 = noise, 1 = data):

| `mu_schedule` | `m(t)` | at `t0 = 1`, `β = 1`, N = 8: `t_k = k/8` |
|---|---|---|
| `constant` (default) | `1` | 1 at every stage |
| `increasing` | `(t² + (1−t)²) / (1−t)²` | 1, 1.0204, 1.1111, 1.36, 2, 3.7778, 10, 50 |
| `decreasing` | `(1−t)² / (t² + (1−t)²)` | the reciprocals |

Exact formulas, no clipping or smoothing; RHSO only optimises at `s_k < 1`, so both are
finite. `constant` (or the field absent) runs the original code path statement for
statement and reproduces its outputs bit for bit. On the JAX loop the non-constant schedules
pass `μ_k` into the differentiated objective as an array argument, so every stage runs the
same computation on new data. The schedule changes no count, grid or execution step, joins
the job id and the output directory (`musched=...`) only when it is not `constant`, is part
of `warmup_key`, and is the `mu_schedule` column of `results.csv`; a non-constant job also
records its per-stage `μ_k` under `rhso_mu_schedule` in `metadata.json`. With
`record_loss_history`, `loss = fidelity + μ_k · R` holds per stage.

`μ` is *not* MPC's `λ`. MPC's `λ` weights the magnitude of an **added control signal** `u`
in a dynamics that RHSO does not have; `μ` weights a **displacement of the state** from
where the trajectory actually was. They are different quantities on different objects, with
different units, and are not interchangeable — which is why the field is separately named
and rejected outside RHSO.

This is an **experimental extension** motivated by observed over-optimisation. It is not
claimed to be theoretically required, and no value of `μ` is claimed to be optimal; `μ = 0`
is a first-class point of the sweep and remains the default.

### The two families differ in the planner, not in the method

**Standard flow (JiT, SiT).** `G_{s_k → 1}` has to be integrated: a differentiable
fixed-step solve over the **remaining suffix of the same outer grid**,
`[s_k, s_{k+1}, …, 1]`, using the repository's existing solver semantics — Euler / Heun /
RK4 and the adapter's own final-step policy, via the shared
`dflow.integrate_flow` / `dflow.flow_step`. No second planning-resolution hyperparameter is
introduced, so planning cost shrinks as `k` grows. Execution then applies **one**
`flow_step` over `[s_k, s_{k+1}]`, under `no_grad`; the Heun→Euler fallback applies only to
the interval that genuinely lands on `s = 1`.

**MeanFlow (pMF, iMF).** The planner is **one direct learned finite-interval transition**
`T_θ(q; s_k → 1)`. The remaining outer intervals are *not* composed for the objective, no
ODE is constructed and no instantaneous velocity is inferred. Execution is exactly one
transition `T_θ(q*; s_k → s_{k+1})`. One forward and one backward per inner iteration at
every `k`, which is what makes MeanFlow models computationally well suited to a
receding-horizon state optimiser — RHSO is available to standard flows, it is simply
dearer there.

Both paths share the same objective, the same fidelity `Φ`, the same
`to_pixels(differentiable=True)` and the same adapter abstraction. Latent models (iMF, SiT)
differentiate through the VAE decoder; there is **no separate pixel/latent RHSO
mathematics** and no model-name special-casing.

### Contracts

* The inner objective path `q → planner → native terminal state → to_pixels(differentiable=True) → A(x) → Φ`
  contains no NumPy conversion, no `detach`, no PIL and no stop-gradient. Model parameters
  stay frozen; only `q` moves.
* **Adam state is rebuilt at every outer stage.** Moments are never carried from `s_k` to
  `s_{k+1}`: the optimisation problem changes when the terminal transport map changes.
* Execution is inference only — no graph is carried into the next outer stage.
* The executed interval always uses the **final post-update `q*`**. The last inner
  objective was evaluated *before* the last Adam update, so its terminal prediction belongs
  to a state that no longer exists and is never reused.
* The anchor is a **stop-gradient constant for the whole stage**: `x_anchor = x.detach()`
  in torch, `jax.lax.stop_gradient(x)` closed over by `loss_fn` in JAX. It never receives a
  gradient or an update. Both families call the same
  `rhso.state_anchor_penalty(B, q, anchor)`, written against the `Backend` abstraction, so
  the normalisation is shared by construction rather than transcribed twice.

### RHSO vs D-Flow vs MPC

| | what is optimised | what is executed after optimising |
|---|---|---|
| **D-Flow** | one starting/intermediate state, globally | the **whole** planned trajectory |
| **MPC** | explicit control variables `u` added to the dynamics, with a control penalty `λ‖u‖²` | one interval, then replan |
| **RHSO** | the **current generative state**, optionally with `μ·R` anchoring it to the state the trajectory reached | one interval, then re-optimise from the state actually reached |

RHSO is *terminal planning + state optimisation + partial execution + replanning*. It is
not implemented by calling D-Flow in a loop; it has its own reconstruction implementation
and reuses only low-level numerical helpers.

### Configuration

```yaml
rhso: {t0: 0.8, beta: [0.5, 1.0, 2.0], num_rhso_steps: 4, num_opt_steps: 10, lr: 0.01}
```

Sweeping the state-regularisation weight (the values below are **syntax, not
recommendations** — nothing here is tuned):

```yaml
rhso:
  t0: 1.0
  beta: 0.5
  num_rhso_steps: 4
  num_opt_steps: 20
  lr: 0.02
  mu: [0.0, 0.01, 0.1, 1.0]        # 0.0 = the unpenalised objective
  optimizer: adam
  phi_normalization: half_mean_squared_per_measurement
```

| scope | fields |
|---|---|
| shared | `t0`, `beta` |
| `rhso` | `num_rhso_steps`, `num_opt_steps`, `lr`, `mu`, `optimizer`, `phi_normalization` |
| `rhso` on a standard flow only | `solver` (same allowed values and semantics as the other standard-flow methods) |

`mu` must be finite and `>= 0`; it defaults to `0.0`, takes part in the job id, and appears
in the output directory name only when it is non-zero (so paths written before the field
existed are unchanged). It is rejected for every other method — including with MPC's `lam`
present in the same file — because no other method here optimises a state in place.

`optimizer: adam` is the only supported value in this iteration. `solver` is meaningless for
MeanFlow RHSO and must be rejected there, exactly as it is for the other MeanFlow methods.
Deliberately **absent**: `lam`, `control_cost_normalization`, `K`, `warm_start`, `tau`,
`alpha_min`, adaptive horizons, adaptive `mu` (beyond the fixed `mu_schedule` shapes above),
and any new noise hyperparameter.

### Cost accounting

Counted honestly per the repository's existing distinctions — one outer stage is **not**
one model evaluation.

**MeanFlow**, with `N` stages and `M` inner iterations:

```
planning model evaluations = N · M
generative backprops       = N · M
optimizer iterations       = N · M
objective evaluations      = N · M
model evaluations total    = N · (M + 1)
```

The `+1` per stage is the execution transition for `q*`.

**Standard flow**: planning cost shrinks with `k`, so it is computed from the solver's real
stage counts over each remaining suffix, plus one executed interval per stage:

```
planning = Σ_k  M · ( (N − k) · stages(solver) − [heun && euler_final_step_for_heun] )
execution = Σ_k ( stages(solver) − [heun && euler_final_step_for_heun && k = N−1] )
```

`rhso.rhso_cost_estimate(values, dynamics_family, euler_final_step_for_heun)` returns
exactly these numbers and is the function `config._estimate_cost` calls, so the dry-run plan
and the measured counters cannot drift apart.

**`mu` does not appear anywhere above, and that is the point.** The state penalty is a
handful of elementwise tensor operations on `q` and a constant; it evaluates no generative
model, so planning evaluations, total model evaluations, generative backprops, optimizer
iterations, objective evaluations and the dry-run estimate are all identical at every `mu`.
A regression check runs the same job at `mu = 0` and `mu > 0` and compares every counter.

### Diagnostics

The scalar Adam differentiates is the **total** objective `Φ + μ·R`, and `loss_history`
records exactly that — so at `μ = 0` the history is what it always was. Two parallel lists
in `ReconstructionStats` split it, on the same per-image scale:

| field | meaning |
|---|---|
| `loss_history` | the optimised total `Φ + μ·R` |
| `fidelity_history` | the terminal measurement fidelity `Φ` alone |
| `state_penalty_history` | the **unweighted** displacement `R` (multiply by `mu` from the result row for the weighted contribution) |

`loss_history[i] == fidelity_history[i] + mu · state_penalty_history[i]` for every recorded
iteration. `R` is stored unweighted so a `mu` sweep can be re-weighted after the fact. All
three land in the per-job `results.npz` when `record_loss_history` is on, and are empty for
every method that has no such split. `mu` itself, and where it came from, are columns in
`results.csv`.

---

## C. How it is wired into `config.py` and `run.py`

**`src/config.py`**

* `beta` joins `t0` in `SHARED_FIELDS`, so it is declared for **every** method; it is in
  `SWEEPABLE_FIELDS` and `BUILTIN_DEFAULTS` (`1.0`), and it is a `JobSpec` field.
* Validated through `schedule.resolve_beta` (finite, `> 0`); a `beta` sweep on a
  single-interval trajectory **warns** rather than fails.
* `beta` and `num_rhso_steps` take part in `job_id`, and `beta` appears in `leaf_dir`
  whenever it is not 1 — so two runs differing only in `beta` can never share an output
  directory or a plotting group.
* `METHOD_DECLARATIONS["rhso"]` declares
  `t0, beta, num_rhso_steps, num_opt_steps, lr, mu, optimizer, phi_normalization, solver`.
  `mu` is sweepable, defaults to `0.0` with provenance `builtin` (the feature *off*, not an
  untuned guess), takes part in `job_id`, appears in `leaf_dir` and the figure label only
  when non-zero, and is rejected for every other method.
  `lam`, `control_cost_normalization` and `K` are rejected with an explanation;
  `optimizer` is restricted to `adam` by `VALID_RHSO_OPTIMIZERS`; `solver` is accepted only
  where the model's capabilities allow one, so a MeanFlow RHSO job refuses it exactly as
  the other MeanFlow methods do. Every entry of `MODEL_CAPABILITIES` supports `rhso`, and
  `rhso` is in `COMPARED_METHODS`.
* `_estimate_cost("rhso", …)` delegates to `rhso.rhso_cost_estimate`, so the dry-run plan
  and the measured counters cannot drift apart.
* `delta` is now **nominal uniform spacing only, and `None` whenever `beta != 1`**. Three
  derived fields are recorded for every method instead:
  `delta_nominal_uniform`, `delta_min`, `delta_max`.
* The dry-run report prints `beta` on every method, `executed dt` as real min–max pairs,
  and reminders that `delta = t0/N` holds only at `beta = 1`.

**`run.py`**

* `warmup_key` includes `beta`, `num_rhso_steps` and `mu`: the first two change the resolved
  time grid and the third changes the objective that is traced and differentiated, so
  two such jobs never share a warm-up and neither compiles inside its measured region.
* `warm_up` reduces RHSO's inner Adam budget to one iteration and **never** reduces
  `num_rhso_steps` or `beta` — every outer stage traces a different computation.
* `RESULT_COLUMNS` gains `beta`, `num_rhso_steps`, `mu`, `hyperparameter_source_mu`,
  `delta_nominal_uniform`, `delta_min` and `delta_max`; `persist_job` writes
  `fidelity_history` and `state_penalty_history` beside `loss_history` in `results.npz`; `METHOD_ORDER` places RHSO after D-Flow; `run_metadata.json` records the
  time-schedule policy.

Nothing else in either file changed, and a configuration that mentions neither `beta` nor
`rhso` resolves exactly as it did before.

---

## D. Direct terminal planning and the theory-validation diagnostics

This section documents the RHSO additions made for the theory-validation experiments:
a configurable **terminal planner**, three families of **diagnostic**, the **batch-4**
memory/runtime interpretation, and the matched-budget experiment.

Nothing here changes what RHSO computes at the settings that already existed. A
configuration that mentions none of the fields below resolves to the same jobs, the same
job ids and the same output directories as before.

### D.1 What the inner objective predicts: `rhso_terminal_mode`

At outer stage `k`, one inner objective is

```
Φ( to_pixels( P(q ; s_k) ) )  +  μ · R(q, x_k)
```

and `rhso_terminal_mode` chooses `P`:

| value | `P(q; s_k)` | who |
|---|---|---|
| `auto` (default) | MeanFlow → `direct`, standard flow → `suffix` | reproduces the pre-field behaviour exactly |
| `direct` | ONE terminal prediction from the current state | pMF and JiT-direct |
| `suffix` | differentiable integration of `[s_k, …, 1]` | legacy JiT/SiT only; **rejected** for MeanFlow |

**pMF direct terminal planning.** `P = T_θ(q; s_k → 1)` — one evaluation of the model's
**learned finite-interval transport map**. This is what MeanFlow RHSO always did; the
field simply names it.

**JiT legacy suffix terminal planning.** `P = G(q; s_k → 1)`, a differentiable fixed-step
solve over the remaining suffix of the outer grid, with the repository's existing solver
semantics. Still available, still the default for a standard flow, and still what an old
config runs. It is **not** used by any new experiment.

**JiT new direct endpoint planning.** `P = x̂₁(q, s_k)` — JiT's own network output. JiT
predicts the clean image and *derives* its velocity as `v = (x̂₁ − x)/max(1−s, t_eps)`;
`JiTAdapter._guided_clean` is now the single authoritative implementation of that guided
prediction, and both `velocity()` and the new `clean_prediction()` call it. There is
therefore one copy of the classifier-free-guidance rule, the interval gating, the dtype
policy and the native-time mapping, and the planner cannot drift away from the sampler.
The clean prediction is **not** obtained by integrating anything.

> **Terminology, and it matters.** `T_θ` is a *learned family of finite-interval transport
> maps*. `x̂₁` is a *direct endpoint prediction* — a one-step terminal surrogate. Both give
> one differentiable terminal prediction per inner objective, and that shared cost profile
> is the point of the comparison, but they are **not** the same mathematical object and the
> repository never labels them as one: `rhso_terminal_planner` records
> `learned_finite_interval_map` or `direct_clean_endpoint_prediction` on every row.

**Execution is unchanged.** In both modes a stage still executes exactly one interval
`s_k → s_{k+1}` — one `flow_step` for a standard flow (including the Heun→Euler policy on
the interval that lands on `s = 1`), one `T_θ(q*; s_k → s_{k+1})` for a MeanFlow. This is
not a new sampler.

**Cost.** A direct objective costs **one terminal-planner evaluation** at every stage, for
both families. A model evaluation is not a network forward: classifier-free guidance still
costs two forwards per JiT prediction, and `network_forwards` is what records that.
`rhso.rhso_cost_estimate` covers all three cases, so the dry-run plan and the measured
counters cannot drift apart.

### D.2 Per-stage optimisation-efficiency diagnostics

`rhso_stage_diagnostics: true` records, for **every real image** and every stage `k`:

| field | meaning |
|---|---|
| `s_from`, `s_to` | `s_k` and `s_{k+1}` |
| `v_pre` | terminal fidelity at the state *entering* the stage, before any Adam update |
| `v_post` | terminal fidelity at the **final** `q`, after all `M` updates |
| `delta`, `delta_per_step` | `V_pre − V_post` and `Δ/M` |
| `theta` | `1 − V_post/V_pre`, NaN when `V_pre` is not positive and finite |
| `anchor_penalty_post` | the displacement `R(q*, x_k)`, whatever `μ` is |
| `stage_seconds` | the stage's **algorithm** time, with diagnostic time removed |

`V_post` is evaluated explicitly. It is **not** read off the last entry of `loss_history`:
that value was computed *before* the final Adam update and belongs to a state that no
longer exists. A test re-runs a stage by hand and confirms the two differ.

Per-image values come from `problems.make_phi_per_sample`, which is the same fidelity the
optimiser sums, written per batch element. `sum_b per_sample == make_phi` holds to
floating-point tolerance (tested for all four normalisations) and the per-sample vector is
never differentiated, so **no gradient anywhere changes**.

### D.3 Execution / replanning consistency

`rhso_consistency_diagnostics: true` measures what executing an optimised state does to the
predicted endpoint. At the optimised `q_k*`:

```
p_k = P(q_k*, s_k)          predict
x_{k+1} = execute one interval
r_k = P(x_{k+1}, s_{k+1})   re-predict from the state actually reached
```

`r_k` is exactly the next stage's `V_pre` prediction, so the pair costs no model call beyond
the two per stage the stage diagnostics already make.

**The measurement is the same; the claim is not.**

| model | `consistency_kind` | what `‖p_k − r_k‖` is |
|---|---|---|
| pMF, iMF | `meanflow_semigroup_defect` | two members of the **learned family of finite-interval maps** applied to the same trajectory: `T(·; s_k → 1)` against `T(·; s_{k+1} → 1) ∘ T(·; s_k → s_{k+1})`. A genuine semigroup defect. |
| JiT-direct | `terminal_prediction_inconsistency` | how far the model's **endpoint prediction** moves once the state is executed. `x̂₁` is a predictor, not a learned family of finite-interval maps, so calling this a semigroup defect would assert structure JiT does not have. |
| JiT-suffix | `suffix_integration_endpoint_shift` | mixes model error with solver discretisation error; reported for completeness only. |

The label and a one-paragraph explanation of it are written into every job's
`metadata.json`, so a later analysis cannot mislabel one as the other.

Recorded per image and stage: `endpoint_shift_l2`, `endpoint_shift_native_rmse`,
`endpoint_shift_native_relative`, `endpoint_shift_pixel_rmse` (both a normalised native
error and a pixel-space RMSE, so no cross-model comparison rests on a
dimensionality-dependent norm), `fidelity_shift_execution` (`V_pre(k+1) − V_post(k)`: what
execution and replanning did to the measurement fidelity of the predicted endpoint) and
`next_stage_recovery` (`Δ_{k+1}`: how much of that the next stage's optimisation recovered).

**The final stage has no successor.** Its consistency, `fidelity_shift_execution` and
`next_stage_recovery` entries are **NaN** by convention rather than fabricated.

### D.4 Jacobian anisotropy diagnostics, and their limits

`rhso_jacobian_diagnostics: true` probes the terminal planner's Jacobian at the state
**entering** each stage — never after an Adam step, and never during the inner loop:

```
pMF         J_k = D_x T_θ(x_{s_k}; s_k → 1)
JiT-direct  J_k = D_x x̂₁(x_{s_k}, s_k)
```

The full Jacobian is never formed (at 256×256×3 it has ~4.3·10¹⁰ entries). Only JVPs, VJPs
and their composition `JᵀJ` are used; both backends build **one** linearisation per stage
and reuse it (`jax.linearize` + `jax.vjp`; the version-independent double-backward trick in
PyTorch), so cost follows the probe budget rather than the state dimension.

**Batch isolation.** The theory concerns one image's Jacobian. Every tangent and cotangent
is masked to a single batch row and only that row of the output is read, so a batch of four
yields four per-image probes — never one condition number for the concatenated batch.
Padded rows are skipped.

**What is claimed, and what is not:**

* `sigma_max_estimate` — power iteration on `JᵀJ`. It estimates the **dominant singular
  scale and nothing else**. Power iteration says nothing about `σ_min`, so no condition
  number is derived from it.
* `gains` — the raw directional gains `g_i = ‖J r_i‖/‖r_i‖` for `rhso_jacobian_probes`
  deterministic random unit directions, kept in full alongside
  `gain_min/max/mean/std/p05/p50/p95`.
* `empirical_gain_ratio = p95/p05` and `empirical_log_anisotropy = log(p95/p05)` — an
  **empirical, sample-based** anisotropy statistic and a *lower bound* on the true condition
  number. Random directions concentrate in high dimension, so the smallest sampled gain is
  almost never near `σ_min`. Nothing here is called `sigma_min` or a condition number.
* These are Jacobians of the **learned** terminal maps. They are not the theoretical AGPP
  Jacobian, and the repository does not claim they are.

Settings: `rhso_jacobian_probes` (default 8), `rhso_jacobian_power_iters` (default 8),
`rhso_jacobian_seed` (default `20240917` — fixed, so a diagnostic run is reproducible).
Directions are seeded per `(seed, job, stage, row)`, so an image's probes do not depend on
which other images share its batch.

These are expensive. Keep them off for anything whose runtime or memory you intend to
report.

### D.5 Diagnostic cost is never algorithmic cost

Stage-end evaluations, consistency predictions, JVPs, VJPs and randomised probes are
measurements *about* the method, not compute the method spends. They are counted and timed
separately:

```
diagnostic_model_evaluations   diagnostic_network_forwards
diagnostic_jvps                diagnostic_vjps                diagnostic_seconds
```

`model_evaluations`, `planning_model_evaluations`, `network_forwards`,
`backprops_through_model` and **`runtime`** keep exactly their previous meaning:
`rhso._finalise` subtracts the diagnostic seconds, so the reported runtime stays *algorithm*
runtime rather than "algorithm + spectral analysis". A regression test runs the same job
with diagnostics on and off and asserts that all eight algorithmic counters are identical
and the reconstruction is bitwise the same.

### D.6 Batch size 4: what the memory and runtime numbers mean

The new suite runs `batch_size: 4` for both models. The optimisation objective is summed
over the batch, so a sample's own gradient is unchanged by its companions, and the final
short chunk still follows the repository's existing repeat-padding rule — padded rows never
appear as diagnostic samples.

`src/memory.py` now reports, in addition to the framework-specific columns it always had,
**one metric comparable across PyTorch and JAX**:

```
gpu_process_baseline_gib   gpu_process_peak_gib
gpu_process_incremental_peak_gib   gpu_process_memory_source
```

sampled through NVML, restricted to this process where the driver exposes per-process
accounting, with a **fresh sampler for every atomic job** and framework-correct
synchronisation at the boundaries. Model loading and the untimed warm-up are outside the
measured region.

* This is a **job peak at batch size 4**. It is not divided by the batch and it is not
  per-image memory. The runtime column at batch 4 is a throughput figure, **not**
  single-image latency.
* When NVML process measurement is unavailable the process fields report `unavailable`.
  JAX's `peak_bytes_in_use` is a **lifetime** high-water mark with no reset API — once an
  earlier job has peaked higher it reports that earlier job — and it is **never**
  substituted for the process metric. It remains available in its own clearly-labelled
  column. This substitution is what previously made pMF's memory look suspiciously
  constant across jobs.
* `gpu_memory_source` and `gpu_process_memory_source` record how each number was obtained.
  Never compare across sources.

### D.7 The matched-budget experiment

`configs/experiments_theory_validation_final100.yaml` holds the main suite: 100 images,
batch 4, both models, all five inverse problems, 70 atomic jobs.

At a fixed total optimisation budget `B = N·M = 160`, the only pairs are

```
(N, M) = (1, 160)   (2, 80)   (4, 40)   (8, 20)
```

Each pair is its own **named experiment block**. That is deliberate and is the least
invasive correct solution: `num_rhso_steps` and `num_opt_steps` are independent sweep axes,
so writing them as two lists in one block would expand to the 16-job Cartesian product
rather than these four matched-budget points. The sweep engine is unchanged; a test asserts
that every resolved pair satisfies `N·M = 160` and that none of the 12 spurious combinations
appears.

The `t0` ablation reuses the same block structure at the main setting (`N=4, M=40`) for
`t0 ∈ {0.8, 0.6, 0.4}`. `t0 = 1.0` is **not** repeated — it is already the `(4, 40)`
matched-budget job. Initialisation follows the repository's existing shared rule unchanged
(`z_t0 = (1−t0)·g(y) + t0·ε`, and the prior noise itself at `t0 = 1`); both `t0` and
`canonical_start_time = 1 − t0` are recorded on every row.

Learning rates are copied verbatim from `configs/experiments_imagenet100_final.yaml` and are
**not** retuned per `N`. A test compares all ten `(model, task)` rates against the frozen
config directly, so the two cannot silently diverge.

### D.8 Persistence

Per job, `results.npz` gains `[image, stage]` arrays — `stage_v_pre`, `stage_v_post`,
`stage_delta`, `stage_delta_per_step`, `stage_theta`, `stage_stage_seconds`, the endpoint
and fidelity-shift metrics, `stage_next_stage_recovery`, and the spectral fields including
`stage_jacobian_gains` of shape `[image, stage, probe]`. The `[image, stage]` structure is
never flattened.

`metadata.json` gains an `rhso_diagnostics` block naming the model, the terminal mode and
planner, the consistency kind **and its meaning in words**, the diagnostic settings and
seed, the batch size, `N`/`M`, `t0` and `canonical_start_time`, the outer times, the
diagnostic cost and both memory sources.

`results.csv` carries job-level **summaries** only (`stage_delta_first/last`,
`stage_theta_first/last`, the mean endpoint shift and recovery, the first/last
`jacobian_sigma_max` and log-anisotropy) plus the diagnostic counters. Large arrays stay in
the per-job artefact. Resume and `--aggregate` are unaffected.

---

## E. The two opt-in extensions added for the six theory experiments

Both are **off unless a configuration asks for them**, and both were added because the
experiments below could not otherwise be launched without editing Python. Everything else
in this document — the schedule, the planner, the objective, the solver policy, the cost
model, the existing diagnostics — is unchanged.

### E.1 `measurement_noise_group` — common random numbers across σ

**Where it goes.** On an experiment block, next to `problem` and `degradation`:

```yaml
experiments:
  denoising_s020_B160_N4:
    problem: denoising
    degradation: {sigma: 0.2}
    measurement_noise_group: theory_noise_sweep_v1
```

**What it does.** The measurement noise is normally seeded from

```
seed(global_seed, 'measurement', problem, params_key, image_id)
```

where `params_key` is the canonical parameter dictionary — **σ included**. A sweep over σ
therefore draws a different ε at every level, and the difference between two noise levels
mixes the effect of σ with the luck of the draw. With a group set, the seed becomes

```
seed(global_seed, 'measurement_paired', group, problem, structural_params_key, image_id)
```

where `structural_params_key` is the same canonical key **with σ removed**. Every σ in the
group then shares one standard-normal realisation per image and merely rescales it:

```
y_σ = A x* + σ ε,     the same ε ~ N(0, I) at every σ
```

**What still separates jobs.** σ remains in `params_key`, so it remains in the problem key
and in the job id: four noise levels are still four problem instances, four measurements
and four sets of jobs. Only the underlying ε is shared. The group itself joins the problem
key and the job id **only when it is set**, so a configuration that never mentions it keeps
every identity, output path and resume artefact it had before.

**What it deliberately does not touch.** Masks, stroke geometry and every operator are
seeded exactly as before, from the full `params_key`. One consequence must be stated
plainly: `random_inpaint`'s mask is drawn per image from a key that includes σ, so a σ
sweep on that task re-draws the mask and `(y − Ax)/σ` is **not** identical across the
sweep. The validator emits a warning saying so whenever a group meets `random_inpaint`.
Pairing is exact for denoising, deblurring, super-resolution and box inpainting, whose
operators do not depend on σ. Fixing this would mean changing existing mask seeding, which
is out of scope and would alter validated results.

**Auditing.** The group appears in `results.csv` as `measurement_noise_group`, and each
job's `metadata.json` records both the group and the seed recipe actually used under
`problem_metadata`.

### E.2 `rhso_gradient_authority_diagnostics` — task-aligned endpoint authority

**Where it goes.** On an RHSO method entry, like the other diagnostics:

```yaml
rhso: {num_rhso_steps: 4, num_opt_steps: 40, lr: 0.01,
       rhso_gradient_authority_diagnostics: true}
```

**What it measures.** §D.4's probes ask how the terminal map stretches *random* directions.
This asks how much authority the state at stage `k` has over the direction the *task*
actually cares about:

```
g_end_k = ∇_{x_1} Φ(x_1; y)   at  x_1 = P_k(x_{t_k})
J_k     = D_{x_{t_k}} P_k(x_{t_k})
A_k     = ‖J_kᵀ g_end_k‖ / ‖g_end_k‖
```

Because `J_kᵀ g_end_k` is exactly `∇_{x_{t_k}} Φ(P_k(x_{t_k}); y)`, **no Jacobian is ever
formed**: one forward pass through the terminal planner, one gradient of the fidelity at
its output, one VJP back through the planner. `P_k` comes from the same
`make_terminal_planner` factory the inner objective uses, so a pMF job measures its learned
finite-interval map and a JiT-direct job measures `x̂₁`.

Recorded per real image and per stage, into the existing `[image, stage]` arrays:

| field | meaning |
| --- | --- |
| `endpoint_fidelity_grad_norm` | `‖g_end_k‖` |
| `state_fidelity_grad_norm` | `‖J_kᵀ g_end_k‖` |
| `gradient_aligned_authority` | the ratio |
| `log_gradient_aligned_authority` | its log |

**Semantics that matter.**

* **Fidelity only.** The state-anchor `mu` penalty is excluded. It describes the
  optimisation problem, not the model's authority over the measurement — and at the state
  entering a stage its gradient is exactly zero anyway.
* **Measured at the pre-optimisation state** entering the stage, the same linearisation
  point as the Jacobian probe, so the two are directly comparable on the same run.
* **Zero-gradient handling.** When `‖g_end_k‖` falls below
  `GRADIENT_AUTHORITY_MIN_ENDPOINT_NORM` (1e-12) the ratio is undefined and is recorded as
  `NaN`, never as a division by an invented epsilon. Both norms stay on the row, and the
  summaries drop `NaN` rather than propagating it.
* **It is a measurement, not the algorithm.** Its model evaluation and VJP land in
  `diagnostic_model_evals` / `diagnostic_vjps`, its time is subtracted from the reported
  runtime, and it changes no sample, no optimiser state and no algorithmic counter —
  exactly the convention §D.5 sets out for the existing probes.
* **It implies the stage bookkeeping**, the same way a consistency measurement does, since
  its numbers live on the per-`(image, stage)` rows the stage bookkeeping creates. Asking
  for it alone therefore measures something rather than silently nothing.
