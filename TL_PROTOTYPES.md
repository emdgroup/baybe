# Transfer Learning Prototypes

This document summarizes the transfer-learning (TL) dispatching prototype implemented
on the `playground/tl-methods-prototypes` branch and records the design decisions
(including the alternatives that were considered and rejected).

## 1. Goal

Enable different transfer-learning strategies for Bayesian optimization that are
selected **solely** through the `TaskParameter`, without changing the way a user
builds a campaign. The following must keep working unchanged:

```python
task_param = TaskParameter(
    name="task",
    values=["source", "target"],
    active_values=["target"],
    override_transfer_learning_mode=TransferLearningMode.<MODE>,
)
searchspace = SearchSpace.from_product([num_param, task_param])
campaign = Campaign(searchspace, objective)  # default recommender + surrogate
```

Switching `<MODE>` must change the TL behavior while the default
`BotorchRecommender` + `GaussianProcessSurrogate` stack stays in place.

## 2. Transfer learning modes

`baybe.parameters.enum.TransferLearningMode` now defines:

| Mode | Status | Realized by |
|------|--------|-------------|
| `INDEX_KERNEL` | implemented | task kernel swap (`IndexKernel`) |
| `POSITIVE_INDEX_KERNEL` | implemented (default) | task kernel swap (`PositiveIndexKernel`) |
| `MEAN_TRANSFER` | implemented | dedicated surrogate (`MeanTransferSurrogate`) |
| `RESIDUAL_LEARNING` | implemented | `ResidualTransferSurrogate` (residual variance only, multi-source) |
| `RESIDUAL_LEARNING_WITH_UNCERTAINTY` | implemented | `ResidualTransferSurrogate` (full chain variance, multi-source) |
| `RGPE` | implemented | `RGPETransferSurrogate` (rank-weighted GP ensemble, multi-source) |

The two families are fundamentally different and therefore dispatch at **different
seams**:

- **Kernel-based modes** (`INDEX_KERNEL`, `POSITIVE_INDEX_KERNEL`) only change which
  task kernel goes into the ICM kernel of the *same* Gaussian process.
- **Surrogate-replacing modes** (`MEAN_TRANSFER`, `RESIDUAL_LEARNING`,
  `RESIDUAL_LEARNING_WITH_UNCERTAINTY`, `RGPE`) need an entirely different surrogate
  architecture that builds a target model from one or more source models.

## 3. Dispatching architecture

### 3.1 Case A - kernel dispatch (in the task kernel factory)

`_BayBETaskKernelFactory._make`
(`baybe/surrogates/gaussian_process/presets/baybe.py`) reads
`searchspace._task_parameter.override_transfer_learning_mode` and returns:

- `IndexKernel` for `INDEX_KERNEL`,
- `PositiveIndexKernel` for `POSITIVE_INDEX_KERNEL` or no override (unchanged default).

`GaussianProcessSurrogate` itself is untouched for these modes: the default
`BayBEKernelFactory` already composes the task kernel into the ICM kernel whenever the
search space contains a task parameter.

### 3.2 Case B - surrogate-replacing dispatch (delegation inside the GP)

Because the default surrogate is a `GaussianProcessSurrogate` and must remain so, the
GP performs the dispatch itself. In `GaussianProcessSurrogate._fit`
(`baybe/surrogates/gaussian_process/core.py`):

1. Read the task parameter's override mode.
2. If an override is set **and** a custom kernel was provided, raise
   `IncompatibleSurrogateError` (overrides only work with the default kernel).
3. If the mode maps to a surrogate-replacing class (via `_tl_replacing_surrogate`),
   construct and fit that delegate, store it in `self._delegate`, and return.
4. Otherwise, follow the normal single-model GP path (Case A handled downstream).

`_posterior` and `to_botorch` route to `self._delegate` when it is set. The registry
`_tl_replacing_surrogate(mode)` maps `MEAN_TRANSFER -> MeanTransferSurrogate`,
`RESIDUAL_LEARNING` / `RESIDUAL_LEARNING_WITH_UNCERTAINTY` ->
`ResidualTransferSurrogate` (with the appropriate `propagate_source_uncertainty` flag),
and `RGPE -> RGPETransferSurrogate`.

## 4. `MeanTransferSurrogate`

Location: `baybe/surrogates/transfer_learning/mean_transfer.py`.

Algorithm (single source, single target):

1. Split `measurements` by task value into a **source** subset (the non-active task
   value) and a **target** subset (the active task value).
2. Build a reduced, task-free search space via `SearchSpace._drop_parameters`.
3. Fit a single-task `source_gp` on the source subset over the reduced space.
4. Fit a single-task `target_gp` on the target subset over the reduced space, using
   `source_gp.posterior_mean_function` as its prior mean (this is the actual "mean
   transfer").
5. On prediction, strip the task column from the incoming full comp-rep candidates and
   query `target_gp` (see 5.3).

Validation is eager: the surrogate raises `IncompatibleSearchSpaceError` if there is no
task parameter, if there is not exactly one source and one target task value, or if
either subset has no measurements.

## 5. Design decisions

### 5.1 Keep `GaussianProcessSurrogate` as the default and let it dispatch
- **Decision:** The GP surrogate dispatches internally (kernel modes downstream in the
  factory; surrogate modes via an internal `_delegate`).
- **Rationale:** The requirement is that a default campaign works unchanged. Since the
  default surrogate is a `GaussianProcessSurrogate`, it is the only per-fit object that
  sees the search space and can dispatch.
- **Rejected alternative:** Making a separate dispatcher surrogate the global default.
  Cleaner separation, but it changes the identity of the default surrogate
  (serialization, `isinstance` checks, existing tests) - too invasive.

### 5.2 Kernel modes vs. surrogate modes dispatch at different seams
- **Decision:** Kernel choice lives in the task kernel factory; surrogate replacement
  lives in a dedicated surrogate reached via delegation.
- **Rationale:** Composition over monolith (AGENTS.md). Forcing both into
  `GaussianProcessSurrogate._fit` would turn it into a god-object.

### 5.3 Strip the task column in `_posterior` (not `FilterFeatures`)
- **Decision:** Override `_posterior` to drop the task column via a tensor
  `index_select` and delegate to `target_gp._posterior`.
- **Rationale:** A local, explicit, testable tensor slice. It does not mutate the
  internals of a fitted model.
- **Rejected alternative - `FilterFeatures` input transform:** The original idea was to
  monkeypatch `target_gp._model.input_transform` with
  `ChainedInputTransform(FilterFeatures(...), existing_transform)` and reassign
  `target_gp._searchspace` / `target_gp._input_scaler`. This was rejected because it
  mutates several private attributes of an already-fitted GP in lockstep (fragile,
  easy to desync) and reaches past BayBE's surrogate boundary. It is only *necessary*
  if acquisition functions must call `.fantasize()` directly on the raw botorch model
  (e.g. `qNIPV`), which the default `qLogEI` does not.
- **Trade-off (documented):** With `_posterior` stripping, the surrogate exposes an
  `AdapterModel` (a plain botorch `Model`) rather than a `GPyTorchModel`, and does not
  support `fantasize`. The `FilterFeatures` route would expose a real `SingleTaskGP`
  and support `fantasize`, at the cost of the fragility above.

### 5.4 Reduced search space reuse (`_drop_parameters`)
- **Decision:** Reuse `SearchSpace._drop_parameters` to build the task-free space for
  the inner GPs, and widen `_ReducedSearchSpace`'s attribute allowlist by exactly four
  members: `transform`, `scaling_bounds`, `get_comp_rep_parameter_indices`, and
  `task_idx`.
- **Rationale:** These are the only additional members the GP fit path touches, and all
  work correctly on the reduced space (per-parameter logic or the correctly-columned
  empty comp-rep). `task_idx` is safe because, with the task parameter dropped, it
  always returns `None` and never performs a real index lookup - this is documented
  inline next to the allowlist.
- **Verified non-issue - decorrelation:** An earlier concern was that the reduced space
  might have a different comp-rep column layout (decorrelation / constant-column
  dropping) than the full space, which would break the strip-in-`_posterior` mapping.
  On inspection, decorrelation is applied **per parameter** in `comp_df`
  (`df_uncorrelated_features`), and there is no subspace-level column reduction.
  Dropping the task parameter therefore cannot change the remaining parameters'
  columns, so the layout is preserved by construction.
- **Rejected alternative - real reduced `SearchSpace.from_product`:** Correct but
  expensive (re-enumerates the grid); unnecessary given the above.
- **Rejected alternative - train inner GPs on the full task space with filtered rows:**
  This keeps the layout but leaves `n_tasks > 1`, so the inner GPs would become
  degenerate multi-task ICM GPs fitted on a constant task column - not the intended
  single-task GPs.
- **Constraints:** Constraints (including task-involving ones) are dropped in the
  reduced space. This is harmless because constraints define the feasible *candidate*
  region (used by the recommender), and are never read during surrogate fitting. The
  outer surrogate and recommender still use the full, constrained search space.

### 5.5 Override + custom kernel is an error
- **Decision:** If a TL override is set together with a custom kernel on the GP, raise
  `IncompatibleSurrogateError`.
- **Rationale:** For now we assume the default kernel; combining the two is
  ambiguous/overspecified. Fail eagerly rather than silently picking one.

### 5.6 `CompositeSurrogate.to_botorch` decides by model type
- **Decision:** `CompositeSurrogate.to_botorch` builds a `ModelListGP` only when every
  constituent model is an actual `GPyTorchModel`, otherwise a `ModelList`.
- **Why it was required:** The default recommender auto-replicates single-output
  surrogates via `_autoreplicate`, so a single-target GP is always wrapped in a
  `CompositeSurrogate`. The previous logic keyed on the *surrogate class*
  (`is_all_instance(..., GaussianProcessSurrogate)`) and always produced a
  `ModelListGP`. A `MEAN_TRANSFER`-delegating GP is still a `GaussianProcessSurrogate`
  instance but exposes an `AdapterModel`, so `ModelListGP` would reject it at runtime.
  Keying on the actual model type fixes this and is behavior-preserving for all
  existing cases (normal GPs expose `SingleTaskGP`; non-GP surrogates already exposed
  `AdapterModel`).
- **Note:** This coupling is a direct consequence of decisions 5.1 + 5.3. The
  `FilterFeatures` route (5.3) would have avoided it.

## 6. Files changed

| File | Change |
|------|--------|
| `baybe/parameters/enum.py` | Added `MEAN_TRANSFER`, `RESIDUAL_LEARNING`, `RGPE` |
| `baybe/surrogates/gaussian_process/presets/baybe.py` | Task kernel dispatch (Case A) |
| `baybe/surrogates/gaussian_process/core.py` | Override guard, `_delegate`, routing of `_posterior`/`to_botorch`, `_tl_replacing_surrogate` |
| `baybe/searchspace/core.py` | Widened `_ReducedSearchSpace` allowlist (documented) |
| `baybe/surrogates/transfer_learning/__init__.py` | New subpackage |
| `baybe/surrogates/transfer_learning/base.py` | Shared multi-source base `_SourceTargetTransferSurrogate` + cold-start fallback |
| `baybe/surrogates/transfer_learning/mean_transfer.py` | New `MeanTransferSurrogate` |
| `baybe/surrogates/transfer_learning/residual_transfer.py` | New `ResidualTransferSurrogate` |
| `baybe/surrogates/transfer_learning/rgpe_transfer.py` | New `RGPETransferSurrogate` |
| `baybe/surrogates/composite.py` | `to_botorch` decides by model type |
| `tests/test_tl_prototypes.py` | New tests |
| `CHANGELOG.md` | Entries |

## 7. Tests

`tests/test_tl_prototypes.py` covers:

- Task-kernel dispatch for each mode (exact kernel type).
- Override + custom kernel raises.
- `MEAN_TRANSFER` delegation (delegate type, inner GPs fitted, posterior shape).
- Outer posterior equals the inner target GP posterior on stripped candidates.
- Invalid source/target cardinalities raise.
- Reduced search space exposes the newly allowed members and `task_idx is None`.
- A full `Campaign.recommend` run with `MEAN_TRANSFER` (exercises the
  auto-replication / `CompositeSurrogate` path that unit tests bypass).
- `RESIDUAL_LEARNING(_WITH_UNCERTAINTY)` delegation, `μ = μ_source + μ_residual`, and
  that the uncertainty variant has a variance no smaller than the residual-only one.
- `RGPE` delegation with multiple sources, that the weights form a convex combination
  over `(sources..., target)`, and a full `Campaign.recommend` run.
- Cold start: with only source data, every surrogate-replacing mode falls back to the
  source posterior; RGPE additionally uses uniform weights (over the sources with no
  target data, and over `(sources..., target)` with a single target point).

## 8. Known limitations / future work

- `MEAN_TRANSFER` supports exactly one source and one target task. `RESIDUAL_LEARNING`
  and `RESIDUAL_LEARNING_WITH_UNCERTAINTY)` support one or more sources and one target
  task (see section 13). `RGPE` supports one or more source tasks and one target task.
- No `fantasize` support for the surrogate-replacing modes (see 5.3); acquisition
  functions that require it (e.g. `qNIPV`) are not supported for these modes.
- Residuals are computed in original target units and passed back through the
  objective's pre-transform when fitting the residual GP. This is exact for linear
  (min/max) `NumericalTarget`s and approximate for bounded/nonlinear target transforms.
- **Cold-start scale limitation:** with zero target points, the mean/residual modes
  predict at the *source* output scale (see section 10). This self-corrects once at
  least one target measurement is available.
- `MeanTransferSurrogate`, `ResidualTransferSurrogate` and `RGPETransferSurrogate` are
  exported from their subpackage but not from the top-level `baybe.surrogates` public
  API (kept internal while prototyping).

## 9. Mean transfer vs. residual transfer: when are they the same model?

Let the source model define a fixed mean function `m_s(x)` (the source GP's posterior
mean). Consider a target GP with kernel `k`, noise `σ²`, training inputs `X` and outputs
`y`.

**Mean transfer** uses `m_s` as the *prior mean* of the target GP. Its posterior at
`x*`:

    μ_MT(x*) = m_s(x*) + k(x*,X)·(K + σ²I)⁻¹·(y − m_s(X))
    Σ_MT(x*) = k(x*,x*) − k(x*,X)·(K + σ²I)⁻¹·k(X,x*)

**Residual transfer** trains a *zero-mean* GP on residuals `r = y − m_s(X)` and adds the
source mean back:

    μ_RT(x*) = m_s(x*) + k(x*,X)·(K + σ²I)⁻¹·r
             = m_s(x*) + k(x*,X)·(K + σ²I)⁻¹·(y − m_s(X)) = μ_MT(x*)
    Σ_RT(x*) = k(x*,x*) − k(x*,X)·(K + σ²I)⁻¹·k(X,x*) = Σ_MT(x*)

So the two are the **exact same model** if and only if **all** of the following hold:

1. the residual GP uses a **zero mean** (not a constant mean),
2. identical kernel, hyperparameters and noise are used for both,
3. the **same output standardization** is applied (same shift/scale on `y` and on `r`),
4. the residual variance alone is reported (no added source variance).

**Why our implementations are genuinely distinct.** `MeanTransferSurrogate` and
`ResidualTransferSurrogate` deliberately break conditions 1, 3 and (optionally) 4:

- The residual GP uses BayBE's default **`ConstantMean`**, not a zero mean (breaks 1),
  so it can absorb a constant offset in the residuals.
- Output standardization (`Standardize`) is fit **on `y`** for mean transfer but **on
  the residuals `r`** for residual transfer (breaks 3). Since residuals typically have a
  smaller spread than `y`, the residual GP is fit on a different scale, which changes the
  learned hyperparameters and hence the predictions.
- `RESIDUAL_LEARNING_WITH_UNCERTAINTY` additionally sums the source and residual
  variances (breaks 4), making the two residual variants differ from each other as well.

Hence the modes `MEAN_TRANSFER`, `RESIDUAL_LEARNING`, and
`RESIDUAL_LEARNING_WITH_UNCERTAINTY` implement three distinct models that coincide only
in the idealized zero-mean / equal-standardization / residual-only-variance limit.

## 10. Shared multi-source base and the cold-start fallback

All surrogate-replacing modes share the base class
`_SourceTargetTransferSurrogate` (`baybe/surrogates/transfer_learning/base.py`). It:

1. splits the measurements by task value into an ordered list of **source** subsets
   (the non-active task values that actually have data) and a **target** subset (the
   active task value, which **may be empty**),
2. builds the reduced, task-free search space (`_drop_parameters`),
3. fits one single-task GP per non-empty source (`_source_gps`), and
4. delegates the target-specific logic to the abstract `_fit_target`.

The number of source tasks is capped per subclass via the `_max_sources` class
variable: `MeanTransferSurrogate` sets `_max_sources = 1` (one source, one target),
while `ResidualTransferSurrogate` and `RGPETransferSurrogate` leave it unbounded. If no
source task has data, the base raises `IncompatibleSearchSpaceError`.

### Why a cold-start fallback is needed

The default recommender uses a `TwoPhaseMetaRecommender` whose switch from the initial
(random) recommender to the Bayesian one is driven by the **total** number of
measurements, which for a TL campaign already includes all the *source* rows. As a
result the surrogate can be invoked with **zero or one** *target* points even though
the campaign has plenty of data. The initial random recommender does **not** protect
the target model here, so the surrogate-replacing modes must degrade gracefully rather
than fail.

### Fallback behavior (source-only prediction)

When the target subset is empty, the base exposes `_source_only_posterior`, which
strips the task column and returns the (single) source GP's posterior directly.
`MeanTransferSurrogate` routes to it when no target data is present. `ResidualTransfer`
handles cold start internally: with no target data the chain contains only source GPs
and their sum is returned directly (see section 13).

**Documented limitation.** With zero target points there is no target data to
recalibrate the output scale, so the prediction is made at the *source* scale. This is
the best available estimate before any target measurement and self-corrects as soon as
one target point is observed.

## 11. `RGPETransferSurrogate`

Location: `baybe/surrogates/transfer_learning/rgpe_transfer.py`. RGPE (ranking-weighted
GP ensemble, Feurer, Letham and Bakshy, ICML 2018 AutoML Workshop) fits one single-task
GP per source task and, once enough target data is available, one on the target task.
The ensemble posterior is a rank-weighted combination of the individual posteriors:

    μ(x*) = Σ_i w_i · μ_i(x*)
    Σ(x*) = Σ_i w_i² · Σ_i(x*)

with non-negative weights `w_i` that sum to one (models with zero weight are dropped and
the rest renormalized before combining).

### Weight estimation

The weights come from a **ranking loss** that counts, for each model, the number of
mis-ordered pairs of observed target points. For the source models the predictions come
directly from `model.posterior`; for the target model they come from
**leave-one-out cross-validation** so that the target model is not unfairly favored by
having seen the points it is scored on. The model with the lowest ranking loss on each
Monte Carlo sample "wins" that sample, and the weight of a model is the fraction of
samples it wins (`num_mc_samples`, default 256, drawn with a `SobolQMCNormalSampler`).

Because the loss depends only on the *ordering* of predictions, the LOOCV is performed
in the target model's transformed (normalized-input / standardized-output) space, which
the monotonic output standardization leaves order-invariant. The LOOCV batch GP reuses
the target model's fitted kernel, mean and likelihood (broadcast across the
leave-one-out batch), so all folds share the target model's hyperparameters.

### Cold start

- **Zero target points:** no target GP is built; the weights are uniform over the source
  GPs (this is the `_source_only`-style fallback generalized to several sources).
- **One target point:** a target GP is built, but a single point cannot rank anything,
  so the weights are uniform over `(sources..., target)`.
- **Two or more target points:** the full ranking-based weights are used.

### Design notes

- **Inferred noise.** The inner GPs use BayBE's default inferred-noise
  `GaussianProcessSurrogate` rather than the fixed-noise model of the original tutorial;
  this keeps the ensemble consistent with the rest of BayBE.
- **Scale robustness.** The rank weights are scale-invariant, but the posterior *mean*
  combination `Σ w_i μ_i` still assumes the source and target models predict on a
  comparable output scale (as they do here, sharing the objective's output space).
- **No `FilterFeatures`.** Like the other modes, RGPE strips the task column in
  `_posterior` (decision 5.3) and therefore does not support `fantasize`.

## 12. Known issues and potential fixes

Findings from a review of the branch (and in particular the RGPE commit). Ordered by
severity.

### 12.1 LOOCV GP is not actually "transform-free" (RGPE)

**Where:** `_loocv_sample_preds` in `rgpe_transfer.py`.

**Issue.** The function documents building a *"transform-free batch GP"* that reuses the
target model's fitted kernel, mean and likelihood in the already-transformed
(normalized-input / standardized-output) space. However, `SingleTaskGP` is constructed
without an explicit `outcome_transform`, and current BoTorch (0.16.x) applies a
**default `Standardize` outcome transform** in that case. Since `train_y_cv` is already
in the target model's standardized space, each leave-one-out fold gets *re-standardized*
per batch. The deep-copied kernel/mean/likelihood hyperparameters were fitted for the
first standardization, so they no longer align exactly with the scale the LOO model
sees — the opposite of the intended "share the fitted hyperparameters" behavior.

**Impact.** Small in practice: re-standardizing already-≈unit-variance data is close to
identity, and the ranking loss only depends on ordering (preserved by any monotonic
transform), so results and tests are largely unaffected. But it contradicts the
documented intent and slightly perturbs the estimated weights.

**Fix.** Make the LOO GP explicitly transform-free:

```python
model = SingleTaskGP(
    train_x_cv,
    train_y_cv,
    covar_module=deepcopy(target_model.covar_module),
    mean_module=deepcopy(target_model.mean_module),
    likelihood=deepcopy(target_model.likelihood),
    outcome_transform=None,
    input_transform=None,
)
```

### 12.2 Redundant weight renormalization (RGPE)

**Where:** `_posterior` in `rgpe_transfer.py`.

**Issue.** The non-zero weights are renormalized (`weights[nonzero] / weights[nonzero].sum()`)
before combining posteriors. In every branch the weights already sum to one (ranking
weights are `bincount / num_samples`; the cold-start fallbacks are uniform), so the
renormalization is a no-op and a minor, unnecessary deviation from the reference
implementation.

**Fix.** Optional — drop the renormalization and simply select the non-zero-weight
models, or keep it as defensive code and note that it is a no-op.

### 12.3 Weight dilution across many sources (RGPE)

**Where:** conceptual, `_compute_rank_weights`.

**Issue.** Like the basic reference implementation, the ensemble does not mitigate
weight dilution: with many source tasks, the target model's weight can be driven toward
zero even when it is the most relevant model. This is already called out in the class
docstring as out of scope.

**Fix.** None required for the prototype. If needed later, add the paper's dilution
mitigation (e.g. discard source models whose ranking loss exceeds the target model's
median loss before computing weights).

### 12.4 Commit-message wording

**Issue.** The commit title *"…no to one target point fallback to source for all TL
modes"* is slightly imprecise: mean/residual transfer fall back to the source only at
**zero** target points and build their target/residual GP from one point onward, while
RGPE uses uniform weights at zero and one point. The behavior is correct; only the
phrasing is loose.

**Fix.** Purely cosmetic — no code change needed.

### 12.5 `MeanTransferSurrogate` OOM on large continuous benchmarks

**Where:** `_build_posterior_mean_module` → `_PosteriorMean.forward` in
`baybe/surrogates/gaussian_process/core.py`.

**Issue.** The target GP's `mean_module` wraps a deep copy of the fitted source GP.
During `fit_gpytorch_mll`, GPyTorch calls `mean_module(train_x_norm)` at every gradient
step (~100–200 iterations). Each call invokes `frozen_model.posterior(x_raw)`, which
allocates the source GP's full `n_source × n_source` kernel matrix. On large continuous
search spaces (e.g. the hartmann benchmarks with large source datasets) this triggers
an OOM kill (exit code 137).

**Root cause.** `train_x_norm` — the target training inputs in the outer GP's
normalized space — is a **fixed constant** throughout MLL optimization. The source GP is
frozen (all `requires_grad = False`). Therefore `forward(train_x_norm)` produces the
same value every gradient step, yet the kernel matrix is reallocated each time.

**Fix** (implemented). Lazy caching inside `_PosteriorMean`:

- A `_train_mean_cache: Tensor | None` attribute is added to `_PosteriorMean`.
- `forward()` branches on `self.training`:
  - **Training mode** (MLL fitting): on the first call, compute and store the result
    under `torch.no_grad()`; return the cached tensor on all subsequent calls.
  - **Eval mode** (prediction): always call the source GP, as `x` is now arbitrary.
- `train(mode=True)` resets `_train_mean_cache = None` so a fresh `.fit()` call
  repopulates the cache from the new target GP's training inputs.
- The expensive `self.gp.posterior(x_raw)` call is extracted into a private helper
  `_eval_source_gp(x)` shared by both branches.

**Correctness.** The cached tensor is detached (no gradient flows from it), which is
correct because no gradient was ever intended to reach the frozen source GP. Gradients
for the target GP's kernel and likelihood hyperparameters flow normally through the
covariance matrix. The `self.training` flag is a reliable signal: `fit_gpytorch_mll`
sets `model.train()` before optimization and `model.eval()` before posterior calls.

**Memory note.** The source GP deep copy is still held in memory for prediction time,
so the double-model memory footprint is unchanged. The fix eliminates the
O(n_optimizer_steps) × O(n_source²) repeated kernel-matrix allocations during fitting.

**Test.** `test_mean_transfer_cache_populated_correctly` in `tests/test_tl_prototypes.py`
verifies that `_train_mean_cache` equals a direct call to `_eval_source_gp(train_x_norm)`
after fitting.

## 13. Multi-source `ResidualTransferSurrogate`

### Algorithm

`ResidualTransferSurrogate` was extended from single-source to multi-source using a
**GP boosting chain**: each GP in the chain corrects the residuals of the previous
stack. Sources are processed in the order imposed by `TaskParameter.values` (always
alphabetical, since `CategoricalParameter` sorts its values on construction).

Given sources `[A, B, …]` (alphabetical) and a target task:

```
chain = []

# First source: fit on raw data
gp_A.fit(reduced_searchspace, objective, data_A)
chain = [gp_A]

# Subsequent sources: fit on residuals w.r.t. current chain sum
r_B  = y_B  - (μ_A(X_B))
gp_B.fit(reduced_searchspace, objective, (X_B, r_B))
chain = [gp_A, gp_B]

# Target: fit on residuals w.r.t. full source chain sum
r_T  = y_T  - (μ_A(X_T) + μ_B(X_T))
gp_T.fit(reduced_searchspace, objective, (X_T, r_T))
chain = [gp_A, gp_B, gp_T]
```

Prediction at `x*`:

    μ(x*) = μ_A(x*) + μ_B(x*) + μ_T(x*)

Variance without propagation (only last GP):

    Σ(x*) = Σ_T(x*)

Variance with propagation (all GPs independent, sum):

    Σ(x*) = Σ_A(x*) + Σ_B(x*) + Σ_T(x*)

The single-source case is a degenerate chain of length 2: `[gp_source, gp_target]`,
which produces identical results to the previous implementation.

### Cold start

If the target task has no measurements, the chain contains only the source GPs
(no target residual GP is appended). The prediction is the sum of all source GP
posteriors. For a single source this matches the previous `_source_only_posterior`
fallback exactly.

### Implementation details

The chain is built entirely in an overridden `_fit` method, bypassing the base class's
source-fitting loop (which would fit all source GPs on raw data). Key points:

- `_max_sources` is removed — no cap on the number of sources.
- `_residual_gp: GaussianProcessSurrogate | None` is replaced by
  `_gp_chain: tuple[GaussianProcessSurrogate, ...]`.
- `_make_residual_measurements(gps, measurements)` is a helper that evaluates the sum
  of posterior means of a list of GPs at `measurements` input points and subtracts from
  the target column(s), returning a residualised copy of the DataFrame.
- `_fit_target` is kept as a no-op to satisfy the abstract base class; it is never
  called because `_fit` is overridden.
- `_posterior` sums means via `torch.stack([...]).sum(dim=0)` and sums covariances
  (when propagating) via `functools.reduce(operator.add, ...)`, both avoiding
  Python's `sum()` initialising from `int(0)`.

### Source ordering caveat

The chain order is determined by `task_param.values`, which is always sorted
alphabetically. Users cannot control the chain order by specifying values in a different
declaration order. Task names should be chosen accordingly if ordering matters
(e.g. `"1_baseline"`, `"2_finetuned"`, `"3_target"`).

### Tests

New and updated tests in `tests/test_tl_prototypes.py`:

- `test_residual_transfer_delegates` — updated to check `len(_gp_chain) == 2`
  (single-source: source GP + target residual GP).
- `test_residual_transfer_mean_is_source_plus_residual` — updated to sum over
  `_gp_chain` instead of accessing `_source_gp`/`_residual_gp` directly.
- `test_residual_transfer_cold_start_uses_source_chain` — verifies that with no target
  data `_gp_chain` has length 1 and the posterior equals the single source GP's output.
- `test_residual_transfer_multisource_chain_length` — 2 sources + target → chain
  length 3.
- `test_residual_transfer_multisource_mean_is_chain_sum` — outer mean equals sum of
  individual GP means with 2 sources.
- `test_residual_transfer_multisource_uncertainty_variant_has_larger_variance` — same
  variance ordering property as the single-source variant, with 2 sources.
