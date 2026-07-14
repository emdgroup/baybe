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
| `RESIDUAL_LEARNING` | implemented | `ResidualTransferSurrogate` (residual variance only) |
| `RESIDUAL_LEARNING_WITH_UNCERTAINTY` | implemented | `ResidualTransferSurrogate` (source + residual variance) |
| `RGPE` | placeholder | - (enum value only) |

The two families are fundamentally different and therefore dispatch at **different
seams**:

- **Kernel-based modes** (`INDEX_KERNEL`, `POSITIVE_INDEX_KERNEL`) only change which
  task kernel goes into the ICM kernel of the *same* Gaussian process.
- **Surrogate-replacing modes** (`MEAN_TRANSFER`, and later `RESIDUAL_LEARNING`,
  `RGPE`) need an entirely different surrogate architecture that builds a target model
  from one or more source models.

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
`_tl_replacing_surrogate(mode)` currently maps only `MEAN_TRANSFER ->
MeanTransferSurrogate`; adding `RGPE`/`RESIDUAL_LEARNING` later is a one-line addition
plus a new surrogate class.

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
| `baybe/surrogates/transfer_learning/mean_transfer.py` | New `MeanTransferSurrogate` |
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

## 8. Known limitations / future work

- `MEAN_TRANSFER` and `RESIDUAL_LEARNING(_WITH_UNCERTAINTY)` support exactly one source
  and one target task.
- No `fantasize` support for the surrogate-replacing modes (see 5.3); acquisition
  functions that require it (e.g. `qNIPV`) are not supported for these modes.
- Residuals are computed in original target units and passed back through the
  objective's pre-transform when fitting the residual GP. This is exact for linear
  (min/max) `NumericalTarget`s and approximate for bounded/nonlinear target transforms.
- `RGPE` is an enum placeholder only; it will add a new surrogate class plus one entry
  in `_tl_replacing_surrogate`.
- `MeanTransferSurrogate` and `ResidualTransferSurrogate` are exported from their
  subpackage but not from the top-level `baybe.surrogates` public API (kept internal
  while prototyping).

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
