# Kernel Override Dispatching: Structured Summary & Implementation Plan

## Context

The `feat/index-kernel-dispatching-via-factory` branch introduces
`TaskParameter.override_transfer_learning_mode`, which lets a user force a
specific TL kernel (`INDEX_KERNEL` or `POSITIVE_INDEX_KERNEL`) regardless of
what kernel/factory is passed to `GaussianProcessSurrogate`. The current
implementation raises `IncompatibleKernelError` when it detects overspecification
(an override + a task-aware factory). The team decided the override should
**always win** instead of raising, stripping the task-kernel component from
whatever was provided and re-attaching the prescribed task kernel.

---

## Design: Three-Case Dispatch in `_resolve_kernel`

When `override_transfer_learning_mode` is set on the `TaskParameter`, we inspect
what sits inside `self.kernel_factory`:

### Case 1: BayBE `Kernel` (detected via `PlainGPComponentFactory.component`)

**Logic:** Manipulate `parameter_names` to exclude the task parameter name.

| Sub-case | Action |
|----------|--------|
| `BasicKernel` with `parameter_names` containing task name | Remove task name from `parameter_names` |
| `BasicKernel` with `parameter_names = None` | Set `parameter_names` to all non-task parameter names |
| `BasicKernel` with only the task param in `parameter_names` | Results in empty names → skip base kernel, return only task kernel |
| `ScaleKernel(base_kernel)` | Apply the above logic to `base_kernel`, re-wrap in ScaleKernel |
| Other `CompositeKernel` (`AdditiveKernel`, `ProductKernel`) | **RAISE** — cannot walk arbitrary trees |

After stripping, convert **both** kernels to gpytorch independently, then
multiply at the gpytorch level (`base_gpytorch * task_gpytorch`). The
multiplication must happen **after** `to_gpytorch()` because:
- Both kernels need the **full** searchspace for correct index resolution in
  `_get_dimensions()` (parameter names → comp-rep column indices)
- Multiplying at BayBE level would create a `ProductKernel`, which calls
  `to_gpytorch` on children with the same searchspace arg — that works too, but
  is unnecessarily indirect and would trigger the "composite kernel" guard

### Case 2: Raw GPyTorch kernel (detected via `PlainGPComponentFactory.component`)

**RAISE** unconditionally. GPyTorch kernels use integer `active_dims`, which
cannot be reliably rewritten without knowing the full comp-rep layout. Too
fragile.

### Case 3: BayBE Kernel Factory (not a `PlainGPComponentFactory`)

**Logic:** Call the factory on a **reduced searchspace** (all parameters minus
`TaskParameter`). Then inspect the result:

| Factory return type | Action |
|---------------------|--------|
| BayBE `Kernel` | Convert to gpytorch (using **full** searchspace for index resolution), multiply with task kernel |
| GPyTorch kernel | **RAISE** — same reason as Case 2 |

This handles `BayBEKernelFactory()` (case 9) correctly: on the reduced
searchspace, `task_idx` is `None`, so it falls through to
`BayBENumericalKernelFactory` which only builds a base kernel.

**Important:** The reduced searchspace is only used for the factory call
(parameter selection logic). The resulting BayBE kernel's `to_gpytorch()` must
receive the **full** searchspace, because `_get_dimensions()` resolves
`parameter_names` → comp-rep column indices, and those indices must match the
actual training tensor layout. Parameter names (e.g. `"x"`) are the same in
both searchspaces, so the lookup works correctly on the full one.

---

## The Reduced SearchSpace: `_ReducedSearchSpace`

Full `SearchSpace.from_product` is expensive and constraint-aware. For the
factory call we only need parameter information. This already exists on `main`
as the private subclass `_ReducedSearchSpace`, constructed via
`SearchSpace._drop_parameters`.

```python
class _ReducedSearchSpace(SearchSpace):
    """A lightweight search space exposing only parameter information.

    Guards attribute access via ``__getattribute__``: only the names listed in
    ``_ALLOWED_ATTRIBUTES`` are accessible; everything else raises
    ``AttributeError``.
    """
```

**Exposed (whitelisted in `_ALLOWED_ATTRIBUTES`):**
- `parameters`, `parameter_names`
- `comp_rep_columns`
- `discrete`, `continuous`
- `constraints`
- `type`
- `n_tasks`, `_task_parameter`
- `_get_n_comp_rep_columns(...)`, `get_parameters_by_name(...)`

**Blocked (any other attribute raises `AttributeError`):**
- `transform()`
- `comp_rep_bounds`, `scaling_bounds`
- `task_idx`
- `get_comp_rep_parameter_indices(...)`
- `discrete.exp_rep`, `discrete.comp_rep`
- Any method requiring the full subspace internals

**Constructor:** Already available on `SearchSpace` (added on `main`):

```python
def _drop_parameters(
    self, names: Collection[str], /
) -> _ReducedSearchSpace:
    """Return a reduced search space without the named parameters."""
```

---

## Method Signature (Revised `_resolve_kernel`)

```python
def _resolve_kernel(self, context: _ModelContext) -> GPyTorchKernel:
    """Resolve the GP kernel, applying TL override if specified."""
```

**Internal flow:**

```
1. task_param = context.searchspace._task_parameter
2. if no task_param or no override → default path (unchanged)
3. Determine override mode (INDEX_KERNEL / POSITIVE_INDEX_KERNEL)
4. Inspect self.kernel_factory:
   a. isinstance(PlainGPComponentFactory)?
      - component is raw gpytorch? → RAISE
      - component is CompositeKernel (not ScaleKernel)? → RAISE
      - component is ScaleKernel? → strip task from inner, re-wrap
      - component is BasicKernel? → strip task from parameter_names
   b. else (factory):
      - Build reduced searchspace via `_drop_parameters({task_name})`
      - Call factory(reduced_ss, objective, measurements)
      - If result is gpytorch → RAISE
      - Else strip/convert as in (a)
5. Convert stripped BayBE kernel → gpytorch via to_gpytorch(searchspace=FULL_ss)
6. Convert task kernel (Index/PositiveIndex) → gpytorch via to_gpytorch(searchspace=FULL_ss)
7. If stripped kernel has empty parameter_names → return task_gpytorch only
8. Else → return base_gpytorch * task_gpytorch (gpytorch-level multiplication)
```

---

## Test Cases (Expected Outcomes)

Given:
```python
task_param = TaskParameter("Task", ["A", "B"],
    override_transfer_learning_mode=TransferLearningMode.INDEX_KERNEL)
num_param = NumericalContinuousParameter("x", (0, 1))
ss = SearchSpace.from_product([num_param, task_param])
surrogate = GaussianProcessSurrogate(kernel_or_factory=...)
```

| # | `kernel_or_factory=` | Expected | Dispatch path |
|---|----------------------|----------|---------------|
| 1 | `MaternKernel(parameter_names=("x", "Task"))` | SUCCESS | Case 1: strip "Task" → `("x",)` |
| 2 | `MaternKernel(parameter_names=("x",))` | SUCCESS | Case 1: no task name, keep as-is |
| 3 | `SmoothedEDBOKernelFactory(parameter_selector=...)` | SUCCESS | Case 3: factory on reduced ss |
| 4 | `MaternKernel()` | SUCCESS | Case 1: `None` → set to `("x",)` |
| 5 | `ScaleKernel(MaternKernel())` | SUCCESS | Case 1 (ScaleKernel): strip inner |
| 6 | `gpytorch.kernels.MaternKernel(active_dims=[0,1])` | **RAISE** | Case 2: gpytorch forbidden |
| 7 | `gpytorch.kernels.MaternKernel()` | **RAISE** | Case 2: gpytorch forbidden |
| 9 | `BayBEKernelFactory()` | SUCCESS | Case 3: on reduced ss, falls to numerical factory |
| 10 | `IndexKernel(parameter_names=("Task",))` | SUCCESS | Case 1: strip → empty → only task kernel |
| 11 | `MaternKernel(p_n=("x",)) * IndexKernel(p_n=("Task",))` | **RAISE** | Case 1: ProductKernel → composite forbidden |

### Verification of Case 9 (Default Factory)

On the reduced search space the task parameter is gone, so `n_tasks == 1` and
`BayBEKernelFactory` dispatches to its numerical branch
(`_BayBENumericalKernelFactory`) → returns a pure numerical kernel (no task
component). This is exactly what we want.

> **Caveat (see Analysis):** `_ReducedSearchSpace` whitelists only a small set
> of attributes (`parameters`, `parameter_names`, `comp_rep_columns`,
> `constraints`, `type`, `n_tasks`, `_task_parameter`, `_get_n_comp_rep_columns`,
> `get_parameters_by_name`). It **blocks** `task_idx`, `comp_rep_bounds`,
> `scaling_bounds`, and `get_comp_rep_parameter_indices`. Any factory reaching
> for one of those on the reduced space raises `AttributeError`. The default
> factory path is safe, but custom-scaled factories (e.g. `SmoothedEDBO`,
> Case 3) must be verified against this whitelist.

---

## Implementation Steps

1. **`_ReducedSearchSpace` / `_drop_parameters`** — already present in
   `baybe/searchspace/core.py` on `main`; no new subclass needed. Reuse
   `SearchSpace._drop_parameters(names)` which returns a `_ReducedSearchSpace`.

2. **Refactor `_resolve_kernel`** (in `baybe/surrogates/gaussian_process/core.py`)
   - Replace current "override raises on overspecification" logic
   - Implement the three-case dispatch described above
   - Helper: `_strip_task_from_kernel(kernel, task_name, non_task_names)` →
     returns modified BayBE Kernel or `None` (if empty)

3. **Remove `IncompatibleKernelError`** (from `baybe/exceptions.py`)
   - Replace with appropriate errors for the "forbidden" cases (gpytorch,
     composite). Could reuse `IncompatibleSearchSpaceError` or introduce a
     narrower `UnsupportedKernelOverrideError`.

4. **Update tests** (`tests/test_kernel_factories.py`)
   - Remove overspecification-raises tests
   - Add parametrized tests for all 11 cases above
   - Test that `_ParameterOnlySearchSpace` blocks constraint access

5. **`_without_task_parameter` removal** → already done; `_resolve_kernel` now
   uses `SearchSpace._drop_parameters({task_param.name})`.

---

## Resolved Design Decisions

1. **Error class**: Keep `IncompatibleKernelError` (or similar) for the
   gpytorch/composite forbidden cases. The semantics change from
   "overspecification" to "unsupported kernel type with override".

2. **`_ReducedSearchSpace` scope**: Exposes `n_tasks`, `_task_parameter`
   etc. On the reduced view (task param removed) these return
   `None`/`1` as expected. This is exactly why Case 9 works: `BayBEKernelFactory`
   sees no task parameter → uses numerical-only factory.

3. **Factory output post-stripping**: NOT needed. The reduced searchspace
   doesn't contain the task parameter, so the factory cannot produce a kernel
   referencing it. Trust the factory output directly.

4. **ScaleKernel depth**: Single level only. `ScaleKernel(BasicKernel)` is
   supported. `ScaleKernel(ScaleKernel(...))` or `ScaleKernel(ProductKernel(...))`
   are composites → raise.

---

## Verification

- Run `tox -e mypy-py310` for type checking
- Run relevant tests: `pytest tests/test_kernel_factories.py tests/test_kernels.py -x`
- Add new parametrized test covering all 11 cases
- Ensure `BayBEKernelFactory()` with override doesn't raise (regression for case 9)

---

## Known Limitation: Default Kernel Factory + Override

### The problem

When `override_transfer_learning_mode` is set, `_resolve_kernel` builds the base
kernel by calling the provided kernel factory on a **reduced** search space
(`SearchSpace._drop_parameters({task_name})`, a `_ReducedSearchSpace`) and then
attaches the prescribed task kernel. `_ReducedSearchSpace` intentionally exposes only
parameter-level information and blocks index-space methods such as
`get_comp_rep_parameter_indices` (these only make sense against a concrete comp-rep
column layout, which the reduced space no longer represents).

The default `BayBEKernelFactory` dispatches (via `_dispatch`) to one of two numerical
factories:

- `_ChenNumericalKernelFactory` when a `SubstanceParameter` is present. It returns a
  BayBE `Kernel` carrying `parameter_names` and defers index resolution to
  `to_gpytorch`, so it **works** on the reduced search space.
- `_CustomScaledNumericalKernelFactory` otherwise (the common case). It returns a
  **raw gpytorch** `MaternKernel` and resolves `active_dims` eagerly via
  `get_comp_rep_parameter_indices`. On the reduced search space this raises
  `UnsupportedSearchSpaceAttributeError`, and even if it did not, its `active_dims`
  would be computed against the reduced (task-free) layout, which is misaligned with
  the full training tensor.

Consequently, the default kernel factory cannot currently be combined with
`override_transfer_learning_mode` on non-substance search spaces.

### Current behaviour

Consistent with the general rule "raise whenever the factory returns a gpytorch kernel
(or otherwise cannot run on the reduced search space)", `_resolve_kernel` raises
`IncompatibleKernelError` in this situation. The reduced-space call is wrapped to catch
`IncompatibleSearchSpaceError` and `UnsupportedSearchSpaceAttributeError` (a dedicated
`AttributeError` subclass raised by `_ReducedSearchSpace`) and re-raise a clear
`IncompatibleKernelError`. Factories that return BayBE `Kernel`s (e.g. the
Chen/substance branch, or user-provided BayBE-kernel factories) remain supported.

### Proposed future solution: lengthscale-constraint support

The only reason `_CustomScaledNumericalKernelFactory` cannot already return a BayBE
`Kernel` (like `_ChenNumericalKernelFactory` does) is that it sets a gpytorch
`lengthscale_constraint=GreaterThan(2.5e-2, transform=None,
initial_value=lengthscale_prior.mode)`, and BayBE's `MaternKernel` has no field to
express such a constraint.

Adding lengthscale-constraint support to BayBE kernels — mirroring the existing `Prior`
pattern (a small `SerialMixin` constraint type with `to_gpytorch()`, a
`lengthscale_constraint` field on `MaternKernel`, and conversion in
`Kernel.to_gpytorch` analogous to how priors are converted) — would let
`_CustomScaledNumericalKernelFactory._make` return a **numerically identical** BayBE
`Kernel`. That kernel would survive the reduced search space (it defers index
resolution) and would make the default factory work with the override, removing this
limitation without changing any GP fits. This is intentionally left as future work.
