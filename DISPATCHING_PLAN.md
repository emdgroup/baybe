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

## The Reduced SearchSpace: `_ParameterOnlySearchSpace`

Full `SearchSpace.from_product` is expensive and constraint-aware. For the
factory call we only need parameter information. Solution: a **private
subclass** that exposes only parameter-related properties.

```python
class _ParameterOnlySearchSpace(SearchSpace):
    """A lightweight searchspace view exposing only parameter information.

    Raises NotImplementedError on any attempt to access constraints,
    subspaces, or transform functionality.
    """
```

**Exposed (working) properties/methods:**
- `parameters`, `parameter_names`
- `comp_rep_columns`, `comp_rep_bounds`, `scaling_bounds`
- `get_comp_rep_parameter_indices(name)`
- `n_tasks`, `task_idx`, `_task_parameter`
- `type` (DISCRETE/CONTINUOUS/HYBRID)

**Blocked (raise):**
- `constraints`, `is_constrained`
- `transform()`
- `discrete.exp_rep`, `discrete.comp_rep`
- Any method requiring the full subspace internals

**Constructor:** A method on `SearchSpace`:

```python
def _without_parameters(
    self, names: Collection[str]
) -> _ParameterOnlySearchSpace:
    """Return a parameter-only view without the named parameters."""
```

This replaces the current `_without_task_parameter()` with a more general
mechanism.

---

## Method Signature (Revised `_resolve_kernel`)

```python
def _resolve_kernel(
    self, context: _ModelContext, train_x: Tensor, train_y: Tensor
) -> GPyTorchKernel:
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
      - Build reduced searchspace
      - Call factory(reduced_ss, train_x, train_y)
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

`BayBEKernelFactory._make` does:
```python
is_multitask = searchspace.task_idx is not None
factory = ICMKernelFactory if is_multitask else BayBENumericalKernelFactory
```
On the reduced searchspace, `task_idx` is `None` → it uses
`BayBENumericalKernelFactory` → returns a pure numerical kernel (no task
component). This is exactly what we want.

---

## Implementation Steps

1. **Add `_ParameterOnlySearchSpace`** (new private subclass in
   `baybe/searchspace/core.py`)
   - Minimal `from_product`-like init from a parameter list
   - Override blocking properties to raise `NotImplementedError`
   - Implement `_without_parameters(names)` on `SearchSpace` returning this type

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

5. **Update `_without_task_parameter`** → replace with `_without_parameters`

---

## Resolved Design Decisions

1. **Error class**: Keep `IncompatibleKernelError` (or similar) for the
   gpytorch/composite forbidden cases. The semantics change from
   "overspecification" to "unsupported kernel type with override".

2. **`_ParameterOnlySearchSpace` scope**: Exposes `n_tasks`, `task_idx`,
   `_task_parameter` etc. On the reduced view (task param removed) these return
   `None`/`1` as expected. This is exactly why Case 9 works: `BayBEKernelFactory`
   sees `task_idx=None` → uses numerical-only factory.

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
