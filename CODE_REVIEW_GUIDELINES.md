# BayBE Code Review Guidelines

These guidelines define the standards an AI code review agent must enforce when
reviewing contributions to the BayBE repository. Every review must evaluate code
against four pillars: **functional correctness**, **adherence to codebase
conventions**, **passing all CI quality gates**, and **general code quality**.

---

## 1. Project Architecture and Design Patterns

### 1.1 `attrs`-Based Class Design

BayBE builds nearly all of its classes with [`attrs`](https://www.attrs.org/).
Reviewers must verify:

- **Use `@define`** (not `@attr.s` or `@dataclass`) for mutable classes. Use
  `@define(frozen=True, slots=False)` for immutable value objects (e.g.,
  parameters).
- **Attribute docstrings** go directly below each attribute declaration,
  separated by blank lines — never in the class docstring:
  ```python
  @define
  class MyClass:
      """Class-level docstring."""

      my_attr: int
      """Attribute-level docstring."""

      other_attr: str
      """Another attribute docstring."""
  ```
- **Default factories and validators** follow the naming convention:
  ```python
  @my_attribute.default
  def _default_my_attribute(self): ...

  @my_attribute.validator
  def _validate_my_attribute(self, attribute, value): ...
  ```
  Validators with `Raises:` sections need at minimum a one-line docstring. They
  may suppress pydoclint warnings using `# noqa: DOC101, DOC103`.
- **`gc.collect()`** must appear at module bottom in any module defining slotted
  `attrs` classes (`@define`), to collect leftover original classes:
  ```python
  # Collect leftover original slotted classes processed by `attrs.define`
  gc.collect()
  ```

### 1.2 Serialization

- Classes that need (de-)serialization should inherit from `SerialMixin` and
  register appropriate hooks with the global `converter` from
  `baybe.serialization`.
- Serialization hooks use `cattrs`. Custom converters must be registered at
  module level after the class definition.
- Roundtrip serialization consistency must be covered by tests (see
  §4.3 on hypothesis strategies).

### 1.3 Method Overrides

- **All** method overrides in subclasses must use
  `@typing_extensions.override`:
  ```python
  from typing_extensions import override

  class Child(Parent):
      @override
      def method(self): ...
  ```
  This is enforced by `mypy` via the `enable_error_code = explicit-override`
  configuration.

### 1.4 Module and Namespace Organization

- Public functionality must be imported into the appropriate high-level
  `__init__.py` namespaces with an explicit `__all__` list (sorted
  alphabetically):
  ```python
  """BayBE parameters."""

  from baybe.parameters.categorical import CategoricalParameter, TaskParameter
  ...

  __all__ = [
      "CategoricalParameter",
      ...
      "TaskParameter",
  ]
  ```
- New subclasses or public functions must be added to the relevant
  `__init__.py` and `__all__`.

### 1.5 Exception Hierarchy

- Custom exceptions live in `baybe/exceptions.py`.
- The project uses a well-defined hierarchy (e.g., `IncompatibilityError` with
  subclasses like `IncompatibleSearchSpaceError`,
  `IncompatibleSurrogateError`). New exceptions should follow this pattern.
- Review that the correct exception type is raised for the situation. Prefer
  specific exception classes over generic `ValueError`/`TypeError` when an
  appropriate custom exception exists.

---

## 2. CI Pipeline Quality Gates

The CI pipeline (`ci.yml`) enforces the following gates. A review must verify
that new code will not break any of them.

### 2.1 Linting (`tox -e lint`)

Linting runs `pre-commit run --all-files`, which triggers:

#### 2.1.1 Ruff (Linting + Formatting)

Configuration: `ruff.toml` (line length 88).

**Selected rule sets:**
- `D` — pydocstyle (Google convention)
- `D401` — imperative mood for docstrings
- `E501` — line-too-long (88 chars)
- `F` — Pyflakes
- `I` — isort
- `E4`, `E7`, `E9` — standard ruff defaults
- `W` — warnings
- `PL` — Pylint

**Globally ignored rules:**
- `D105` (magic method docstrings), `D107` (`__init__` docstrings)
- `PLR0912` (too many branches), `PLR0913` (too many arguments),
  `PLR0915` (too many statements), `PLR2004` (magic values)
- `PLC0415` (import outside top level), `E703` (trailing semicolons)

**Per-file ignores:**
- `examples/*`: `E266`, `E501`, `D100`, `D401`
- `tests/*`: `D102`, `D103`, `D401`
- `baybe/exceptions.py`: `D205`, `D212`, `D415`

Reviewers must ensure:
- All imports are sorted (`isort`-compatible order via ruff).
- Line length does not exceed 88 characters.
- Docstrings follow Google style conventions.
- No unused imports or variables.

#### 2.1.2 pydoclint (via flake8)

Configuration: `pydoclint.toml`.

- Style: Google
- `arg-type-hints-in-docstring = False` — do **not** repeat type hints in
  docstrings.
- `check-return-types = False` — do **not** repeat return types in docstrings.
- `check-class-attributes = False` — class attribute docstring checking is
  disabled (compatibility with `attrs`).

Reviewers must ensure docstrings do **not** contain redundant type information
that belongs in type annotations.

#### 2.1.3 pyupgrade

- Configured with `--py310-plus`.
- Reviewers should verify that code uses modern Python 3.10+ syntax:
  - `X | Y` union types instead of `Union[X, Y]` (in runtime positions).
  - `match`/`case` where appropriate.
  - Use `from __future__ import annotations` for stringified annotations.

#### 2.1.4 uv-lock

- The `uv.lock` file must remain in sync with `pyproject.toml`. Any dependency
  change requires a lock file update.

### 2.2 Type Checking (`tox -e mypy`)

Configuration: `mypy.ini`.

- Packages checked: `baybe`, `benchmarks`.
- `enable_error_code = explicit-override` — all overridden methods must use
  `@override`.
- `disable_error_code = type-abstract` — avoids false positives for
  `type[Protocol]`.
- Some modules are currently excluded (`baybe/simulation`,
  `baybe/utils/dataframe.py`, `baybe/recommenders/naive.py`,
  `baybe/simulation.py`), but new code should not be added to this exclusion
  list without strong justification.

Reviewers must check:
- All function signatures have complete type annotations.
- `attrs` fields have type annotations.
- No new `# type: ignore` comments without a justifying inline comment
  (e.g., referencing a known upstream issue).
- `TYPE_CHECKING` imports are used correctly for heavy or circular
  dependencies.

### 2.3 Tests (`tox -e fulltest` / `tox -e coretest`)

- `fulltest`: Runs with all extras installed, against Python 3.10 and 3.13.
- `coretest`: Runs with core dependencies only, against Python 3.10 and 3.14.
- Coverage thresholds:
  - **Overall: 70%** minimum.
  - **Per-file: 45%** minimum.
- Tests must not emit warnings (`-p no:warnings` is used).

Reviewers must ensure:
- New public functionality has corresponding tests.
- Tests use existing fixture patterns from `conftest.py` (named fixtures with
  `fixture_` prefix and explicit `name` parameter).
- Test functions have docstrings (except in `tests/` where `D102`/`D103` are
  suppressed — but descriptive docstrings are still preferred).
- Validation tests check error paths (`pytest.raises`) with meaningful
  parametrization using `pytest.param(..., id="...")`.
- The test does not introduce flaky behavior. Numerical instability should be
  handled with retry logic (see `conftest.py`'s `run_iterations`).

### 2.4 Dependency Audit (`tox -e audit`)

- `pip-audit` scans all installed packages.
- New dependencies must not introduce known vulnerabilities.
- Dependency version pins must include an inline comment if they are set for
  audit/security reasons (see `# see AUDIT NOTE` pattern in `pyproject.toml`).

### 2.5 Documentation Build (`tox -e docs`)

- Built with Sphinx using the Furo theme.
- Docs build from examples — new examples must be executable.
- Cross-references must use Sphinx roles:
  `` :class:`baybe.parameters.base.Parameter` ``,
  `` :func:`baybe.utils.dataframe.filter_df` ``, etc.
- Literals use double backticks: ``` ``MyLiteral`` ```.

### 2.6 Changelog

- Every PR must add an entry to `CHANGELOG.md` (follows
  [Keep a Changelog](https://keepachangelog.com/) format).
- Sections: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
  `Breaking Changes`.
- The CI checks this via a diff against `main`.

---

## 3. Functional Correctness

Before evaluating style, conventions, or CI compliance, reviewers must verify
that the contribution actually achieves what it claims to do.

### 3.1 Issue and PR Description Alignment

- The code changes **must resolve the issue** described in the PR description or
  linked ticket. A contribution that passes all quality gates but does not
  address the stated problem is incomplete.
- Reviewers must read the PR description and any linked issues, then verify that
  the implementation matches the described intent.
- If the PR description is vague or missing, request clarification before
  approving.
- Partial fixes must be explicitly called out — a PR should not be approved as
  resolving an issue if it only addresses a subset of the requirements without
  acknowledgment.

### 3.2 Test Coverage for New Features

- **Every new feature must include tests** that demonstrate the feature works as
  intended. A feature without tests is not considered complete, regardless of
  code quality.
- Tests must cover:
  - The **primary use case** (happy path).
  - Relevant **edge cases** and **boundary conditions**.
  - **Error handling paths** where the feature is expected to reject invalid
    input or degrade gracefully.
- For bug fixes, at least one test must be added that **would have failed**
  before the fix and passes after it, to prevent regressions.

---

## 4. Code Quality Standards

### 4.1 Docstrings

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):

- **Every** public function, method, and class must have a docstring.
- Structure: short one-line summary (imperative mood), optional extended
  description, then `Args`, `Returns`, `Raises`, etc.
- **Do not** repeat type hints in docstrings — types belong in annotations only.
- Reference objects using Sphinx roles:
  `` :class:`path.to.Class` ``, `` :func:`path.to.function` ``.
- Use double backticks for literal values: ``` ``True`` ```, ``` ``None`` ```.
- `attrs` validators that raise exceptions should have a `Raises:` section.

### 4.2 Type Annotations

- All function signatures (parameters and return types) must be annotated.
- Use `from __future__ import annotations` for forward references and modern
  union syntax (`X | Y`).
- Use `TYPE_CHECKING` blocks for imports needed only for type checking:
  ```python
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from heavy_module import HeavyType
  ```
- Use `typing_extensions` for features not yet in `typing` for the minimum
  supported Python version (3.10).

### 4.3 Testing Patterns

#### Property-Based Testing (Hypothesis)

- Serialization roundtrip tests must use hypothesis strategies defined in
  `tests/hypothesis_strategies/`.
- When adding a new class that supports serialization, add a corresponding
  hypothesis strategy and a roundtrip test.
- Hypothesis settings: CI uses `deadline=500, max_examples=100`.

#### Fixture Conventions

- Fixture functions use `fixture_` prefix with an explicit `name` parameter:
  ```python
  @pytest.fixture(name="my_object")
  def fixture_my_object():
      """Descriptive docstring."""
      ...
  ```
- Default fixtures are defined in `conftest.py` and can be overridden per-test
  module.

#### Parametrization

- Use `pytest.param(..., id="descriptive_id")` for all parametrized test cases.
- Validation tests follow the pattern of checking expected exception types:
  ```python
  @pytest.mark.parametrize(
      ("input", "error"),
      [
          param(..., ValueError, id="descriptive_case"),
      ],
  )
  def test_invalid_input(input, error):
      """Providing invalid input raises an exception."""
      with pytest.raises(error):
          ...
  ```

#### Optional Dependencies

- Tests for optional features must be guarded with `pytest.mark.skipif`:
  ```python
  @pytest.mark.skipif(
      not CHEM_INSTALLED, reason="Optional chem dependency not installed."
  )
  ```

### 4.4 Import Conventions

- Standard library, then third-party, then local — enforced by `ruff` isort.
- Use `from __future__ import annotations` in modules that need forward
  references.
- Heavy imports (e.g., `torch`, `botorch`) that are not always needed should
  be lazy or behind `TYPE_CHECKING`.
- Public modules re-export through `__init__.py` with `__all__`.

### 4.5 Error Handling

- Use specific custom exceptions from `baybe.exceptions` where applicable.
- Error messages should be user-friendly, explaining what went wrong and
  ideally what to do about it.
- Pattern: include the problematic value in the error message:
  ```python
  raise ValueError(
      f"Parameter '{self.name}' has tolerance {tolerance}, "
      f"but the maximum allowed tolerance is {max_tol}."
  )
  ```

### 4.6 Private vs Public API

- Private attributes and methods use a leading underscore (`_`).
- `attrs` fields that expose a different public interface use `alias`:
  ```python
  _values: tuple[float, ...] = field(alias="values", ...)
  ```
- Private utility functions at module level also use leading underscores.

### 4.7 Dependency and Feature Flags

- Optional features are gated through `baybe._optional.info` (e.g.,
  `CHEM_INSTALLED`, `POLARS_INSTALLED`).
- Settings that control optional acceleration use the `AutoBool` pattern
  (`True` / `False` / `Auto`).
- New optional dependencies must:
  - Be added to the appropriate optional dependency group in `pyproject.toml`.
  - Have an `_optional/info` check.
  - Raise `OptionalImportError` with a helpful installation message.

### 4.8 Numerical Considerations

- The codebase supports configurable floating-point precision via
  `active_settings.DTypeFloatNumpy` and `active_settings.DTypeFloatTorch`.
- New numerical code must respect these settings rather than hardcoding
  `float64` or `float32`.
- Be alert to numerical stability issues, especially in surrogate model
  fitting. The test suite uses retry logic with `tenacity` for known
  intermittent `LinAlgError` issues.

---

## 5. Review Checklist

For every pull request, verify each of the following:

### Functional Correctness
- [ ] Changes actually resolve the issue described in the PR description or
  linked ticket
- [ ] Implementation matches the stated intent — no gaps or unaddressed
  requirements
- [ ] New features include tests covering the primary use case, edge cases, and
  error paths
- [ ] Bug fixes include a regression test that fails without the fix

### Structure and Design
- [ ] New classes use `@define` (or `@define(frozen=True, slots=False)` for
  immutable types)
- [ ] `gc.collect()` present at module bottom for modules with `attrs` classes
- [ ] Method overrides use `@typing_extensions.override`
- [ ] New public symbols exported in `__init__.py` with `__all__`
- [ ] Custom exceptions added to `baybe/exceptions.py` if needed

### Docstrings and Documentation
- [ ] All public functions/classes/methods have Google-style docstrings
- [ ] Docstring summary is imperative mood
- [ ] No type hints duplicated in docstrings
- [ ] Sphinx cross-references used for referencing classes/functions
- [ ] `CHANGELOG.md` updated with the change

### Type Safety
- [ ] All function signatures fully annotated
- [ ] `from __future__ import annotations` used where needed
- [ ] No unwarranted `# type: ignore` without explaining comment
- [ ] `TYPE_CHECKING` used for heavy/circular import-only types

### Code Conventions
- [ ] Line length ≤ 88 characters
- [ ] Imports sorted (stdlib → third-party → local)
- [ ] Python 3.10+ syntax used (`X | Y` unions, `match`/`case` where fitting)
- [ ] Private members prefixed with `_`
- [ ] `attrs` validators/defaults follow naming convention
- [ ] Error messages include relevant context values

### Testing
- [ ] New functionality has corresponding tests
- [ ] Validation tests cover error/edge cases with `pytest.raises`
- [ ] `pytest.param(..., id="...")` used for parametrized cases
- [ ] Serialization roundtrip test added (if applicable)
- [ ] Hypothesis strategies added (if new serializable class)
- [ ] Optional-dependency tests guarded with `skipif`
- [ ] No test flakiness introduced

### Dependencies and Security
- [ ] No new known vulnerabilities (`pip-audit`)
- [ ] Optional dependencies properly gated with `_optional/info`
- [ ] No credentials or secrets in the code
- [ ] Lock file updated if dependencies changed

### Formatting and Linting
- [ ] Passes `ruff check` and `ruff format`
- [ ] Passes `pydoclint` checks
- [ ] Passes `pyupgrade --py310-plus`
- [ ] Passes `mypy`

---

## 6. Common Pitfalls

These are patterns that frequently cause CI failures or review rejections:

1. **Missing `gc.collect()`** at module bottom in files with `@define` classes.
2. **Missing `@override`** on overridden methods — caught by `mypy`.
3. **Type hints in docstrings** — `pydoclint` expects them only in annotations.
4. **Non-imperative docstring summaries** — ruff `D401` requires imperative
   mood (e.g., "Return the value" not "Returns the value").
5. **Hardcoded `float64`/`float32`** instead of using `active_settings` dtypes.
6. **Adding new public classes** without updating `__init__.py` and `__all__`.
7. **Forgetting the `CHANGELOG.md`** entry.
8. **Broad exception catches** (`except Exception`) without re-raising or
   specific handling.
9. **Using `Union[X, Y]`** instead of `X | Y` in a `__future__ annotations`
   context.
10. **Not gating optional imports** — any import from an optional dependency
    must be wrapped in a try/except or lazy import pattern with
    `OptionalImportError`.
