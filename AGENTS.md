# BayBE Coding Guide for AI Agents

## 1. Project Overview
**BayBE** (Bayesian Back End) is a Python library for Bayesian optimization and
experimental design. It targets Python 3.10+.

### Tech Stack
| Role | Library |
|------|---------|
| Data modeling | `attrs` (`@define`) |
| Serialization | `cattrs` + custom `SerialMixin` |
| Bayesian optimization | `botorch`, `gpytorch` |
| Tensor backend | `torch` |
| Numerics | `numpy`, `pandas`, `scipy`, `scikit-learn` |
| Testing | `pytest`, `hypothesis` |
| Linting/formatting | `ruff`, `mypy`, `pydoclint`, `pyupgrade` |

### Directory Layout
- `baybe/` — Main source package with many subpackages containing structures and objects
  such as parameters, constraints, targets, etc.
- `tests/` — Test suite. `conftest.py` has the central fixture system.
- `docs/` — Sphinx documentation (Furo theme, MyST-Parser). `docs/userguide/` has
  concept-oriented pages.
- `examples/` — Runnable example scripts (jupytext percent format).
- `benchmarks/` — Benchmarking framework.
- `.github/workflows/` — CI/CD (lint, test, docs, release, benchmark).

## 2. Architecture and Design Principles
- Campaign is the single stateful class: All other domain objects are frozen or
  stateless after construction. Do not add mutable state outside `Campaign`
- Comp-rep vs. exp-rep boundary: The comp-rep transition happens in the surrogate
  layer, not the recommender. `comp` is reserved for this — never reuse it for
  "component" or "comparison"
- Composition over monolith: Use builder pattern (e.g., acquisition functions) and
  factory classmethods (`from_product`, `from_dataframe`, `from_parameter`) instead of
  if-else chains
- Explicit over implicit: No silent errors — raise immediately. Validate eagerly
  at construction time. Side effects after validation only
- Separation of concerns: Target transformation ("what/how to transform") is
  separate from objective ("how to combine targets"). Surrogate owns data
  scaling/transformation; recommender owns recommendation logic.
- Design for generalizability. Keep fields private when API may change
- Keep classes focused — attributes relevant to the general case only
- Prefer BayBE types over raw external types in interfaces
- Keep raw data types consistent; defer conversions to point of use

## 3. Class Design
### attrs Only
All domain classes use `attrs` `@define`. No dataclasses, no Pydantic.
- Immutable value objects (parameters, kernels, priors, transformations, objectives,
  targets): `@define(frozen=True, slots=False)`.
- Mutable stateful objects (campaign, surrogates, recommenders): `@define`.
- `slots=False` required with `frozen=True` when `cached_property` is needed. See 
  `attrs` issue #164
- Also use `slots=False` when monkeypatching is needed (e.g., `register_hooks`)

### Inheritance: ABC + SerialMixin + Protocol
1. `Protocol` (`@runtime_checkable`): External/duck-typed interfaces. Always
   `__slots__ = ()`.
2. ABC: Shared behavior. Inherits from `ABC` and `SerialMixin`
3. Concrete classes: Inherit from ABC.

### Fields and Methods
- Use `field()` with `validator=`, `converter=`, `default=`, `factory=`, `alias=`.
- Private fields: `_` prefix, typically `init=False`.
- Store each piece of information once — no data duplication.
- Use `attrs.evolve()` for modified copies of frozen objects.
- Use `on_setattr` hooks for cache invalidation on mutable objects.
- `ClassVar[bool]` for capability flags (`supports_transfer_learning`, etc.).
- Order class content like this: 1) Attributes, 2) validators and post_init, 3)
  properties, 4) methods. Within each group use alphabetical order.

### Attribute Docstrings
String literals immediately below field declarations, blank lines between attributes.

### Module Epilogue
Every module using `@define` must end with:
`gc.collect()` with comment "Collect leftover original slotted classes..."

### Factory Classmethods
Name descriptively: `from_product`, `from_dataframe`, `from_parameter`, `from_config`,
`from_json`, `from_dict`, `from_preset`.

### classproperty
Custom `@classproperty` from `baybe.utils.basic` for class-level computed properties.

## 4. Naming Conventions
| Element | Convention | Examples |
|---------|------------|---------|
| Variables/functions | `snake_case` | `batch_size`, `add_measurements` |
| Classes | `PascalCase` | `Campaign`, `BotorchRecommender` |
| Constants | `SCREAMING_SNAKE_CASE` | `_RECOMMENDED`, `_TYPE_FIELD` |
| Private members | `_` prefix | `_cached_recommendation`, `_fit()` |
| Booleans | `is_`/`has_`/`supports_` | `is_numerical`, `supports_transfer_learning` |
| Counts | `n_` prefix | `n_batches_done`, `n_grid_points` |
| Protocols | `Protocol` suffix | `RecommenderProtocol` |
| Default/validator methods | `_default_<attr>` / `_validate_<attr>` | |

### Reserved Terminology
| Term | Reserved Meaning                                               |
|------|----------------------------------------------------------------|
| `task` | Transfer learning context only                                 |
| `comp` / `comp_rep` | Computational representation                                   |
| `exp_rep` | Experimental representation                                    |
| `configuration` | Assignment of values to ALL parameters (not individual values) |

### Abbreviations
- Full words in public API: `searchspace`, `objective`, `measurements`.
- Short forms in local contexts: `df`, `p` (parameter), `t` (target), `rec`.
- Established: `comp_rep`, `exp_rep`, `acqf`.

### Additional Rules
- Use `self.__class__.__name__` not hardcoded class names in errors/repr.
- No private field names (`_attr`) in user-facing messages — use
  `fields(type(self)).attr.alias`.
- Method names start with verbs. Comments capitalize first word.

## 5. Type Annotations
- **Full coverage**: All signatures including returns. Every field annotated.
- Uses `from __future__ import annotations` where needed (PEP 563).
- Modern syntax: `X | Y` (not `Union`), `X | None` (not `Optional`).
- `TypeAlias` for complex types, `ClassVar[T]` for class metadata.
- `TypeVar` with `_co` suffix for covariant. `Literal[...]` for constrained strings.
- `NoReturn` for deprecation properties that always raise.
- Always use `@typing_extensions.override` on overridden methods (enforced by
  mypy `explicit-override`).
- Use positional-only (`/`) and keyword-only (`*`) markers where appropriate.
- Heavy/optional imports for type annotations go in `if TYPE_CHECKING:` blocks.
- `# type: ignore` must include specific error code and explaining comment.
- Use `==` for equality, `is` for identity/sentinels. Prefer `tuple` over `list` for
  immutable returns. Prefer generic types (e.g., `Collection` over `Sequence` when
  only `in` is needed).

## 6. Import Conventions
- Order (enforced by ruff isort): `__future__` | stdlib | third-party | local.
  `import X` before `from X import ...` within each group.
- Lazy imports: `torch`, `botorch`, `gpytorch`, `scipy`, `sklearn` must be
  lazy-loaded inside function bodies. CI tests assert `torch` is not loaded after
  importing BayBE objects. Non-negotiable.
- `TYPE_CHECKING` guard: Imports only for annotations go inside
  `if TYPE_CHECKING:`.
- Aliasing: BoTorch types use `Bo` prefix (e.g., `BoAcquisitionFunction`).
- Remove unused imports immediately. Module-level imports preferred except for
  optional dependency gating.

## 7. Docstring and Comment Style
### Google-Style Docstrings (enforced by ruff + pydoclint)
- Module: One-line description.
- Class: Describe purpose. Attribute docs go below each field, not in class
  docstring.
- Method/function: `Args:`, `Returns:`, `Raises:` sections.
- Types from annotations only (not in docstrings). All raised exceptions documented.
- Imperative mood for summary line (D401). D105/D107 exempted.
- Even private properties and `cached_property` need docstrings.
- Overridden methods inherit docstrings via `@override`.
- Sphinx roles for cross-refs: `:func:`, `:class:`, `:meth:`. Double backticks for
  literals.
- Attrs validators get `# noqa: DOC101, DOC103` (pydoclint confused by
  `(self, attribute, value)` signature).

### Comment Patterns
- `# TODO:` — planned improvements.
- `# IMPROVE:` — enhancement ideas (project-specific).
- `# FIXME[tag]:` — known problems.
- `# NOTE:` — important context.
- `##### Section #####` separators in rare occasions where justified.
- `# >>>>>>>>>> Deprecation` / `# <<<<<<<<<< Deprecation` block delimiters.
- Do not use other comment styles such as `-----`.

## 8. Error Handling
- Custom errors and warnings can be found in `baybe/exceptions.py`.
- f-strings with self-documenting expressions: `f"{batch_size=}"`.
- Always chain exceptions: `raise ... from ex`.
- Warnings originate at source level. Use specific warning classes, not bare `Warning`.

## 9. Serialization
- Built on `cattrs` with central global `Converter` in `baybe/serialization/core.py`.
- `SerialMixin` provides `to_dict()`/`from_dict()`, `to_json()`/`from_json()`.
- Polymorphic deserialization via `"type"` field on abstract bases.
- DataFrames serialized via pickle + base64.
- `Campaign` includes `"version"` field. `"constructor"` key enables alternative
  classmethod deserialization.
- Register custom cattrs hooks at module level after class definition.
- Always copy caller-provided dicts before `pop()`/mutation.

## 10. Deprecation Strategy
Three tiers:
1. Soft: `warnings.warn(..., DeprecationWarning)` — message says what to use
   instead and when removal happens.
2. Hard: `DeprecationError` — fully removed features that must interrupt execution.
3. Serialization-level: Custom cattrs hooks redirect legacy class names.

- Deprecated class names become wrapper functions that warn and delegate.
- Mark blocks with `# >>>>>>>>>> Deprecation` / `# <<<<<<<<<< Deprecation`.
- Old env var names auto-translated with warning at settings init.
- Private cross-module imports acceptable temporarily during deprecation; clean up
  when expired.
- Test in `tests/test_deprecations.py`.

## 11. Validation Patterns
- Inline validators: `field(validator=(instance_of(str), min_len(1)))`, `in_()`,
  `deep_iterable()`, custom `finite_float`, `gt()`.
- Method validators: `@_field.validator` with `# noqa: DOC101, DOC103` for
  validators needing `self` access.
- Cross-field: `__attrs_post_init__` when validation involves multiple fields.
- Converters: `field(converter=to_searchspace)` for automatic type coercion.
- Reusable validators in `baybe/utils/validation.py`: `finite_float`,
  `non_nan_float`, `non_inf_float`, `validate_not_nan`, `validate_target_input`,
  `validate_parameter_input`, `validate_object_names`.
- Cache invalidation: `on_setattr` hooks on mutable fields.

## 12. Optional Dependencies
1. Detection (`baybe/_optional/info.py`): `importlib.util.find_spec()` sets boolean
   flags (`CHEM_INSTALLED`, `ONNX_INSTALLED`, etc.) without importing.
2. Guarded imports (`baybe/_optional/<dep>.py`): Import or raise
   `OptionalImportError` with pip install instructions.
3. In tests: `@pytest.mark.skipif(not CHEM_INSTALLED, ...)`. Use `pytestmark` for
   module-level skips.

## 13. Public API
- Every package defines `__all__` (always a `list`). Only concrete user-facing classes
  exported.
- Top-level exports: `Campaign`, `Settings`, `active_settings`, `__version__`.
- Subpackage `__init__.py` re-exports for convenient imports.
- Aliases allowed (e.g., `EI = ExpectedImprovement`).

## 14. Tooling and CI
| Tool | Config | Purpose |
|------|--------|---------|
| `ruff` | `ruff.toml` | Lint (D, E4, E7, E9, E501, F, I, W, PL) + format. Line length 88. |
| `flake8`+`pydoclint` | `pydoclint.toml` | Docstring consistency (DOC rules only) |
| `mypy` | `mypy.ini` | Type checking. `explicit-override`, `unused-ignore` enabled. |
| `pyupgrade` | `.pre-commit-config.yaml` | Python 3.10+ syntax |

Pre-commit hooks: pydoclint, ruff (lint+format), uv-lock, pyupgrade.
Tox environments: `fulltest`, `coretest`, `lint`, `mypy`, `audit`, `docs`.
Coverage: 70% overall, 45% per-file minimum.
CI runs on push/PR to `main`/`dev/**`: changelog check, lint, typecheck, audit,
coretest, fulltest. Docs and benchmarks on releases.
For each development, ensure `tox -e mypy-py310` runs without problems.

## 15. Git and PR Workflow
- Feature branches from `main`. `dev/**` for development. Naming: `bug/`, `refactor/`,
  `feature/`, `benchmarks/`, `docs/`.
- Rebase preferred for linear history.
- CHANGELOG.md updated in every PR (CI enforced). Specific entries, complete
  sentences. Commit named "Update CHANGELOG" as last commit.
- Use imperative in commit header, e.g. "Add", "Fix", "Rework", "Handle", "Adjust", etc.
- Keep commit body short and informative. Do not add commit body if it has no
  additional info compared to the header.
- Pre-commit must pass. Clean history: squash add/revert pairs, no debug prints.
- Create a separate commit for logical steps, neither per-file nor one giant commit.
- When implementing a fix for some pre-existing issue, make a separate commit with
  explanation.

## 16. Anti-Patterns to Avoid
- No monolithic if-else chains — use builder/strategy patterns.
- No position-based argument matching — use name-based.
- No dead/unreachable code.
- No data duplication.
- No eager imports of heavy deps (torch, scipy, sklearn, botorch) — lazy-load.
- Do not mix reserved terminology (`task` = TL only, `comp` = comp-rep only).
- No hardcoded enum values in comments — link the enum.
- No private field names in user-facing messages — use public alias.
- No hardcoded class names in repr/errors — use `self.__class__.__name__`.
- No silent errors. No mutation of caller-provided dicts.
- No stale cross-references. No inconsistent terminology in docstrings.
- No conftest pollution — prefer local fixtures.
- Tests must test what they claim. No duplicated test logic — parametrize.