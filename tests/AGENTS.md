# AGENTS.md — BayBE Testing Conventions

## Framework
**pytest exclusively.** No `unittest.TestCase`. Use `assert`, `pytest.raises`,
`pytest.warns`, `pytest.mark.*`. Config in `pytest.ini`: `--doctest-modules`,
`xfail_strict=True`.

## Test Structure
**Standalone functions**, no test classes.
- Files: `test_<feature>.py`.
- Functions: `test_<descriptive_name>` in `snake_case`.
- Fixtures: `fixture_` prefix with explicit `name=` parameter for reusability:
  `@pytest.fixture(name="campaign") def fixture_campaign(...):`

## Assertion Patterns
- Use `assert` for most checks, no failure message required.
- `pytest.raises(ex, match=...)` for exceptions and `pytest.warns(w, match=...)`. Always
  include a match string to only catch the specific exception.
- `pytest.warns(WarningType)` for warnings.
- `contextlib.nullcontext()` for conditional expectations:
  `with pytest.raises(...) if error_expected else nullcontext():`
- Use pandas helpers like `assert_frame_equal`, `assert_series_equal`,
  `assert_index_equal`.
- Use `assert_roundtrip_consistency` for serialization tests

## Fixture Architecture
- Central `tests/conftest.py` uses **name-based indirection**:
  ```
  parameter_names -> parameters -> searchspace -> campaign
  target_names -> targets -> objective -> campaign
  constraint_names -> constraints -> searchspace -> campaign
  ```
- Tests can override by parametrizing `*_names` fixtures:
  `@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])`
- `ongoing_campaign` is cached per session (keyed by the campaign's JSON) and
  deep-copied per test; identical configurations share measurements, and the campaign
  must be serializable.
- **Prefer local fixtures** over adding to `conftest.py`. The autouse `reset_settings` fixture 
  restores global state before every test. Only one top-level `conftest.py` exists.
  ```python
  @pytest.fixture(name="campaign")
  def fixture_campaign():
      """A minimal campaign for local test use."""
      ...
  ```

## Parametrization
- Avoid creating multiple test functions with similar content. Generalize content
  and make heavy (and possibly stacked) use of `@pytest.mark.parametrize` to test
  required configurations.
- Always provide short human-readable IDs via `pytest.param(..., id="...")` or `ids=`.
- Use `batch_size` 2 for multi-point cases; larger only if the size itself is under
  test.
- Don't parametrize over settings ignored in the given configuration, and never
  directly over sets (non-deterministic order breaks `pytest-xdist`); use `sorted(...)`.
- Use `FAST_OPTIMIZATION_SETTINGS` for test-local `BotorchRecommender`s unless
  optimization quality is under test.
- Tests run in parallel: no order dependence or shared state; write files only to
  `tmp_path`.

## Hypothesis
- Full strategy library in `tests/hypothesis_strategies/` for all domain types.
- Create a new hypothesis strategy and a `@given` roundtrip serialization test for
  every new user-facing class.
- Combined with parametrize via `@given(data=st.data())` + `data.draw(strategy)`.

## Markers
- `@pytest.mark.slow`: Skipped with `--fast` CLI flag
- `@pytest.mark.skipif(not CHEM_INSTALLED, ...)`: Optional dependency gating
- `pytestmark = pytest.mark.skipif(...)`: Module-level skip

## Retry, Mocking, and Test Data
- Use mocking (`unittest.mock`, `Mock(spec=...)`, `patch.object(Class, "method")`,
  `Mock(wraps=...)`, `monkeypatch.setenv/setattr`) for caching verification, env vars
  and suppressing side effects.
- Create testing data/measurements with the `create_fake_input` or
  `add_fake_measurements` utilities

## Test Organization
| Directory | Purpose |
|-----------|---------|
| `tests/` | Top-level feature tests |
| `tests/validation/` | Input validation (per-component) |
| `tests/serialization/` | Roundtrip serialization (per-component) |
| `tests/integration/` | Cross-component integration |
| `tests/constraints/` | Constraint-specific tests |
| `tests/hypothesis_strategies/` | Hypothesis strategy definitions |
| `tests/docs/` | Doc build + example execution tests |
| `tests/utils/` | Utility function tests |