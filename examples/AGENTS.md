# AGENTS.md — BayBE Example Conventions

## Format
Examples are `.py` files, converted to notebooks, executed, then rendered as Markdown
in Sphinx docs. Must be syntactically valid, runnable end-to-end, and fast in CI via
`SMOKE_TEST`.

## Structure
Every example follows this pattern:
1. Title: `# # Short Descriptive Text in Telegraph Style`
2. Introduction: Comment lines describing what the example demonstrates
3. Section headers: `## Subheading` in comments
4. Imports: stdlib / third-party / BayBE / example utils (same ordering as main
   codebase)
5. SMOKE_TEST: `SMOKE_TEST = "SMOKE_TEST" in os.environ` then conditional
   constants: `N_DOE_ITERATIONS = 2 if SMOKE_TEST else 20`
6. Constants: `SCREAMING_SNAKE_CASE` (`DIMENSION`, `BATCH_SIZE`)
7. Inline docs: Explanatory comments between code blocks
8. Print results: `print(recommendation)`
9. Fake measurements: Non-simulation examples use
   `add_fake_measurements(recommendation, campaign.targets)`

## Cross-References
You can embed Sphinx admonitions and cross-references in comments:
"# See {class}`~baybe.campaign.Campaign` for details."

## Directory Layout
Organized by category (`Basics/`, `Backtesting/`, `Constraints_Continuous/`,
`Multi_Target/`, `Serialization/`, `Transfer_Learning/`, etc.). Each category has a
`*_Header.md` for the docs toctree. Shared utilities in `examples/utils.py`.