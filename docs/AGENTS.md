# AGENTS.md — BayBE Documentation Conventions

## Toolchain
Sphinx + Furo theme + MyST-Parser (Markdown). Also: sphinx-autodoc-typehints,
sphinxcontrib-bibtex, jupytext (example conversion), autosummary (API reference).
Config in `docs/conf.py`.

## Structure
- `docs/concepts/` — Concept-oriented pages (getting recommendations, simulation,
  serialization, etc.)
- `docs/components/` — Component reference pages (campaigns, parameters, targets,
  objectives, constraints, etc.)
- `docs/faq.md` — FAQ with dropdown directives
- `docs/known_issues.md` — Known issues with fix suggestions
- `docs/templates/` — RST templates for autosummary
- `docs/scripts/` — Build scripts (`build_documentation.py`, `build_examples.py`)

## Concepts and Components
- Concepts contain explanations for high-level ideas and workflows
- Components contain reference pages for specific BayBE building blocks
- Provide runnable code snippets, use admonitions, link to relevant examples. Keep
  the snippets very concise 
- Use fenced code blocks with triple backticks (e.g. ```python) for runnable code that
  is also tested in the doc tests
- Use fenced code blocks with tildes (e.g. ~~~python) for code examples that cannot be
  tested, e.g. if they reference objects that do not exist within the snippet or the
  test fixtures

## MyST Syntax Reference
- Cross-refs: "{class}`~baybe.campaign.Campaign`",
  "{meth}`~baybe.campaign.Campaign.recommend`", "{doc}`/components/campaigns`".
  The `~` shortens display to final component
- Admonitions: `{admonition} Title` with `:class: note|important|tip|warning|caution|info`.
- Anchors: `(ANCHOR_NAME)=` above a heading, reference via "{ref}`ANCHOR_NAME`".
- For math use: `$...$` inline, `$$...$$` display, `{math}` directive.

## Writing Style
- Precise wording ("possible" vs "optional" — "optional" has typing meaning).
- Use consistent terminology within a page (don't mix `search space` / `searchspace` /
  `SearchSpace`).
- Double backticks for literals. Keep cross-references up-to-date when paths change.