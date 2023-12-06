# Contributing to BayBE

**All contributions to BayBE are welcome!**

... no matter if bug fixes, new features, or just typo corrections.

To shorten the overall development and review process, this page contains are a
few sections that can make your life easier.

## General Workflow

To implement your contributions in a local development environment,
we recommend the following workflow:

1. Clone a [fork](https://github.com/emdgroup/BayBE/fork) of the repository to 
   your local machine.

1. Create and activate a virtual python environment using one of the supported 
   python versions.

1. Change into the root folder of the cloned repository and install an editable version
   including all development dependencies:
   ```console
   pip install -e '.[dev]'
   ```

1. Run our tests to verify everything works as expected:
   ```console
   pytest
   ```

1. Install our [pre-commit](https://pre-commit.com/) hooks:
   ```console
   pre-commit install
   ```

1. Create a new branch for your contribution:
   ```console
   git checkout -b <your_branch_name>
   ```

1. **Implement your changes.**

1. Optional but recommended to prevent complaints from our CI pipeline:
   **Test your code.**

   There are several test environments you can run via `tox`, each corresponding to a
   [developer tool](#developer-tools) in a certain Python version. 
   You can retrieve all available environments via `tox list`.
   For more information, see our [README about tests](https://github.com/emdgroup/baybe/blob/main/tests/README.md).
   
   For instance, running all code tests in Python 3.11 can be achieved via:
   ```console
   tox -e fulltest-py311
   ```

   If you want to challenge your machine, you can run all checks in all Python versions
   in parallel via:
   ```console
   tox -p
   ```
   
   This can be considered the ultimate one-stop check to make sure your code is ready 
   for merge. 

1. Push the updated branch back to your fork:
   ```console
   git push origin
   ```

1. Open a pull request via Github's web page.

(developer-tools)=
## Developer Tools

In order to maintain a high code quality, we use a variety of code developer tools. 
When following the above described workflow, [pre-commit](https://pre-commit.com/)
will automatically trigger (most) necessary checks during your development process.
In any case, these checks are also conducted in our CI pipeline, which must pass 
before your pull request is considered ready for review. 
If you have questions or problems, simply ask for advice.

| Tool                                                                                            | Purpose                                   |
|:------------------------------------------------------------------------------------------------|:------------------------------------------|
| [ruff](https://docs.astral.sh/ruff/)                                                            | code linting and formatting               |
| [mypy](https://mypy.readthedocs.io/)                                                            | static type checking                      |
| [pydocstyle](http://www.pydocstyle.org/)   <br/> [pydoclint](https://github.com/jsh9/pydoclint) | analyzing docstrings                      |
| [typos](https://github.com/crate-ci/typos)                                                      | basic spell checking                      |
| [pytest](https://docs.pytest.org/)                                                              | testing                                   |
| [pytest-cov](https://pytest-cov.readthedocs.io/)                                                | measuring test coverage                   |
| [sphinx](https://www.sphinx-doc.org/)                                                           | generating our documentation              |
| [pip-audit](https://github.com/pypa/pip-audit)                                                  | detecting vulnerabilities in dependencies |
| [tox](https://tox.wiki/)                                                                        | orchestrating all the above               |

Executing a specific one of these tools is easiest by using the corresponding
[tox](https://tox.wiki/) environment,
```console
tox -e <env>
```
where `<env>` is any of the environment names found via `tox list`.

(code-design)=
## Code Design

When reading BayBE's code, you will notice certain re-occurring design patterns.
These patterns are by no means enforced, but following them can streamline your
own development process:

* We build most our classes with [attrs](https://www.attrs.org/), which is useful 
  for lean class design and attribute validation.
* Our (de-)serialization machinery is built upon [cattrs](https://catt.rs/), separating
  object serialization from class design.
* The modular nature of BayBE's components is reflected in our test suite through
  the use of [hypothesis](https://hypothesis.readthedocs.io/) property tests.

## Extending BayBE's Functionality

For most parts, BayBE's code and functional components are organized into different 
subpackages.
When extending its functionality (for instance, by adding new component subclasses),
make sure that the newly written code is well integrated into the existing package and
module hierarchy.
In particular, public functionality should be imported into the appropriate high-level
namespaces for easier user import. For an example, see our
[parameter namespace](https://github.com/emdgroup/baybe/blob/main/baybe/parameters/__init__.py).

## Writing Docstrings

Our docstrings generally follow the 
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Basic style and consistency checks are automatically performed via 
[pre-commit](https://pre-commit.com/) during development and in our CI pipeline.

Apart from that, we generally recommend adhering to the following guideline:

- Each function should have a docstring containing:
    * a short one-line summary at the top, 
    * an optional extended summary or description below and
    * all relevant sections (`Args`, `Raises`, ...).
  
  Potential exceptions are functions whose docstring is to be fully inherited from a 
  parent class.
  In this case, use `# noqa: D102` to disable the automatic checks locally.

- Use type hints (for variables/constants, attributes, function/method signatures, ...).
  Avoid repeating type hints in docstrings.

- When referencing objects (classes, functions, ...),
  use ``:<key>:`path.to.function` `` where `<key>` is to be replaced with the 
  respective keyword (`class`, `func`, ...)

- Use double backticks for literals like in ``` ``MyString`` ```.

### Docstrings for `attrs` classes 

- Place attribute docstrings below the attribute declaration, not in the class 
  docstring.
  Separate different attributes using a blank line.
  For example:
  ```python
  @define
  class Cookies:
    """A delicious recipe for chocolate-banana cookies."""
  
    chocolate: float
    """Chocolate is naturally measured in terms of floats."""
  
    bananas: int
    """For bananas, we use integers, of course."""
  ```

- Unless another more specific name is suitable, use our default naming convention for 
  `attrs` built-ins (defaults, converters, validators):
  ```python
  @my_attribute.default
  def _default_my_attribute(self): ...
  
  @my_attribute.converter
  def _convert_my_attribute(self): ...
  
  @my_attribute.validator
  def _validate_my_attribute(self, attribute, value): ...
  ```
  A one-line docstring suffices for these methods, but they should have a `Raises:` 
  section if applicable. Linter warnings regarding missing attribute docstrings can be 
  silenced using `# noqa: DOC101, DOC103`.
  