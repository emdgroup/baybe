# Contributing to the development of BayBE

This overview describes the basic aspects that are relevant when developing code for BayBE.
Note that this is still **under development**.

## Writing docstrings

The docstrings that are used for BayBE are based on the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
These docstrings are used to automatically create the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/index.html).

The infrastructure used to host the current documentation as well as the design decision that were taken when developing BayBE make it necessary to adhere to the following guidelines when writing docstrings.

- The basic rules of the Google Python Style Guide apply, so most importantly:
    * docstrings need to have a short one-line summary at the top, 
    * an optional extended summary or description below and
    * all relevant sections (`Args`, `Raises`, ...).
- Each function needs to have a docstring. The only exception are functions that inherit their docstring from a parent class. In this case, the following comments should be added:
    * At the end of the line containing the `def` for the function, add `# noqa: D102` to disable that error.
    * Have the comment `# See base class.` as the first line of the function.
- Note that `pydocstyle` does not enforce docstrings for private methods. If a private function has a docstring, `pydocstyle` acknowledges it and checks it.
- Function signatures need to have type hints for both inputs and the return type.
- Type hints should not be added to the docstrings.
- When referencing another class, function, or similar, use the syntax ``:func:`path.to.function` `` where `func` should be replaced by the respective keyword.
- When parts of the comment should appear as `code` in the docstring, use double backticks ``.
- Since we use [attrs](https://www.attrs.org/en/stable/) for writing classes,  initialization functions are not documented. Instance attributes thus need to be documented using a docstring in the line below their declaration.
- Class variables are documented by adding a docstring in the line below their declaration.
- When an inherited class sets one of the instance attributes, this attribute needs to be documented in the docstring of the inherited class.
- Magic functions do not require a docstring.
- Some special rules apply to writing docstrings for validators:
    * All validators should begin with `_validate`.
    * The docstring of a validator should contain a one-line description of what is being validated as well as a `Raises:` section.
    * If necessary, a validator's docstring can contain a more detailed additional description.
    * Validators should **not** have an `Args:` section. 
    * Since these guidelines raise errors for  [pydoclint](https://github.com/jsh9/pydoclint), add `# noqa: DOC101, DOC103` to the same line as the `def` keyword of the declared validator to disable the errors.
- For custom [cattrs](https://catt.rs/) (un-)structuring hooks, a one-line docstring is sufficient.

## Adding functionality
For most parts, BayBE's code is organized into different subpackages. When 
extending its functionality (for instance, by adding new component subclasses), make 
sure that the newly written code is well integrated into the existing package and 
module hierarchy. In particular, public functionality should be imported into the 
appropriate high-level namespaces for easy user import. For an example, see the
[parameter namespace](baybe.parameters).
