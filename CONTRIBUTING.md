# Contributing to the development of BayBE

This overview describes the basic aspects that are relevant when developing code for BayBE.
Note that this is stil **under development**.

## Writing docstrings

The docstrings that are used for BayBE are based on the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
These docstrings are used to automatically create the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/index.html).

The infrastructure used to host the current documentation as well as the design decision that were taken when developing BayBE make it necessary to adhere to the following guidelines when writing docstrings.

- The basic rules of the Google Python Style Guide apply. That is, docstrings should have a short one-line summary at the top, an optional extended summary or description below and all relevant sections.
- Each function needs to have a docstring. The only exception are functions that inherit their docstring from a parent class. In this case, the following comments should be added:
    * At the end of the line containing the `def` for the function, add `# noqa: D102` to disable the pydocstyle error `D102: Missing docstring in public method`.
    * Have the comment `# See base class.` as the first line of the function. This makes it clearer for developers that the docstring of this function can be found in the parent function.
- It is important to mention that `pydocstyle` does not enforce docstrings for private methods and that this is enforced via `pylint`. If a private function has a docstring, `pydocstyle` acknowledges it and also checks it.
- Function signatures need to have type hints for both inputs and the return type.
- Since Sphinx uses these to deduce the type hints in the documentation, type hints should not be added to the docstrings.
- When making a reference to another class, function, or similar, use the syntax ``:py:func:`path.to.function` `` where `func` should be replaced by the respective keyword. Note that `path.to.function` should be the path that you would use if you were to perform an import, e.g., `baybe.surrogate.Surrogate.posterior`.
- When parts of the comment should appear as `code` in the docstring, it is necessary to use triple backticks ``` when enclosing the corresponding parts.
- Since we mainly use [attrs](https://www.attrs.org/en/stable/) for writing classes, the documentation of initialization functions needs to be done in the class docstring. In particular, instance attributes need to be documented there.
- Class variables are documented by adding a docstring in the line below their declaration.
- When an inherited class sets one of the instance attributes, this attribute needs to be documented in the docstring of the inherited class.
- All functions that are used for validation, serialization, initializing defaults and so on should have a one line docstring only. The description of what these functions actually do has to be provided by comments in the code itself.
- Magic functions as well as `__init__` functions do not require a docstring.
- Some special rules apply to writing docstrings for validators:
    * All validators should begin with `_validate`.
    * The docstring of a validator should contain a one-line description of what is being validated as well as a `Raises:` section with one line for each individual error.
    * If necessary, a validator's docstring can contain an additional description of how the validation is being carried out.
    * In particular, validators should **not** have an `Args:` section. Since having a `Raises:` but not an `Args:` section raises errors for  [pydoclint](https://github.com/jsh9/pydoclint), the two errors `DOC101` and `DOC103` need to be exlcuded for validators manually using `noqa`. Note that the corresponding comment `# noqa: DOC101, DOC103` needs to be one the same line as the `def` keyword of the declared validator.