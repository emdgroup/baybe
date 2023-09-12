# Contributing to the development of BayBE

This overview describes the basic aspects that are relevant when developing code for BayBE.
Note that this is stil **under development**.

## Writing docstrings

The docstrings that are used for BayBE are based on the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
These docstrings are used to automatically create the documentation using [Sphinx](https://www.sphinx-doc.org/en/master/index.html).

The infrastructure used to host the current documentation as well as the design decision that were taken when developing BayBE make it necessary to adhere to the following guidelines when writing docstrings.

- The basic rules of the Google Python Style Guide apply. That is, docstrings should have a short one-line summary at the top, an optional extended summary or description below and all relevant sections.
- Each function needs to have a docstring.
- Function signatures need to have type hints for both inputs and the return type.
- Since Sphinx uses these to deduce the type hints in the documentation, type hints should not be added to the docstrings.
- When making a reference to another class, function, or similar, use the syntax ``:py:func:`path.to.function` `` where `func` should be replaced by the respective keyword. Note that `path.to.function` should be the path that you would use if you were to perform an import, e.g., `baybe.surrogate.Surrogate.posterior`.
- When parts of the comment should appear as `code` in the docstring, it is necessary to use triple backticks ``` when enclosing the corresponding parts.
- Since we mainly use [attrs](https://www.attrs.org/en/stable/) for writing classes, the documentation of initialization functions needs to be done in the class docstring. In particular, instance attributes need to be documented there.
- Class variables are documented by adding a docstring in the line below their declaration.
- When an inherited class sets one of the instance attributes, this attribute needs to be documented in the docstring of the inherited class.