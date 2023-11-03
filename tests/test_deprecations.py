# pylint: disable=import-outside-toplevel,unused-import

"""Deprecation tests."""

import pytest


def test_deprecated_baybe_class():
    """Using the deprecated ```BayBE``` class should raise a warning."""
    with pytest.warns(DeprecationWarning):
        from baybe import BayBE

        try:
            BayBE()
        except TypeError:
            pass


def test_moved_objective():
    """Importing ```Objective``` from ```baybe.targets``` should raise a warning."""
    with pytest.warns(DeprecationWarning):
        from baybe.targets import Objective

        try:
            Objective()
        except TypeError:
            pass


def test_renamed_surrogate():
    """Importing from ```baybe.surrogate``` should raise a warning."""
    with pytest.warns(DeprecationWarning):
        from baybe.surrogate import Surrogate  # noqa: F401
