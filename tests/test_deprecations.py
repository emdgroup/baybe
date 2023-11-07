# pylint: disable=import-outside-toplevel,unused-import
"""Deprecation tests."""

import pytest

from baybe import BayBE
from baybe.searchspace import SearchSpace
from baybe.targets import Objective


def test_deprecated_baybe_class(parameters, objective):
    """Using the deprecated ```BayBE``` class should raise a warning."""
    with pytest.warns(DeprecationWarning):
        BayBE(SearchSpace.from_product(parameters), objective)


def test_moved_objective(targets):
    """Importing ```Objective``` from ```baybe.targets``` should raise a warning."""
    with pytest.warns(DeprecationWarning):
        Objective(mode="SINGLE", targets=targets)


def test_renamed_surrogate():
    """Importing from ```baybe.surrogate``` should raise a warning."""
    with pytest.warns(DeprecationWarning):
        from baybe.surrogate import GaussianProcessSurrogate  # noqa: F401
