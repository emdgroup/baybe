"""Import tests."""

import importlib
import pkgutil

import pytest


def find_modules() -> list[str]:
    """Return all BayBE module names."""
    package = importlib.import_module("baybe")
    return [
        name
        for _, name, _ in pkgutil.walk_packages(
            path=package.__path__, prefix=package.__name__ + "."
        )
    ]


@pytest.mark.parametrize("module", find_modules())
def test_imports(module: str):
    """All modules can be imported without throwing errors."""
    importlib.import_module(module)
