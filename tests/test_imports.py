"""Tests for module imports."""

import importlib
import os
import pkgutil
import subprocess
import sys
from collections.abc import Sequence

import pytest
from pytest import param

pytestmark = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Only possible in FULLTEST environment.",
)

_EAGER_LOADING_EXIT_CODE = 42


def find_modules() -> list[str]:
    """Return all BayBE module names."""
    package = importlib.import_module("baybe")
    return [
        name
        for _, name, _ in pkgutil.walk_packages(
            path=package.__path__, prefix=package.__name__ + "."
        )
    ]


def make_import_check(modules: Sequence[str], target: str) -> str:
    """Create code that tests if importing the given modules also imports the target.

    Args:
        modules: The modules to be imported by the created code.
        target: The target module whose presence is to be checked after the import.

    Returns:
        Code that signals the presence of the target via a non-zero exit code.
    """
    imports = "\n".join([f"import {module}" for module in modules])
    return "\n".join(
        [
            "import sys",
            f"{imports}",
            f"hit = '{target}' in sys.modules.keys()",
            f"exit({_EAGER_LOADING_EXIT_CODE} if hit else 0)",
        ]
    )


@pytest.mark.parametrize("module", find_modules())
def test_imports(module: str):
    """All modules can be imported without throwing errors."""
    importlib.import_module(module)


WHITELISTS = {
    "torch": [
        "baybe.acquisition.partial",
        "baybe.surrogates._adapter",
        "baybe.utils.botorch_wrapper",
        "baybe.utils.torch",
    ],
}
"""Modules (dict values) for which certain imports (dict keys) are permitted."""


@pytest.mark.parametrize(
    ("target", "whitelist"), [param(k, v, id=k) for k, v in WHITELISTS.items()]
)
def test_lazy_loading(target: str, whitelist: Sequence[str]):
    """The target does not appear in the module list after loading BayBE modules."""
    all_modules = find_modules()
    assert (w in all_modules for w in whitelist)
    modules = [i for i in all_modules if i not in whitelist]
    code = make_import_check(modules, target)
    python_interpreter = sys.executable
    result = subprocess.call([python_interpreter, "-c", code])
    assert result == 0


@pytest.mark.parametrize(
    ("target", "module"),
    [param(k, m, id=f"{k}-{m}") for k, v in WHITELISTS.items() for m in v],
)
def test_whitelist_modules_are_true_positives(target, module):
    """The whitelisted modules actually import the target."""
    code = make_import_check([module], target)
    python_interpreter = sys.executable
    result = subprocess.call([python_interpreter, "-c", code])
    assert result == _EAGER_LOADING_EXIT_CODE
