"""Import tests."""

import importlib
import pkgutil
import subprocess
import sys
from collections.abc import Sequence

import pytest

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
        str: Code that signals the presence of the target via a non-zero exit code.
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


WHITELIST = [
    "baybe.acquisition.adapter",
    "baybe.acquisition.partial",
    "baybe.utils.botorch_wrapper",
    "baybe.utils.torch",
]
"""List of modules that are allowed to import torch."""


def test_torch_lazy_loading():
    """Torch does not appear in the module list after loading BayBE modules."""
    modules = [i for i in find_modules() if i not in WHITELIST]
    code = make_import_check(modules, "torch")
    python_interpreter = sys.executable
    result = subprocess.call([python_interpreter, "-c", code])
    assert result == 0
