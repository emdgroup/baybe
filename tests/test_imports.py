"""Import tests."""

import importlib
import pkgutil
import subprocess
import sys

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


WHITELIST = [
    "baybe.acquisition.adapter",
    "baybe.acquisition.partial",
    "baybe.utils.botorch_wrapper",
    "baybe.utils.torch",
]
"""List of modules that are allowed to import torch."""


def test_torch_lazy_loading():
    """Torch does not appear in the module list after loading BayBE modules."""
    imps = "\n".join([f"import {i}" for i in find_modules() if i not in WHITELIST])
    code = "\n".join(
        [
            "import sys",
            f"{imps}",
            "exit(int('torch' in sys.modules.keys()))",
        ]
    )
    python_interpreter = sys.executable
    result = subprocess.call([python_interpreter, "-c", code])
    assert result == 0
