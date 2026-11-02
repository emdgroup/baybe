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


def make_import_check(modules: Sequence[str], targets: Sequence[str]) -> str:
    """Create code that tests if importing the given modules also imports the targets.

    Args:
        modules: The modules to be imported by the created code.
        targets: The target modules whose presence is to be checked after the import.

    Returns:
        Code that signals the presence of all targets via a non-zero exit code. The
        modules are imported one by one and the first module after whose import all
        targets are present is reported on stderr. If no such module exists, the
        missing targets are reported instead.
    """
    return "\n".join(
        [
            "import importlib",
            "import sys",
            f"targets = {list(targets)!r}",
            f"for module in {list(modules)!r}:",
            "    importlib.import_module(module)",
            "    if all(t in sys.modules for t in targets):",
            "        print(f'Importing {module!r} loads {targets}', file=sys.stderr)",
            f"        exit({_EAGER_LOADING_EXIT_CODE})",
            "missing = [t for t in targets if t not in sys.modules]",
            "print(f'Not loaded: {missing}', file=sys.stderr)",
            "exit(0)",
        ]
    )


_modules = find_modules()


@pytest.mark.parametrize("module", _modules)
def test_imports(module: str):
    """All modules can be imported without throwing errors."""
    importlib.import_module(module)


WHITELISTS = {
    "torch": [
        "baybe.acquisition.partial",
        "baybe.acquisition._builder",
        "baybe.objectives.botorch",
        "baybe.surrogates._adapter",
        "baybe.surrogates.gaussian_process.components._gpytorch",
        "baybe.utils.torch",
    ],
    "scipy": [
        "baybe._optional.chem",
        "baybe._optional.insights",
        "baybe._optional.ngboost",
        "baybe.acquisition._builder",
        "baybe.acquisition.partial",
        "baybe.insights",
        "baybe.insights.shap",
        "baybe.objectives.botorch",
        "baybe.surrogates._adapter",
        "baybe.surrogates.gaussian_process.components._gpytorch",
        "baybe.utils.chemistry",
        "baybe.utils.clustering_algorithms",
        "baybe.utils.clustering_algorithms.third_party",
        "baybe.utils.clustering_algorithms.third_party.kmedoids",
    ],
    "sklearn": [
        "baybe._optional.chem",
        "baybe._optional.insights",
        "baybe._optional.ngboost",
        "baybe.insights",
        "baybe.insights.shap",
        "baybe.utils.chemistry",
        "baybe.utils.clustering_algorithms",
        "baybe.utils.clustering_algorithms.third_party",
        "baybe.utils.clustering_algorithms.third_party.kmedoids",
    ],
}
"""Modules (dict values) for which certain imports (dict keys) are permitted."""


@pytest.mark.parametrize(
    ("target", "whitelist"), [param(k, v, id=k) for k, v in WHITELISTS.items()]
)
def test_lazy_loading(target: str, whitelist: Sequence[str]):
    """The target does not appear in the module list after loading BayBE modules."""
    all_modules = find_modules()
    unknown = [w for w in whitelist if w not in all_modules]
    assert not unknown, f"Unknown whitelisted modules: {unknown}"

    modules = [m for m in all_modules if m not in whitelist]
    code = make_import_check(modules, [target])
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


_WHITELISTED_TARGETS: dict[str, list[str]] = {
    m: [t for t, ms in WHITELISTS.items() if m in ms]
    for ms in WHITELISTS.values()
    for m in ms
}
"""The inverted whitelist, mapping modules to their permitted imports."""


@pytest.mark.parametrize(
    ("module", "targets"),
    [param(m, t, id=m) for m, t in _WHITELISTED_TARGETS.items()],
)
def test_whitelist_modules_are_true_positives(module, targets):
    """The whitelisted modules actually import all targets they are whitelisted for.

    All targets of a module are checked within a single subprocess to avoid paying
    the interpreter startup and import costs once per target.
    """
    code = make_import_check([module], targets)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == _EAGER_LOADING_EXIT_CODE, result.stderr
