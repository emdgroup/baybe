"""Information about availability of optional dependencies."""

import os
import sys
from contextlib import contextmanager
from importlib.util import find_spec


@contextmanager
def exclude_sys_path(path: str, /):  # noqa: DOC402, DOC404
    """Temporarily remove a specified path from `sys.path`.

    Args:
        path: The path to exclude from the search.
    """
    original_sys_path = sys.path[:]
    if path in sys.path:
        sys.path.remove(path)
    try:
        yield
    finally:
        sys.path = original_sys_path


# Individual packages
with exclude_sys_path(os.getcwd()):
    FLAKE8_INSTALLED = find_spec("flake8") is not None
    LIME_INSTALLED = find_spec("lime") is not None
    ONNX_INSTALLED = find_spec("onnxruntime") is not None
    POLARS_INSTALLED = find_spec("polars") is not None
    PRE_COMMIT_INSTALLED = find_spec("pre_commit") is not None
    PYDOCLINT_INSTALLED = find_spec("pydoclint") is not None
    RUFF_INSTALLED = find_spec("ruff") is not None
    SHAP_INSTALLED = find_spec("shap") is not None
    SKFP_INSTALLED = find_spec("skfp") is not None  # scikit-fingerprints
    STREAMLIT_INSTALLED = find_spec("streamlit") is not None
    XYZPY_INSTALLED = find_spec("xyzpy") is not None

# TODO: `find_spec` does not work for `typos` (probably because it's a rust package),
#   neither seem any versions of `pip show typos` / `pip freeze | grep typos` /
#   `any(pkg.name == "typos" for pkg in pkgutils.iter_modules())` called as a subprocess
#   work in CI (probably because stuff is running inside tox and/or paths need to be
#   configured properly). As a temporary workaround – since the test collection does not
#   directly depend on the flag – we thus simply set it to `True`.
TYPOS_INSTALLED = True

# Information on whether all required packages for certain functionality are available
CHEM_INSTALLED = SKFP_INSTALLED
INSIGHTS_INSTALLED = SHAP_INSTALLED and LIME_INSTALLED
LINT_INSTALLED = all(
    (
        FLAKE8_INSTALLED,
        PRE_COMMIT_INSTALLED,
        PYDOCLINT_INSTALLED,
        RUFF_INSTALLED,
        TYPOS_INSTALLED,
    )
)
