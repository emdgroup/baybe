"""Test if all examples can be run without error."""

import os
import runpy
from pathlib import Path

import pytest

from baybe.surrogates import _ONNX_INSTALLED

from ..conftest import _CHEM_INSTALLED

# Run these tests in reduced settings
_SMOKE_TEST_CACHE = os.environ.get("SMOKE_TEST", None)
os.environ["SMOKE_TEST"] = "true"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CHEM_INSTALLED and _ONNX_INSTALLED), reason="skipped for core tests"
)
@pytest.mark.parametrize("example", Path("examples/").rglob("*.py"))
def test_example(example: Path):
    """Test an individual example by running it."""
    runpy.run_path(str(example))


if _SMOKE_TEST_CACHE is not None:
    os.environ["SMOKE_TEST"] = _SMOKE_TEST_CACHE
else:
    os.environ.pop("SMOKE_TEST")
