"""Test if all examples can be run without error."""

import os
import subprocess
from pathlib import Path

import pytest

from baybe._optional.info import CHEM_INSTALLED, ONNX_INSTALLED

# Run these tests in reduced settings
_SMOKE_TEST_CACHE = os.environ.get("SMOKE_TEST", None)
os.environ["SMOKE_TEST"] = "true"


paths = [str(x) for x in Path("examples/").rglob("*.py")]


@pytest.mark.slow
@pytest.mark.skipif(
    not (CHEM_INSTALLED and ONNX_INSTALLED), reason="skipped for core tests"
)
@pytest.mark.parametrize("example", paths, ids=paths)
def test_example(example: str):
    """Test an individual example by running it.

    This runs in a separate process and isolated environment due to problems caused by
    monkeypatching in some examples affecting other tests if they were executed in the
    same environment.
    """
    subprocess.run(["python", example], check=True)


if _SMOKE_TEST_CACHE is not None:
    os.environ["SMOKE_TEST"] = _SMOKE_TEST_CACHE
else:
    os.environ.pop("SMOKE_TEST")
