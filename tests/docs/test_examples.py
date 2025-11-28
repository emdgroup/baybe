"""Test if all examples can be run without error."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from baybe._optional.info import CHEM_INSTALLED, ONNX_INSTALLED

# Run these tests in reduced settings
_SMOKE_TEST_CACHE = os.environ.get("SMOKE_TEST", None)
os.environ["SMOKE_TEST"] = "true"


paths = [str(x) for x in Path("examples/").rglob("*.py")]

# Examples that are known to fail due to BoTorch limitations
# https://github.com/meta-pytorch/botorch/issues/3085
KNOWN_FAILING_EXAMPLES = {
    "examples/Transfer_Learning/basic_transfer_learning.py",
}


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
    if example in KNOWN_FAILING_EXAMPLES:
        pytest.xfail(
            reason="BoTorch MultiTaskGP cannot predict for tasks not in training "
            "data. See: https://github.com/meta-pytorch/botorch/issues/3085"
        )

    env = os.environ | {
        "PYTHONPATH": os.getcwd(),  # to ensure examples are found
        "MPLBACKEND": "Agg",  # avoid popups resulting from plt.show()
    }
    subprocess.run([sys.executable, example], check=True, env=env)


if _SMOKE_TEST_CACHE is not None:
    os.environ["SMOKE_TEST"] = _SMOKE_TEST_CACHE
else:
    os.environ.pop("SMOKE_TEST")
