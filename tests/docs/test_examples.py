"""Test if all examples can be run without error."""

import runpy
from pathlib import Path

import pytest

from baybe.surrogates import _ONNX_INSTALLED
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

_CHEM_INSTALLED = _MORDRED_INSTALLED and _RDKIT_INSTALLED


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CHEM_INSTALLED and _ONNX_INSTALLED), reason="skipped for core tests"
)
@pytest.mark.parametrize("example", Path("examples/").rglob("*.py"))
def test_example(example: Path):
    """Test an individual example by running it."""
    runpy.run_path(str(example))
