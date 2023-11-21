"""Test if all examples can be run without error."""

import runpy
from pathlib import Path

import pytest

from baybe.surrogates import _ONNX_INSTALLED
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

_CHEM_INSTALLED = _MORDRED_INSTALLED and _RDKIT_INSTALLED


example_list = list(Path(".").glob("examples/*/*.py"))

EXAMPLES = [str(file) for file in example_list]


if _CHEM_INSTALLED and _ONNX_INSTALLED:

    @pytest.mark.slow
    @pytest.mark.parametrize("example", EXAMPLES)
    def test_example(example: str):
        """Test an individual example by running it."""
        runpy.run_path(example)
