"""
Test if all examples can be run without error.
"""

import runpy
from pathlib import Path

import pytest

example_list = list(Path(".").glob("baybe/examples/*/*.py"))

EXAMPLES = [str(file) for file in example_list]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example(example: str):
    """Test an individual example by running it"""
    runpy.run_path(example)
