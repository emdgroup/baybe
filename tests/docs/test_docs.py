"""Test the code provided in the docs."""

from pathlib import Path

import pytest

from .utils import extract_code_blocks


def test_readme():
    """The blocks in the README become a valid python script when concatenated."""
    readme_code = extract_code_blocks("README.md")
    exec(readme_code)


@pytest.mark.parametrize("file", Path("docs/userguide/").rglob("*.md"))
def test_userguide(file: Path):
    """The blocks in the user guide become a valid python script when concatenated."""
    userguide_code = extract_code_blocks(file)
    exec(userguide_code)
