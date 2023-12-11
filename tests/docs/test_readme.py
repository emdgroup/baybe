"""Test the code provided in the README."""

from .utils import extract_code_blocks


def test_readme():
    """The blocks in the README become a valid python script when concatenated."""
    readme_code = extract_code_blocks("README.md")
    exec(readme_code)
