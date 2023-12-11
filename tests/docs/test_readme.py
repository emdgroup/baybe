"""Test the code provided in the README."""

import re


def extract_code_blocks(filename: str) -> str:
    """Extract all python code blocks from the specified file into a single string."""
    with open(filename, "r") as file:
        contents = file.read()

    code_blocks = re.findall(r"```python\s+(.*?)\s+```", contents, flags=re.DOTALL)
    concatenated_code = "\n".join(code_blocks)

    return concatenated_code


def test_readme():
    """The blocks in the README become a valid python script when concatenated."""
    readme_code = extract_code_blocks("README.md")
    exec(readme_code)
