"""Utilities for doc testing."""

import re


def extract_code_blocks(filename: str) -> str:
    """Extract all python code blocks from the specified file into a single string."""
    with open(filename, "r") as file:
        contents = file.read()

    code_blocks = re.findall(r"```python\s+(.*?)\s+```", contents, flags=re.DOTALL)
    concatenated_code = "\n".join(code_blocks)

    return concatenated_code
