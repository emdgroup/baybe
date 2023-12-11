"""Utilities for doc testing."""

import re
from pathlib import Path
from typing import Union


def extract_code_blocks(path: Union[str, Path]) -> str:
    """Extract all python code blocks from the specified file into a single string."""
    contents = Path(path).read_text()
    code_blocks = re.findall(r"```python\s+(.*?)\s+```", contents, flags=re.DOTALL)
    concatenated_code = "\n".join(code_blocks)

    return concatenated_code
