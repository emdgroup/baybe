"""Utilities for doc testing."""

import re
from pathlib import Path
from typing import List, Union


def extract_code_blocks(
    path: Union[str, Path], include_tilde: bool = True
) -> List[str]:
    """Extract all python code blocks from the specified file."""
    contents = Path(path).read_text()
    pattern = (
        r"(?:```|~~~)python\s+(.*?)\s+(?:```|~~~)"
        if include_tilde
        else r"```python\s+(.*?)\s+```"
    )
    code_blocks = re.findall(pattern, contents, flags=re.DOTALL)

    return code_blocks
