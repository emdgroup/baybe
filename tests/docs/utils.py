"""Utilities for doc testing."""

import re
from pathlib import Path
from textwrap import dedent


def extract_code_blocks(path: str | Path, include_tilde: bool = True) -> list[str]:
    """Extract all python code blocks from the specified file."""
    contents = Path(path).read_text()
    pattern = (
        r"\s*(?:```|~~~)python\n*(.*?)\n*\s*(?:```|~~~)"
        if include_tilde
        else r"\s*```python\n*(.*?)\n*\s*```"
    )
    code_blocks = [dedent(c) for c in re.findall(pattern, contents, flags=re.DOTALL)]

    return code_blocks
