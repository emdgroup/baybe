"""Utilities for doc testing."""

import re
from pathlib import Path
from typing import List, Union


def extract_code_blocks(path: Union[str, Path]) -> List[str]:
    """Extract all python code blocks from the specified file."""
    contents = Path(path).read_text()
    code_blocks = re.findall(r"```python\s+(.*?)\s+```", contents, flags=re.DOTALL)

    return code_blocks
