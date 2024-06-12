"""Information about availability of optional dependencies."""

import os
import sys
from contextlib import contextmanager
from importlib.util import find_spec


@contextmanager
def exclude_sys_path(path: str, /):  # noqa: DOC402, DOC404
    """Temporarily remove a specified path from `sys.path`.

    Args:
        path: The path to exclude from the search.
    """
    original_sys_path = sys.path[:]
    if path in sys.path:
        sys.path.remove(path)
    try:
        yield
    finally:
        sys.path = original_sys_path


# Individual packages
with exclude_sys_path(os.getcwd()):
    MORDRED_INSTALLED = find_spec("mordred") is not None
    ONNX_INSTALLED = find_spec("onnxruntime") is not None
    RDKIT_INSTALLED = find_spec("rdkit") is not None
    STREAMLIT_INSTALLED = find_spec("streamlit") is not None
    XYZPY_INSTALLED = find_spec("xyzpy") is not None

# Package combinations
CHEM_INSTALLED = MORDRED_INSTALLED and RDKIT_INSTALLED
