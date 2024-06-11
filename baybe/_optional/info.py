"""Information about availability of optional dependencies."""

from importlib.util import find_spec

# Individual packages
MORDRED_INSTALLED = find_spec("mordred") is not None
ONNX_INSTALLED = find_spec("onnxruntime") is not None
RDKIT_INSTALLED = find_spec("rdkit") is not None
STREAMLIT_INSTALLED = find_spec("streamlit") is not None
XYZPY_INSTALLED = find_spec("xyzpy") is not None

# Package combinations
CHEM_INSTALLED = MORDRED_INSTALLED and RDKIT_INSTALLED
