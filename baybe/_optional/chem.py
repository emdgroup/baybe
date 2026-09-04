"""Optional chemistry imports."""

from types import ModuleType

from baybe.exceptions import OptionalImportError

try:
    from rdkit import Chem
    from skfp.bases import BaseFingerprintTransformer
    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

except ModuleNotFoundError as ex:
    raise OptionalImportError(name="scikit-fingerprints", group="chem") from ex

__all__ = [
    "BaseFingerprintTransformer",
    "Chem",
    "ConformerGenerator",
    "fingerprints",  # lazily imported via __getattr__  # noqa: F822
    "MolFromSmilesTransformer",
]


def __getattr__(name: str) -> ModuleType:
    """Lazily import torch-dependent skfp submodules."""
    if name == "fingerprints":
        try:
            from skfp import fingerprints
        except ModuleNotFoundError as ex:
            raise OptionalImportError(name="scikit-fingerprints", group="chem") from ex
        return fingerprints
    raise AttributeError(name)
