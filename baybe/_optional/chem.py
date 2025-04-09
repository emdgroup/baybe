"""Optional chemistry imports."""

from baybe.exceptions import OptionalImportError

try:
    from rdkit import Chem
    from skfp import fingerprints
    from skfp.bases import BaseFingerprintTransformer
    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

except ModuleNotFoundError as ex:
    raise OptionalImportError(name="scikit-fingerprints", group="chem") from ex

__all__ = [
    "Chem",
    "fingerprints",
    "BaseFingerprintTransformer",
    "ConformerGenerator",
    "MolFromSmilesTransformer",
]
