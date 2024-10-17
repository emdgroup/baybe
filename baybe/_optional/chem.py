"""Optional chemistry imports."""

from baybe.exceptions import OptionalImportError

try:
    from rdkit import Chem
    from skfp import fingerprints
    from skfp.bases import BaseFingerprintTransformer
    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Chemistry functionality is unavailable because the necessary optional "
        "dependencies are not installed. "
        "Consider installing BayBE with 'chem' dependency, "
        "e.g. via `pip install baybe[chem]`."
    ) from ex

__all__ = [
    "Chem",
    "fingerprints",
    "BaseFingerprintTransformer",
    "ConformerGenerator",
    "MolFromSmilesTransformer",
]
