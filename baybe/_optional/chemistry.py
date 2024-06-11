"""Optional chemistry imports."""

from baybe.exceptions import OptionalImportError

try:
    from mordred import Calculator, descriptors
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Chemistry functionality is unavailable because the necessary optional "
        "dependencies are not installed. "
        "Consider installing BayBE with 'chem' dependency, "
        "e.g. via `pip install baybe[chem]`."
    ) from ex

__all__ = [
    "descriptors",
    "Calculator",
    "Chem",
    "GetMorganFingerprintAsBitVect",
    "RDLogger",
]
