"""Optional chemistry imports."""

from baybe.exceptions import OptionalImportError

try:
    from mordred import Calculator, descriptors  # noqa: F401
    from rdkit import Chem, RDLogger  # noqa: F401
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect  # noqa: F401
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Chemistry functionality is unavailable because the necessary optional "
        "dependencies are not installed. "
        "Consider installing BayBE with 'chem' dependency, "
        "e.g. via `pip install baybe[chem]`."
    ) from ex
