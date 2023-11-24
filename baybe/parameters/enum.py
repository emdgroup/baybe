"""Parameter-related enumerations."""

from enum import Enum


class ParameterEncoding(Enum):
    """Generic base class for all parameter encodings."""


class CategoricalEncoding(ParameterEncoding):
    """Available encodings for categorical parameters."""

    OHE = "OHE"
    INT = "INT"


class SubstanceEncoding(ParameterEncoding):
    """Available encodings for substance parameters."""

    MORDRED = "MORDRED"
    RDKIT = "RDKIT"
    MORGAN_FP = "MORGAN_FP"


class CustomEncoding(ParameterEncoding):
    """Available encodings for custom parameters."""

    CUSTOM = "CUSTOM"
