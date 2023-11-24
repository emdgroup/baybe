"""Parameter-related enumerations."""

from enum import Enum


class ParameterEncoding(Enum):
    """Generic base class for all parameter encodings."""


class CategoricalEncoding(ParameterEncoding):
    """Available encodings for categorical parameters."""

    OHE = "OHE"
    """One-hot encoding."""

    INT = "INT"
    """Integer encoding."""


class SubstanceEncoding(ParameterEncoding):
    """Available encodings for substance parameters."""

    MORDRED = "MORDRED"
    """Encoding based on Mordred chemical descriptors."""

    RDKIT = "RDKIT"
    """Encoding based on RDKit chemical descriptors."""

    MORGAN_FP = "MORGAN_FP"
    """Encoding based on Morgan molecule fingerprints."""


class CustomEncoding(ParameterEncoding):
    """Available encodings for custom parameters."""

    CUSTOM = "CUSTOM"
    """User-defined encoding."""
