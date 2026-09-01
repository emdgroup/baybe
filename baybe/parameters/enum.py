"""Parameter-related enumerations."""

from __future__ import annotations

from enum import Enum, Flag, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter


class _ParameterKind(Flag):
    """Flag enum encoding the kind of a parameter.

    Can be used to express compatibility (e.g. Gaussian process kernel factories)
    with different parameter types via bitwise combination of flags.
    """

    REGULAR = auto()
    """Regular parameter undergoing no special treatment."""

    TASK = auto()
    """Task parameter for transfer learning."""

    FIDELITY = auto()
    """Fidelity parameter for multi-fidelity modelling."""

    @staticmethod
    def from_parameter(parameter: Parameter) -> _ParameterKind:
        """Determine the kind of a parameter from its type."""
        from baybe.parameters.categorical import TaskParameter
        from baybe.parameters.fidelity import (
            CategoricalFidelityParameter,
            NumericalDiscreteFidelityParameter,
        )

        if isinstance(parameter, TaskParameter):
            return _ParameterKind.TASK
        if isinstance(
            parameter,
            (CategoricalFidelityParameter, NumericalDiscreteFidelityParameter),
        ):
            return _ParameterKind.FIDELITY
        return _ParameterKind.REGULAR


class ParameterEncoding(Enum):
    """Generic base class for all parameter encodings."""


class CategoricalEncoding(ParameterEncoding):
    """Available encodings for categorical parameters."""

    OHE = "OHE"
    """One-hot encoding."""

    INT = "INT"
    """Integer encoding."""


class CustomEncoding(ParameterEncoding):
    """Available encodings for custom parameters."""

    CUSTOM = "CUSTOM"
    """User-defined encoding."""


class SubstanceEncoding(ParameterEncoding):
    """Available encodings for substance parameters from
    `scikit-fingerprints <https://scikit-fingerprints.readthedocs.io/>`_ package.
    """  # noqa: D205

    ATOMPAIR = "ATOMPAIR"
    """:class:`skfp.fingerprints.AtomPairFingerprint`"""

    AUTOCORR = "AUTOCORR"
    """:class:`skfp.fingerprints.AutocorrFingerprint`"""

    AVALON = "AVALON"
    """:class:`skfp.fingerprints.AvalonFingerprint`"""

    BCUT2D = "BCUT2D"
    """:class:`skfp.fingerprints.BCUT2DFingerprint`"""

    E3FP = "E3FP"
    """:class:`skfp.fingerprints.E3FPFingerprint`"""

    ECFP = "ECFP"
    """:class:`skfp.fingerprints.ECFPFingerprint`"""

    ELECTROSHAPE = "ELECTROSHAPE"
    """:class:`skfp.fingerprints.ElectroShapeFingerprint`"""

    MORGAN_FP = "MORGAN_FP"
    """
    Deprecated! Uses :class:`skfp.fingerprints.ECFPFingerprint` with ``fp_size=1024``
    and ``radius=4``.
    """

    ERG = "ERG"
    """:class:`skfp.fingerprints.ERGFingerprint`"""

    ESTATE = "ESTATE"
    """:class:`skfp.fingerprints.EStateFingerprint`"""

    FUNCTIONALGROUPS = "FUNCTIONALGROUPS"
    """:class:`skfp.fingerprints.FunctionalGroupsFingerprint`"""

    GETAWAY = "GETAWAY"
    """:class:`skfp.fingerprints.GETAWAYFingerprint`"""

    GHOSECRIPPEN = "GHOSECRIPPEN"
    """:class:`skfp.fingerprints.GhoseCrippenFingerprint`"""

    KLEKOTAROTH = "KLEKOTAROTH"
    """:class:`skfp.fingerprints.KlekotaRothFingerprint`"""

    LAGGNER = "LAGGNER"
    """:class:`skfp.fingerprints.LaggnerFingerprint`"""

    LAYERED = "LAYERED"
    """:class:`skfp.fingerprints.LayeredFingerprint`"""

    LINGO = "LINGO"
    """:class:`skfp.fingerprints.LingoFingerprint`"""

    MACCS = "MACCS"
    """:class:`skfp.fingerprints.MACCSFingerprint`"""

    MAP = "MAP"
    """:class:`skfp.fingerprints.MAPFingerprint`"""

    MHFP = "MHFP"
    """:class:`skfp.fingerprints.MHFPFingerprint`"""

    MORSE = "MORSE"
    """:class:`skfp.fingerprints.MORSEFingerprint`"""

    MQNS = "MQNS"
    """:class:`skfp.fingerprints.MQNsFingerprint`"""

    MORDRED = "MORDRED"
    """:class:`skfp.fingerprints.MordredFingerprint`"""

    PATTERN = "PATTERN"
    """:class:`skfp.fingerprints.PatternFingerprint`"""

    PHARMACOPHORE = "PHARMACOPHORE"
    """:class:`skfp.fingerprints.PharmacophoreFingerprint`"""

    PHYSIOCHEMICALPROPERTIES = "PHYSIOCHEMICALPROPERTIES"
    """:class:`skfp.fingerprints.PhysiochemicalPropertiesFingerprint`"""

    PUBCHEM = "PUBCHEM"
    """:class:`skfp.fingerprints.PubChemFingerprint`"""

    RDF = "RDF"
    """:class:`skfp.fingerprints.RDFFingerprint`"""

    RDKIT = "RDKIT"
    """Deprecated! Uses :class:`skfp.fingerprints.RDKit2DDescriptors`."""

    RDKITFINGERPRINT = "RDKITFINGERPRINT"
    """:class:`skfp.fingerprints.RDKitFingerprint`"""

    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    """:class:`skfp.fingerprints.RDKit2DDescriptorsFingerprint`"""

    SECFP = "SECFP"
    """:class:`skfp.fingerprints.SECFPFingerprint`"""

    TOPOLOGICALTORSION = "TOPOLOGICALTORSION"
    """:class:`skfp.fingerprints.TopologicalTorsionFingerprint`"""

    USR = "USR"
    """:class:`skfp.fingerprints.USRFingerprint`"""

    USRCAT = "USRCAT"
    """:class:`skfp.fingerprints.USRCATFingerprint`"""

    VSA = "VSA"
    """:class:`skfp.fingerprints.VSAFingerprint`"""

    WHIM = "WHIM"
    """:class:`skfp.fingerprints.WHIMFingerprint`"""
