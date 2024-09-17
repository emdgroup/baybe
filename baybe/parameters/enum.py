"""Parameter-related enumerations."""

import warnings
from enum import Enum


class ParameterEncoding(Enum):
    """Generic base class for all parameter encodings."""


class CategoricalEncoding(ParameterEncoding):
    """Available encodings for categorical parameters."""

    OHE = "OHE"
    """One-hot encoding."""

    INT = "INT"
    """Integer encoding."""


class SubstanceEncodingAliases(ParameterEncoding):
    """Aliases for SubstanceEncoding values."""

    MORGAN_FP = "ECFP"


class SubstanceEncoding(ParameterEncoding):
    """Available encodings for substance parameters."""

    AtomPairFingerprint = "ATOMPAIR"
    AutocorrFingerprint = "AUTOCORR"
    AvalonFingerprint = "AVALON"
    E3FPFingerprint = "E3FP"
    ECFPFingerprint = "ECFP"
    ERGFingerprint = "ERG"
    EStateFingerprint = "ESTATE"
    FunctionalGroupsFingerprint = "FUNCTIONALGROUPS"
    GETAWAYFingerprint = "GETAWAY"
    GhoseCrippenFingerprint = "GHOSECRIPPEN"
    KlekotaRothFingerprint = "KLEKOTAROTH"
    LaggnerFingerprint = "LAGGNER"
    LayeredFingerprint = "LAYERED"
    LingoFingerprint = "LINGO"
    MACCSFingerprint = "MACCS"
    MAPFingerprint = "MAP"
    MHFPFingerprint = "MHFP"
    MORSEFingerprint = "MORSE"
    MQNsFingerprint = "MQNS"
    MordredFingerprint = "MORDRED"
    PatternFingerprint = "PATTERN"
    PharmacophoreFingerprint = "PHARMACOPHORE"
    PhysiochemicalPropertiesFingerprint = "PHYSIOCHEMICALPROPERTIES"
    PubChemFingerprint = "PUBCHEM"
    RDFFingerprint = "RDF"
    RDKit2DDescriptorsFingerprint = "RDKIT2DDESCRIPTORS"
    RDKitFingerprint = "RDKIT"
    SECFPFingerprint = "SECFP"
    TopologicalTorsionFingerprint = "TOPOLOGICALTORSION"
    USRCATFingerprint = "USRCAT"
    USRFingerprint = "USR"
    WHIMFingerprint = "WHIM"

    @classmethod
    def _missing_(cls, value):
        """Backward compatibility of enum values.

        Enable backwards compatibility of value names that
        differ between SKFP and previous version.
        """
        if value in SubstanceEncodingAliases.__members__:
            replace = SubstanceEncodingAliases[str(value)].value
            warnings.warn(
                f"Fingerprint name {value} has changed and will be disabled in "
                f"a future version. Use fingerprint name {replace} instead.",
                DeprecationWarning,
            )
            return cls(replace)
        else:
            return super()._missing_(value)


class CustomEncoding(ParameterEncoding):
    """Available encodings for custom parameters."""

    CUSTOM = "CUSTOM"
    """User-defined encoding."""
