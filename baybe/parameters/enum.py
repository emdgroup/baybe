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


class CustomEncoding(ParameterEncoding):
    """Available encodings for custom parameters."""

    CUSTOM = "CUSTOM"
    """User-defined encoding."""


class SubstanceEncoding(ParameterEncoding):
    """Available encodings for substance parameters from scikit-fingerprints package.

    For more information on individual fingerprints refer to
    scikit-fingerprints package.
    """

    ATOMPAIR = "ATOMPAIR"
    """AtomPairFingerprint."""

    AUTOCORR = "AUTOCORR"
    """AutocorrFingerprint."""

    AVALON = "AVALON"
    """AvalonFingerprint."""

    E3FP = "E3FP"
    """E3FPFingerprint."""

    ECFP = "ECFP"
    """ECFPFingerprint."""

    MORGAN_FP = "MORGAN_FP"
    """Deprecated!
    As a substitution, ECFP fingerprint
    with fp_size=1024 and radius=4 will be used.
    """

    ERG = "ERG"
    """ERGFingerprint."""

    ESTATE = "ESTATE"
    """EStateFingerprint."""

    FUNCTIONALGROUPS = "FUNCTIONALGROUPS"
    """FunctionalGroupsFingerprint."""

    GETAWAY = "GETAWAY"
    """GETAWAYFingerprint."""

    GHOSECRIPPEN = "GHOSECRIPPEN"
    """GhoseCrippenFingerprint."""

    KLEKOTAROTH = "KLEKOTAROTH"
    """KlekotaRothFingerprint."""

    LAGGNER = "LAGGNER"
    """LaggnerFingerprint."""

    LAYERED = "LAYERED"
    """LayeredFingerprint."""

    LINGO = "LINGO"
    """LingoFingerprint."""

    MACCS = "MACCS"
    """MACCSFingerprint."""

    MAP = "MAP"
    """MAPFingerprint."""

    MHFP = "MHFP"
    """MHFPFingerprint."""

    MORSE = "MORSE"
    """MORSEFingerprint."""

    MQNS = "MQNS"
    """MQNsFingerprint."""

    MORDRED = "MORDRED"
    """MordredFingerprint."""

    PATTERN = "PATTERN"
    """PatternFingerprint."""

    PHARMACOPHORE = "PHARMACOPHORE"
    """PharmacophoreFingerprint."""

    PHYSIOCHEMICALPROPERTIES = "PHYSIOCHEMICALPROPERTIES"
    """PhysiochemicalPropertiesFingerprint."""

    PUBCHEM = "PUBCHEM"
    """PubChemFingerprint."""

    RDF = "RDF"
    """RDFFingerprint."""

    RDKIT = "RDKIT"
    """Deprecated! As a substitution, RDKit2DDescriptors will be used.
    """

    RDKITFINGERPRINT = "RDKITFINGERPRINT"
    """RDKitFingerprint."""

    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    """RDKit2DDescriptorsFingerprint."""

    SECFP = "SECFP"
    """SECFPFingerprint."""

    TOPOLOGICALTORSION = "TOPOLOGICALTORSION"
    """TopologicalTorsionFingerprint."""

    USR = "USR"
    """USRFingerprint."""

    USRCAT = "USRCAT"
    """USRCATFingerprint."""

    WHIM = "WHIM"
    """WHIMFingerprint."""
