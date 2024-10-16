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
    """Available encodings for substance parameters."""

    ATOMPAIR = "ATOMPAIR"
    """AtomPairFingerprint from scikit-fingerprints."""

    AUTOCORR = "AUTOCORR"
    """AutocorrFingerprint from scikit-fingerprints."""

    AVALON = "AVALON"
    """AvalonFingerprint from scikit-fingerprints."""

    E3FP = "E3FP"
    """E3FPFingerprint from scikit-fingerprints."""

    ECFP = "ECFP"
    """ECFPFingerprint from scikit-fingerprints."""

    MORGAN_FP = "MORGAN_FP"
    """Deprecated!"""

    ERG = "ERG"
    """ERGFingerprint from scikit-fingerprints."""

    ESTATE = "ESTATE"
    """EStateFingerprint from scikit-fingerprints."""

    FUNCTIONALGROUPS = "FUNCTIONALGROUPS"
    """FunctionalGroupsFingerprint from scikit-fingerprints."""

    GETAWAY = "GETAWAY"
    """GETAWAYFingerprint from scikit-fingerprints."""

    GHOSECRIPPEN = "GHOSECRIPPEN"
    """GhoseCrippenFingerprint from scikit-fingerprints."""

    KLEKOTAROTH = "KLEKOTAROTH"
    """KlekotaRothFingerprint from scikit-fingerprints."""

    LAGGNER = "LAGGNER"
    """LaggnerFingerprint from scikit-fingerprints."""

    LAYERED = "LAYERED"
    """LayeredFingerprint from scikit-fingerprints."""

    LINGO = "LINGO"
    """LingoFingerprint from scikit-fingerprints."""

    MACCS = "MACCS"
    """MACCSFingerprint from scikit-fingerprints."""

    MAP = "MAP"
    """MAPFingerprint from scikit-fingerprints."""

    MHFP = "MHFP"
    """MHFPFingerprint from scikit-fingerprints."""

    MORSE = "MORSE"
    """MORSEFingerprint from scikit-fingerprints."""

    MQNS = "MQNS"
    """MQNsFingerprint from scikit-fingerprints."""

    MORDRED = "MORDRED"
    """MordredFingerprint from scikit-fingerprints."""

    PATTERN = "PATTERN"
    """PatternFingerprint from scikit-fingerprints."""

    PHARMACOPHORE = "PHARMACOPHORE"
    """PharmacophoreFingerprint from scikit-fingerprints."""

    PHYSIOCHEMICALPROPERTIES = "PHYSIOCHEMICALPROPERTIES"
    """PhysiochemicalPropertiesFingerprint from scikit-fingerprints."""

    PUBCHEM = "PUBCHEM"
    """PubChemFingerprint from scikit-fingerprints."""

    RDF = "RDF"
    """RDFFingerprint from scikit-fingerprints."""

    RDKIT = "RDKIT"
    """Deprecated!"""

    RDKITFINGERPRINT = "RDKITFINGERPRINT"
    """RDKitFingerprint from scikit-fingerprints."""

    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    """RDKit2DDescriptorsFingerprint from scikit-fingerprints."""

    SECFP = "SECFP"
    """SECFPFingerprint from scikit-fingerprints."""

    TOPOLOGICALTORSION = "TOPOLOGICALTORSION"
    """TopologicalTorsionFingerprint from scikit-fingerprints."""

    USR = "USR"
    """USRFingerprint from scikit-fingerprints."""

    USRCAT = "USRCAT"
    """USRCATFingerprint from scikit-fingerprints."""

    WHIM = "WHIM"
    """WHIMFingerprint from scikit-fingerprints."""


fingerprint_name_map: dict[str, str] = {
    "ATOMPAIR": "AtomPairFingerprint",
    "AUTOCORR": "AutocorrFingerprint",
    "AVALON": "AvalonFingerprint",
    "E3FP": "E3FPFingerprint",
    "ECFP": "ECFPFingerprint",
    "ERG": "ERGFingerprint",
    "ESTATE": "EStateFingerprint",
    "FUNCTIONALGROUPS": "FunctionalGroupsFingerprint",
    "GETAWAY": "GETAWAYFingerprint",
    "GHOSECRIPPEN": "GhoseCrippenFingerprint",
    "KLEKOTAROTH": "KlekotaRothFingerprint",
    "LAGGNER": "LaggnerFingerprint",
    "LAYERED": "LayeredFingerprint",
    "LINGO": "LingoFingerprint",
    "MACCS": "MACCSFingerprint",
    "MAP": "MAPFingerprint",
    "MHFP": "MHFPFingerprint",
    "MORGAN_FP": "ECFPFingerprint",  # Deprecated!
    "MORSE": "MORSEFingerprint",
    "MQNS": "MQNsFingerprint",
    "MORDRED": "MordredFingerprint",
    "PATTERN": "PatternFingerprint",
    "PHARMACOPHORE": "PharmacophoreFingerprint",
    "PHYSIOCHEMICALPROPERTIES": "PhysiochemicalPropertiesFingerprint",
    "PUBCHEM": "PubChemFingerprint",
    "RDF": "RDFFingerprint",
    "RDKIT": "RDKit2DDescriptorsFingerprint",  # Deprecated!
    "RDKITFINGERPRINT": "RDKitFingerprint",
    "RDKIT2DDESCRIPTORS": "RDKit2DDescriptorsFingerprint",
    "SECFP": "SECFPFingerprint",
    "TOPOLOGICALTORSION": "TopologicalTorsionFingerprint",
    "USRCAT": "USRCATFingerprint",
    "USR": "USRFingerprint",
    "WHIM": "WHIMFingerprint",
}
"""Mapping of substance parameter encoding names to fingerprint classes."""
