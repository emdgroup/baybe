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
    AUTOCORR = "AUTOCORR"
    AVALON = "AVALON"
    E3FP = "E3FP"
    ECFP = "ECFP"
    MORGAN_FP = "MORGAN_FP"
    ERG = "ERG"
    ESTATE = "ESTATE"
    FUNCTIONALGROUPS = "FUNCTIONALGROUPS"
    GETAWAY = "GETAWAY"
    GHOSECRIPPEN = "GHOSECRIPPEN"
    KLEKOTAROTH = "KLEKOTAROTH"
    LAGGNER = "LAGGNER"
    LAYERED = "LAYERED"
    LINGO = "LINGO"
    MACCS = "MACCS"
    MAP = "MAP"
    MHFP = "MHFP"
    MORSE = "MORSE"
    MQNS = "MQNS"
    MORDRED = "MORDRED"
    PATTERN = "PATTERN"
    PHARMACOPHORE = "PHARMACOPHORE"
    PHYSIOCHEMICALPROPERTIES = "PHYSIOCHEMICALPROPERTIES"
    PUBCHEM = "PUBCHEM"
    RDF = "RDF"
    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    RDKIT = "RDKIT"
    SECFP = "SECFP"
    TOPOLOGICALTORSION = "TOPOLOGICALTORSION"
    USRCAT = "USRCAT"
    USR = "USR"
    WHIM = "WHIM"


fingerprint_name_map: dict[str, str] = {
    "ATOMPAIR": "AtomPairFingerprint",
    "AUTOCORR": "AutocorrFingerprint",
    "AVALON": "AvalonFingerprint",
    "E3FP": "E3FPFingerprint",
    "ECFP": "ECFPFingerprint",
    "MORGAN_FP": "ECFPFingerprint",
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
    "MORSE": "MORSEFingerprint",
    "MQNS": "MQNsFingerprint",
    "MORDRED": "MordredFingerprint",
    "PATTERN": "PatternFingerprint",
    "PHARMACOPHORE": "PharmacophoreFingerprint",
    "PHYSIOCHEMICALPROPERTIES": "PhysiochemicalPropertiesFingerprint",
    "PUBCHEM": "PubChemFingerprint",
    "RDF": "RDFFingerprint",
    "RDKIT2DDESCRIPTORS": "RDKit2DDescriptorsFingerprint",
    "RDKIT": "RDKitFingerprint",
    "SECFP": "SECFPFingerprint",
    "TOPOLOGICALTORSION": "TopologicalTorsionFingerprint",
    "USRCAT": "USRCATFingerprint",
    "USR": "USRFingerprint",
    "WHIM": "WHIMFingerprint",
}
"""Mapping of substance parameter encoding names to fingerprint classes."""
