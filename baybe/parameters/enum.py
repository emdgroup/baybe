"""Parameter-related enumerations."""

from enum import Enum

from baybe._optional.info import CHEM_INSTALLED


class ParameterEncoding(Enum):
    """Generic base class for all parameter encodings."""


class CategoricalEncoding(ParameterEncoding):
    """Available encodings for categorical parameters."""

    OHE = "OHE"
    """One-hot encoding."""

    INT = "INT"
    """Integer encoding."""


# TODO Ideally, this should be turned into a class that can:
#  - return default when CHEM not installed
#  - check if enum is fingerprint
PARAM_SUFFIX_FINGERPRINT = "Fingerprint"

if CHEM_INSTALLED:
    import inspect

    from baybe._optional.chem import BaseFingerprintTransformer, skfp_fingerprints

    AVAILABLE_SKFP_FP = dict(
        inspect.getmembers(
            skfp_fingerprints,
            lambda x: inspect.isclass(x) and issubclass(x, BaseFingerprintTransformer),
        )
    )
    AVAILABLE_SKFP_FP["Default"] = AVAILABLE_SKFP_FP["MordredFingerprint"]
else:
    AVAILABLE_SKFP_FP = {"Default": None}

AVAILABLE_SKFP_FP = {
    (
        name
        if name.endswith(PARAM_SUFFIX_FINGERPRINT)
        else name + PARAM_SUFFIX_FINGERPRINT
    ): fp
    for name, fp in AVAILABLE_SKFP_FP.items()
}

SubstanceEncoding = ParameterEncoding(
    value="SubstanceEncoding", names={k: k for k in AVAILABLE_SKFP_FP.keys()}
)
"""Available encodings for substance parameters."""


class CustomEncoding(ParameterEncoding):
    """Available encodings for custom parameters."""

    CUSTOM = "CUSTOM"
    """User-defined encoding."""
