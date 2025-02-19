"""Optional import for the insights subpackage."""

from baybe.exceptions import OptionalImportError

try:
    import shap
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Explainer functionality is unavailable because 'insights' is not installed."
        " Consider installing BayBE with 'insights' dependency, e.g. via "
        "`pip install baybe[insights]`."
    ) from ex

__all__ = [
    "shap",
]
