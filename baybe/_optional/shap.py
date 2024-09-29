"""Optional SHAP import."""

from baybe.exceptions import OptionalImportError

try:
    import shap
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Shapley functionality is unavailable because 'shap' is not installed. "
        "Consider installing BayBE with 'shap' dependency, e.g. via "
        "`pip install baybe[shap]`."
    ) from ex

__all__ = [
    "shap",
]
