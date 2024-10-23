"""Optional import for diagnostics utilities."""

from baybe.exceptions import OptionalImportError

try:
    import shap
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Explainer functionality is unavailable because 'diagnostics' is not installed."
        " Consider installing BayBE with 'diagnostics' dependency, e.g. via "
        "`pip install baybe[diagnostics]`."
    ) from ex

__all__ = [
    "shap",
]
