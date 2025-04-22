"""Optional import for the insights subpackage."""

from baybe.exceptions import OptionalImportError

try:
    import shap
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="shap", group="insights") from ex

__all__ = [
    "shap",
]
