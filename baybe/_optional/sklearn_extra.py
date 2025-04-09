"""Optional scikit-learn-extra import."""

from baybe.exceptions import OptionalImportError

try:
    from sklearn_extra.cluster import KMedoids
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="scikit-learn-extra") from ex

__all__ = [
    "KMedoids",
]
