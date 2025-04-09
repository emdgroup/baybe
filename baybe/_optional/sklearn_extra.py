"""Optional scikit-learn-extra import."""

from baybe._optional.info import _ERROR_MESSAGE
from baybe.exceptions import OptionalImportError

try:
    from sklearn_extra.cluster import KMedoids
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        _ERROR_MESSAGE.format(package="scikit-learn-extra"),
        name="scikit-learn-extra",
    ) from ex

__all__ = [
    "KMedoids",
]
