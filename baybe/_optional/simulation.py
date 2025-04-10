"""Optional xyzpy import."""

from baybe.exceptions import OptionalImportError

try:
    import xyzpy
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="xyzpy", group="simulation") from ex

__all__ = [
    "xyzpy",
]
