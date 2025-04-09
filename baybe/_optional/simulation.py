"""Optional xyzpy import."""

from baybe.exceptions import OptionalImportError

try:
    import xyzpy
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Batch scenario simulation is unavailable because 'xyzpy' is not "
        "installed. "
        "Consider installing BayBE with 'simulation' dependency, "
        "e.g. via `pip install baybe[simulation]`.",
        name="xyzpy",
    ) from ex

__all__ = [
    "xyzpy",
]
