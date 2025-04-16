"""Optional polars import."""

from baybe.exceptions import OptionalImportError

try:
    import polars
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="polars", group="polars") from ex

__all__ = [
    "polars",
]
