"""Optional polars import."""

from baybe.exceptions import OptionalImportError

try:
    import polars
except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Polars functionality is unavailable because 'polars' is not installed. "
        "Consider installing BayBE with 'polars' dependency, e.g. via "
        "`pip install baybe[polars]`."
    ) from ex

__all__ = [
    "polars",
]
