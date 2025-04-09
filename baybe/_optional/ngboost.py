"""Optional ngboost import."""

from baybe.exceptions import OptionalImportError

try:
    from ngboost import NGBRegressor
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="ngboost") from ex

__all__ = [
    "NGBRegressor",
]
