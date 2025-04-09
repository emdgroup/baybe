"""Optional ngboost import."""

from baybe._optional.info import _ERROR_MESSAGE
from baybe.exceptions import OptionalImportError

try:
    from ngboost import NGBRegressor
except ModuleNotFoundError as ex:
    raise OptionalImportError(_ERROR_MESSAGE.format(package="ngboost")) from ex

__all__ = [
    "NGBRegressor",
]
