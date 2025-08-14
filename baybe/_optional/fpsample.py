"""Optional fpsample imports."""

from baybe.exceptions import OptionalImportError

try:
    from fpsample import fps_sampling
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="fpsample") from ex

__all__ = ["fps_sampling"]
