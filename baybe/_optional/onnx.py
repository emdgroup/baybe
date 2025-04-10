"""Optional ONNX import."""

from baybe.exceptions import OptionalImportError

try:
    import onnxruntime
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="onnxruntime", group="onnx") from ex

__all__ = [
    "onnxruntime",
]
