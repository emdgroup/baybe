"""Optional ONNX import."""

from baybe.exceptions import OptionalImportError

try:
    import onnx
    import onnxruntime

    try:
        from onnx.helper import split_complex_to_pairs  # noqa: F401
    except ImportError:
        from onnx.helper import _split_complex_to_pairs

        onnx.helper.split_complex_to_pairs = _split_complex_to_pairs
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="onnxruntime", group="onnx") from ex

__all__ = [
    "onnxruntime",
]
