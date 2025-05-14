"""Optional ONNX import."""

from packaging.version import Version

from baybe.exceptions import OptionalImportError

try:
    import onnx
    import onnxruntime

    # NOTE: In version 1.18.0, the function `onnx.helper.split_complex_to_pairs` was
    # replaced by the "private" function `_split_complex_to_pairs`.
    # Not all dependencies, in particulat "skl2onnx" have been updated to use the
    # new function. To ensure backwards compatibility, we check the version of onnx
    # and if it is lower than 1.18.0, we import the old function.
    # This is a temporary fix until all dependencies are updated.
    # It is important to note that the main branch of "skl2onnx" already uses the new
    # function, so this issue will be resolved in the next release.
    onnx_version = onnx.__version__
    if Version(onnx_version) >= Version("1.18.0"):
        from onnx.helper import _split_complex_to_pairs

        onnx.helper.split_complex_to_pairs = _split_complex_to_pairs
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="onnxruntime", group="onnx") from ex

__all__ = [
    "onnxruntime",
]
