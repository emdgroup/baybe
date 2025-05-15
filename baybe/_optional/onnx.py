"""Optional ONNX import."""

from packaging.version import Version

from baybe.exceptions import OptionalImportError


def patch_onnx():
    """Monkeypatch onnx module to fix backward compatibility issues."""
    # NOTE: In version 1.18.0, the function `onnx.helper.split_complex_to_pairs`
    #   became private, resulting in a backward-incompatible change.
    #   Because not all consuming packages (e.g. "skl2onnx") have been updated,
    #   we temporarily monkeypatch the module.

    import onnx

    if Version(onnx.__version__) >= Version("1.18.0"):
        from onnx.helper import _split_complex_to_pairs

        onnx.helper.split_complex_to_pairs = _split_complex_to_pairs


try:
    import onnxruntime

    patch_onnx()
except ModuleNotFoundError as ex:
    raise OptionalImportError(name="onnxruntime", group="onnx") from ex

__all__ = [
    "onnxruntime",
]
