"""Tests for JSON serialization."""

from contextlib import nullcontext
from json import JSONDecodeError
from pathlib import Path

import pytest

from baybe.parameters.numerical import NumericalDiscreteParameter


@pytest.mark.parametrize("overwrite", [False, True], ids=["no_overwrite", "overwrite"])
@pytest.mark.parametrize("exists", [False, True], ids=["no_exists", "exists"])
@pytest.mark.parametrize("as_string", [False, True], ids=["path", "string"])
def test_json_serialization_to_file(
    tmp_path, exists: bool, overwrite: bool, as_string: bool
):
    """JSON (de)serialization to/from a file works in all its variants."""
    file_path = tmp_path / "tempfile"
    if exists:
        file_path.touch()
    if as_string:
        file_path = str(file_path)
    with (
        pytest.raises(FileExistsError, match="explicitly set")
        if exists and not overwrite
        else nullcontext()
    ):
        p = NumericalDiscreteParameter("p", [0, 1])
        p.to_json(file_path, overwrite=overwrite)
        p2 = NumericalDiscreteParameter.from_json(file_path)
        assert p == p2


def test_json_serialization_to_filelike(tmp_path):
    """Objects can be JSON-(de)serialized to/from file-likes."""
    file_like = tmp_path / "tempfile"
    with open(file_like, "w") as f:
        p = NumericalDiscreteParameter("p", [0, 1])
        p.to_json(f)
    with open(file_like) as f:
        p2 = NumericalDiscreteParameter.from_json(f)
    assert p == p2


@pytest.mark.parametrize(
    "channel", ["string", "file_as_string", "file_as_path", "filelike"]
)
def test_invalid_json_parsing(tmp_path, channel: str):
    """JSON decoding errors are propagated regardless of the source format."""
    invalid_json = r"{missing_quotes: abc}"
    if channel == "string":
        source = invalid_json
    else:
        source = tmp_path / "tempfile"
        Path(source).write_text(invalid_json)
        if channel == "filelike":
            source = open(source)
        if channel == "file_as_string":
            source = str(source)
        if channel == "file_as_path":
            source = Path(source)

    with pytest.raises(
        JSONDecodeError, match="Expecting property name enclosed in double quotes"
    ):
        NumericalDiscreteParameter.from_json(source)
