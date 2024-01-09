"""Test the code provided in the docs."""

import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from .utils import extract_code_blocks

doc_files = ["README.md", *Path("docs/userguide/").rglob("*.md")]
"""Files whose code blocks are to be checked."""


@pytest.mark.parametrize("file", doc_files)
def test_code_executability(file: Path):
    """The code blocks in the file become a valid python script when concatenated.

    Blocks surrounded with "triple-tilde" are ignored.
    """
    userguide_code = "\n".join(extract_code_blocks(file, include_tilde=False))
    exec(userguide_code)


@pytest.mark.parametrize("file", doc_files)
def test_code_format(file: Path):
    """The code blocks in the file are properly formatted.

    If it fails, run `pytest` with the `-s` flag to show the necessary format changes.
    """
    code_blocks = extract_code_blocks(file)
    success = True
    for block in code_blocks:
        with NamedTemporaryFile("w+") as f:
            f.write(block)
            f.write("\n")  # the code blocks contain no empty line at the end
            f.flush()
            result = subprocess.run(
                ["ruff", "format", "--check", f"{f.name}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode:
                success = False
                print("\nOriginal Code")
                print("-------------")
                print(block, "\n")
                print("\nReformatted Code")
                print("----------------")
                subprocess.run(
                    ["ruff", "format", f"{f.name}"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                f.seek(0)
                print(f.read(), "\n")
    assert success
