"""Test the code provided in the docs."""

import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from baybe._optional.info import CHEM_INSTALLED, LINT_INSTALLED
from tests.docs.utils import extract_code_blocks

doc_files = list(map(str, [Path("README.md"), *Path("docs/userguide/").rglob("*.md")]))
"""Files whose code blocks are to be checked."""

doc_files_pseudocode = list(map(str, [Path("docs/userguide/campaigns.md")]))
"""Files containing pseudocode that needs to be checked with injected fixtures."""


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize("file", doc_files, ids=doc_files)
def test_code_executability(file: Path, campaign, searchspace, objective, recommender):
    """The code blocks in the file become a valid python script when concatenated.

    Blocks surrounded with "triple-tilde" are ignored. Fixtures made available to this
    test will be available in the executed code too.
    """
    userguide_code = "\n".join(extract_code_blocks(file, include_tilde=False))

    # Create a fixed namespace, which is provided to exec as both global and local
    # name space. This ensures that all snippets are executed in their own fresh
    # environment unaffected by other snippets. The space for globals and locals must
    # be the same, as otherwise exec uses separate scopes for specific patterns within
    # the snippet (e.g. list comprehensions) causing unknown name errors despite
    # correct import.
    namespace = {
        "__builtins__": __builtins__,
        "campaign": campaign,
        "searchspace": searchspace,
        "objective": objective,
        "recommender": recommender,
    }
    exec(userguide_code, namespace, namespace)


# TODO: Needs a refactoring (files codeblocks should be auto-detected)
@pytest.mark.parametrize("file", doc_files_pseudocode, ids=doc_files_pseudocode)
def test_pseudocode_executability(file: Path, searchspace, objective, recommender):
    """The pseudocode blocks in the file are a valid python script when using fixtures.

    Blocks surrounded with "triple-backticks" are included.
    """
    userguide_pseudocode = "\n".join(extract_code_blocks(file, include_tilde=True))
    exec(userguide_pseudocode)


@pytest.mark.skipif(
    not LINT_INSTALLED, reason="Optional lint dependency not installed."
)
@pytest.mark.parametrize("file", doc_files, ids=doc_files)
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
