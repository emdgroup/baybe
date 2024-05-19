"""Utility for checking the links of the documentation."""

from subprocess import check_call


def check_links() -> None:
    """Check whether the links of the documentation are valid."""
    link_call = [
        "sphinx-build",
        "-b",
        "linkcheck",
        "docs",
        "docs/build",
    ]

    check_call(link_call)
