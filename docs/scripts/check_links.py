"""Utility for checking the external links of the documentation.

This script can be run standalone via ``tox -e linkcheck`` or directly with
``uv run --locked --extra docs docs/scripts/check_links.py``.
"""

from subprocess import check_call


def check_links() -> None:
    """Check whether the external links of the documentation are valid."""
    link_call = [
        "sphinx-build",
        "-b",
        "linkcheck",
        "docs",
        "docs/build",
    ]

    check_call(link_call)


if __name__ == "__main__":
    check_links()
