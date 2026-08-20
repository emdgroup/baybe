"""Utility for checking the cross-references of the documentation."""

from subprocess import check_call


def check_links() -> None:
    """Check that documentation cross-references resolve (external links: lychee)."""
    check_call(
        [
            "sphinx-build",
            "-b",
            "dummy",
            "docs",
            "docs/build",
            "-n",
            "-W",
        ]
    )
