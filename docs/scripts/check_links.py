"""Utility for checking the external links of the documentation."""

from subprocess import check_call


def check_links() -> None:
    """Check whether the external links of the documentation are valid."""
    link_call = [
        "sphinx-build",
        "-b",
        "linkcheck",
        "docs",
        "docs/build",
        "-j",  # Build in parallel
        "auto",
    ]

    check_call(link_call)


if __name__ == "__main__":
    check_links()
