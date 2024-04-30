"""Utility for checking the links of the documentation."""

from subprocess import check_call, run


def check_links(force: bool = False) -> None:
    """Check whether the links of the documentation are valid.

    Args:
        force: Force-check the links, even if there are errors or wanrings.
    """
    # The call for checking external links.
    link_call = [
        "sphinx-build",
        "-b",
        "linkcheck",
        "docs",
        "docs/build",
    ]

    if force:
        print("Force-building the documentation, ignoring errors and warnings.")
        # In force mode, we do not want to fail, even if an error code is returned.
        # Hence, we use run instead of check_call
        run(link_call, check=False)
    else:
        check_call(link_call)
