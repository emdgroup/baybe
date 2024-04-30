"""Utility for building the documentation."""

from subprocess import check_call, run


def build_documentation(force: bool = False) -> None:
    """Build the documentation.

    Args:
        force: Force-build the documentation, ignoring any errors or warnings.
    """
    # The actual call that will be made to build the documentation
    building_call = [
        "sphinx-build",
        "-b",
        "html",
        "docs",
        "docs/build",
        "-n",  # Being nitpicky
        "-W",  # Fail when encountering an error or a warning
    ]

    if force:
        print("Force-building the documentation, ignoring errors and warnings.")
        # In force mode, we do not want to fail, even if an error code is returned.
        # Hence, we use run instead of check_call
        run(building_call + ["--keep-going"], check=False)
    else:
        check_call(building_call)
