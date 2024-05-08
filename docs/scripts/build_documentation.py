"""Utility for building the documentation."""
import argparse
import os
import pathlib
from subprocess import check_call, run

from utils import adjust_pictures

from baybe.telemetry import VARNAME_TELEMETRY_ENABLED

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--ignore_examples",
    help="Ignore the examples by not executing them.",
    action="store_true",
)
parser.add_argument(
    "-w",
    "--include_warnings",
    help="Include warnings when processing the examples. The default is ignoring them.",
    action="store_true",
)
parser.add_argument(
    "-f",
    "--force",
    help="Force-build the documentation, even when there are warnings or errors.",
    action="store_true",
)

# Parse input arguments
args = parser.parse_args()
IGNORE_EXAMPLES = args.ignore_examples
INCLUDE_WARNINGS = args.include_warnings
FORCE = args.force


def build_documentation(force: bool = False) -> None:
    """Build the documentation.

    Args:
        force: Force-build the documentation, ignoring any errors or warnings.
    """
    # Directory where the documentation is build.
    build_dir = pathlib.Path("docs/build")

    # The actual call that will be made to build the documentation
    building_call = [
        "sphinx-build",
        "-b",
        "html",
        "docs",
        build_dir,
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


if __name__ == "__main__":
    if not INCLUDE_WARNINGS:
        os.environ["PYTHONWARNINGS"] = "ignore"

    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"

    build_documentation(force=FORCE)

    # CLEANUP
    # Adjust the banner in the index and the README
    adjust_pictures(
        "docs/build/index.html",
        match="banner",
        light_version="banner2",
        dark_version="banner1",
    )
    # Adjust the chemical encoding example picture in the index and the README
    adjust_pictures(
        "docs/build/index.html",
        match="full_lookup",
        light_version="full_lookup_light",
        dark_version="full_lookup_dark",
    )
