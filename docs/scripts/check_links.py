"""Utility for checking the links of the documentation."""

import argparse
import os
from subprocess import check_call, run

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


if __name__ == "__main__":
    if not INCLUDE_WARNINGS:
        os.environ["PYTHONWARNINGS"] = "ignore"

    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"

    check_links(force=FORCE)
