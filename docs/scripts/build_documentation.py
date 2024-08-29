"""Utility for building the documentation."""

import argparse
import os
import pathlib
from subprocess import check_call, run

from build_examples import build_examples
from check_links import check_links
from utils import adjust_pictures

from baybe.telemetry import VARNAME_TELEMETRY_ENABLED

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--run-examples",
    help="Re-run the examples.",
    action="store_true",
)
parser.add_argument(
    "-l",
    "--no_linkcheck",
    help="Do not check the links.",
    action="store_true",
)
parser.add_argument(
    "-r",
    "--full-rebuild",
    help="Perform a full rebuild, independent of `-e` and `-l` flags.",
    action="store_true",
)
parser.add_argument(
    "-w",
    "--include-warnings",
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
RUN_EXAMPLES = args.run_examples
LINKCHECK = not args.no_linkcheck
FULL_REBUILD = args.full_rebuild
INCLUDE_WARNINGS = args.include_warnings
FORCE = args.force


def build_documentation(
    run_examples: bool = False,
    verify_links: bool = False,
    full_rebuild: bool = False,
    force: bool = False,
) -> None:
    """Build the documentation.

    A full build of the documentation consists of converting the examples into jupyter
    notebooks, executing them, transforming them into markdown files, as well as
    checking all links and performing the actual ``sphinx-build``. Such a full build can
    be triggered using the ``full_rebuild`` flag.
    If this flag is not set, this function tries to re-use as much of potentially
    existing structures like already built examples as possible. This behavior can be
    changed by using the other flags.

    Args:
        run_examples: Fully recalculate the examples. If this is ``False`` and no
            folder containing an already built set of examples is found, dummy files
            replicating the structure of the examples are created.
        verify_links: Check both internal and external links.
        full_rebuild: Perform a full rebuild of the documentation, including a
            recalculation of the examples and checking the links. Note that this option
            ignores the choices for ``run_examples`` and ``check_links`` if set to
            ``True.
        force: Force-build the steps, ignoring any errors or warnings.
    """
    examples_directory = pathlib.Path("docs/examples")
    examples_exist = examples_directory.is_dir()

    rerun_examples = run_examples or full_rebuild
    perform_linkcheck = verify_links or full_rebuild

    if rerun_examples:
        build_examples(
            destination_directory=examples_directory,
            dummy=False,
            remove_dir=examples_exist,
        )
    elif not examples_exist:
        # Perform dummy-build of examples in the case that they should not be
        # re-calculated and the folder does not exist.
        build_examples(
            destination_directory=examples_directory,
            dummy=True,
            remove_dir=examples_exist,
        )

    if perform_linkcheck:
        check_links()

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

    print(f"{LINKCHECK=}")

    build_documentation(
        run_examples=RUN_EXAMPLES,
        verify_links=LINKCHECK,
        full_rebuild=FULL_REBUILD,
        force=FORCE,
    )

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
        match="_images/full_lookup_light",
        light_version="_images/full_lookup_light",
        dark_version="_images/full_lookup_dark",
    )
