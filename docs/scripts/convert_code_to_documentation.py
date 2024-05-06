"""Automatic conversion of python code to markdown files for the documentation."""

import argparse
import os
import pathlib
import shutil
from subprocess import check_call, run

from utils import adjust_pictures, create_example_documentation

from baybe.telemetry import VARNAME_TELEMETRY_ENABLED

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--ignore_examples",
    help="Ignore the examples and do not include them into the documentation.",
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

# We adjust the environment variable if we decide to ignore warnings
if not INCLUDE_WARNINGS:
    os.environ["PYTHONWARNINGS"] = "ignore"

# Disable telemtetry
os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
# Directory where Sphinx builds the documentation
build_dir = pathlib.Path("docs/build")

# The call for checking external links.
link_call = [
    "sphinx-build",
    "-b",
    "linkcheck",
    "docs",
    build_dir,
]
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

# Process examples if required.
# Note that ignoring of examples now happens by simply not executing them.
create_example_documentation(
    example_dest_dir="docs/examples", ignore_examples=IGNORE_EXAMPLES
)


if FORCE:
    print("Force-building the documentation, ignoring errors and warnings.")
    # In force mode, we do not want to fail, even if an error code is returned.
    # Hence, we use run instead of check_call
    run(link_call, check=False)
    run(building_call + ["--keep-going"], check=False)
else:
    check_call(link_call)
    check_call(building_call)

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


# Delete the created markdown files of the examples.
example_directory = pathlib.Path("docs/examples")
if example_directory.is_dir():
    shutil.rmtree(example_directory)
