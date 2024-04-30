"""Automatic conversion of python code to markdown files for the documentation."""

import argparse
import os
import pathlib
import shutil

from build_documentation import build_documentation
from build_examples import build_examples
from check_links import check_links
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

# We adjust the environment variable if we decide to ignore warnings
if not INCLUDE_WARNINGS:
    os.environ["PYTHONWARNINGS"] = "ignore"

# Disable telemtetry
os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
# Directory where Sphinx builds the documentation
build_dir = pathlib.Path("docs/build")


# Process examples if required.
# Note that ignoring of examples now happens by simply not executing them.
build_examples(example_dest_dir="docs/examples", ignore_examples=IGNORE_EXAMPLES)

check_links(force=FORCE)
build_documentation(force=FORCE)

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


# Delete the created markdown files of the examples.
example_directory = pathlib.Path("docs/examples")
if example_directory.is_dir():
    shutil.rmtree(example_directory)
