"""Automatic conversion of python code to markdown files for the documentation."""

import argparse
import os
import pathlib
import shutil
from subprocess import check_call, run

from utils import adjust_banner, create_example_documentation

from baybe.telemetry import VARNAME_TELEMETRY_ENABLED

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--target_dir",
    help="Destination directory in which the build will be saved (relative).\
    Note that building the documentation actually happens within the doc folder.\
    After building the documentation, it will be copied to this folder.\
    Default is a subfolder 'docs' placed in `build`.",
    default="./build/docs",
)
parser.add_argument(
    "-p",
    "--include_private",
    help="Include private methods in the documentation. Default is false.",
    action="store_true",
)
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
DESTINATION_DIR = args.target_dir
INCLUDE_PRIVATE = args.include_private
IGNORE_EXAMPLES = args.ignore_examples
INCLUDE_WARNINGS = args.include_warnings
FORCE = args.force

# We adjust the environment variable if we decide to ignore warnings
if not INCLUDE_WARNINGS:
    os.environ["PYTHONWARNINGS"] = "ignore"

# Disable telemtetry
os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
# Directories where Sphinx will always put the build, sdk and autosummary data
build_dir = pathlib.Path("docs/build")
sdk_dir = pathlib.Path("docs/sdk")
autosummary_dir = pathlib.Path("docs/_autosummary")
destination_dir = pathlib.Path(DESTINATION_DIR)

# Collect all of the directories and delete them if they still exist.
directories = [sdk_dir, autosummary_dir, build_dir, destination_dir]

for directory in directories:
    if directory.is_dir():
        shutil.rmtree(directory)

# The call for checking external links.
link_call = [
    "sphinx-build",
    "-b",
    "linkcheck",
    "docs",
    build_dir,
    "-D",
    f"autodoc_default_options.private_members={INCLUDE_PRIVATE}",
]
# The actual call that will be made to build the documentation
building_call = [
    "sphinx-build",
    "-b",
    "html",
    "docs",
    build_dir,
    "-D",
    f"autodoc_default_options.private_members={INCLUDE_PRIVATE}",
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
adjust_banner("docs/build/index.html", light_banner="banner2", dark_banner="banner1")
adjust_banner(
    "docs/build/misc/readme_link.html", light_banner="banner2", dark_banner="banner1"
)

# Clean the other files
for directory in [sdk_dir, autosummary_dir]:
    if directory.is_dir():
        shutil.rmtree(directory)

documentation = pathlib.Path(build_dir)
shutil.move(documentation, destination_dir)

# Delete the created markdown files of the examples.
example_directory = pathlib.Path("docs/examples")
if example_directory.is_dir():
    shutil.rmtree(example_directory)
