"""Automatic conversion of python code to markdown files for the documentation."""

import argparse
import os
import pathlib
import shutil
from subprocess import check_call, DEVNULL

from baybe.telemetry import VARNAME_TELEMETRY_ENABLED

parser = argparse.ArgumentParser()
parser.add_argument(
    "--html",
    help="Use html instead of markdown. Default is false.",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--target_dir",
    help="Destination directory in which the build will be saved.\
    That is, a folder named 'build' will be created and this folder contains the\
    markdown resp. html files. Note that this folder is being deleted if it already\
    exists!\
    Default is a subfolder 'build' which is being placed in the current folder.",
    default="./build",
)
parser.add_argument(
    "-p",
    "--include_private",
    help="Include private methods in the documentation. Default is false.",
    action="store_true",
)
parser.add_argument(
    "--do_not_prettify",
    help="Flag for denoting that the routines used to make the output look prettier"
    "should not be used.",
    action="store_true",
)
parser.add_argument(
    "--debug",
    help="Activate debugging mode by not surpressing the output of conversion.",
    action="store_true",
)

# Parse input arguments
args = parser.parse_args()
USE_HTML = args.html
DIR = args.target_dir
DEBUG = args.debug
INCLUDE_PRIVATE = args.include_private
PRETTIFY = not args.do_not_prettify

# Additional options for the sphinx-apidoc
private_members = "private-members" if INCLUDE_PRIVATE else ""

sphinx_apidoc_options = ["members", "show-inheritance", private_members]

# Only use options that were actually set
os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(filter(None, sphinx_apidoc_options))
os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
# Directories where Sphinx will always put the build, sdk and autosummary data
build_dir = pathlib.Path("docs/build")
sdk_dir = pathlib.Path("docs/sdk")
autosummary_dir = pathlib.Path("docs/misc/_autosummary")
# Output destination
destination_dir = pathlib.Path(DIR)

# Collect all of the directories and delete them if they still exist.
# Note that destination_dir is last here as we re-use this later while ignoring this
directores = [build_dir, sdk_dir, autosummary_dir, destination_dir]

for directory in directores:
    if directory.is_dir():
        shutil.rmtree(directory)


# The actual call that will be made to build the documentation
call = ["sphinx-build", "-b", "html", "docs", "docs/build"]
# For some weird reason, we need to call sphinx-build twice
if not DEBUG:
    check_call(call, stderr=DEVNULL, stdout=DEVNULL)
    check_call(call, stderr=DEVNULL, stdout=DEVNULL)
else:
    check_call(call)
    check_call(call)

# Copy the files to the intended location
documentation = pathlib.Path(build_dir)


shutil.move(documentation, destination_dir)

# Clean the other files
for dir in directores[:-1]:
    if dir.is_dir():
        shutil.rmtree(dir)
