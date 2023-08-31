"""
Automatic transformation of example files written in python into markdown files
Using this files requires the following additional packages:
- p2j: Necessary for converting the python source code to jupyter notebooks
    Available via `pip install p2j`
- jupyterlab: Necessary for the execution of the transformed jupyter notebooks
    Available via  `pip install jupyterlab`
- nbconvert: Necessary for converting the jupyter notebook to markdown format
    Available via `pip install nbconvert`
"""

import argparse
import pathlib
import re
import shutil
from subprocess import check_call, DEVNULL, STDOUT
from textwrap import fill

from tqdm import tqdm


def format_displayed_names(text: str) -> str:
    """Format all file names and keys such that they look better.

    Currently, this replaces underscores by blanks and capitalizes the input.

    Args:
        text: The input text

    Returns:
        str: The string with all underscores replaced by blanks
    """
    return text.replace("_", " ").capitalize()


# Script to transform all .py files in .md files in the examples folder
# Create a new folder named examples_markdown to store the markdown files

# First, we parse potential arguments the user might want to give.
# These are a different folder name and the option whether or not the headers should
# be written
parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--target_dir",
    help="Choose a different target directory. Default is `SDK Examples`.",
    default="SDK Examples",
)
parser.add_argument(
    "--write_headers",
    help="Decide whether eleventy headers are written. Default is to not write them",
    action="store_true",
)
parser.add_argument(
    "-lw",
    "--line_width",
    help="Choose the line width for. Default is 90.",
    default=90,
    type=int,
)
args = parser.parse_args()
# Folder where the .md files created are stored
# Default name is examples_markdown, optional name can be provided
DESTINATION_DIR_NAME = args.target_dir
WRITE_HEADERS = args.write_headers
LINE_WIDTH = args.line_width
destination_dir = pathlib.Path(DESTINATION_DIR_NAME)

# if the destination directory already exists it is deleted
if destination_dir.is_dir():
    shutil.rmtree(destination_dir)

# Copy the examples folder in the destination directory
shutil.copytree("examples", destination_dir)

# List all directories in the examples folder
directories = [d for d in destination_dir.iterdir() if d.is_dir()]

# List all files in top-level folder as it determines the example order entry
top_level_files = [
    f
    for f in destination_dir.parent.iterdir()
    if f.is_file() and not f.name.startswith(".")
]

# Write the markdown file for the SDK Example folder itself.
# Only necessary if eleventy headers should be written.
if WRITE_HEADERS:
    with open(
        destination_dir / f"{destination_dir.name}.md", "w", encoding="UTF-8"
    ) as f:
        f.write(
            "---"
            + "\neleventyNavigation:"
            + f"\n  key: {destination_dir.name}"
            + f"\n  order: {len(top_level_files)}"
            + "\n  parent: Python SDK"
            + "\nlayout: layout.njk"
            + "\npermalink: baybe/sdk/examples/"
            + f"\ntitle: {destination_dir.name}"
            + "\n---\n\n "
        )
        f.write("These are examples on using the BayBE SDK")

# Iterate over the directories
directory_order = 3  # pylint: disable=invalid-name
for directory in (pbar := tqdm(directories)):

    # Set description of progressbar
    pbar.set_description("Overall progress")

    # Write the file for the sub-directory itself if required
    if WRITE_HEADERS:
        with open(directory / f"{directory.name}.md", "w", encoding="UTF-8") as f:
            if "Basics" in directory.name:
                order = 1  # pylint: disable=invalid-name
            elif "Searchspaces" in directory.name:
                order = 2  # pylint: disable=invalid-name
            else:
                order = directory_order  # pylint: disable=invalid-name
                directory_order = directory_order + 1  # pylint: disable=invalid-name

            f.write(
                "---"
                + "\neleventyNavigation:"
                + f"\n  key: {format_displayed_names(directory.name)}"
                + f"\n  order: {order}"
                + f"\n  parent: {destination_dir.name}"
                + "\nlayout: layout.njk"
                + f"\npermalink: baybe/sdk/examples/{directory.name}/"
                + f"\ntitle: {format_displayed_names(directory.name)}"
                + "\n---\n\n "
            )

    # list all .py files in the subdirectory that need to be converted
    py_files = list(directory.glob("**/*.py"))

    for file_index, file in enumerate(inner_pbar := tqdm(py_files, leave=False)):

        # Set description for progress bar
        inner_pbar.set_description(
            f"Progressing {str(directory)[len(DESTINATION_DIR_NAME)+1:]}"
        )

        # Create the Markdown file:

        # 1. Convert the file to jupyter notebook
        check_call(["p2j", file], stdout=DEVNULL, stderr=STDOUT)

        notebook_path = file.with_suffix(".ipynb")

        # 2. Execute the notebook
        check_call(
            [
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--inplace",
                notebook_path,
            ],
            stdout=DEVNULL,
            stderr=STDOUT,
        )

        # 3. Convert the notebook to markdown
        check_call(
            ["jupyter", "nbconvert", "--to", "markdown", notebook_path],
            stdout=DEVNULL,
            stderr=STDOUT,
        )

        markdown_path = file.with_suffix(".md")

        # 4. Add lines at the top of the .md file

        # Collect information about the file
        filename = file.stem
        directory_name = directory.name
        order = file_index + 1

        # Write the information collected at the top of the .md file if required
        formatted_file_name = format_displayed_names(filename)

        LINES_TO_ADD = (
            (
                "---"
                + "\neleventyNavigation:"
                + f"\n  key: {formatted_file_name}"
                + f"\n  order: {order}"
                + f"\n  parent: {format_displayed_names(directory.name)}"
                + "\nlayout: layout.njk"
                + f"\npermalink: baybe/sdk/examples/{directory_name}/{filename}_ex/"
                + f"\ntitle: {formatted_file_name}"
                + "\n---\n\n "
            )
            if WRITE_HEADERS
            else ""
        )
        with open(markdown_path, "r+", encoding="UTF-8") as f:
            content = f.readlines()

            formatted_content = []

            # Some manual cleanup of the lines
            for line in content:
                # Any lines starting with '![png]' or `pylint`is removed
                if line.startswith(("![png]", "# pylint", "pylint")):
                    continue

                if WRITE_HEADERS and "(./" in line:
                    # Step 1: Include one additional ../
                    line = line.replace("(./", "(./../")
                    # Step 2: Get position of the end of the [...](...) part
                    end = [m.start() for m in re.finditer(re.escape(")"), line)][0]
                    # Step 3: Replace with the permalink ending
                    line = line[: end - 3] + "_ex" + line[end:]
                # Very long lines need to be wrapped.
                # Note that such lines can only be printed output of the jupyter
                # notebook as our comments and python code cannot become so long.
                # Thus, the formatting here is safe.
                if len(line) > LINE_WIDTH and line.startswith("    "):
                    line = fill(  # pylint: disable=invalid-name
                        line, width=LINE_WIDTH, subsequent_indent="    "
                    )
                formatted_content.append(line)
            # The final file is then written
            f.seek(0)
            f.write(LINES_TO_ADD)
            f.writelines(formatted_content)

# 5. Remove remaining files and subdirectories from the destination directory

# Remove any not markdown files
for file in destination_dir.glob("**/*"):
    if file.is_file() and file.suffix != ".md":
        file.unlink(file)


# Remove any remaining empty subdirectories
for subdirectory in destination_dir.glob("*/*"):
    if subdirectory.is_dir() and not any(subdirectory.iterdir()):
        subdirectory.rmdir()
