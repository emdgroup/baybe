"""
Automatic transformation of example files written in python into markdown files
"""

import argparse
import os
import pathlib
import shutil

from tqdm import tqdm

# Script to transform all .py files in .md files in the examples folder
# Create a new folder named examples_markdown to store the markdown files

# First, we parse potential arguments the user might want to give.
# These are a different folder name and the option whether or not the headers should
# be written
parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--target_dir",
    help="Choose a different target directory. Default is SDKExamples.",
    default="SDKExamples",
)
parser.add_argument(
    "--write_headers",
    help="Decide whether eleventy headers should be written. Default is to write them",
    action=argparse.BooleanOptionalAction,
    default=True,
)
args = parser.parse_args()
# Folder where the .md files created are stored
# Default name is examples_markdown, optional name can be provided
DESTINATION_DIR_NAME = args.target_dir
WRITE_HEADERS = args.write_headers
destination_dir = pathlib.Path(DESTINATION_DIR_NAME)

# if the destination directory already exists it is deleted
if destination_dir.is_dir():
    shutil.rmtree(destination_dir)

# Copy the examples folder in the destination directory
shutil.copytree("examples", destination_dir)

# List all directories in the examples folder
directories = [d for d in destination_dir.iterdir() if d.is_dir()]

# Write the markdown file for the SDK Example folder itself.
# Only necessary if eleventy headers should be written.
if WRITE_HEADERS:
    with open(
        rf"{destination_dir}/{destination_dir.name}.md", "w", encoding="UTF-8"
    ) as f:
        f.write(
            "---"
            + "\neleventyNavigation:"
            + "\n  key: SDK Examples"
            + f"\n  order: {len(directories)+1}"
            + "\n  parent: Python SDK"
            + "\nlayout: layout.njk"
            + "\npermalink: baybe/sdk/examples/"
            + "\ntitle: SDK Examples"
            + "\n---\n\n "
        )
        f.write("These are examples on using the BayBE SDK")

# Iterate over the directories
for directory in (pbar := tqdm(directories)):

    # Set description of progressbar
    pbar.set_description("Overall progress")

    # list all .py files in the subdirectory that need to be converted
    py_files = list(directory.glob("**/*.py"))

    for file_index, file in enumerate(inner_pbar := tqdm(py_files, leave=False)):

        # Set description for progress bar
        inner_pbar.set_description(
            f"Progressing {str(directory)[len(DESTINATION_DIR_NAME)+1:]}"
        )

        # Create the Markdown file:

        # 1. Convert the file to jupyter notebook
        os.system(rf"p2j {file} >/dev/null 2>&1")

        notebook_path = file.with_suffix(".ipynb")

        # 2. Execute the notebook
        os.system(
            f"jupyter nbconvert --execute --to notebook --inplace {notebook_path}"
            + ">/dev/null 2>&1"
        )

        # 3. Convert the notebook to markdown
        os.system(rf"jupyter nbconvert --to markdown {notebook_path} >/dev/null 2>&1")

        markdown_path = file.with_suffix(".md")

        # 4. Add lines at the top of the .md file

        # Collect information about the file
        filename = file.stem
        directory_name = directory.name
        order = file_index + 1

        # Write the information collected at the top of the .md file if required
        if WRITE_HEADERS:
            formatted_name = filename.replace("_", " ").capitalize()

            LINES_TO_ADD = (
                "---"
                + "\neleventyNavigation:"
                + f"\n  key: {formatted_name}"
                + f"\n  order: {order}"
                + f"\n  parent: {directory_name}"
                + "\nlayout: layout.njk"
                + f"\npermalink: baybe/sdk/examples/{directory_name}/{filename}_ex/"
                + f"\ntitle: {formatted_name}"
                + "\n---\n\n "
            )
            with open(rf"{markdown_path}", "r+", encoding="UTF-8") as f:
                content = f.readlines()

                # Any lines starting with '![png]' is removed
                content = [line for line in content if not line.startswith("![png]")]

                # The final file is then written
                f.seek(0)
                f.write(LINES_TO_ADD)
                f.writelines(content)

    # Write the file for the sub-directory itself if required
    if WRITE_HEADERS:
        with open(rf"{directory}/{directory.name}.md", "w", encoding="UTF-8") as f:
            f.write(
                "---"
                + "\neleventyNavigation:"
                + f"\n  key: {directory.name}"
                + f"\n  order: {order+1}"
                + "\n  parent: SDK Examples"
                + "\nlayout: layout.njk"
                + f"\npermalink: baybe/sdk/examples/{directory.name}/"
                + f"\ntitle: {directory.name}"
                + "\n---\n\n "
            )

# 5. Remove remaining files and subdirectories from the destination directory

# remove any not markdown files
for file in destination_dir.glob("**/*"):
    if file.is_file() and file.suffix != ".md":
        file.unlink(file)


# Remove any remaining empty subdirectories
for subdirectory in destination_dir.glob("*/*"):
    if subdirectory.is_dir() and not any(subdirectory.iterdir()):
        subdirectory.rmdir()
