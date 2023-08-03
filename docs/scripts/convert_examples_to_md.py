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
import os
import pathlib
import re
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
    help="Choose a different target directory. Default is `SDK Examples`.",
    default="SDK Examples",
)
parser.add_argument(
    "--write_headers",
    help="Decide whether eleventy headers are written. Default is to not write them",
    action="store_true",
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
            + f"\n  key: {DESTINATION_DIR_NAME}"
            + f"\n  order: {len(directories)+1}"
            + "\n  parent: Python SDK"
            + "\nlayout: layout.njk"
            + "\npermalink: baybe/sdk/examples/"
            + f"\ntitle: {DESTINATION_DIR_NAME}"
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
        os.system(rf"p2j '{file}' >/dev/null 2>&1")

        notebook_path = file.with_suffix(".ipynb")

        # 2. Execute the notebook
        os.system(
            f"jupyter nbconvert --execute --to notebook --inplace '{notebook_path}'"
            + ">/dev/null 2>&1"
        )

        # 3. Convert the notebook to markdown
        os.system(rf"jupyter nbconvert --to markdown '{notebook_path}' >/dev/null 2>&1")

        markdown_path = file.with_suffix(".md")

        # 4. Add lines at the top of the .md file

        # Collect information about the file
        filename = file.stem
        directory_name = directory.name
        order = file_index + 1

        # Write the information collected at the top of the .md file if required
        formatted_name = filename.replace("_", " ").capitalize()

        LINES_TO_ADD = (
            (
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
            if WRITE_HEADERS
            else ""
        )
        with open(rf"{markdown_path}", "r+", encoding="UTF-8") as f:
            content = f.readlines()

            formatted_content = []

            # Some manual cleanup of the lines
            for line in content:
                # Any lines starting with '![png]' is removed
                if line.startswith("![png]"):
                    continue

                # For long lines, we manually insert line breaks.
                # These are always inserted after the last "," in between two "=".
                # This is necessary as we were not able to find a tool that auto formats
                # the outputs generated by the notebook execution which tend to look
                # ugly without post-processing
                if len(line) > 110:
                    # Get positions of "="
                    equal_signs = [m.start() for m in re.finditer("=", line)]
                    # Only necessary to format if there is more than a single "="
                    if len(equal_signs) > 1:
                        # Get all commata that appear before the last "="
                        commata = [m.start() for m in re.finditer(",", line)]
                        commata = [c for c in commata if c < equal_signs[-1]]
                        if commata:
                            # pyline insists on having this all capital
                            comma = 0  # pylint: disable=C0103
                            comma_positions = []
                            # Calculate positions of last commata between two "="
                            for i in equal_signs[:-1]:
                                while commata[comma + 1] < i:
                                    comma += 1
                                comma_positions.append(commata[comma])
                            # Get the individual chunks of text between the commata
                            chunks = [line[0 : comma_positions[0]]]
                            for i in range(len(comma_positions) - 1):
                                chunks.append(
                                    line[
                                        comma_positions[i] + 2 : comma_positions[i + 1]
                                    ]
                                )
                            chunks = list(filter(None, chunks))
                            # Join everything together
                            line = ",\n    ".join(chunks)  # pylint: disable=C0103
                formatted_content.append(line)

            # The final file is then written
            f.seek(0)
            f.write(LINES_TO_ADD)
            f.writelines(formatted_content)

    # Write the file for the sub-directory itself if required
    if WRITE_HEADERS:
        with open(rf"{directory}/{directory.name}.md", "w", encoding="UTF-8") as f:
            f.write(
                "---"
                + "\neleventyNavigation:"
                + f"\n  key: {directory.name}"
                + f"\n  order: {order+1}"
                + f"\n  parent: {DESTINATION_DIR_NAME}"
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
