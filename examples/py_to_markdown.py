"""
Automatic transformation of example files written in python into markdown files
"""

import os
import pathlib
import shutil


# Script to transform all .py files in .md files in the examples folder
# Create a new folder named examples_markdown to store the markdown files

# Folder where the .md files created are stored
destination_dir = pathlib.Path("examples_markdown")

# if the destination directory already exists it is deleted
if destination_dir.is_dir():
    shutil.rmtree(destination_dir)

# Copy the examples folder in the destination directory
shutil.copytree("examples", destination_dir)


# list all directories in the examples folder
directories = [d for d in destination_dir.iterdir() if d.is_dir()]

# Iterate over the directories
for directory in directories:
    # list all .py files in the subdirectory that need to be converted
    py_files = list(directory.glob("**/*.py"))

    for file_index, file in enumerate(py_files):
        # Create the Markdown file:

        # 1. Convert the file to jupyter notebook
        os.system(rf"p2j {file}")

        notebook_path = file.with_suffix(".ipynb")

        # 2. Execute the notebook
        os.system(
            rf"jupyter nbconvert --execute --to notebook --inplace {notebook_path}"
        )

        # 3. Convert the notebook to markdown
        os.system(rf"jupyter nbconvert --to markdown {notebook_path}")

        markdown_path = file.with_suffix(".md")

        # 4. Add lines at the top of the .md file

        # Collect information about the file
        filename = file.stem
        directory_name = directory.name
        order = file_index + 1

        # Write the information collected at the top of the .md file

        LINES_TO_ADD = (
            "---"
            + "\neleventyNavigation:"
            + "\n  key: "
            + filename
            + "\n  order: "
            + str(order)
            + "\n  parent: Examples/"
            + directory_name
            + "\nlayout: layout.njk"
            + "\npermalink: baybe/sdk/examples/"
            + directory_name
            + "\ntitle: "
            + filename
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

# 5. remove remaining not markdown files and subdirectories from the destination directory

# remove any not markdown files
for file in destination_dir.glob("**/*"):
    if file.is_file() and file.suffix != ".md":
        file.unlink(file)


# Remove any remaining empty subdirectories
for subdirectory in destination_dir.glob("*/*"):
    if subdirectory.is_dir() and not any(subdirectory.iterdir()):
        subdirectory.rmdir()
