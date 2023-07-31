"""
Automatic transformation of example files written in python into markdown files
"""


import glob
import os
import shutil

# Script to transform all .py files in .md files in the examples folder
# Create a new folder named examples_markdown to store the markdown files

# Folder where the .md files created are stored
DESTINATION_DIR = "examples_markdown/"

# if the destination directory already exist it is deleted
if os.path.isdir(DESTINATION_DIR):
    shutil.rmtree(DESTINATION_DIR)

# Copy the examples folder in the destination directory
shutil.copytree("examples/", DESTINATION_DIR)

# list all directories in the examples folder

directories = [
    d
    for d in os.listdir("examples_markdown/")
    if os.path.isdir(os.path.join("examples_markdown", d))
]


# Iterate over the directories
for directory in directories:
    path = os.path.join("examples_markdown", directory, "*.py")

    # list all .py files in the subdirectory that need to be converted
    py_files = [os.path.split(fpath)[1] for fpath in glob.glob(path)]

    for file_index, file in enumerate(py_files):
        # Collect information about the file
        FILENAME = os.path.splitext(file)[0]
        ORDER = file_index + 1

        # Create the Markdown file:

        # 1. Convert the file to jupyter notebook
        os.system(rf"p2j examples_markdown\{directory}\{file}")
        NOTEBOOK = FILENAME + ".ipynb"

        # 2. Execute the notebook
        os.system(
            "jupyter nbconvert --execute --to notebook --inplace"
            + rf" examples_markdown\{directory}\{NOTEBOOK}"
        )
        MARKDOWN = FILENAME + ".md"

        # 3. Convert the notebook to markdown
        os.system(
            rf"jupyter nbconvert --to markdown examples_markdown\{directory}\{NOTEBOOK}"
        )

        # 4. Delete the no-longer-necessary notebook
        os.remove(rf"examples_markdown\{directory}\{NOTEBOOK}")

        # 5. Add lines at the top of the .md file

        LINES_TO_ADD = (
            "---"
            + "\neleventyNavigation:"
            + "\n  key: "
            + FILENAME
            + "\n  order: "
            + str(ORDER)
            + "\n  parent: Examples/"
            + directory
            + "\nlayout: layout.njk"
            + "\npermalink: baybe/sdk/examples/"
            + directory
            + "\ntitle: "
            + FILENAME
            + "\n---\n\n "
        )
        with open(
            rf"examples_markdown\{directory}\{MARKDOWN}", "r+", encoding="UTF-8"
        ) as f:
            content = f.read()

            f.seek(0)
            f.write(LINES_TO_ADD)
            f.write(content)

        # 6. Remove the no-longer-necessary .py file
        os.remove(rf"examples_markdown\{directory}\{file}")

# 7. remove remaining not markdown files from the destination directory
files = glob.glob("examples_markdown/**/*")
other_files = [
    file for file in files if os.path.splitext(os.path.split(file)[1])[1] != ".md"
]
for file in other_files:
    os.remove(file)

os.remove("examples_markdown/py_to_markdown.py")
