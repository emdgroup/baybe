"""
Automatic transformation of example files written in python into markdown files
"""


import glob
import os

# Script to transform all .py files in .md files in the examples folder

# list all directories in the examples folder
directories = [
    d for d in os.listdir("examples/") if os.path.isdir(os.path.join("examples", d))
]

# Iterate over the directories
for directory in directories:
    path = os.path.join("examples", directory, "*.py")

    # list all .py files in the subdirectory that need to be converted
    py_files = [os.path.split(fpath)[1] for fpath in glob.glob(path)]

    for file_index, file in enumerate(py_files):
        # Collect information about the file
        FILENAME = os.path.splitext(file)[0]
        ORDER = file_index + 1

        # Create the Markdown file:

        # 1. Convert the file to jupyter notebook
        os.system(rf"p2j examples\{directory}\{file}")
        NOTEBOOK = FILENAME + ".ipynb"

        # 2. Execute the notebook
        os.system(
            rf"jupyter nbconvert --execute --to notebook --inplace examples\{directory}\{NOTEBOOK}"
        )
        MARKDOWN = FILENAME + ".md"

        # 3. Convert the notebook to markdown
        os.system(rf"jupyter nbconvert --to markdown examples\{directory}\{NOTEBOOK}")

        # 4. Delete the no-longer-necessary notebook
        os.remove(rf"examples\{directory}\{NOTEBOOK}")

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
        with open(rf"examples\{directory}\{MARKDOWN}", "r+", encoding="UTF-8") as f:
            content = f.read()
            f.seek(0)
            f.write(LINES_TO_ADD)
            f.write(content)
