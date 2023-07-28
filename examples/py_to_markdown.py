"""
Automatic transformation of example files written in python into markdown files
"""


import os

# Script to transform .py files in .md files in the examples folder

# Information about the file
DIRECTORY = "Basics"
FILE = "baybe_object.py"
FILENAME = "baybe_object"
ORDER = 1

# Create the Markdown file:

# 1. Convert the file to jupyter notebook
os.system(rf"p2j examples\{DIRECTORY}\{FILE}")
NOTEBOOK = FILENAME + ".ipynb"

# 2. Execute the notebook
os.system(
    rf"jupyter nbconvert --execute --to notebook --inplace examples\{DIRECTORY}\{NOTEBOOK}"
)
MARKDOWN = FILENAME + ".md"

# 3. Convert the notebook to markdown
os.system(rf"jupyter nbconvert --to markdown examples\{DIRECTORY}\{NOTEBOOK}")

# 4. Delete the no-longer-necessary notebook
os.remove(rf"examples\{DIRECTORY}\{NOTEBOOK}")

# %. Add lines at the top of the .md file

LINES_TO_ADD = (
    "---"
    + "\neleventyNavigation:"
    + "\n  key: "
    + FILENAME
    + "\n  order: "
    + str(ORDER)
    + "\n  parent: Examples/"
    + DIRECTORY
    + "\nlayout: layout.njk"
    + "\npermalink: baybe/sdk/examples/"
    + DIRECTORY
    + "\ntitle: "
    + FILENAME
    + "\n---\n\n "
)
with open(rf"examples\{DIRECTORY}\{MARKDOWN}", "r+", encoding="UTF-8") as f:
    content = f.read()
    f.seek(0)
    f.write(LINES_TO_ADD)
    f.write(content)
