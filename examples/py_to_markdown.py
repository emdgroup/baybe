"""
Automatic transformation of example files written in python into markdown files
"""


import os

# Script to transform .py files in .md files in the examples folder

# Information about the file
DIRECTORY = "Basics"
FILE = "baybe_object.py"
FILENAME = "baybe_object"


# Create the Markdown file:

# 1. Convert the file to jupyter notebook
os.system(rf"p2j examples\ {DIRECTORY} \ {FILE}")
NOTEBOOK = FILENAME + ".ipynb"

# 2. Execute the notebook
os.system(
    rf"jupyter nbconvert --execute --to notebook --inplace examples\ {DIRECTORY} \ {NOTEBOOK}"
)

# 3. Convert the notebook to markdown
os.system(rf"jupyter nbconvert --to markdown examples\ {DIRECTORY} \ {NOTEBOOK}")

# 4. Delete the no-longer-necessary notebook
os.remove(rf"examples\ {DIRECTORY} \ {NOTEBOOK}")
