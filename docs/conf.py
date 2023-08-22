# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import shutil
import sys

# We need to "trick" sphinx due to it thinking that decorated classes are just aliases
# We thus need to import and later define some specific names
from baybe.surrogate import (
    BayesianLinearSurrogate,
    NGBoostSurrogate,
    RandomForestSurrogate,
)

# -- Path setup --------------------------------------------------------------

__location__ = os.path.dirname(__file__)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.join(__location__, "../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BayBE"
copyright = "2022-2023, Merck KGaA"
author = "Martin Fitzner, Adrian Šošić, Alexander Hopp, Alex Lee"


# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See Issue:
# https://github.com/readthedocs/readthedocs.org/issues/1139
# DON'T FORGET: Check the box "Install your project inside a virtualenv using
# setup.py install" in the RTD Advanced Settings.
# Additionally it helps us to avoid running apidoc manually

try:  # for Sphinx >= 1.7
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc

output_dir = os.path.join(__location__, "sdk")
baybe_module_dir = os.path.join(__location__, "../baybe")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    args = ["--implicit-namespaces", "-M", "-T", "-e", "-f", "-o", output_dir]

    apidoc.main(
        [
            *args,
            baybe_module_dir,
            baybe_module_dir + "/__init__.py",
        ]
    )
except Exception as e:
    print(f"Running `sphinx-apidoc` failed!\n{e}")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Note that the order here is important! E.g. napoleon needs to be loaded before
# autodoc_typehints
extensions = [
    "sphinx.ext.napoleon",  # Necessary for numpy/google docstrings to work
    "sphinx_toolbox.more_autodoc.typehints",  # Proper typehints 1
    "sphinx_autodoc_typehints",  # Proper typehints 2
    "sphinx.ext.autodoc",  # Crucial for autodocumentation
    "sphinx_autodoc_defaultargs",  # Automatic documentation of default values
    "sphinx.ext.intersphinx",  # Links to other documentations like numpy, python,...
    "sphinx_markdown_builder",  # Necessary for building the markdown files.
]

# Necessary additional code for dealing with defaults.
rst_prolog = """
.. |default| raw:: html

    Default:"""

# Not sure about this, seems to be default stuff that is always used
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Enable markdown
# Note that we do not need additional configuration here.
extensions.append("myst_parser")

# The suffix of source filename
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# On purpose not deleted since this will become relevant once we write the html version

# html_theme = "furo"
# html_static_path = ["_static"]


# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "sklearn_extra": ("https://scikit-learn-extra.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

# --- Options for autodoc typehints and autodoc -------------------------------
# https://pypi.org/project/sphinx-autodoc-typehints/

autodoc_typehints = "both"
# Separate class names and init functions
autodoc_class_signature = "separated"
# Preserve the defaults
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,  # Also document members of classes
    "member-order": "bysource",  # Order as in source files (alternative is alphabetic)
    "exclude-members": "__init__",  # Do not include inits
}
autodoc_typehints_description_target = "documented_params"


# Magic function doing magic stuff
def setup(app):
    # This avoids that sphinx refers to all surrogate models via alias
    RandomForestSurrogate.__name__ = "RandomForestSurrogate"
    BayesianLinearSurrogate.__name__ = "BayesianLinearSurrogate"
    NGBoostSurrogate.__name__ = "NGBoostSurrogate"
