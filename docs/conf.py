# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import shutil
import sys

# We need to "trick" sphinx due to it thinking that decorated classes are just aliases
# We thus need to import and later define some specific names
from baybe.surrogates import get_available_surrogates


# -- Path setup --------------------------------------------------------------

__location__ = os.path.dirname(__file__)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Seems to be not necessary at the moment
# sys.path.insert(0, os.path.join(__location__, "../examples"))

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
    "sphinx_autodoc_typehints",  # Proper typehints
    "sphinx.ext.autodoc",  # Crucial for autodocumentation
    "sphinx.ext.autosummary",  # Autosummary
    "sphinx.ext.intersphinx",  # Links to other documentations like numpy, python,...
    "sphinx.ext.viewcode",  # Links to code
]


# Tell sphinx where to find the templates
templates_path = ["templates"]
# Tell sphinx which files should be excluded
exclude_patterns = ["_build", "sdk"]

# Enable markdown
# Note that we do not need additional configuration here.
extensions.append("myst_parser")

# The suffix of source filename
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# We use the read-the-docs theme
html_theme = "sphinx_rtd_theme"
# We want to have links to the source
html_show_sourcelink = True


# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

# Mappings to all external packages that we want to have clickable links to
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "sklearn_extra": ("https://scikit-learn-extra.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "rdkit": ("https://rdkit.org/docs/", None),
}

# --- Options for autodoc typehints and autodoc -------------------------------
# https://pypi.org/project/sphinx-autodoc-typehints/

# For some reason, sphinx does not like it if we use the -D option to just tell it
# that we want to include private members. We thus manually verify whether the option
# was set.
private_members = "True" in sys.argv[sys.argv.index("-D") + 1]

# Represent typehints whenever possible. Partly necessary for proper display
autodoc_typehints = "both"
# Separate class names and init functions. Necessary choice due to the github issue
# regarding defaults https://github.com/sphinx-toolbox/sphinx-toolbox/issues/146
# TODO Revisit this once we have decided how to handle defaults
autodoc_class_signature = "separated"
# Do not preserve the defaults as we want them to be evaluated due to attrs
# Seems to be bugged for attrs defaults, see
# https://github.com/sphinx-toolbox/sphinx-toolbox/issues/146
autodoc_preserve_defaults = False
# Set the default options that should be used for autodoc
autodoc_default_options = {
    # Order by type (function, attribute...), required for proper inheritance
    "member-order": "groupwise",
    # Include private members if this was requested
    "private-members": private_members,
}
# Only show parameters that are documented.
autodoc_typehints_description_target = "documented_params"


# This function enables us to hook into the internal sphinx processes
# These allow us to change docstrings (resp. how they are processed) which is currently
# necessary in four cases:
# 1. Call get_available_surrogates() as this is necessary for properly rendering them
# 2. Add a note that __init__ functions are created automatically and thus, do not
#   have a source code available.
# 3. Remove all of the serialization functions from the documentation of the various
#   inherited classes, but keep them for SerialMixin
# 4. Some classes have empty __init__ (e.g. utils.basic.Dummy). We put some dummy string
#   into these.
# 5. Skip the fit method for ONNX surrogates since they cannot be fit
def setup(app):  # noqa: D103
    # We initialize all available surrogates once as this sets their name
    # Corresponds to point 1
    get_available_surrogates()
    # Some docstrings need to be processed after reading them
    # Corresponds to points 2, 4
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    # Some functions should be skipped.
    # Corresponds to points 3, 5
    app.connect("autodoc-skip-member", autodoc_skip_member)


def autodoc_process_docstring(app, what, name, obj, options, lines):  # noqa: D103
    no_init = "Thus, source code for this method is not available"
    if len(lines) > 0 and lines[0].startswith("Method generated by attrs"):
        lines.append(no_init)
    if "__init__" in name and len(lines) == 0:
        lines.append("This is a dummy initialization function.")


def autodoc_skip_member(app, what, name, obj, skip, options):  # noqa: D103
    # This function can be used to skip specific members for the documentation.
    # Currently, it used to prevent the serialization-related functions which appear
    # in nearly all classes (due to inheritance) from appearing in any class other
    # than SerialMixin.
    # Since we do not want to exclude them everywhere, the logic is to check whether
    # the current member is one of the corresponding functions and ig the currently
    # processed document cor‚responds to SerialMixin.
    # NOTE The functions will remain visible in the Full list of available methods, so
    # users will still be able to see them.
    exclusions = (
        "to_json",
        "to_dict",
        "from_json",
        "from_dict",
    )
    current_doc = app.env.temp_data.get("docname")
    if current_doc is not None:
        if name in exclusions:
            return "SerialMixin" not in current_doc
        # Furthermore, we skip the fit method for ONNX models
        if name == "fit":
            return "surrogates.custom" in current_doc
    return None
