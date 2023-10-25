"""Automatic conversion of python code to markdown files for the documentation."""

import argparse
import os
import pathlib
import re
import shutil
from subprocess import check_call, DEVNULL

parser = argparse.ArgumentParser()
parser.add_argument(
    "--html",
    help="Use html instead of markdown. Default is false.",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--target_dir",
    help="Destination directory in which the build will be saved.\
    That is, a folder named 'build' will be created and this folder contains the\
    markdown resp. html files. Note that this folder is being deleted if it already\
    exists!\
    Default is a subfolder 'build' which is being placed in the current folder.",
    default="./build",
)
parser.add_argument(
    "-p",
    "--include_private",
    help="Include private methods in the documentation. Default is false.",
    action="store_true",
)
parser.add_argument(
    "--do_not_prettify",
    help="Flag for denoting that the routines used to make the output look prettier"
    "should not be used.",
    action="store_true",
)

# Parse input arguments
args = parser.parse_args()
USE_HTML = args.html
DIR = args.target_dir
INCLUDE_PRIVATE = args.include_private
PRETTIFY = not args.do_not_prettify

# Additional options for the sphinx-apidoc
private_members = "private-members" if INCLUDE_PRIVATE else ""

sphinx_apidoc_options = ["members", "show-inheritance", private_members]

# Only use options that were actually set
os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(filter(None, sphinx_apidoc_options))
os.environ["BAYBE_TELEMETRY_ENABLED"] = "false"
build_dir = pathlib.Path("docs/build")
sdk_dir = pathlib.Path("docs/sdk")
# Output destination
destination_dir = pathlib.Path(DIR)

# If the folders already exist we delete them
if build_dir.is_dir():
    shutil.rmtree(build_dir)

if sdk_dir.is_dir():
    shutil.rmtree(sdk_dir)

if destination_dir.is_dir():
    shutil.rmtree(destination_dir)

# The actual call that will be made to build the documentation
call = (
    ["sphinx-build", "-b", "html", "docs", "docs/build"]
    if USE_HTML
    else ["sphinx-build", "-M", "markdown", "docs", "docs/build"]
)
check_call(call, stderr=DEVNULL, stdout=DEVNULL)

# Get the path to the actual markdown files
markdown_files = list(pathlib.Path("docs/build/markdown/sdk").glob("*.md"))

# Small trick to ensure that baybe.md will be the first file: Sort via length
markdown_files.sort(key=lambda x: str(x)[:-3].replace(".", ""))

for order, file in enumerate(markdown_files):

    # Get the actual file name by removing the first parts of the string and the ending
    NAME = str(file)[len("docs/build/markdown/sdk/") : -3]

    # Eleventy header that is added
    # The permalink is constructed by removing all . as this gives unique names that
    # can easily be identified and used for e.g. creating links later on.
    LINES_TO_ADD = (
        "---"
        + "\neleventyNavigation:"
        + f"\n  key: {NAME}"
        + f"\n  order: {order}"
        + "\n  parent: Documentation"
        + "\nlayout: layout.njk"
        + f"\npermalink: baybe/sdk/docs/{NAME.replace('.md', '').replace('.', '')}/"
        + f"\ntitle: {NAME}"
        + "\n---\n\n "
    )

    # If we currently consider the baybe.md file, we change the LINES_TO_ADD since we
    # want this to be the file for the full folder
    if NAME == "baybe":
        # Get the name of the destination directory
        directory_name = destination_dir.name

        # Next, we get the number of top-level files such that the documentation will be
        # placed properly after already existing files
        # List all files in top-level folder as it determines the order entry
        top_level_files = [
            f
            for f in destination_dir.parent.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        LINES_TO_ADD = (
            "---"
            + "\neleventyNavigation:"
            + f"\n  key: {directory_name}"
            + f"\n  order: {len(top_level_files)+1}"
            + "\n  parent: Python SDK"
            + "\nlayout: layout.njk"
            + "\npermalink: baybe/sdk/docs/"
            + f"\ntitle: {directory_name}"
            + "\n---\n\n "
        )

    with open(file, "r+", encoding="UTF-8") as f:
        content = f.readlines()

        # List for storing the formatted lines
        formatted_content = []

        # Flag denoting whether it is necessary to skip lines due to them containing
        # information about the return type of properties
        skip_property_line = False
        # Flag for skipping two lines at once. Necessary since the line for the return
        # text and the actual return type are two lines
        double_skip = False

        # Some manual cleanup of the lines
        # There are two major aspects that we need to take care of:
        # 1. Make the lines look a bit better
        # 2. Adjust links such that they work for the documentation
        for line in content:
            # PART 1: MAKE THINGS LOOK NICER IF THIS IS CHOSEN
            if PRETTIFY:

                # There are situations in which we need to skip some lines.

                # NOTE: It seems like sphinx does not add the return type hint for
                # cached properties like in comp_df. Thus, there is an additional check
                # that sets skip_property_line=Fals again if we encounter another
                # heading
                if skip_property_line and line.startswith("####"):
                    skip_property_line = False
                # We thus check if we are still in such a situation
                if skip_property_line and line.startswith("* **Return type:**"):
                    skip_property_line = False
                    double_skip = True
                    continue

                # If we need to skip two lines, we skip the second on now.
                if double_skip:
                    double_skip = False
                    continue

                # 1. Remove ugly "*"
                if line.startswith("#### ") and "*" in line:
                    line = line.replace("*", "")

                # 2. Replacement for class variables types
                line = line.replace("[`typing.ClassVar`]", "[`ClassVar`]")
                line = line.replace("[`typing.Type`]", "[`Type`]")

                # 3. Do a similar replacement for the specific object types
                line = line.replace(
                    "`baybe.searchspace.SearchSpaceType`", "`SearchSpaceType`"
                )

                # 3. Handling of attr defaults
                # sphinx cannot  handle attr attributes that do not use the 'default'
                # keyword. We thus manually remove the corresponding parts in the
                # documentation and manually add links.

                # Factory attributes creates tis as default value so those
                # lines are skipped
                if "_Nothing.NOTHING" in line:
                    continue
                # Remove the "~" sign that is in front of the ScikitLearnModel
                line = line.replace("~_ScikitLearn", "_ScikitLearn")
                # Remove the "~" sign that is in front ot the _T class var
                line = line.replace("`~_T`", "`_T`")
                # Insert the correct link manually.
                if "Default: SequentialGreedyRecommender" in line:
                    line = line.replace(
                        "SequentialGreedyRecommender\n",
                        "[`SequentialGreedyRecommender`]",
                    )
                    line = (
                        line + "(/baybe/sdk/docs/baybestrategiesbayesian/#class-baybe."
                        "strategies.bayesian.sequentialgreedyrecommender)\n"
                    )

                # 4. numpy ArrayLike alias
                # It seems like the current version of sphinx does not properly handle
                # aliases, specifically when using autodoc_typehints which we need.
                # See https://github.com/sphinx-doc/sphinx/issues/10455 for the open
                # issue and https://stackoverflow.com/questions/
                # 73223417/type-aliases-in-type-hints-are-not-preserved
                # for a comment regarding the autodoc issue
                # In our case, this is not a severe problem as we only rarely use
                # ArrayLike and thus simply manually replace the corresponding lines.
                if " * **arr**" in line and "`Union`" in line:
                    # Get the first part with the convoluted type hint and the second
                    # part with the actual description
                    _, description = line.split("–")
                    # Construct the first part of the line manually
                    type_alias = (
                        "  * **arr** ([`ArrayLike`](https://numpy.org/devdocs/"
                        + "reference/typing.html#numpy.typing.ArrayLike)) - "
                    )
                    line = type_alias + description
                if "* **x**" in line and "`Union`" in line:
                    # Get the first part with the convoluted type hint and the second
                    # part with the actual description
                    _, description = line.split("–")
                    # Construct the first part of the line manually
                    type_alias = (
                        "  * **x** ([`ArrayLike`](https://numpy.org/devdocs/"
                        + "reference/typing.html#numpy.typing.ArrayLike)) - "
                    )
                    line = type_alias + description
                if "* **y**" in line and "`Union`" in line:
                    # Get the first part with the convoluted type hint and the second
                    # part with the actual description
                    _, description = line.split("–")
                    # Construct the first part of the line manually
                    type_alias = (
                        "  * **y** ([`ArrayLike`](https://numpy.org/devdocs/"
                        + "reference/typing.html#numpy.typing.ArrayLike)) - "
                    )
                    line = type_alias + description

                # 5. Return annotations for properties
                # Sphinx shows the return type also for properties which is not
                # necessary. Thus, whenever we encounter a property, we skip the next
                # line that contains a return
                # NOTE: It seems like sphinx does not add the return type hint for
                # cached properties like in comp_df. Thus, there is an additional check
                # that sets skip_property_line=Fals again if we encounter another
                # heading
                if line.startswith("#### property") or line.startswith(
                    "#### abstract property"
                ):
                    skip_property_line = True

                # None return types
                # If a function returns "None", then this should be part of the doc
                if line.startswith("  [`None`]"):
                    # We need to get rid of the previous line and then continue
                    formatted_content = formatted_content[:-1]
                    continue

                # Multi-line enumerations in docstrings
                # This happens e.g. in simulation.py
                if ">   :" in line:
                    line = line.replace(">   :", ">    ")

            # PART 2: FORMAT LINKS
            # We have to take care of different kinds of links.
            # 1. Links to classes in other files
            # 2. Links to other files
            # 3. Links to classes in the same file
            # 4. Links to functions
            # The fourth one cannot be fixed without severe manual fixes since the
            # signature of the functions is part of the permalink. The intended solution
            # for this is to remove links to functions, which does not work yet and is
            # hence a TODO

            # We begin with links to other files. These are characterized by the link
            # starting with the string "(baybe."
            if "(baybe." in line:
                # We begin by extracting the part in the () parentheses
                start = line.index("(baybe.") + 1
                end = line.index(")", start)
                link = line[start:end]

                # Depending on link, there are no different types of links
                # 1. Other files
                # These are characterized by having no occurrence of #
                if link.count("#") == 0:
                    # Here, only the link to the file itself needs to be replaced
                    # We get rid of the ending and manually construct the new_link
                    # Note that this uses how the permalinks are created above
                    link = link[:-3].lower()
                    new_link = f"/baybe/sdk/docs/{link.replace('.','')}"
                # 2. Classes in other files
                # These have the form "otherfile#class" and are thus characterized by
                # having one # in link
                if link.count("#") == 1:
                    permalink, classlink = link.split("#")
                    # Remove .md suffix
                    permalink = permalink[:-3]
                    # Get the permalink to the file
                    permalink = f"/baybe/sdk/docs/{permalink.replace('.','')}/"
                    # Add the link to the class
                    classlink = "#class-" + classlink
                    # Put everything together
                    new_link = (permalink + classlink).lower()
                # Replace the old link by the new one
                line = re.sub(r"\(baybe.(.*?)\)", f"({new_link})", line)
            # Second case: Links within files
            # These are characterized by the link starting with "(#baybe"
            if "(#baybe" in line:
                # To create the link, we require the permalink of the current file.
                # Since these have the format #baybe.filename we can get them as follows
                start = line.index("#baybe.") + 1
                next_point = line.index(".", start + 6)
                link = line[start:next_point]
                permalink = f"/baybe/sdk/docs/{link.replace('.','')}/#class-"
                # Now we need to get the link itself
                start = line.index("#baybe.") + 1
                closing_parentheses = line.index(")", start)
                link = line[start:closing_parentheses]
                new_link = (permalink + link).lower()
                # Replace the old link by the new one
                line = re.sub(r"\(#baybe.(.*?)\)", f"({new_link})", line)

            # Add the (potentially formatted) line to the list
            formatted_content.append(line)
        f.close()
    # We rewrite the file here completely as we are not just adding or removing single
    # lines
    with open(file, "w", encoding="UTF-8") as f:
        f.truncate(0)
        f.write(LINES_TO_ADD)
        f.writelines(formatted_content)
        f.close()
    # Check if we need to rename the file.
    # This happens since we want to have the original baybe.md file as the file for the
    # folder.
    if NAME == "baybe":
        file = file.rename(file.parent / f"{directory_name}.md")

# Copy the files to the intended location
# Get only the corresponding markdown resp. html files
if USE_HTML:
    documentation = pathlib.Path(build_dir)
else:
    documentation = pathlib.Path(build_dir / "markdown" / "sdk")

shutil.move(documentation, destination_dir)

# Clean the other files
if build_dir.is_dir():
    shutil.rmtree(build_dir)

if sdk_dir.is_dir():
    shutil.rmtree(sdk_dir)
