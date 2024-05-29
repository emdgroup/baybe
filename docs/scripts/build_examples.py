"""Utility for creating the examples."""

import shutil
import textwrap
from pathlib import Path
from subprocess import DEVNULL, STDOUT, check_call

from tqdm import tqdm

# TODO full rebuild option


def build_examples(destination_directory: Path, dummy: bool, remove_dir: bool):
    """Create the documentation version of the examples files.

    Note that this deletes the destination directory if it already exists.

    Args:
        destination_directory: The destination directory.
        dummy: Only build a dummy version of the files.
        remove_dir: Remove the examples directory if it already exists.

    Raises:
        OSError: If the directory already exists but should not be removed.
    """
    # if the destination directory already exists it is deleted
    if destination_directory.is_dir():
        if remove_dir:
            shutil.rmtree(destination_directory)
        else:
            raise OSError("Destination directory exists but should not be removed.")

    # Copy the examples folder in the destination directory
    shutil.copytree("examples", destination_directory)

    # For the toctree of the top level example folder, we need to keep track of all
    # folders. We thus write the header here and populate it during the execution of the
    # examples
    ex_file = """# Examples\n\n```{toctree}\n:maxdepth: 2\n\n"""

    # List all directories in the examples folder
    ex_directories = [d for d in destination_directory.iterdir() if d.is_dir()]

    # This list contains the order of the examples as we want to have them in the end.
    # The examples that should be the first ones are already included here and skipped
    # later on. ALl other are just included.
    ex_order = [
        "Basics<Basics/Basics>\n",
        "Searchspaces<Searchspaces/Searchspaces>\n",
        "Constraints Discrete<Constraints_Discrete/Constraints_Discrete>\n",
        "Constraints Continuous<Constraints_Continuous/Constraints_Continuous>\n",
        "Multi Target<Multi_Target/Multi_Target>\n",
        "Serialization<Serialization/Serialization>\n",
        "Custom Surrogates<Custom_Surrogates/Custom_Surrogates>\n",
    ]

    # Iterate over the directories.
    for sub_directory in (pbar := tqdm(ex_directories)):
        # Get the name of the current folder
        # Format it by replacing underscores and capitalizing the words
        folder_name = sub_directory.stem
        formatted = " ".join(word.capitalize() for word in folder_name.split("_"))

        # Create the link to the folder to the top level toctree.
        ex_file_entry = formatted + f"<{folder_name}/{folder_name}>\n"
        # Add it to the list of examples if it is not already contained
        if ex_file_entry not in ex_order:
            ex_order.append(ex_file_entry)

        # We need to create a file for the inclusion of the folder.
        # We thus get the content of the corresponding header file.
        header_folder_name = sub_directory / f"{folder_name}_Header.md"
        header = header_folder_name.read_text()

        subdir_toctree = header + "\n```{toctree}\n:maxdepth: 1\n\n"

        # Set description of progressbar
        pbar.set_description("Overall progress")

        # list all .py files in the subdirectory that need to be converted
        py_files = list(sub_directory.glob("**/*.py"))

        # Iterate through the individual example files
        for file in (inner_pbar := tqdm(py_files, leave=False)):
            # Include the name of the file to the toctree
            # Format it by replacing underscores and capitalizing the words
            file_name = file.stem

            formatted = " ".join(word.capitalize() for word in file_name.split("_"))
            # Remove duplicate "constraints" for the files in the constraints folder.
            if "Constraints" in folder_name and "Constraints" in formatted:
                formatted = formatted.replace("Constraints", "")

            # Also format the Prodsum name to Product/Sum
            if "Prodsum" in formatted:
                formatted = formatted.replace("Prodsum", "Product/Sum")
            subdir_toctree += formatted + f"<{file_name}>\n"

            # If we ignore the examples, we do not want to actually execute or convert
            # anything. Still, due to existing links, it is necessary to construct a
            # dummy file and then continue.
            if dummy:
                markdown_path = file.with_suffix(".md")
                # Rewrite the file
                with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
                    markdown_file.writelines("# DUMMY FILE")
                continue

            # Set description for progress bar
            inner_pbar.set_description(f"Progressing {folder_name}")

            # Create the Markdown file:

            # 1. Convert the file to jupyter notebook
            check_call(
                ["jupytext", "--to", "notebook", file], stdout=DEVNULL, stderr=STDOUT
            )

            notebook_path = file.with_suffix(".ipynb")

            # 2. Execute the notebook and convert to markdown.
            # This is only done if we decide not to ignore the examples.
            # The creation of the files themselves and converting them to markdown still
            # happens since we need the files to check for link integrity.
            convert_execute = [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--inplace",
                notebook_path,
            ]
            convert_execute.append("--execute")

            to_markdown = ["jupyter", "nbconvert", "--to", "markdown", notebook_path]

            check_call(convert_execute, stdout=DEVNULL, stderr=STDOUT)
            check_call(
                to_markdown,
                stdout=DEVNULL,
                stderr=STDOUT,
            )

            # CLEANUP
            markdown_path = file.with_suffix(".md")
            # We wrap lines which are too long as long as they do not contain a link.
            # To discover whether a line contains a link, we check if the string "]("
            # is contained.
            with open(markdown_path, encoding="UTF-8") as markdown_file:
                content = markdown_file.read()
                wrapped_lines = []
                ignored_substrings = (
                    "![svg]",
                    "![png]",
                    "<Figure size",
                    "it/s",
                    "s/it",
                )
                for line in content.splitlines():
                    if any(substring in line for substring in ignored_substrings):
                        continue
                    if len(line) > 88 and "](" not in line:
                        wrapped = textwrap.wrap(line, width=88)
                        wrapped_lines.extend(wrapped)
                    else:
                        wrapped_lines.append(line)

            # Add a manual new line to each of the lines
            lines = [line + "\n" for line in wrapped_lines]
            # Delete lines we do not want to have in our documentation
            # lines = [line for line in lines if "![svg]" not in line]
            # We check whether pre-built light and dark plots exist. If so, we append
            # corresponding lines to our markdown file for including them.
            light_figure = Path(sub_directory / (file_name + "_light.svg"))
            dark_figure = Path(sub_directory / (file_name + "_dark.svg"))
            if light_figure.is_file() and dark_figure.is_file():
                lines.append(f"```{{image}} {file_name}_light.svg\n")
                lines.append(":align: center\n")
                lines.append(":class: only-light\n")
                lines.append("```\n")
                lines.append(f"```{{image}} {file_name}_dark.svg\n")
                lines.append(":align: center\n")
                lines.append(":class: only-dark\n")
                lines.append("```\n")

            # Rewrite the file
            with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
                markdown_file.writelines(lines)

        # Write last line of toctree file for this directory and write the file
        subdir_toctree += "```"
        with open(
            sub_directory / f"{sub_directory.name}.md", "w", encoding="UTF-8"
        ) as f:
            f.write(subdir_toctree)

    # Append the ordered list of examples to the file for the top level folder
    ex_file += "".join(ex_order)
    # Write last line of top level toctree file and write the file
    ex_file += "```"
    with open(
        destination_directory / f"{destination_directory.name}.md",
        "w",
        encoding="UTF-8",
    ) as f:
        f.write(ex_file)

    # Remove remaining files and subdirectories from the destination directory
    # Remove any not markdown files
    for file in destination_directory.glob("**/*"):
        if file.is_file():
            if file.suffix not in (".md", ".svg") or "Header" in file.name:
                file.unlink(file)

    # Remove any remaining empty subdirectories
    for subdirectory in destination_directory.glob("*/*"):
        if subdirectory.is_dir() and not any(subdirectory.iterdir()):
            subdirectory.rmdir()
