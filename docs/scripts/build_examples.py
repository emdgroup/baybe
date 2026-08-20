"""Utility for creating the examples."""

import os
import shutil
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from subprocess import DEVNULL, STDOUT, check_call

from tqdm import tqdm

# TODO full rebuild option


def _convert_example(file: Path, sub_directory: Path) -> None:
    """Convert a single example file into its documentation markdown page.

    Args:
        file: The example ``.py`` file to convert.
        sub_directory: The example folder the file belongs to (used to locate figures).
    """
    file_name = file.stem

    # Convert the file to a jupyter notebook
    check_call(["jupytext", "--to", "notebook", file], stdout=DEVNULL, stderr=STDOUT)

    notebook_path = file.with_suffix(".ipynb")

    # Execute the notebook, then convert it to markdown
    env = os.environ | {"PYTHONPATH": os.getcwd()}
    convert_execute = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--inplace",
        "--execute",
        notebook_path,
    ]
    check_call(convert_execute, stdout=DEVNULL, stderr=STDOUT, env=env)
    to_markdown = ["jupyter", "nbconvert", "--to", "markdown", notebook_path]
    check_call(to_markdown, stdout=DEVNULL, stderr=STDOUT, env=env)

    # Wrap long lines, except those containing a link (detected via "](")
    markdown_path = file.with_suffix(".md")
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
            # Skip formatter control lines
            if "fmt: off" in line or "fmt: on" in line:
                continue
            if any(substring in line for substring in ignored_substrings):
                continue
            if (
                len(line) > 88
                and "](" not in line
                and not line.lstrip().startswith("#")
            ):
                wrapped = textwrap.wrap(line, width=88)
                wrapped_lines.extend(wrapped)
            else:
                wrapped_lines.append(line)

    lines = [line + "\n" for line in wrapped_lines]
    # Append light/dark figures if both exist, else a single figure if present
    light_figure = Path(sub_directory / (file_name + "_light.svg"))
    dark_figure = Path(sub_directory / (file_name + "_dark.svg"))
    figure = Path(sub_directory / (file_name + ".svg"))
    if light_figure.is_file() and dark_figure.is_file():
        lines.append(f"```{{image}} {file_name}_light.svg\n")
        lines.append(":align: center\n")
        lines.append(":class: only-light\n")
        lines.append("```\n")
        lines.append(f"```{{image}} {file_name}_dark.svg\n")
        lines.append(":align: center\n")
        lines.append(":class: only-dark\n")
        lines.append("```\n")
    elif figure.is_file():
        lines.append(f"```{{image}} {file_name}.svg\n")
        lines.append(":align: center\n")
        lines.append("```\n")

    with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
        markdown_file.writelines(lines)


def build_examples(
    destination_directory: Path,
    dummy: bool,
    remove_dir: bool,
    max_workers: int | None = None,
):
    """Create the documentation version of the examples files.

    Note that this deletes the destination directory if it already exists.

    Args:
        destination_directory: The destination directory.
        dummy: Only build a dummy version of the files.
        remove_dir: Remove the examples directory if it already exists.
        max_workers: Number of examples to convert concurrently. Defaults to the number
            of available CPUs. Lower it if the build runs out of memory.

    Raises:
        OSError: If the directory already exists but should not be removed.
    """
    # if the destination directory already exists it is deleted
    if destination_directory.is_dir():
        if remove_dir:
            shutil.rmtree(destination_directory)
        else:
            raise OSError("Destination directory exists but should not be removed.")

    # Copy the examples folder in the destination directory. "__pycache__" might be
    # present in the examples folder and needs to be ignored
    def ignore_pycache(_, contents: list[str]):
        return [item for item in contents if item == "__pycache__"]

    shutil.copytree("examples", destination_directory, ignore=ignore_pycache)

    # For the toctree of the top level example folder, we need to keep track of all
    # folders. We thus write the header here and populate it during the execution of the
    # examples
    ex_file = """# Examples\n\n```{toctree}\n:maxdepth: 2\n\n"""

    ex_directories = [d for d in destination_directory.iterdir() if d.is_dir()]

    # Desired example order; entries listed here come first, all others are appended
    ex_order = [
        "Basics<Basics/Basics>\n",
        "Searchspaces<Searchspaces/Searchspaces>\n",
        "Constraints Discrete<Constraints_Discrete/Constraints_Discrete>\n",
        "Constraints Continuous<Constraints_Continuous/Constraints_Continuous>\n",
        "Multi Target<Multi_Target/Multi_Target>\n",
        "Serialization<Serialization/Serialization>\n",
        "Custom Surrogates<Custom_Surrogates/Custom_Surrogates>\n",
    ]

    # Files to convert, collected across all folders so conversion can be batched
    files_to_convert: list[tuple[Path, Path]] = []

    # First pass (sequential): build the toctree files and collect the conversion work
    for sub_directory in ex_directories:
        # Folder name, formatted for display (underscores to spaces, capitalized)
        folder_name = sub_directory.stem
        formatted = " ".join(word.capitalize() for word in folder_name.split("_"))

        # Add the folder to the top level toctree if not already present
        ex_file_entry = formatted + f"<{folder_name}/{folder_name}>\n"
        if ex_file_entry not in ex_order:
            ex_order.append(ex_file_entry)

        # Start the folder's toctree from its header file
        header_folder_name = sub_directory / f"{folder_name}_Header.md"
        header = header_folder_name.read_text()

        subdir_toctree = header + "\n```{toctree}\n:maxdepth: 1\n\n"

        py_files = list(sub_directory.glob("**/*.py"))

        for file in py_files:
            # Add the file to the folder's toctree, with a formatted display name
            file_name = file.stem

            formatted = " ".join(word.capitalize() for word in file_name.split("_"))
            # Drop duplicate "Constraints" in the constraints folders
            if "Constraints" in folder_name and "Constraints" in formatted:
                formatted = formatted.replace("Constraints", "")

            # Format "Prodsum" as "Product/Sum"
            if "Prodsum" in formatted:
                formatted = formatted.replace("Prodsum", "Product/Sum")
            subdir_toctree += formatted + f"<{file_name}>\n"

            # In dummy mode, write a placeholder so links still resolve
            if dummy:
                markdown_path = file.with_suffix(".md")
                with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
                    markdown_file.writelines("# DUMMY FILE")
                continue

            files_to_convert.append((file, sub_directory))

        # Write last line of toctree file for this directory and write the file
        subdir_toctree += "```"
        with open(
            sub_directory / f"{sub_directory.name}.md", "w", encoding="UTF-8"
        ) as f:
            f.write(subdir_toctree)

    # Second pass: convert the independent example files concurrently
    if files_to_convert:
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        workers = max(1, min(max_workers, len(files_to_convert)))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_convert_example, file, sub_directory): file
                for file, sub_directory in files_to_convert
            }
            # `result()` re-raises so a failed example aborts the build
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Converting examples"
            ):
                future.result()

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
