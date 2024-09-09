"""This file contains some utility function for building the documentation."""

import re
import sys
from pathlib import Path

from packaging.version import Version


def adjust_pictures(
    file_path: str, match: str, light_version: str, dark_version: str
) -> None:
    """Adjust pictures to have different versions for light and dark furo style.

    Args:
        file_path: The (relative) path to the file that needs to be adjusted. Typically
            the index and README_link file.
        match: The expression that identifies the picture name as referenced in the
            file (do not include the file extension)
        light_version: The name of the light mode picture version.
        dark_version: The name of the dark mode picture version.
    """
    with open(file_path) as file:
        lines = file.readlines()

    line_index = None
    for i, line in enumerate(lines):
        if match in line:
            line_index = i
            break

    if line_index is not None:
        line = lines[line_index]
        light_line = line
        light_line = light_line.replace(  # For replacing the banner
            '"reference external"', '"reference external only-light"'
        )
        light_line = light_line.replace(  # For replacing the example plot
            'img alt="Substance Encoding Example" ',
            'img alt="Substance Encoding Example" class="only-light align-center" ',
        )
        lines[line_index] = light_line
        dark_line = light_line.replace("light", "dark")
        dark_line = dark_line.replace(light_version, dark_version)
        lines.insert(line_index + 1, dark_line)

        with open(file_path, "w") as file:
            file.writelines(lines)


def add_version_to_selector_page(version: str) -> None:
    """Add the newly built version to the version selection overview.

    Args:
        version: The version that should be added.
    """
    indent = "        "
    new_line = (
        f"{indent}<li><a href="
        f'"https://emdgroup.github.io/baybe/{version}/">{version}'
        "</a></li>\n"
    )
    file = Path("versions.html")
    modified_lines = []

    # Regular expression pattern to match the version number in the format X.Y.Z
    pattern = r"\b\d+\.\d+\.\d+\b"

    # Indices for memorizing the index of the line containing "stable" and the last
    # version that was larger than the one to be inserted
    stable_index = -1
    last_version_index = -1
    with file.open(mode="r") as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            # We might find strings that look like version numbers without wanting to
            # We thus need to potentially reset `last_version_index`
            if "<h1>Available versions</h1>" in line:
                last_version_index = -1

            # Search for the version number in the current line
            if version_number := re.search(pattern, line):
                # Memorize whether we saw a larger version number, implying that this is
                # a hotfix
                if Version(version_number.group()) > Version(version):
                    last_version_index = ind

            # Add the already existing line
            modified_lines.append(line)

            # Slightly weird string since the word "stable" appears is other places
            if "stable</a></li>" in line:
                stable_index = ind

    # We never found a larger number than the current, so we just insert after stable
    index = last_version_index + 1 if last_version_index != -1 else stable_index + 1
    modified_lines.insert(index, new_line)
    with file.open(mode="w") as f:
        f.writelines(modified_lines)


def adjust_version_to_html(version: str) -> None:
    """Adjust the shown version in the HTML files.

    This method includes the current version in the sidebar of all HTML files and
    additionally adds a banner warning when the documentation is not ``stable``.

    Args:
        version: The version that should be injected into the sidebar.
    """
    prefix = '<li class="toctree-l1">'
    html_class = "reference external"
    link = "https://emdgroup.github.io/baybe/versions"
    new_line = (
        f'{prefix}<a class={html_class} href="{link}">Version: {version}</a></li>'
    )
    link_to_stable = '<a href="https://emdgroup.github.io/baybe/stable">stable</a>'
    # The content of the announcement that might be added.
    announcement = (
        f"You are not viewing the documentation of the {link_to_stable} version."
    )
    # Actual HTML code that will be inserted
    announcement_html = (
        """<div class="announcement">\n"""
        """    <aside class="announcement-content">\n"""
        f"""        {announcement}\n"""
        """    </aside>\n"""
        """</div>\n"""
    )
    add_announcement = version != "stable"
    path = Path(version)
    if path.exists():
        # Recursively check all HTML files
        for file in path.rglob("*.html"):
            modified_lines = []
            with file.open(mode="r") as f:
                lines = f.readlines()
                for line in lines:
                    # Check if we need to add the announcement
                    if add_announcement:
                        # Add announcement at correct position
                        if line.strip() == '<div class="page">':
                            modified_lines.insert(-2, announcement_html)
                    if "Versions</a></li>" in line:
                        modified_lines.append(new_line)
                    else:
                        modified_lines.append(line)
            with file.open(mode="w") as f:
                f.writelines(modified_lines)


def check_for_hotfix(tags: list[str], version: str):
    """Check whether the current build corresponds to a hotfix."""
    split_tags = tags.split("\n")
    split_tags.sort(key=Version)
    print(Version(version) < Version(split_tags[-1]), end="")


if __name__ == "__main__":
    chosen_method = sys.argv[1]
    version = sys.argv[2]
    if chosen_method == "selector_page":
        add_version_to_selector_page(version)
    elif chosen_method == "html":
        adjust_version_to_html(version)
    elif chosen_method == "hotfix":
        check_for_hotfix(tags=sys.argv[3], version=version)
    else:
        print(f"Invalid arguments {sys.argv} were chosen!")
