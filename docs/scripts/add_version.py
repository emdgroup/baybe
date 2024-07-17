"""Utilities for adding versions when building the documentation."""

import sys
from pathlib import Path


def add_version_to_selector_page(version: str) -> None:
    """Add the newly built version to the version selection overview.

    Args:
        version: The version that should be added.
    """
    indent = "        "
    new_line = (
        f"{indent}<li><a href="
        f'"https://avhopp.github.io/baybe_dev/{version}/">{version}'
        "</a></li>\n"
    )
    file = Path("versions.html")
    modified_lines = []
    with file.open(mode="r") as f:
        lines = f.readlines()
        for line in lines:
            modified_lines.append(line)
            # Add new line at correct position which is in the first line after stable
            if "Stable" in line:
                modified_lines.append(new_line)
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
    link = "https://avhopp.github.io/baybe_dev/versions"
    line_to_replace = f'{prefix}<a class={html_class} href="{link}">Versions</a></li>'
    new_line = (
        f'{prefix}<a class={html_class} href="{link}">Version: {version}</a></li>'
    )
    link_to_stable = '<a href="https://avhopp.github.io/baybe_dev/stable">stable</a>'
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
                    if line.strip() == line_to_replace:
                        modified_lines.append(new_line)
                    else:
                        modified_lines.append(line)
            with file.open(mode="w") as f:
                f.writelines(modified_lines)


if __name__ == "__main__":
    chosen_method = sys.argv[1]
    version = sys.argv[2]
    if chosen_method == "selector_page":
        print(f"Adding {version=} to version selector page")
        add_version_to_selector_page(version)
    elif chosen_method == "html":
        adjust_version_to_html(version)
        print(f"Adding {version=} to HTML")
    else:
        print(f"Invalid arguments {sys.argv} were chosen!")
