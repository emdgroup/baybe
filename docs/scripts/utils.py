"""This file contains some utility function for building the documentation."""


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
