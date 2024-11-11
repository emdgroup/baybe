"""This module contains utility functions for the result module."""

from typing import Any


def _convert_metadata_to_string(metadata: dict[Any, Any]) -> dict[str, str]:
    """Convert the metadata to a string representation.

    The function will convert the metadata to a string representation.
    """
    for key, value in metadata.items():
        sanitized_key = str(key).replace(" ", "_")
        metadata[sanitized_key] = str(value)
    return metadata
