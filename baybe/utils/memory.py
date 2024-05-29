"""Utilities for memory usage."""


def bytes_to_human_readable(num: float, /) -> tuple[float, str]:
    """Turn a float number representing a memory byte size into a human-readable format.

    Args:
        num: The number representing a memory size in bytes.

    Returns:
        A tuple with the converted number and its determined human-readable unit.
    """
    for unit in ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(num) < 1024.0:
            return num, unit
        num /= 1024.0
    return round(num, 2), "YB"
