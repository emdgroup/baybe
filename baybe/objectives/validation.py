"""Validation functionality for objectives."""

from collections.abc import Collection

from baybe.targets.base import Target


def validate_target_names(  # noqa: DOC101, DOC103
    _, __, targets: Collection[Target]
) -> None:
    """An attrs-compatible validator to assert unique target names.

    Raises:
        ValueError: If the given collection contains targets with the same name.
    """  # noqa: D401
    if len(names := [t.name for t in targets]) != len(set(names)):
        raise ValueError("All targets must have unique names.")
