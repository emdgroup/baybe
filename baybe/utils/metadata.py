"""Generic metadata system for BayBE objects."""

from __future__ import annotations

from typing import Any

from attrs import define, field, fields
from attrs.validators import deep_mapping, instance_of
from attrs.validators import optional as optional_v

from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class Metadata(SerialMixin):
    """Metadata providing description, unit, and other information for BayBE objects."""

    description: str | None = field(
        default=None, validator=optional_v(instance_of(str))
    )
    """A description of the object."""

    unit: str | None = field(default=None, validator=optional_v(instance_of(str)))
    """The unit of measurement for the object."""

    misc: dict[str, Any] = field(
        factory=dict,
        validator=deep_mapping(
            mapping_validator=instance_of(dict),
            key_validator=instance_of(str),
            # FIXME: https://github.com/python-attrs/attrs/issues/1246
            value_validator=lambda *x: None,
        ),
    )
    """Additional user-defined metadata."""


def to_metadata(value: dict[str, Any] | Metadata, /) -> Metadata:
    """Convert a dictionary to :class:`Metadata` (with :class:`Metadata` passthrough).

    Args:
        value: The metadata input.

    Returns:
        The :class:`Metadata` instance.

    Raises:
        TypeError: If the input is not a dictionary or :class:`Metadata`.
    """
    if isinstance(value, Metadata):
        return value

    if not isinstance(value, dict):
        raise TypeError(
            f"The input must be a dictionary or a '{Metadata.__name__}' instance. "
            f"Got: {type(value)}"
        )

    # Separate known fields from unknown ones
    flds = fields(Metadata)
    value = value.copy()
    known_fields = {
        fld: value.pop(fld, None) for fld in (flds.description.name, flds.unit.name)
    }

    return Metadata(**known_fields, misc=value)
