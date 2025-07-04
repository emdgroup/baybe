"""Generic metadata system for BayBE objects."""

from __future__ import annotations

from typing import Any

import cattrs
from attrs import define, field, fields
from attrs.validators import deep_mapping, instance_of
from attrs.validators import optional as optional_v

from baybe.serialization import SerialMixin, converter
from baybe.utils.basic import classproperty


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

    @misc.validator
    def _validate_misc(self, _, value: dict[str, Any]) -> None:
        if inv := set(value).intersection(self._explicit_fields):
            raise ValueError(
                f"Miscellaneous metadata cannot contain the following fields: {inv}. "
                f"Use the corresponding attributes instead."
            )

    @classproperty
    def _explicit_fields(self) -> set[str]:
        """The explicit metadata fields."""  # noqa: D401
        flds = fields(Metadata)
        return {fld.name for fld in flds if fld.name != flds.misc.name}


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
    return converter.structure(value, Metadata)


@converter.register_structure_hook
def _separate_metadata_fields(dct: dict[str, Any], _: type[Metadata]) -> Metadata:
    """Separate known fields from miscellaneous metadata."""
    dct = dct.copy()
    explicit = {fld: dct.pop(fld, None) for fld in Metadata._explicit_fields}
    return Metadata(**explicit, misc=dct)


@converter.register_unstructure_hook
def _flatten_misc_metadata(metadata: Metadata) -> dict[str, Any]:
    """Flatten the metadata for serialization."""
    fn = cattrs.gen.make_dict_unstructure_fn(Metadata, converter)
    dct = fn(metadata)
    fld = fields(Metadata).misc.name
    dct = dct | dct.pop(fld)
    return dct
