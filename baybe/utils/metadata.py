"""Generic metadata system for BayBE objects."""

from __future__ import annotations

from typing import Any, TypeVar

import cattrs
from attrs import AttrsInstance, define, field, fields
from attrs.validators import deep_mapping, instance_of
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.serialization import SerialMixin, converter
from baybe.serialization.core import _TYPE_FIELD
from baybe.utils.basic import classproperty

_TMetaData = TypeVar("_TMetaData", bound="Metadata")


@define(frozen=True)
class Metadata(SerialMixin):
    """Metadata class providing basic information for BayBE objects."""

    description: str | None = field(
        default=None, validator=optional_v(instance_of(str))
    )
    """A description of the object."""

    misc: dict[str, Any] = field(
        factory=dict,
        validator=deep_mapping(
            mapping_validator=instance_of(dict),
            key_validator=instance_of(str),
            # FIXME: https://github.com/python-attrs/attrs/issues/1246
            value_validator=lambda *x: None,
        ),
        kw_only=True,
    )
    """Additional user-defined metadata."""

    @misc.validator
    def _validate_misc(self, _, value: dict[str, Any]) -> None:
        if inv := set(value).intersection(self._explicit_fields | {_TYPE_FIELD}):
            raise ValueError(
                f"Miscellaneous metadata cannot contain the following fields: {inv}. "
                f"Use the corresponding attributes instead."
            )

    @classproperty
    def _explicit_fields(cls: type[AttrsInstance]) -> set[str]:
        """The explicit metadata fields."""  # noqa: D401
        flds = fields(cls)
        return {fld.name for fld in flds if fld.name != flds.misc.name}

    @property
    def is_empty(self) -> bool:
        """Check if metadata contains any meaningful information."""
        return self.description is None and not self.misc


@define(frozen=True)
class MeasurableMetadata(Metadata):
    """Class providing metadata for BayBE :class:`Parameter` objects."""

    unit: str | None = field(default=None, validator=optional_v(instance_of(str)))
    """The unit of measurement for the parameter."""

    @override
    @property
    def is_empty(self) -> bool:
        """Check if metadata contains any meaningful information."""
        return super().is_empty and self.unit is None


def to_metadata(
    value: dict[str, Any] | _TMetaData | None, cls: type[_TMetaData], /
) -> _TMetaData:
    """Convert a dictionary to :class:`Metadata` (with :class:`Metadata` passthrough).

    Args:
        value: The metadata input.
        cls: The specific :class:`Metadata` subclass to convert to.

    Returns:
        The created metadata instance of the requested :class:`Metadata` subclass.

    Raises:
        TypeError: If the input is not a dictionary or of the specified
            :class:`Metadata` type.
    """
    if value is None:
        return cls()

    if isinstance(value, cls):
        return value

    if not isinstance(value, dict):
        raise TypeError(
            f"The input must be a dictionary or a '{cls.__name__}' instance. "
            f"Got: {type(value)}"
        )

    # Separate known fields from unknown ones
    return converter.structure(value, cls)


ConvertibleToMeasurableMetadata = MeasurableMetadata | dict[str, Any] | None
"""A type alias for objects that can be converted to :class:`MeasurableMetadata`."""


@converter.register_structure_hook
def _separate_metadata_fields(dct: dict[str, Any], cls: type[Metadata]) -> Metadata:
    """Separate known fields from miscellaneous metadata."""
    dct = dct.copy()
    dct.pop(_TYPE_FIELD, None)
    explicit = {fld: dct.pop(fld, None) for fld in cls._explicit_fields}
    return cls(**explicit, misc=dct)


@converter.register_unstructure_hook
def _flatten_misc_metadata(metadata: Metadata) -> dict[str, Any]:
    """Flatten the metadata for serialization."""
    cls = type(metadata)
    fn = cattrs.gen.make_dict_unstructure_fn(cls, converter)
    dct = fn(metadata)
    dct = dct | dct.pop(fields(Metadata).misc.name)
    return dct
