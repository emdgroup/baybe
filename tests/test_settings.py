"""Tests for settings management."""

from typing import Any

import pytest
from attrs import Attribute, evolve, fields

from baybe import Settings, settings


def toggle(value: Any, /) -> Any:
    """Toggle a given settings value."""
    match value:
        case bool():
            return not value
    raise ValueError(f"Undefined toggling operation for type '{type(value)}'.")


def invalidate(value: Any, /) -> Any:
    """Create an invalid value for a given setting."""
    match value:
        case bool():
            return "invalid"
    raise ValueError(f"Undefined invalidation operation for type '{type(value)}'.")


@pytest.mark.parametrize("attribute", Settings.attributes, ids=lambda a: a.name)
def test_direct_setting(attribute: Attribute):
    """Attributes of the global settings object can be directly modified."""
    original_value = getattr(settings, attribute.name)
    assert original_value == attribute.default

    new_value = toggle(attribute.default)
    setattr(settings, attribute.name, new_value)
    assert getattr(settings, attribute.name) == new_value


def test_direct_setting_unknown_attribute():
    """Attempting to apply an unknown setting raises an error."""
    with pytest.raises(AttributeError):
        settings.unknown_setting = True


@pytest.mark.parametrize("attribute", fields(Settings), ids=lambda a: a.name)
def test_invalid_setting(attribute: Attribute):
    """Attempting to apply an invalid settings value raises an error."""
    with pytest.raises(TypeError):
        setattr(settings, attribute.name, invalidate(attribute.default))


def test_setting_via_instantiation():
    """Applying joint settings can be done via settings instantiation."""
    attributes = Settings.attributes

    # Collect original and new settings values
    original_values = {fld.name: getattr(settings, fld.name) for fld in attributes}
    overwrites = {key: toggle(value) for key, value in original_values.items()}

    # Instantiating a new settings object immediately applies its settings globally
    s = Settings(**overwrites)
    for fld in attributes:
        assert getattr(s, fld.name) == overwrites[fld.name]
        assert getattr(settings, fld.name) == overwrites[fld.name]

    # The same applies for evolving a settings object (here, we roll back the changes)
    s2 = evolve(s, **original_values)
    for fld in attributes:
        assert getattr(s, fld.name) == overwrites[fld.name]
        assert getattr(s2, fld.name) == original_values[fld.name]
        assert getattr(settings, fld.name) == original_values[fld.name]
