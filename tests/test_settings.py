"""Tests for settings management."""

from typing import Any

import pytest
from attrs import Attribute, evolve

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


@pytest.mark.parametrize("attribute", Settings.attributes, ids=lambda a: a.name)
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

    # A collection of settings can be applied immediately
    s = Settings(**overwrites, apply_immediately=True)
    for fld in attributes:
        assert getattr(s, fld.name) == overwrites[fld.name]
        assert getattr(settings, fld.name) == overwrites[fld.name]

    # The same applies for evolving a settings object (here, we roll back the changes)
    s2 = evolve(s, **original_values, apply_immediately=True)
    for fld in attributes:
        assert getattr(s, fld.name) == overwrites[fld.name]
        assert getattr(s2, fld.name) == original_values[fld.name]
        assert getattr(settings, fld.name) == original_values[fld.name]


@pytest.mark.parametrize("immediately", [True, False])
def test_setting_via_context_manager(immediately: bool):
    """Settings are rolled back after exiting a settings context."""
    attributes = Settings.attributes

    # Collect original and new settings values
    original_values = {fld.name: getattr(settings, fld.name) for fld in attributes}
    overwrites = {key: toggle(value) for key, value in original_values.items()}

    # The settings of a new object are only applied immediately if specified
    s = Settings(apply_immediately=immediately, **overwrites)
    for fld in attributes:
        assert getattr(s, fld.name) == overwrites[fld.name]
        reference = overwrites if immediately else original_values
        assert getattr(settings, fld.name) == reference[fld.name]

    # The new settings are applied within the context
    with s:
        for fld in attributes:
            assert getattr(s, fld.name) == overwrites[fld.name]
            assert getattr(settings, fld.name) == overwrites[fld.name]

    # After exiting the context, the original settings are restored
    for fld in attributes:
        reference = overwrites if immediately else original_values
        assert getattr(settings, fld.name) == reference[fld.name]


def test_setting_via_decorator():
    """Settings can be enabled by decorating callables."""
    attributes = Settings.attributes

    # Collect original and new settings values
    original_values = {fld.name: getattr(settings, fld.name) for fld in attributes}
    overwrites = {key: toggle(value) for key, value in original_values.items()}

    @Settings(**overwrites)
    def func():
        for fld in attributes:
            assert getattr(settings, fld.name) == overwrites[fld.name]

    # The new settings are applied within the function
    func()

    # After exiting the function, the original settings are restored
    for fld in attributes:
        assert getattr(settings, fld.name) == original_values[fld.name]
