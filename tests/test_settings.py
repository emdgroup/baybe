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


def assert_attribute_values(obj: Any, attributes: dict[str, Any], /) -> None:
    """Assert that the attributes of an object match the expected values."""
    for key, expected in attributes.items():
        actual = getattr(obj, key)
        assert actual == expected, (
            f"Attribute '{key}' expected to be '{expected}' but got '{actual}'."
        )


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
    assert_attribute_values(s, overwrites)
    assert_attribute_values(settings, overwrites)

    # The same applies for evolving a settings object (here, we roll back the changes)
    s2 = evolve(s, **original_values, apply_immediately=True)
    assert_attribute_values(s, overwrites)
    assert_attribute_values(s2, original_values)
    assert_attribute_values(settings, original_values)


@pytest.mark.parametrize("immediately", [True, False])
def test_setting_via_context(immediately: bool):
    """Settings are rolled back after exiting a settings context."""
    attributes = Settings.attributes

    # Collect original and new settings values
    original_values = {fld.name: getattr(settings, fld.name) for fld in attributes}
    overwrites = {key: toggle(value) for key, value in original_values.items()}

    # The settings of a new object are only applied immediately if specified
    s = Settings(apply_immediately=immediately, **overwrites)
    reference = overwrites if immediately else original_values
    assert_attribute_values(s, overwrites)
    assert_attribute_values(settings, reference)

    # The new settings are applied within the context
    with s:
        assert_attribute_values(s, overwrites)
        assert_attribute_values(settings, overwrites)

    # After exiting the context, the original settings are restored
    assert_attribute_values(s, overwrites)
    assert_attribute_values(settings, reference)


def test_nested_contexts():
    """Settings can be nested and properly restored in LIFO order."""
    original_value = settings.dataframe_validation

    with Settings(dataframe_validation=True):
        assert settings.dataframe_validation

        with Settings(dataframe_validation=False):
            assert not settings.dataframe_validation

        # After exiting inner context, outer setting should be restored
        assert settings.dataframe_validation

    # After exiting outer context, original setting should be restored
    assert settings.dataframe_validation == original_value


def test_exception_during_context_settings():
    """Exceptions raised inside a context are propagated and settings are restored."""
    original_value = settings.dataframe_validation

    class CustomError(Exception):
        """A custom exception for testing purposes."""

    # The custom exception is properly propagated
    with pytest.raises(CustomError, match="Test exception"):
        with Settings(dataframe_validation=not original_value):
            assert settings.dataframe_validation == (not original_value)
            raise CustomError("Test exception")

    # Settings are restored despite the exception
    assert settings.dataframe_validation == original_value


def test_setting_via_decorator():
    """Settings can be enabled by decorating callables."""
    attributes = Settings.attributes

    # Collect original and new settings values
    original_values = {fld.name: getattr(settings, fld.name) for fld in attributes}
    overwrites = {key: toggle(value) for key, value in original_values.items()}

    @Settings(**overwrites)
    def func():
        assert_attribute_values(settings, overwrites)

    # The new settings are applied within the function
    func()

    # After exiting the function, the original settings are restored
    assert_attribute_values(settings, original_values)
