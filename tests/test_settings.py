"""Tests for settings management."""

import os
from pathlib import Path
from typing import Any

import pytest
from attrs import Attribute, evolve

from baybe import Settings, active_settings


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


@pytest.fixture()
def original_values():
    """The original settings values."""
    return {fld.name: getattr(active_settings, fld.name) for fld in Settings.attributes}


@pytest.fixture()
def toggled_values():
    """Toggled settings values (i.e. differing from the original values)."""
    return {
        fld.name: toggle(getattr(active_settings, fld.name))
        for fld in Settings.attributes
    }


def test_setting_unknown_attribute():
    """Attempting to activate an unknown setting raises an error."""
    with pytest.raises(AttributeError):
        active_settings.unknown_setting = True
    with pytest.raises(TypeError):
        Settings(unknown_setting=True)


@pytest.mark.parametrize("attribute", Settings.attributes, ids=lambda a: a.name)
def test_invalid_setting(attribute: Attribute):
    """Attempting to activate an invalid settings value raises an error."""
    original_value = getattr(active_settings, attribute.name)
    with pytest.raises(ValueError):
        setattr(active_settings, attribute.name, invalidate(original_value))
    with pytest.raises(ValueError):
        Settings(**{attribute.name: invalidate(original_value)})


@pytest.mark.parametrize("attribute", Settings.attributes, ids=lambda a: a.name)
def test_direct_setting(attribute: Attribute):
    """Attributes of the global settings object can be directly modified."""
    original_value = getattr(active_settings, attribute.name)
    new_value = toggle(original_value)
    assert original_value != new_value
    setattr(active_settings, attribute.name, new_value)
    assert getattr(active_settings, attribute.name) == new_value


def test_setting_via_instantiation(original_values, toggled_values):
    """Activating settings jointly can be done via settings instantiation."""
    # A collection of settings can be activated immediately
    s = Settings(**toggled_values, activate_immediately=True)
    assert_attribute_values(s, toggled_values)
    assert_attribute_values(active_settings, toggled_values)

    # The same applies for evolving a settings object (here, we roll back the changes)
    s2 = evolve(s, **original_values, activate_immediately=True)
    assert_attribute_values(s, toggled_values)
    assert_attribute_values(s2, original_values)
    assert_attribute_values(active_settings, original_values)


def test_sequential_setting_via_instantiation(original_values):
    """New settings have previous settings as their default.

    Settings can be activated sequentially, one attribute at a time. That is, instead of
    using the attribute defaults for unspecified attributes, the values of the current
    settings are used.
    """
    # The growing collection of all modified attributes
    modified: dict[str, Any] = {}

    for attr in Settings.attributes:
        # Modify one attribute at a time
        change = {attr.name: toggle(original_values[attr.name])}
        modified.update(change)
        s = Settings(**change, activate_immediately=True)

        # The new object carries the currently modified attribute and all previous ones
        assert_attribute_values(s, original_values | modified)
        assert_attribute_values(active_settings, original_values | modified)


@pytest.mark.parametrize("immediately", [True, False])
def test_setting_via_context(immediately: bool, original_values, toggled_values):
    """Settings are rolled back after exiting a settings context."""
    # The settings of a new object are only activated immediately if specified
    s = Settings(activate_immediately=immediately, **toggled_values)
    reference = toggled_values if immediately else original_values
    assert_attribute_values(s, toggled_values)
    assert_attribute_values(active_settings, reference)

    # The new settings are activated within the context
    with s:
        assert_attribute_values(s, toggled_values)
        assert_attribute_values(active_settings, toggled_values)

    # After exiting the context, the original settings are restored
    assert_attribute_values(s, toggled_values)
    assert_attribute_values(active_settings, reference)


def test_nested_contexts():
    """Settings can be nested and properly restored in LIFO order."""
    original_value = active_settings.dataframe_validation

    with Settings(dataframe_validation=True):
        assert active_settings.dataframe_validation

        with Settings(dataframe_validation=False):
            assert not active_settings.dataframe_validation

        # After exiting inner context, outer setting should be restored
        assert active_settings.dataframe_validation

    # After exiting outer context, original setting should be restored
    assert active_settings.dataframe_validation == original_value


def test_exception_during_context_settings():
    """Exceptions raised inside a context are propagated and settings are restored."""
    original_value = active_settings.dataframe_validation

    class CustomError(Exception):
        """A custom exception for testing purposes."""

    # The custom exception is properly propagated
    with pytest.raises(CustomError, match="Test exception"):
        with Settings(dataframe_validation=not original_value):
            assert active_settings.dataframe_validation == (not original_value)
            raise CustomError("Test exception")

    # Settings are restored despite the exception
    assert active_settings.dataframe_validation == original_value


def test_setting_via_decorator(original_values, toggled_values):
    """Settings can be enabled by decorating callables."""

    @Settings(**toggled_values)
    def func():
        assert_attribute_values(active_settings, toggled_values)

    # The new settings are active during the function call
    func()

    # After exiting the function, the original settings are restored
    assert_attribute_values(active_settings, original_values)


def test_unknown_environment_variable(monkeypatch):
    """Unknown environment variables raise an error upon settings instantiation."""
    monkeypatch.setenv("BAYBE_UNKNOWN_SETTING", "True")
    with pytest.raises(RuntimeError, match="BAYBE_UNKNOWN_SETTING"):
        Settings()


@pytest.mark.parametrize("use_environment", [True, False], ids=["env", "no_env"])
@pytest.mark.parametrize(
    "restore_defaults", [True, False], ids=["restore", "no_restore"]
)
@pytest.mark.parametrize(
    "pass_explicit", [True, False], ids=["explicit", "no_explicit"]
)
def test_settings_initialization(
    monkeypatch,
    use_environment: bool,
    restore_defaults: bool,
    pass_explicit: bool,
):
    """The settings initialization can be configured via control flags."""
    # The different sources for the cache_directory setting
    cache_directory_original = active_settings.cache_directory
    cache_directory_env = Path("env")
    cache_directory_previous = Path("previous")
    cache_directory_explicit = Path("explicit")

    # Prepare environment
    env_key = "BAYBE_CACHE_DIRECTORY"
    assert env_key not in os.environ
    monkeypatch.setenv(env_key, str(cache_directory_env))

    # Prepare "previous" settings
    active_settings.cache_directory = cache_directory_previous

    # Create the "next" settings
    s = Settings(
        use_environment=use_environment,
        restore_defaults=restore_defaults,
        **{"cache_directory": cache_directory_explicit} if pass_explicit else {},
    )

    # The source used to initialize the setting value is controlled by the flags
    if pass_explicit:
        expected = cache_directory_explicit
    elif use_environment:
        expected = cache_directory_env
    elif restore_defaults:
        expected = cache_directory_original
    else:
        expected = cache_directory_previous
    assert_attribute_values(s, {"cache_directory": expected})
