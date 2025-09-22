"""BayBE settings."""

from __future__ import annotations

import os
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any

from attrs import Attribute, Factory, define, field, fields
from attrs.validators import instance_of

from baybe.utils.basic import classproperty
from baybe.utils.boolean import strtobool

if TYPE_CHECKING:
    from types import TracebackType

# The temporary assignment to `None` is needed because the object is already referenced
# in the `Settings` class body
settings: Settings = None  # type: ignore[assignment]
"""The global settings instance controlling execution behavior."""


class _SlottedContextDecorator:
    """Like :class:`contextlib.ContextDecorator` but with `__slots__`.

    The code has been copied from the Python standard library.
    """

    __slots__ = ()

    def _recreate_cm(self):
        return self

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)

        return inner


def _to_bool(value: Any) -> bool:
    """Convert Booleans and strings representing Booleans to actual Booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return strtobool(value)
    raise TypeError(f"Cannot convert value of type '{type(value)}' to Boolean.")


def load_environment(cls: type[Settings], fields: list[Attribute]) -> list[Attribute]:
    """Replace default values with environment variable lookups, if enabled."""
    results = []
    for fld in fields:
        if fld.name.startswith("_"):
            results.append(fld)
            continue

        # We use a factory here because the environment variables should be lookup up
        # at instantiation time, not at class definition time
        def _default(self: Settings) -> Any:
            if self._use_environment:
                env_name = f"BAYBE_{fld.name.upper()}"
                return os.getenv(env_name, fld.default)
            return fld.default

        results.append(fld.evolve(default=Factory(_default, takes_self=True)))
    return results


@define(kw_only=True, field_transformer=load_environment)
class Settings(_SlottedContextDecorator):
    """BayBE settings."""

    _previous_settings: Settings | None = field(default=None, init=False)
    """The previously applied settings (used for context management)."""

    _use_environment: bool = field(default=True, validator=instance_of(bool))
    """Controls if environment variables shall be used to initialize settings."""

    _apply_immediately: bool = field(default=False, validator=instance_of(bool))
    """Controls if settings are applied immediately upon instantiation."""

    dataframe_validation: bool = field(default=True, converter=_to_bool)
    """Controls if dataframe content is validated against the recommendation context."""

    def __attrs_post_init__(self):
        if settings is None:
            # If we arrive here, we are in the initialization of the global object
            # --> Nothing to do
            return

        if self._apply_immediately:
            self._apply()

    def __enter__(self) -> Settings:
        self._apply()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._restore_previous()

    @classproperty
    def attributes(cls) -> tuple[Attribute, ...]:
        """The available settings."""  # noqa: D401
        return tuple(fld for fld in fields(Settings) if not fld.name.startswith("_"))

    def _apply(self) -> None:
        """Apply the settings globally."""
        self._previous_settings = deepcopy(settings)
        self.overwrite(settings)

    def _restore_previous(self) -> None:
        """Restore the previous settings."""
        assert self._previous_settings is not None
        self._previous_settings.overwrite(settings)
        self._previous_settings = None

    def overwrite(self, target: Settings) -> None:
        """Overwrite the settings of another :class:`Settings` object."""
        for fld in self.attributes:
            setattr(target, fld.name, getattr(self, fld.name))


settings = Settings()
