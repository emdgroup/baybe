"""BayBE settings."""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from attrs import Attribute, Factory, define, field, fields
from attrs.validators import instance_of

from baybe.utils.basic import classproperty
from baybe.utils.boolean import strtobool

if TYPE_CHECKING:
    from types import TracebackType

# The temporary assignment to `None` is needed because the object is already referenced
# in the `Settings` class body
active_settings: Settings = None  # type: ignore[assignment]
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


def adjust_defaults(cls: type[Settings], fields: list[Attribute]) -> list[Attribute]:
    """Replace default values with the appropriate source, controlled via flags."""
    results = []
    for fld in fields:
        if fld.name.startswith("_"):
            results.append(fld)
            continue

        # We use a factory here because the environment variables should be lookup up
        # at instantiation time, not at class definition time
        def make_default_factory(fld: Attribute) -> Any:
            def _(self: Settings) -> Any:
                if self._restore_defaults:
                    default = fld.default
                else:
                    # Here, the current global settings value is used as default, to
                    # enable updating settings one attribute at a time (the fallback to
                    # the default happens when the global settings object is itself
                    # being created)
                    default = getattr(active_settings, fld.name, fld.default)

                if self._restore_environment:
                    # If enabled, the environment values take precedence for the default
                    env_name = f"BAYBE_{fld.name.upper()}"
                    return os.getenv(env_name, default)

                return default

            return Factory(_, takes_self=True)

        results.append(fld.evolve(default=make_default_factory(fld)))
    return results


@define(kw_only=True, field_transformer=adjust_defaults)
class Settings(_SlottedContextDecorator):
    """BayBE settings."""

    _previous_settings: Settings | None = field(default=None, init=False)
    """The previously active settings (used for context management)."""

    _restore_defaults: bool = field(default=False, validator=instance_of(bool))
    """Controls if settings shall be restored to their default values."""

    _restore_environment: bool = field(default=False, validator=instance_of(bool))
    """Controls if environment variables shall be used to initialize settings."""

    _activate_immediately: bool = field(default=False, validator=instance_of(bool))
    """Controls if settings are activated immediately upon instantiation."""

    dataframe_validation: bool = field(default=True, converter=_to_bool)
    """Controls if dataframe content is validated against the recommendation context."""

    use_polars: bool = field(default=False, converter=_to_bool)
    """Controls if polars acceleration is to be used, if available."""

    cache_directory: Path = field(
        converter=Path, default=Path(tempfile.gettempdir()) / ".baybe_cache"
    )
    """Controls which directory is used for caching."""

    def __attrs_pre_init__(self) -> None:
        env_vars = {name for name in os.environ if name.startswith("BAYBE_")}
        unknown = env_vars - {f"BAYBE_{attr.name.upper()}" for attr in self.attributes}
        if unknown:
            raise RuntimeError(f"Unknown environment variables: {unknown}")

    def __attrs_post_init__(self) -> None:
        if active_settings is None:
            # If we arrive here, we are in the initialization of the global object
            # --> Nothing to do
            return

        if self._activate_immediately:
            self.activate()

    def __enter__(self) -> Settings:
        self.activate()
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

    def activate(self) -> None:
        """Activate the settings globally."""
        self._previous_settings = deepcopy(active_settings)
        self.overwrite(active_settings)

    def _restore_previous(self) -> None:
        """Restore the previous settings."""
        assert self._previous_settings is not None
        self._previous_settings.overwrite(active_settings)
        self._previous_settings = None

    def overwrite(self, target: Settings) -> None:
        """Overwrite the settings of another :class:`Settings` object."""
        for fld in self.attributes:
            setattr(target, fld.name, getattr(self, fld.name))


active_settings = Settings(restore_environment=True)
