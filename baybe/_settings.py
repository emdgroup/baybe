"""BayBE settings."""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING

from attrs import Attribute, define, field, fields
from attrs.validators import instance_of

from baybe.utils.basic import classproperty

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


@define(kw_only=True)
class Settings(_SlottedContextDecorator):
    """BayBE settings."""

    _previous_settings: Settings | None = field(default=None, init=False)
    """The previously applied settings (used for context management)."""

    dataframe_validation: bool = field(default=True, validator=instance_of(bool))
    """Controls if dataframe content is validated against the recommendation context."""

    def __attrs_post_init__(self):
        if settings is None:
            # If we arrive here, we are in the initialization of the global object
            # --> Nothing to do
            return

        # Store the global settings at the time of instantiation for later rollback
        self._previous_settings = deepcopy(settings)

        # Each new instance immediately applies its settings globally
        self.overwrite(settings)

    def __enter__(self) -> Settings:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._previous_settings is not None
        self._previous_settings.overwrite(settings)

    @classproperty
    def attributes(cls) -> tuple[Attribute, ...]:
        """The available settings."""  # noqa: D401
        return tuple(fld for fld in fields(Settings) if not fld.name.startswith("_"))

    def overwrite(self, target: Settings) -> None:
        """Overwrite the settings of another :class:`Settings` object."""
        for fld in self.attributes:
            setattr(target, fld.name, getattr(self, fld.name))


settings = Settings()
