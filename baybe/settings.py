"""BayBE settings."""

from __future__ import annotations

import os
import random
import tempfile
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from attrs import Attribute, Factory, define, field, fields
from attrs.validators import instance_of
from attrs.validators import optional as optional_v
from typing_extensions import Self

from baybe._optional.info import FPSAMPLE_INSTALLED, POLARS_INSTALLED
from baybe.exceptions import OptionalImportError
from baybe.utils.basic import AutoBool, classproperty
from baybe.utils.boolean import strtobool

if TYPE_CHECKING:
    from types import TracebackType

    import torch

# The temporary assignment to `None` is needed because the object is already referenced
# in the `Settings` class body
active_settings: Settings = None  # type: ignore[assignment]
"""The global settings instance controlling execution behavior."""

_MISSING_PACKAGE_ERROR_MESSAGE = (
    "The setting 'use_{package_name}' cannot be set to 'True' because '{package_name}' "
    "is not installed. Either install 'polars' or set 'use_{package_name}' "
    "to 'False'/'Auto'."
)

_ENV_VARS_WHITELIST = {
    "BAYBE_TEST_ENV",  # defines testing scope
}
"""The collection of whitelisted **additional** environment variables allowed."""


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


@define(frozen=True)
class _RandomState:
    """Container for the random states of all managed numeric libraries."""

    state_builtin = field(factory=random.getstate)
    """The state of the built-in random number generator."""

    state_numpy = field(factory=np.random.get_state)
    """The state of the Numpy random number generator."""

    state_torch = field()  # set by default method below (for lazy torch loading)
    """The state of the Torch random number generator."""

    @state_torch.default
    def _default_state_torch(self) -> Any:
        """Get the current Torch random state using a lazy import."""
        import torch

        return torch.get_rng_state()

    @classmethod
    def from_seed(cls, seed: int) -> Self:
        """Get the random state corresponding to a given seed value."""
        import torch

        # Remember the current state
        previous_state = cls()

        # Set the requested seed and extract the corresponding state
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        new_state = cls()

        # Restore the previous state
        previous_state.activate()

        return new_state

    def activate(self) -> None:
        """Activate the random state."""
        import torch

        random.setstate(self.state_builtin)
        np.random.set_state(self.state_numpy)
        torch.set_rng_state(self.state_torch)


def _activate_random_seed(
    self: Settings, attribute: Attribute, value: int | None
) -> None:
    """Activate the random seed, remembering the previous state."""
    if value is None:
        if self._previous_random_state is not None:
            self._previous_random_state.activate()
        return
    self._previous_random_state = _RandomState()
    _RandomState.from_seed(value).activate()


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

    _previous_random_state: _RandomState | None = field(init=False, default=None)
    """The previously set random state."""

    dataframe_validation: bool = field(default=True, converter=_to_bool)
    """Controls if dataframe content is validated against the recommendation context."""

    random_seed: int | None = field(
        default=None,
        validator=optional_v(instance_of(int)),
        on_setattr=_activate_random_seed,
    )
    """The used random seed."""

    use_polars: AutoBool = field(
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
    )
    """Controls if polars acceleration is to be used, if available."""

    use_fpsample: AutoBool = field(
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
    )
    """Controls if fpsample acceleration is to be used, if available."""

    use_numpy_single_precision: bool = field(default=False, validator=instance_of(bool))
    """Controls if numpy arrays are created with single or double precision."""

    use_torch_single_precision: bool = field(default=False, validator=instance_of(bool))
    """Controls if torch tensors are created with single or double precision."""

    parallelize_simulations: bool = field(default=True, validator=instance_of(bool))
    """Controls if simulation runs are parallelized in `xyzpy <https://xyzpy.readthedocs.io/en/latest/index.html>`_."""

    cache_directory: Path = field(
        converter=Path, default=Path(tempfile.gettempdir()) / ".baybe_cache"
    )
    """Controls which directory is used for caching."""

    def __attrs_pre_init__(self) -> None:
        env_vars = {name for name in os.environ if name.startswith("BAYBE_")}
        unknown = env_vars - (
            {f"BAYBE_{attr.name.upper()}" for attr in self.attributes}
            | _ENV_VARS_WHITELIST
        )
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

    @use_polars.validator
    def _validate_use_polars(self, _, value: AutoBool) -> None:
        if value is AutoBool.TRUE and not POLARS_INSTALLED:
            raise OptionalImportError(
                _MISSING_PACKAGE_ERROR_MESSAGE.format(package_name="polars")
            )

    @use_fpsample.validator
    def _validate_use_fpsample(self, _, value: AutoBool) -> None:
        if value is AutoBool.TRUE and not FPSAMPLE_INSTALLED:
            raise OptionalImportError(
                _MISSING_PACKAGE_ERROR_MESSAGE.format(package_name="fpsample")
            )

    @property
    def is_polars_enabled(self) -> bool:
        """Indicates if polars is enabled (i.e., installed and set to be used)."""
        return self.use_polars.evaluate(lambda: POLARS_INSTALLED)

    @property
    def is_fpsample_enabled(self) -> bool:
        """Indicates if fpsample is enabled (i.e., installed and set to be used)."""
        return self.use_fpsample.evaluate(lambda: FPSAMPLE_INSTALLED)

    @property
    def DTypeFloatNumpy(self) -> type[np.floating]:
        """The floating point data type used for numpy arrays."""
        return np.float32 if self.use_numpy_single_precision else np.float64

    @property
    def DTypeFloatTorch(self) -> torch.dtype:
        """The floating point data type used for torch tensors."""
        import torch

        return torch.float32 if self.use_torch_single_precision else torch.float64

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
