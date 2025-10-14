"""BayBE settings."""

from __future__ import annotations

import gc
import os
import random
import tempfile
import warnings
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
from attrs import Attribute, Factory, define, field, fields
from attrs.converters import optional as optional_c
from attrs.setters import validate
from attrs.validators import in_, instance_of
from attrs.validators import optional as optional_v
from typing_extensions import Self

from baybe._optional.info import FPSAMPLE_INSTALLED, POLARS_INSTALLED
from baybe.exceptions import OptionalImportError
from baybe.utils.basic import classproperty
from baybe.utils.boolean import AutoBool, strtobool

if TYPE_CHECKING:
    from types import TracebackType

    import torch

    _TSeed = TypeVar("_TSeed", int, None)

# The temporary assignment to `None` is needed because the object is already referenced
# in the `Settings` class body
active_settings: Settings = None  # type: ignore[assignment]
"""The global settings instance controlling execution behavior."""

_MISSING_PACKAGE_ERROR_MESSAGE = (
    "The setting 'use_{package_name}' cannot be set to 'True' because '{package_name}' "
    "is not installed. Either install '{package_name}' or set 'use_{package_name}' "
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
        if fld.name in cls._internal_attributes:
            results.append(fld)
            continue

        # We use a factory here because the environment variables should be lookup up
        # at instantiation time, not at class definition time
        def make_default_factory(fld: Attribute) -> Any:
            # TODO: https://github.com/python-attrs/attrs/issues/1479
            name = fld.alias or fld.name

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
                    env_name = f"BAYBE_{name.upper()}"
                    value = os.getenv(env_name, default)
                    if fld.type == "bool":
                        value = _to_bool(value)
                    return value

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

    def activate(self) -> None:
        """Activate the random state."""
        import torch

        random.setstate(self.state_builtin)
        np.random.set_state(self.state_numpy)
        torch.set_rng_state(self.state_torch)

    @classmethod
    def activate_from_seed(cls, seed: int) -> Self:
        """Active the random state corresponding to a given seed."""
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        return cls()


def _on_set_random_seed(instance: Settings, __: Attribute, value: _TSeed) -> _TSeed:
    """Activate the given random seed on attribute change."""
    if id(instance) == Settings._global_settings_id and value is not None:
        _RandomState.activate_from_seed(value)
    return value


@define(kw_only=True, field_transformer=adjust_defaults)
class Settings(_SlottedContextDecorator):
    """BayBE settings."""

    # >>>>> Internal
    _global_settings_id: ClassVar[int]
    """The id of the global settings instance.

    Useful to identify if an action is performed on the global or a local instance."""

    _previous_random_state: _RandomState | None = field(init=False, default=None)
    """The previously set random state."""

    _previous_settings: Settings | None = field(default=None, init=False)
    """The previously active settings (used for context management)."""
    # <<<<< Internal

    # >>>>> Control flags
    _restore_defaults: bool = field(default=False, validator=instance_of(bool))
    """Controls if settings shall be restored to their default values."""

    _restore_environment: bool = field(default=False, validator=instance_of(bool))
    """Controls if environment variables shall be used to initialize settings."""
    # <<<<< Control flags

    # >>>>> Settings attributes
    cache_campaign_recommendations: bool = field(
        default=True, validator=instance_of(bool)
    )
    """Controls if campaigns cache their latest recommendation."""

    cache_directory: Path | None = field(
        default=Path(tempfile.gettempdir()) / ".baybe_cache", converter=optional_c(Path)
    )
    """The directory used for caching. Set to ``None`` to disable caching."""

    float_precision_numpy: int = field(
        default=64, converter=int, validator=in_((16, 32, 64))
    )
    """The floating point precision used for NumPy arrays."""

    float_precision_torch: int = field(
        default=64, converter=int, validator=in_((16, 32, 64))
    )
    """The floating point precision used for Torch tensors."""

    parallelize_simulations: bool = field(default=True, validator=instance_of(bool))
    """Controls if simulation runs in `xyzpy <https://xyzpy.readthedocs.io/en/latest/index.html>`_ are executed in parallel."""  # noqa: E501

    preprocess_dataframes: bool = field(default=True, validator=instance_of(bool))
    """Controls if dataframe content is validated and normalized before used."""

    random_seed: int | None = field(
        default=None,
        validator=optional_v(instance_of(int)),
        on_setattr=[validate, _on_set_random_seed],
    )
    """The used random seed."""

    _use_fpsample: AutoBool = field(
        alias="use_fpsample",
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
    )
    """Controls if `fpsample <https://github.com/leonardodalinky/fpsample>`_ acceleration is to be used, if available."""  # noqa: E501

    _use_polars_for_constraints: AutoBool = field(
        alias="use_polars_for_constraints",
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
    )
    """Controls if polars acceleration is to be used for constraints, if available."""
    # <<<<< Settings attributes

    def __attrs_pre_init__(self) -> None:
        # >>>>> Deprecation
        flds = fields(Settings)
        pairs: list[tuple[str, Attribute]] = [
            ("BAYBE_NUMPY_USE_SINGLE_PRECISION", flds.float_precision_numpy),
            ("BAYBE_TORCH_USE_SINGLE_PRECISION", flds.float_precision_torch),
            ("BAYBE_DEACTIVATE_POLARS", flds._use_polars_for_constraints),
            ("BAYBE_PARALLEL_SIMULATION_RUNS", flds.parallelize_simulations),
            ("BAYBE_CACHE_DIR", flds.cache_directory),
        ]
        for env_var, fld in pairs:
            if (val := os.environ.pop(env_var, None)) is not None:
                warnings.warn(
                    f"The environment variable '{env_var}' has "
                    f"been deprecated and support will be dropped in a future version. "
                    f"Please use 'BAYBE_{(fld.alias or fld.name).upper()}' instead. "
                    f"For now, we've automatically handled the translation for you.",
                    DeprecationWarning,
                )
                if env_var.endswith("SINGLE_PRECISION"):
                    new_value = "32" if _to_bool(val) else "64"
                elif env_var.endswith("POLARS"):
                    new_value = "false" if _to_bool(val) else "true"
                elif env_var.endswith("SIMULATION_RUNS"):
                    new_value = val
                elif env_var.endswith("CACHE_DIR"):
                    new_value = val
                os.environ[f"BAYBE_{(fld.alias or fld.name).upper()}"] = new_value
        # <<<<< Deprecation

        env_vars = {name for name in os.environ if name.startswith("BAYBE_")}
        unknown = env_vars - (
            {f"BAYBE_{attr.alias.upper()}" for attr in self._settings_attributes}
            | _ENV_VARS_WHITELIST
        )
        if unknown:
            raise RuntimeError(f"Unknown environment variables: {unknown}")

    def __enter__(self) -> Settings:
        self.activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.restore_previous()

    @_use_polars_for_constraints.validator
    def _validate_use_polars_for_constraints(self, _, value: AutoBool) -> None:
        if value is AutoBool.TRUE and not POLARS_INSTALLED:
            raise OptionalImportError(
                _MISSING_PACKAGE_ERROR_MESSAGE.format(package_name="polars")
            )

    @_use_fpsample.validator
    def _validate_use_fpsample(self, _, value: AutoBool) -> None:
        if value is AutoBool.TRUE and not FPSAMPLE_INSTALLED:
            raise OptionalImportError(
                _MISSING_PACKAGE_ERROR_MESSAGE.format(package_name="fpsample")
            )

    @property
    def use_polars_for_constraints(self) -> bool:
        """Indicates if Polars is enabled (i.e., installed and set to be used)."""
        return self._use_polars_for_constraints.evaluate(lambda: POLARS_INSTALLED)

    @property
    def use_fpsample(self) -> bool:
        """Indicates if `fpsample <https://github.com/leonardodalinky/fpsample>`_  is enabled (i.e., installed and set to be used)."""  # noqa: E501
        return self._use_fpsample.evaluate(lambda: FPSAMPLE_INSTALLED)

    @property
    def DTypeFloatNumpy(self) -> type[np.floating]:
        """The floating point data type used for NumPy arrays."""
        return getattr(np, f"float{self.float_precision_numpy}")

    @property
    def DTypeFloatTorch(self) -> torch.dtype:
        """The floating point data type used for Torch tensors."""
        import torch

        return getattr(torch, f"float{self.float_precision_torch}")

    @classproperty
    def _internal_attributes(cls) -> frozenset[str]:
        """The names of the internal attributes not representing settings."""  # noqa: D401
        # IMPROVE: This approach is not type-safe but the set is needed already at
        #   class definition time, which means we cannot use `attrs.fields` or similar.
        #   Perhaps `typing.Annotated` can be used, if there's an elegant way to
        #   resolve the stringified types coming from `__future__.annotations`?
        return frozenset(
            {
                "_previous_random_state",
                "_previous_settings",
                "_restore_defaults",
                "_restore_environment",
            }
        )

    @classproperty
    def _settings_attributes(cls) -> tuple[Attribute, ...]:
        """The attributes representing the available settings."""  # noqa: D401
        return tuple(
            fld
            for fld in fields(Settings)
            if fld.name not in Settings._internal_attributes
        )

    def activate(self) -> Settings:
        """Activate the settings globally."""
        self._previous_settings = deepcopy(active_settings)
        self.overwrite(active_settings)
        if self.random_seed is not None:
            _RandomState.activate_from_seed(self.random_seed)
        return self

    def restore_previous(self) -> None:
        """Restore the previous settings."""
        if self._previous_settings is None:
            raise RuntimeError(
                "The settings have not yet been activated, "
                "so there are no previous settings to restore."
            )
        self._previous_settings.overwrite(active_settings)
        self._previous_settings = None

    def overwrite(self, target: Settings) -> None:
        """Overwrite the settings of another :class:`Settings` object."""
        for fld in self._settings_attributes:
            setattr(target, fld.name, getattr(self, fld.name))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()


active_settings = Settings(restore_environment=True)
Settings._global_settings_id = id(active_settings)
"""The currently active global settings instance."""
