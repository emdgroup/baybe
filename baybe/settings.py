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
from attrs import Attribute, Converter, Factory, cmp_using, define, field, fields
from attrs.setters import validate
from attrs.validators import instance_of
from attrs.validators import optional as optional_v
from typing_extensions import Self

from baybe._optional.info import FPSAMPLE_INSTALLED, POLARS_INSTALLED
from baybe.exceptions import NotAllowedError, OptionalImportError
from baybe.utils.basic import classproperty
from baybe.utils.boolean import AutoBool, to_bool

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    _TSeed = TypeVar("_TSeed", int, None)

_RANDOM_SEED_ATTRIBUTE_NAME = "random_seed"

# The temporary assignment to `None` is needed because the object is already referenced
# in the `Settings` class body
active_settings: Settings = None  # type: ignore[assignment]
"""The global settings instance controlling execution behavior."""

_MISSING_PACKAGE_ERROR_MESSAGE = (
    "The setting 'use_{package_name}' cannot be set to 'True' because '{package_name}' "
    "is not installed. Either install '{package_name}' or set 'use_{package_name}' "
    "to 'False'/'Auto'."
)


def _validate_whitelist_env_vars(vars: dict[str, str], /) -> None:
    """Validate the values of non-settings environment variables."""
    if (value := vars.pop("BAYBE_TEST_ENV", None)) is not None:
        if value not in {"CORETEST", "FULLTEST", "GPUTEST"}:
            raise ValueError(
                f"Allowed values for 'BAYBE_TEST_ENV' are "
                f"'CORETEST', 'FULLTEST', and 'GPUTEST'. Given: '{value}'"
            )
    if vars:
        raise RuntimeError(f"Unknown 'BAYBE_*' environment variables: {set(vars)}")


def _lazy_torch_equal(a: Tensor, b: Tensor, /) -> bool:
    """Equality check for tensors with lazy torch import."""
    import torch

    return torch.equal(a, b)


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


def adjust_defaults(cls: type[Settings], fields: list[Attribute]) -> list[Attribute]:
    """Replace default values with the appropriate source, controlled via flags."""
    results = []
    for fld in fields:
        if fld.name in (*cls._non_setting_attributes, _RANDOM_SEED_ATTRIBUTE_NAME):
            results.append(fld)
            continue

        # We use a factory here because the environment variables should be looked up
        # at instantiation time, not at class definition time
        def make_default_factory(fld: Attribute) -> Any:
            # TODO: https://github.com/python-attrs/attrs/issues/1479
            name = fld.alias or fld.name

            def get_default_value(self: Settings) -> Any:
                """Dynamically retrieve the default value for the field.

                Depending on the control flags, the value is retrieved either from the
                field specification itself, from the corresponding environment variable,
                or from the current global settings object.
                """
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
                        value = to_bool(value)
                    return value

                return default

            return Factory(get_default_value, takes_self=True)

        results.append(fld.evolve(default=make_default_factory(fld)))

    return results


@define(frozen=True)
class _RandomState:
    """Container for the random states of all managed numeric libraries."""

    state_python = field(init=False, factory=random.getstate)
    """The state of the Python random number generator."""

    state_numpy = field(
        init=False,
        factory=np.random.get_state,
        eq=cmp_using(
            eq=lambda s1, s2: all(np.array_equal(a, b) for a, b in zip(s1, s2))
        ),
    )
    """The state of the Numpy random number generator."""

    state_torch: Tensor = field(init=False, eq=cmp_using(eq=_lazy_torch_equal))
    """The state of the Torch random number generator."""
    # Note: initialized by attrs default method below (for lazy torch loading)

    @state_torch.default
    def _default_state_torch(self) -> Tensor:
        """Get the current Torch random state using a lazy import."""
        import torch

        return torch.get_rng_state()

    def activate(self) -> None:
        """Activate the random state."""
        import torch

        random.setstate(self.state_python)
        np.random.set_state(self.state_numpy)
        torch.set_rng_state(self.state_torch)

    @staticmethod
    def _activate_seed(seed: int) -> None:
        """Seed all random number generators."""
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def from_seed(cls, seed: int, *, activate: bool = False) -> Self:
        """Create a random state corresponding to a given seed."""
        if activate:
            cls._activate_seed(seed)
            return cls()

        backup = cls()
        cls._activate_seed(seed)
        state = cls()
        backup.activate()
        return state


def _on_set_random_seed(instance: Settings, __: Attribute, value: _TSeed) -> _TSeed:
    """Activate the given random seed on attribute change."""
    if id(instance) == Settings._global_settings_id and value is not None:
        _RandomState.from_seed(value, activate=True)

    return value


def _convert_cache_directory(
    value: str | Path | None, field: Attribute, /
) -> Path | None:
    """Attrs converter for the cache directory setting."""
    if value is None or value == "":
        return None
    try:
        return Path(value)
    except Exception as ex:
        raise type(ex)(
            f"Cannot set '{field.alias}' to '{value}'. "
            f"Expected 'None' or a path-like object."
        ) from ex


@define(kw_only=True, field_transformer=adjust_defaults)
class Settings(_SlottedContextDecorator):
    """BayBE settings."""

    # ----- Internal ----- #
    _global_settings_id: ClassVar[int]
    """The id of the global settings instance.

    Useful to identify if an action is performed on the global or a local instance."""

    _previous_settings: Settings | None = field(default=None, init=False)
    """The previously active settings (used for context management)."""

    _previous_random_state: _RandomState | None = field(default=None, init=False)
    """The previous random state (used for context management)."""

    # ----- Control flags ----- #
    _restore_defaults: bool = field(default=False, validator=instance_of(bool))
    """Controls if settings shall be restored to their default values."""

    _restore_environment: bool = field(default=False, validator=instance_of(bool))
    """Controls if environment variables shall be used to initialize settings."""

    # ----- Settings attributes ----- #
    cache_campaign_recommendations: bool = field(
        default=True, validator=instance_of(bool)
    )
    """Controls if :class:`~baybe.campaign.Campaign` objects cache their latest set of
    recommendations."""

    cache_directory: Path | None = field(
        default=Path(tempfile.gettempdir()) / ".baybe_cache",
        converter=Converter(_convert_cache_directory, takes_field=True),  # type: ignore[misc]
    )
    """The directory used for persistent caching on disk. Set to ``""`` or ``None`` to disable caching."""  # noqa: E501

    parallelize_simulation_runs: bool = field(default=True, validator=instance_of(bool))
    """Controls if simulation runs with `xyzpy <https://xyzpy.readthedocs.io/>`_ are executed in parallel."""  # noqa: E501

    preprocess_dataframes: bool = field(default=True, validator=instance_of(bool))
    """Controls if incoming user dataframes are preprocessed (i.e., dtype-converted and validated) before use."""  # noqa: E501

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
    """Controls if `polars <https://pola.rs/>`_ acceleration is to be used for constraints, if available."""  # noqa: E501

    use_single_precision_numpy: bool = field(default=False, validator=instance_of(bool))
    """Controls the floating point precision used for `numpy <https://numpy.org/>`_ arrays."""  # noqa: E501

    use_single_precision_torch: bool = field(default=False, validator=instance_of(bool))
    """Controls the floating point precision used for `torch <https://pytorch.org/>`_ tensors."""  # noqa: E501

    def __attrs_pre_init__(self) -> None:
        # >>>>> Deprecation
        flds = fields(Settings)
        pairs: list[tuple[str, Attribute]] = [
            ("BAYBE_NUMPY_USE_SINGLE_PRECISION", flds.use_single_precision_numpy),
            ("BAYBE_TORCH_USE_SINGLE_PRECISION", flds.use_single_precision_torch),
            ("BAYBE_DEACTIVATE_POLARS", flds._use_polars_for_constraints),
            ("BAYBE_PARALLEL_SIMULATION_RUNS", flds.parallelize_simulation_runs),
            ("BAYBE_CACHE_DIR", flds.cache_directory),
        ]
        for env_var, fld in pairs:
            if (value := os.environ.pop(env_var, None)) is not None:
                warnings.warn(
                    f"The environment variable '{env_var}' has "
                    f"been deprecated and support will be dropped in a future version. "
                    f"Please use 'BAYBE_{(fld.alias or fld.name).upper()}' instead. "
                    f"For now, we've automatically handled the translation for you.",
                    DeprecationWarning,
                )
                if env_var.endswith("POLARS"):
                    value = "false" if to_bool(value) else "true"
                elif env_var.endswith("SIMULATION_RUNS"):
                    value = "true" if to_bool(value) else "false"
                os.environ[f"BAYBE_{(fld.alias or fld.name).upper()}"] = value
        # <<<<< Deprecation

        known_env_vars = {
            f"BAYBE_{attr.alias.upper()}" for attr in self._settings_attributes
        }
        _validate_whitelist_env_vars(
            {
                k: v
                for k, v in os.environ.items()
                if k.startswith("BAYBE_") and k not in known_env_vars
            }
        )

    def __enter__(self) -> Settings:
        self.activate()
        return self

    def __exit__(self, *args) -> None:
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
        """Indicates if ``polars`` is enabled (i.e., installed and set to be used)."""
        return self._use_polars_for_constraints.evaluate(lambda: POLARS_INSTALLED)

    @use_polars_for_constraints.setter
    def use_polars_for_constraints(self, value: AutoBool | bool, /) -> None:
        # Note: uses attrs converter
        self._use_polars_for_constraints = value  # type: ignore[assignment]

    @property
    def use_fpsample(self) -> bool:
        """Indicates if ``fpsample`` is enabled (i.e., installed and set to be used)."""  # noqa: E501
        return self._use_fpsample.evaluate(lambda: FPSAMPLE_INSTALLED)

    @use_fpsample.setter
    def use_fpsample(self, value: AutoBool | bool, /) -> None:
        # Note: uses attrs converter
        self._use_fpsample = value  # type: ignore[assignment]

    @property
    def DTypeFloatNumpy(self) -> type[np.floating]:
        """The floating point precision used for ``numpy`` arrays."""
        return np.float32 if self.use_single_precision_numpy else np.float64

    @property
    def DTypeFloatTorch(self) -> torch.dtype:
        """The floating point precision used for ``torch`` tensors."""
        import torch

        return torch.float32 if self.use_single_precision_torch else torch.float64

    @classproperty
    def _non_setting_attributes(cls) -> frozenset[str]:
        """The names of attributes that do not represent user-facing settings."""  # noqa: D401
        # IMPROVE: This approach is not type-safe but the set is needed already at
        #   class definition time, which means we cannot use `attrs.fields` or similar.
        #   Perhaps `typing.Annotated` can be used, if there's an elegant way to
        #   resolve the stringified types coming from `__future__.annotations`?
        return frozenset(
            {
                "_previous_settings",
                "_previous_random_state",
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
            if fld.name not in Settings._non_setting_attributes
        )

    def activate(self) -> Settings:
        """Activate the settings globally."""
        if id(self) == Settings._global_settings_id:
            raise NotAllowedError(
                f"Calling '{self.activate.__name__}' on the global settings "
                f"object is not allowed since it is always active."
            )

        # Store the previous state only if it's actually required for settings
        # restoration later on (see `restore_previous` method)
        if self.random_seed is not None:
            self._previous_random_state = _RandomState()

        self._previous_settings = deepcopy(active_settings)
        self.overwrite(active_settings)

        return self

    def restore_previous(self) -> None:
        """Restore the previous settings."""
        if id(self) == Settings._global_settings_id:
            raise NotAllowedError(
                f"Calling '{self.restore_previous.__name__}' on the global settings "
                f"object is not supported."
            )

        if self._previous_settings is None:
            raise RuntimeError(
                "The settings have not yet been activated, "
                "so there are no previous settings to restore."
            )

        # When restoring, we do not want to re-sync the random state back to
        # the seed value of the previous setting, since the random state has
        # potentially progressed in the meantime ...
        self._previous_settings.overwrite(active_settings, keep_random_state=True)

        # ... Instead, we restore the random state from setting activation time, but
        # only when randomness control was actually part of the settings configurations
        # and the state was altered in the first place.
        if self.random_seed is not None:
            self._previous_random_state.activate()
            self._previous_random_state = None

        # Clear backup attribute
        self._previous_settings = None

    def overwrite(self, target: Settings, keep_random_state: bool = False) -> None:
        """Overwrite the settings of another :class:`Settings` object."""
        if keep_random_state:
            state = _RandomState()

        for fld in self._settings_attributes:
            setattr(target, fld.name, getattr(self, fld.name))

        if keep_random_state:
            state.activate()


# Since there is critical code hardcoded against the attribute name, we
# ensure that the attribute exists as a sanity check (in case of future name edits)
assert _RANDOM_SEED_ATTRIBUTE_NAME in (fld.name for fld in fields(Settings))

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()


active_settings = Settings(restore_environment=True)
"""The currently active global settings instance."""

# Set the global settings id for later reference
Settings._global_settings_id = id(active_settings)

# Special handling of the random seed:
# The automatic adoption of seed values from the environment or the active settings
# object as default value for new settings objects is skipped in the class logic to
# enable proper progression of random states. However, we still want that a given
# environmental seed populates the active settings object (and *only* that object) upon
# session start, so we manually set it here.
if (
    _seed := os.environ.get(f"BAYBE_{_RANDOM_SEED_ATTRIBUTE_NAME.upper()}", None)
) is not None:
    active_settings.random_seed = int(_seed)
