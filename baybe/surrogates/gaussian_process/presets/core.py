"""Preset configurations for Gaussian process surrogates."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate


class GaussianProcessPreset(Enum):
    """Available Gaussian process surrogate presets."""

    BAYBE = "BAYBE"
    """Recreates the default settings of the Gaussian process surrogate class."""


def make_gp_from_preset(preset: GaussianProcessPreset) -> GaussianProcessSurrogate:
    """Create a :class:`GaussianProcessSurrogate` from a :class:`GaussianProcessPreset."""  # noqa: E501
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate

    if preset is GaussianProcessPreset.BAYBE:
        return GaussianProcessSurrogate()

    raise ValueError(
        f"Unknown '{GaussianProcessPreset.__name__}' with name '{preset.name}'."
    )
