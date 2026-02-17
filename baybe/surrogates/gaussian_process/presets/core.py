"""Preset configurations for Gaussian process surrogates."""

from __future__ import annotations

import importlib
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate


class GaussianProcessPreset(Enum):
    """Available Gaussian process surrogate presets."""

    BAYBE = "BAYBE"
    """The default BayBE settings of the Gaussian process surrogate class."""

    EDBO = "EDBO"
    """The EDBO settings."""

    EDBO_SMOOTHED = "EDBO_SMOOTHED"
    """A smoothed version of the EDBO settings."""


def make_gp_from_preset(preset: GaussianProcessPreset) -> GaussianProcessSurrogate:
    """Create a :class:`GaussianProcessSurrogate` from a :class:`GaussianProcessPreset."""  # noqa: E501
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate

    if preset is GaussianProcessPreset.BAYBE:
        return GaussianProcessSurrogate()

    module_name = f"baybe.surrogates.gaussian_process.presets.{preset.value.lower()}"
    module = importlib.import_module(module_name)

    PresetKernelFactory = getattr(module, "PresetKernelFactory")
    PresetMeanFactory = getattr(module, "PresetMeanFactory")
    PresetLikelihoodFactory = getattr(module, "PresetLikelihoodFactory")

    return GaussianProcessSurrogate(
        PresetKernelFactory(), PresetMeanFactory(), PresetLikelihoodFactory()
    )
