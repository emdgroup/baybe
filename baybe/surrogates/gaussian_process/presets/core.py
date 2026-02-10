"""Preset configurations for Gaussian process surrogates."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, assert_never

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
    from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory
    from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
        SmoothedEDBOKernelFactory,
    )

    # TODO: Pass mean and likelihood settings once possible
    if preset is GaussianProcessPreset.BAYBE:
        return GaussianProcessSurrogate()
    if preset is GaussianProcessPreset.EDBO:
        return GaussianProcessSurrogate(kernel_or_factory=EDBOKernelFactory())
    if preset is GaussianProcessPreset.EDBO_SMOOTHED:
        return GaussianProcessSurrogate(kernel_or_factory=SmoothedEDBOKernelFactory())

    assert_never(preset)
