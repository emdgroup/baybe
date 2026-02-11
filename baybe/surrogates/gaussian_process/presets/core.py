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

    if preset is GaussianProcessPreset.BAYBE:
        return GaussianProcessSurrogate()

    if preset is GaussianProcessPreset.EDBO:
        from baybe.surrogates.gaussian_process.presets.edbo import (
            EDBOKernelFactory as PresetKernelFactory,
        )
        from baybe.surrogates.gaussian_process.presets.edbo import (
            EDBOLikelihoodFactory as PresetLikelihoodFactory,
        )
        from baybe.surrogates.gaussian_process.presets.edbo import (
            EDBOMeanFactory as PresetMeanFactory,
        )

    elif preset is GaussianProcessPreset.EDBO_SMOOTHED:
        from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
            SmoothedEDBOKernelFactory as PresetKernelFactory,
        )
        from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
            SmoothedEDBOLikelihoodFactory as PresetLikelihoodFactory,
        )
        from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
            SmoothedEDBOMeanFactory as PresetMeanFactory,
        )

    else:
        assert_never(preset)

    return GaussianProcessSurrogate(
        PresetKernelFactory(), PresetMeanFactory(), PresetLikelihoodFactory()
    )
