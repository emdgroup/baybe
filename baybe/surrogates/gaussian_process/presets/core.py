"""Preset configurations for Gaussian process surrogates."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baybe.surrogates.base import Surrogate

from surrogates.gaussian_process.core import GaussianProcessSurrogate


class GaussianProcessPreset(Enum):
    """Available Gaussian process surrogate presets."""

    BAYBE = "BAYBE"
    """The default BayBE settings of the Gaussian process surrogate class."""

    EDBO = "EDBO"
    """The EDBO settings."""

    EDBO_SMOOTHED = "EDBO_SMOOTHED"
    """A smoothed version of the EDBO settings."""

    BOTORCH_STMF = "BOTORCH_STMF"
    """Recreates the default settings of the BOTORCH SingleTaskMultiFidelityGP."""


def make_gp_from_preset(preset: GaussianProcessPreset) -> Surrogate:
    """Create a :class:`GaussianProcessSurrogate` from a :class:`GaussianProcessPreset."""  # noqa: E501
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
    from baybe.surrogates.gaussian_process.multi_fidelity import (
        GaussianProcessSurrogateSTMF,
        MultiFidelityGaussianProcessSurrogate,
    )

    if preset is GaussianProcessPreset.BAYBE:
        return GaussianProcessSurrogate()

    if preset is GaussianProcessPreset.MFGP:
        return MultiFidelityGaussianProcessSurrogate()

    if preset is GaussianProcessPreset.BOTORCH_STMF:
        return GaussianProcessSurrogateSTMF()

    raise ValueError(
        f"Unknown '{GaussianProcessPreset.__name__}' with name '{preset.name}'."
    )
