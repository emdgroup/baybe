"""Preset configurations for Gaussian process surrogates."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baybe.surrogates.base import Surrogate


class GaussianProcessPreset(Enum):
    """Available Gaussian process surrogate presets."""

    BAYBE = "BAYBE"
    """Recreates the default settings of the Gaussian process surrogate class."""

    BOTORCH_STMF = "BOTORCH_STMF"
    """Recreates the default settings of the BOTORCH SingleTaskMultiFidelityGP."""

    MFGP = "MFGP"
    """Recreates the default settings of the MFUCB approach with a default Matern
    kernel over design subspace and a full rank index kernel over fidelities."""


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
