"""Preset configurations for Gaussian process surrogates."""

from __future__ import annotations

from enum import Enum


class GaussianProcessPreset(Enum):
    """Available Gaussian process surrogate presets."""

    BAYBE = "BAYBE"
    """The default BayBE settings of the Gaussian process surrogate class."""

    BOTORCH = "BOTORCH"
    """The BoTorch settings."""

    EDBO = "EDBO"
    """The EDBO settings."""

    EDBO_SMOOTHED = "EDBO_SMOOTHED"
    """A smoothed version of the EDBO settings."""
