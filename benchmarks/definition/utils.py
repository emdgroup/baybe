"""File containing utility funcionts and enums for benchmark definitions."""

from enum import Enum


class RunMode(Enum):
    """Enum for different run modes of benchmarks.

    This collects the different run modes that can be used to adjust the selection of
    setting parameters for benchmarks. The purpose is to allow for different
    configurations or behaviors of benchmarks based on predefined modes which encode
    settings where for example more or less DoE-iterations are necessary.
    """

    DEFAULT = "DEFAULT"
    """Default run mode for benchmarks with performance relevant settings."""

    SMOKETEST = "SMOKETEST"
    """Minimal run mode for verifying that the benchmarks are executable."""

    @classmethod
    def from_string(cls, value: str, /) -> "RunMode":
        """Convert a string to a RunMode, case-insensitively."""
        try:
            return cls[value.upper()]
        except KeyError as ex:
            raise ValueError(f"Invalid run mode: {value}") from ex
