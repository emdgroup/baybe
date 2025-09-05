"""Basic benchmark configuration."""

import logging
import sys
import time
from abc import ABC
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

import attrs
from attrs import Attribute, define, field
from attrs.validators import deep_mapping, instance_of
from cattrs import override
from cattrs.gen import make_dict_unstructure_fn
from pandas import DataFrame

from benchmarks.result import Result, ResultMetadata
from benchmarks.serialization import BenchmarkSerialization, converter

logger = logging.getLogger(__name__)

stdout = logging.StreamHandler(stream=sys.stdout)

logger.addHandler(stdout)
logger.setLevel(logging.INFO)


class RunMode(Enum):
    """Enum for different run modes of benchmarks.

    This collects the different run modes that can be used to adjust the selection of
    setting parameters for benchmarks. The purpose is to allow for different
    configurations or behaviors of benchmarks based on predefined modes which encode
    settings where for example more or less DoE-iterations are necessary.
    """

    STANDARD = "STANDARD"
    """Default run mode for benchmarks with performance relevant settings."""

    SMOKETEST = "SMOKETEST"
    """Minimal run mode for verifying that the benchmarks are executable."""

    @classmethod
    def from_string(cls, value: str):
        """Convert a string to a RunMode, case-insensitively."""
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid run mode: {value}")


@define(kw_only=True, frozen=True)
class BenchmarkSettings(ABC, BenchmarkSerialization):
    """The basic benchmark configuration."""

    random_seed: int = field(validator=instance_of(int), default=1337)
    """The used random seed."""

    runmode: RunMode = field(
        default=RunMode.STANDARD,
        converter=RunMode,
        validator=instance_of(RunMode),
    )
    """Mode to run benchmark in.
    Can be used to adjust the selection of setting parameters.
    """


MANDATORY_RUNMODES = frozenset({RunMode.STANDARD, RunMode.SMOKETEST})
"""Mandatory runmodes that must be defined in all benchmark settings."""


def make_runmode_attr_validator(
    value_type: type, /
) -> Callable[[BenchmarkSettings, Attribute, dict[RunMode, Any]], None]:
    """Make validator for attributes that depend on the runmode.

    Make sure that all required runmodes are defined
    and that values are of the desired type.
    """

    def runmode_attr_validator(
        instance: BenchmarkSettings, attribute: Attribute, value: dict[RunMode, Any]
    ) -> None:
        # Check type
        VALID_DICT = deep_mapping(
            key_validator=instance_of(RunMode),
            value_validator=instance_of(value_type),
            mapping_validator=instance_of(dict),
        )
        VALID_DICT(instance, attribute, value)

        missing_runmodes = {rm.value for rm in MANDATORY_RUNMODES if rm not in value}

        if missing_runmodes:
            raise ValueError(
                f"Runmode(s) {missing_runmodes} must be defined in {attribute.name}."
            )

    return runmode_attr_validator


BenchmarkSettingsType = TypeVar("BenchmarkSettingsType", bound=BenchmarkSettings)


@define(frozen=True)
class Benchmark(Generic[BenchmarkSettingsType], BenchmarkSerialization):
    """The base class for all benchmark definitions."""

    function: Callable[[BenchmarkSettingsType], DataFrame] = field()
    """The callable containing the benchmarking logic."""

    settings: BenchmarkSettingsType = field()
    """The benchmark configuration."""

    @function.validator
    def _validate_function(self, _, function) -> None:
        if function.__doc__ is None:
            raise ValueError("The benchmark function must have a docstring.")

    @property
    def name(self) -> str:
        """The name of the benchmark function."""
        return self.function.__name__

    @property
    def description(self) -> str:
        """The description of the benchmark function."""
        assert self.function.__doc__ is not None
        return self.function.__doc__

    def __call__(self, runmode: RunMode | None = None) -> Result:
        """Execute the benchmark and return the result.

        Args:
            runmode: The run mode to use for the benchmark.
                If not specified, the runmode previously set in the
                settings will be used.

        Returns:
            The result of the benchmark execution.
        """
        used_settings = (
            attrs.evolve(self.settings, runmode=runmode) if runmode else self.settings
        )

        start_datetime = datetime.now(timezone.utc)

        logger.info(
            "=" * 80
            + f"\nRunning benchmark '{self.name}' with "
            + f"random seed {used_settings.random_seed} "
            + f"in runmode {used_settings.runmode}.\n"
        )

        start_sec = time.perf_counter()
        result = self.function(used_settings)
        stop_sec = time.perf_counter()

        duration = timedelta(seconds=stop_sec - start_sec)

        logger.info(
            f"\nFinished benchmark '{self.name}' after {duration} "
            + f"with random seed {used_settings.random_seed}.\n"
            + "=" * 80
        )

        metadata = ResultMetadata(
            start_datetime=start_datetime,
            duration=duration,
        )

        return Result(self.name, result, metadata)


@converter.register_unstructure_hook
def unstructure_benchmark(benchmark: Benchmark) -> dict:
    """Unstructure a benchmark instance."""
    fn = make_dict_unstructure_fn(
        type(benchmark), converter, function=override(omit=True)
    )
    return {
        "name": benchmark.name,
        "description": benchmark.description,
        **fn(benchmark),
    }
