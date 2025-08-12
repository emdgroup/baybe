"""Basic benchmark configuration."""

import logging
import sys
import time
from abc import ABC
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from attr import Attribute
from attr.validators import optional
from attrs import define, field
from attrs.validators import instance_of
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

    This collects all modes that must be defined for all runmode-dependent attributes of
    any settings instance.
    """

    STANDARD = "standard"
    RUNTHROUGH = "runthrough"


@define(kw_only=True)
class BenchmarkSettings(ABC, BenchmarkSerialization):
    """The basic benchmark configuration."""

    random_seed: int = field(validator=instance_of(int), default=1337)
    """The used random seed."""

    runmode: str | None = field(
        validator=optional(instance_of(str)),
        default=None,
        converter=lambda x: RunMode.STANDARD.value if x is None else x,
    )
    """Mode to run benchmark in.
    Can be used to adjust the selection of setting parameters.
    """


def make_runmode_attr_validator(
    value_type: type,
) -> Callable[[BenchmarkSettings, Attribute, dict[str, Any]], None]:
    """Maker validator for attributes that depend on the runmode.

    Make sure that all required runmodes are defined
    and that values are of the desired type.
    """

    def runmode_attr_validator(
        instance: BenchmarkSettings, attribute: Attribute, value: dict[str, Any]
    ) -> None:
        # Check type
        if (
            not isinstance(value, dict)
            or not all(isinstance(key, str) for key in value.keys())
            or not all(isinstance(val, value_type) for val in value.values())
        ):
            raise TypeError(
                f"{str(attribute.name)} mut be of type Dict[str,{value_type.__name__}"
            )

        # Check obligatory runmodes
        missing_runmodes = []
        for rm in RunMode:
            runmode = rm.value
            if runmode not in value:
                missing_runmodes.append(runmode)
        if len(missing_runmodes) > 0:
            raise ValueError(
                "Missing settings for runmodes: "
                + ", ".join([rm for rm in missing_runmodes])
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

    def __call__(self, runmode: str | None = None) -> Result:
        """Execute the benchmark and return the result."""
        self.settings.runmode = runmode

        start_datetime = datetime.now(timezone.utc)

        logger.info(
            "=" * 80
            + f"\nRunning benchmark '{self.name}' with "
            + f"random seed {self.settings.random_seed} "
            + f"in runmode {self.settings.runmode}.\n"
        )

        start_sec = time.perf_counter()
        result = self.function(self.settings)
        stop_sec = time.perf_counter()

        duration = timedelta(seconds=stop_sec - start_sec)

        logger.info(
            f"\nFinished benchmark '{self.name}' after {duration} "
            + f"with random seed {self.settings.random_seed}.\n"
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
