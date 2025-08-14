"""Basic result classes for benchmarking."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

import psutil
import torch
from attrs import define, field
from attrs.validators import deep_mapping, instance_of
from cattrs.gen import make_dict_unstructure_fn
from pandas import DataFrame

from benchmarks.result import ResultMetadata
from benchmarks.serialization import BenchmarkSerialization, converter


@define(frozen=True)
class Result(BenchmarkSerialization):
    """A single benchmarking result."""

    benchmark_identifier: str = field(validator=instance_of(str))
    """The identifier of the benchmark that produced the result."""

    data: DataFrame = field(validator=instance_of(DataFrame))
    """The result of the benchmarked callable."""

    metadata: ResultMetadata = field(validator=instance_of(ResultMetadata))
    """The metadata associated with the benchmark result."""

    python_env: dict[str, str] = field(
        init=False,
        validator=deep_mapping(instance_of(str), instance_of(str), instance_of(dict)),
    )
    """The Python environment in which the benchmark was executed."""

    python_version: str = field(
        init=False, default=sys.version, validator=instance_of(str)
    )
    """The Python version with which the benchmark was executed."""

    benchmarking_system_details: dict[str, str] = field(
        init=False,
        validator=deep_mapping(instance_of(str), instance_of(str), instance_of(dict)),
    )
    """Technical details of the system where the benchmarking data was created."""

    @python_env.default
    def _default_python_env(self) -> dict[str, str]:
        installed_packages = importlib.metadata.distributions()
        return {dist.name: dist.version for dist in installed_packages}

    @benchmarking_system_details.default
    def _default_benchmarking_system_technical_details(self) -> dict[str, str]:
        """Get the technical details of the system where the benchmark was executed."""
        cpu_info = platform.processor()
        cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count(logical=True)
        cpu_frequency = psutil.cpu_freq()

        gpu_device_properties: dict[str, str] = {}
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_info = torch.cuda.get_device_properties()
            gpu_device_properties = {
                "gpu_name": gpu_name,
                "gpu_total_memory": str(gpu_info.total_memory // (1024**2)) + " MB",
                "gpu_compute_capability": str(gpu_info.major)
                + "."
                + str(gpu_info.minor),
            }

        return {
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "arch": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "RAM": str(psutil.virtual_memory().total // (1024**2)) + " MB",
            "version": platform.version(),
            "node": platform.node(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "python_revision": platform.python_revision(),
            "linkage_format": platform.architecture()[1],
            "cpu_count": str(psutil.cpu_count(logical=False)),
            "cpu_logical_count": str(psutil.cpu_count(logical=True)),
            "swap_memory_total": str(psutil.swap_memory().total // (1024**2)) + " MB",
            "processor": cpu_info,
            "cpu_count_physical": str(cpu_count),
            "cpu_count_logical": str(logical_cpu_count),
            "cpu_frequency_current": str(cpu_frequency.current) + " MHz",
            "cpu_frequency_min": str(cpu_frequency.min) + " MHz",
            "cpu_frequency_max": str(cpu_frequency.max) + " MHz",
        } | gpu_device_properties


converter.register_unstructure_hook(
    Result,
    make_dict_unstructure_fn(Result, converter, _cattrs_include_init_false=True),
)
