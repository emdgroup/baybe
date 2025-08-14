"""Basic result classes for benchmarking."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

import cpuinfo
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
        cpu_info = cpuinfo.get_cpu_info()

        cpu_brand = cpu_info.get("brand_raw", "Unknown CPU")
        clock_advertised = cpu_info.get("hz_advertised_friendly", "Unknown Clock Speed")
        clock_max = cpu_info.get("hz_actual_friendly", "Unknown Max Clock Speed")
        layer_1_cache_size = str(
            cpu_info.get("l1_data_cache_size", "Unknown L1 Cache Size")
        )
        layer_2_cache_size = str(cpu_info.get("l2_cache_size", "Unknown L2 Cache Size"))
        layer_3_cache_size = str(cpu_info.get("l3_cache_size", "Unknown L3 Cache Size"))

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
            "cpu_brand": cpu_brand,
            "clock_advertised": clock_advertised,
            "clock_max": clock_max,
            "layer_1_cache_size": layer_1_cache_size,
            "layer_2_cache_size": layer_2_cache_size,
            "layer_3_cache_size": layer_3_cache_size,
        } | gpu_device_properties


converter.register_unstructure_hook(
    Result,
    make_dict_unstructure_fn(Result, converter, _cattrs_include_init_false=True),
)
