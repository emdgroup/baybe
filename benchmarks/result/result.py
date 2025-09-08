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

from benchmarks.definition.utils import RunMode
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

    runmode: RunMode = field(validator=instance_of(RunMode))
    """The mode which governed the benchmark settings."""

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
    """Technical details of the system on which the benchmark was executed."""

    @python_env.default
    def _default_python_env(self) -> dict[str, str]:
        installed_packages = importlib.metadata.distributions()
        return {dist.name: dist.version for dist in installed_packages}

    def _get_pytorch_device_info(self) -> dict[str, str]:
        """Get information about the PyTorch device in use."""
        info: dict[str, str] = {}

        used_module = torch.get_device_module()
        module_name = getattr(used_module, "__name__", str(used_module))
        info["pytorch_device_module"] = module_name

        backend_key = module_name.rsplit(".", 1)[-1].lower()

        if backend_key == "cuda":
            backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            name = torch.cuda.get_device_name(idx)
            info.update(
                {
                    "gpu_backend": backend,
                    "gpu_count": str(torch.cuda.device_count()),
                    "gpu_index": str(idx),
                    "gpu_name": str(name),
                    "gpu_total_memory": f"{props.total_memory // (1024**2)} MB",
                }
            )
            if backend == "CUDA":
                info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
                if torch.version.cuda:
                    info["cuda_version"] = str(torch.version.cuda)
            else:
                hip_ver = getattr(torch.version, "hip", None)
                if hip_ver:
                    info["hip_version"] = str(hip_ver)

        elif backend_key == "mps":
            is_avail = (
                torch.backends.mps.is_available()
                if hasattr(torch.backends, "mps")
                else False
            )
            is_built = (
                torch.backends.mps.is_built()
                if hasattr(torch.backends, "mps")
                else False
            )
            info.update(
                {
                    "gpu_backend": "MPS",
                    "mps_is_available": str(is_avail),
                    "mps_is_built": str(is_built),
                    "gpu_count": "1" if is_avail else "0",
                    "gpu_index": "0" if is_avail else "-1",
                    "gpu_name": "Apple MPS",
                }
            )
            if hasattr(torch.mps, "driver_allocated_memory"):
                info["mps_driver_allocated_memory"] = (
                    f"{torch.mps.driver_allocated_memory() // (1024**2)} MB"
                )
            if hasattr(torch.mps, "current_allocated_memory"):
                info["mps_current_allocated_memory"] = (
                    f"{torch.mps.current_allocated_memory() // (1024**2)} MB"
                )

        return info

    @benchmarking_system_details.default
    def _default_benchmarking_system_technical_details(self) -> dict[str, str]:
        """Get the technical details of the system where the benchmark was executed."""
        cpu_info = platform.processor()
        cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count(logical=True)
        cpu_frequency = psutil.cpu_freq()

        cpu_frequency_info: dict[str, str] = {}
        if cpu_frequency:
            cpu_frequency_info = {
                "cpu_frequency_current": str(cpu_frequency.current) + " MHz",
                "cpu_frequency_min": str(cpu_frequency.min) + " MHz",
                "cpu_frequency_max": str(cpu_frequency.max) + " MHz",
            }

        gpu_device_properties: dict[str, str] = self._get_pytorch_device_info()
        return (
            {
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
                "swap_memory_total": str(psutil.swap_memory().total // (1024**2))
                + " MB",
                "processor": cpu_info,
                "cpu_count_physical": str(cpu_count),
                "cpu_count_logical": str(logical_cpu_count),
            }
            | gpu_device_properties
            | cpu_frequency_info
        )


converter.register_unstructure_hook(
    Result,
    make_dict_unstructure_fn(Result, converter, _cattrs_include_init_false=True),
)
