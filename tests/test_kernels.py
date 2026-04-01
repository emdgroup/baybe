"""Kernel tests."""

from typing import Any

import numpy as np
from attrs import asdict, has
from hypothesis import given

from baybe.kernels.base import BasicKernel, Kernel
from baybe.kernels.basic import IndexKernel
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from tests.hypothesis_strategies.kernels import kernels

# TODO: Consider deprecating these attribute names to avoid inconsistencies
_RENAME_DICT: dict[str, str] = {
    "base_kernels": "kernels",
}
"""Dictionary for resolving name differences between BayBE and GPyTorch attributes."""


def _collect_parameter_names(kernel: Kernel) -> set[str]:
    """Collect all parameter names involved in a kernel structure.

    Args:
        kernel: A BayBE kernel (basic or composite).

    Returns:
        A set of all parameter names found in the kernel structure.
    """
    parameter_names = set()

    # If it's a BasicKernel, add its parameter_names
    if isinstance(kernel, BasicKernel) and kernel.parameter_names is not None:
        parameter_names.update(kernel.parameter_names)

    # Recursively collect from composite kernels
    kernel_dict = asdict(kernel, recurse=False)
    for value in kernel_dict.values():
        # Handle single nested kernel (e.g., ScaleKernel.base_kernel)
        if isinstance(value, Kernel):
            parameter_names.update(_collect_parameter_names(value))
        # Handle tuple of kernels (e.g., AdditiveKernel.base_kernels)
        elif isinstance(value, tuple) and all(isinstance(k, Kernel) for k in value):
            for k in value:
                parameter_names.update(_collect_parameter_names(k))

    return parameter_names


def validate_gpytorch_kernel_components(  # noqa: DOC501
    obj: Any, mapped: Any, searchspace: SearchSpace
) -> None:
    """Validate that all kernel components are correctly translated to GPyTorch.

    Args:
        obj: An object occurring as part of a BayBE kernel.
        mapped: The corresponding object in the translated GPyTorch kernel.
        searchspace: The search space used for the translation.
    """
    # Assert that the kernel kwargs are correctly mapped
    if isinstance(obj, BasicKernel):
        active_dims, ard_num_dims = obj._get_dimensions(searchspace)

        assert mapped.ard_num_dims == ard_num_dims
        assert active_dims == (
            tuple(mapped.active_dims.tolist())
            if mapped.active_dims is not None
            else None
        )

    # Compare attribute by attribute
    for name, component in asdict(obj, recurse=False).items():
        # If the BayBE component is `None`, the GPyTorch component might not even
        # exist, so we skip
        if component is None:
            continue

        # Skip BayBE-only attributes that have no GPyTorch counterpart
        if isinstance(obj, Kernel) and name in obj._whitelisted_attributes:
            continue

        # Resolve attribute naming differences
        mapped_name = _RENAME_DICT.get(name, name)

        # If the attribute does not exist in the GPyTorch version, it must have some
        # special handling on the GPyTorch side ...
        if (mapped_component := getattr(mapped, mapped_name, None)) is None:
            # The number of tasks is reflected by the the constructed covariance matrix
            if isinstance(obj, IndexKernel) and name == "num_tasks":
                assert mapped.covar_factor.shape[-2] == component
                continue

            # The rank is reflected by the the constructed covariance matrix
            elif isinstance(obj, IndexKernel) and name == "rank":
                assert mapped.covar_factor.shape[-1] == component
                continue

            # Initial values are directly applied. Because setting initial values
            # involves going through constraint transformations on GPyTorch side (i.e.,
            # difference between `<attr>` and `raw_<attr>`), the numerical values will
            # not be exact, so we check only for approximate matches.
            elif name.endswith("_initial_value"):
                assert np.allclose(
                    component,
                    getattr(mapped, name.removesuffix("_initial_value"))
                    .detach()
                    .numpy(),
                )
                continue

            raise AssertionError(f"Kernel component not correctly mapped: {name}")

        # If the component is itself another attrs object, recurse
        elif has(component):
            validate_gpytorch_kernel_components(
                component, mapped_component, searchspace
            )

        # Same for collections of BayBE objects (coming from composite kernels)
        elif isinstance(component, tuple) and all(has(c) for c in component):
            for c, m in zip(component, mapped_component):
                validate_gpytorch_kernel_components(c, m, searchspace)

        # On the lowest component level, simply check for equality
        else:
            assert component == mapped_component


@given(kernels())
def test_kernel_assembly(kernel: Kernel):
    """Turning a BayBE kernel into a GPyTorch kernel raises no errors and all its
    components are translated correctly."""  # noqa

    # Create a search space containing parameters referenced by the kernel
    parameter_names = _collect_parameter_names(kernel)
    if not parameter_names:
        parameter_names = ["x"]
    searchspace = SearchSpace.from_product(
        [NumericalContinuousParameter(name, (0, 1)) for name in parameter_names]
    )

    k = kernel.to_gpytorch(searchspace)
    validate_gpytorch_kernel_components(kernel, k, searchspace)
