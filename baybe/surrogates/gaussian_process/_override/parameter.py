"""Resolution of parameter-specific kernel overrides."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels.base import Kernel

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace
    from baybe.surrogates.gaussian_process.core import _ModelContext


def extract_parameter_overrides(
    context: _ModelContext,
) -> list[tuple[str, GPyTorchKernel]]:
    """Extract the regular parameter-specific kernel overrides.

    Args:
        context: The model context providing the search space.

    Returns:
        A ``(parameter_name, kernel)`` pair for each parameter with an override.
    """
    return [
        (p.name, make_parameter_override_kernel(p, context.searchspace))
        for p in context.searchspace.parameters
        if p.kernel_override is not None
    ]


def make_parameter_override_kernel(
    parameter: Parameter, searchspace: SearchSpace
) -> GPyTorchKernel:
    """Create the kernel factor for a parameter's override.

    Args:
        parameter: The parameter carrying the override.
        searchspace: The search space the kernel operates on.

    Returns:
        The GPyTorch kernel bound to the parameter's dimensions.
    """
    override = parameter.kernel_override
    assert override is not None

    # BayBE kernels resolve their own dimensions; raw kernels are bound manually.
    if isinstance(override, Kernel):
        return override.to_gpytorch(searchspace)
    indices = searchspace.get_comp_rep_parameter_indices(parameter.name)
    return bind_gpytorch_override(override, indices, parameter.name)


def bind_gpytorch_override(
    override: GPyTorchKernel, indices: tuple[int, ...], name: str
) -> GPyTorchKernel:
    """Copy a raw GPyTorch override and bind it to the given dimensions.

    Args:
        override: The provided GPyTorch kernel (must not specify active dimensions).
        indices: The computational column indices of the owning parameter.
        name: The owning parameter name (for error messages).

    Raises:
        IncompatibleOverrideError: If the kernel specifies active dimensions or an
            incompatible number of ARD dimensions.

    Returns:
        A copy of the kernel bound to the given dimensions.
    """
    import torch
    from gpytorch.kernels import Kernel as GPyTorchKernel

    for kernel in override.modules():
        if not isinstance(kernel, GPyTorchKernel):
            continue
        if kernel.active_dims is not None:
            raise IncompatibleOverrideError(
                f"The GPyTorch kernel override for parameter '{name}' must not "
                f"specify 'active_dims'."
            )
        if kernel.ard_num_dims not in (None, len(indices)):
            raise IncompatibleOverrideError(
                f"The GPyTorch kernel override for parameter '{name}' specifies "
                f"{kernel.ard_num_dims} ARD dimensions, but the parameter has "
                f"{len(indices)} computational dimensions."
            )
    result = deepcopy(override)
    result.active_dims = torch.tensor(indices)
    return result
