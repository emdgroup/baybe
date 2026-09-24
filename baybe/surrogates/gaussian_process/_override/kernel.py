"""Kernel override resolution for Gaussian process surrogates."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, NoReturn

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels.base import Kernel

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace
    from baybe.surrogates.gaussian_process.components.kernel import (
        KernelFactoryProtocol,
    )
    from baybe.surrogates.gaussian_process.core import _ModelContext


def extract_parameter_kernel_overrides(
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
        if p.override_kernel is not None
    ]


def make_parameter_override_kernel(
    parameter: Parameter, searchspace: SearchSpace
) -> GPyTorchKernel:
    """Create the kernel factor for a parameter's override.

    Args:
        parameter: The parameter carrying a non-``None`` kernel override, as ensured
            by :func:`extract_parameter_kernel_overrides`.
        searchspace: The search space the kernel operates on.

    Returns:
        The GPyTorch kernel bound to the parameter's dimensions.
    """
    override = parameter.override_kernel
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
    # TODO[typing]: GPyTorch annotates `active_dims` with the constructor's tuple
    #   type, but `register_buffer` stores a tensor at runtime.
    result.active_dims = torch.tensor(indices)  # pyrefly: ignore[bad-assignment]
    return result


def get_active_dimensions(kernel: GPyTorchKernel, searchspace: SearchSpace) -> set[int]:
    """Get the input columns used by a kernel, including standard composites.

    Args:
        kernel: The resolved kernel to inspect.
        searchspace: The search space defining the full model input.

    Returns:
        The active input column indices.
    """
    from gpytorch.kernels import AdditiveKernel, ProductKernel, ScaleKernel

    if kernel.active_dims is not None:
        # TODO[typing]: GPyTorch annotates `active_dims` with the constructor's tuple
        #   type, but `register_buffer` stores a tensor at runtime.
        return set(kernel.active_dims.tolist())  # pyrefly: ignore[missing-attribute]
    if isinstance(kernel, (AdditiveKernel, ProductKernel)):
        return set().union(
            # TODO[typing]: Iterating a `ModuleList` yields the `Module` base type.
            *(get_active_dimensions(k, searchspace) for k in kernel.kernels)  # pyrefly: ignore[bad-argument-type]
        )
    if isinstance(kernel, ScaleKernel):
        return get_active_dimensions(kernel.base_kernel, searchspace)
    return set(range(len(searchspace.comp_rep_columns)))


def reduce_kernel_spec(
    component: Kernel | GPyTorchKernel,
    excluded_names: set[str],
    searchspace: SearchSpace,
    factory: KernelFactoryProtocol,
) -> Kernel | None:
    """Remove the excluded parameters from a fixed BayBE kernel.

    Args:
        component: The kernel specification to reduce.
        excluded_names: The names of the parameters to remove.
        searchspace: The search space the kernel operates on.
        factory: The originating factory (for error messages).

    Returns:
        The reduced kernel, or ``None`` if nothing remains.
    """
    if not isinstance(component, Kernel):
        raise_incompatible_override(excluded_names, factory)
    # Reduction can exhaust the scope and return None despite a non-None input.
    spec: Kernel | None = component
    for name in excluded_names:
        if spec is None:
            break
        try:
            spec = spec._without_parameter(name, searchspace)
        except TypeError as ex:
            raise_incompatible_override(excluded_names, factory, ex)

    return spec


def raise_incompatible_override(
    parameter_names: set[str],
    factory: KernelFactoryProtocol,
    cause: Exception | None = None,
) -> NoReturn:
    """Raise an error for a surrogate kernel that cannot be reduced.

    Args:
        parameter_names: The overridden parameter names.
        factory: The offending kernel factory.
        cause: The underlying exception, if any.

    Raises:
        IncompatibleOverrideError: Always.
    """
    raise IncompatibleOverrideError(
        f"Kernel overrides for {sorted(parameter_names)} require a surrogate "
        f"kernel (factory) that can exclude these parameters. "
        f"'{type(factory).__name__}' does not satisfy this requirement."
    ) from cause
