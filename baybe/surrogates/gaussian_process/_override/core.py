"""Shared helpers for kernel override resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels.base import Kernel

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.searchspace import SearchSpace
    from baybe.surrogates.gaussian_process.components.kernel import (
        KernelFactoryProtocol,
    )


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
        # TODO[typing]: https://github.com/facebook/pyrefly/issues/3988
        return set(kernel.active_dims.tolist())  # pyrefly: ignore[not-callable]
    if isinstance(kernel, (AdditiveKernel, ProductKernel)):
        return set().union(
            *(get_active_dimensions(k, searchspace) for k in kernel.kernels)
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
