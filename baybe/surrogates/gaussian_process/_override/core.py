"""Shared helpers for kernel override resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from baybe.exceptions import IncompatibleOverrideError
from baybe.kernels.base import Kernel

if TYPE_CHECKING:
    from baybe.searchspace import SearchSpace


def reduce_kernel_spec(
    component: object,
    excluded_names: set[str],
    searchspace: SearchSpace,
    factory: object,
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
    parameter_names: set[str], factory: object, cause: Exception | None = None
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
