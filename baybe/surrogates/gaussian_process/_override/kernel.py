"""Kernel override resolution for Gaussian process surrogates."""

from __future__ import annotations

from collections.abc import Iterator
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
        override: The provided GPyTorch kernel.
        indices: The computational column indices of the owning parameter.
        name: The owning parameter name (for error messages).

    Raises:
        IncompatibleOverrideError: If the kernel specifies an incompatible number of
            ARD dimensions.

    Returns:
        A copy of the kernel bound to the given dimensions.
    """
    import torch

    for kernel in (override, *override.sub_kernels()):
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


def iter_gpytorch_kernel_tree(
    kernel: GPyTorchKernel, n_columns: int, /
) -> Iterator[tuple[GPyTorchKernel, tuple[int, ...]]]:
    """Iterate over all kernels of a GPyTorch kernel tree with their input columns.

    In contrast to :meth:`gpytorch.kernels.Kernel.named_sub_kernels`, each kernel is
    yielded together with the columns of the full model input it actually acts on.
    These are not simply the kernel's ``active_dims``, since:

    * ``active_dims`` are relative to the (already sliced) input of the parent kernel.
    * A :class:`~gpytorch.kernels.ScaleKernel` passes its sliced input directly to the
      ``forward`` method of its base kernel, bypassing the base kernel's own slicing.

    Example:
        For ``ProductKernel(RBFKernel(active_dims=[0]), ScaleKernel(MaternKernel(
        active_dims=[1, 2])))`` on three columns, the yielded pairs are the product
        kernel with ``(0, 1, 2)``, the RBF kernel with ``(0,)``, and both the scale
        kernel (which adopts the ``active_dims`` of its base kernel) and the Matérn
        kernel with ``(1, 2)``.

    Args:
        kernel: The root of the kernel tree.
        n_columns: The number of columns of the full model input.

    Yields:
        Each kernel of the tree (in depth-first order, starting with the root) together
        with the indices of the model input columns it acts on.
    """
    from gpytorch.kernels import Kernel, ScaleKernel
    from torch.nn import ModuleList

    def _iterate(
        kernel: GPyTorchKernel, columns: tuple[int, ...], sliced: bool
    ) -> Iterator[tuple[GPyTorchKernel, tuple[int, ...]]]:
        if sliced and (active_dims := kernel.active_dims) is not None:
            # TODO[typing]: GPyTorch annotates `active_dims` with the constructor's
            #   tuple type, but `register_buffer` stores a tensor at runtime.
            columns = tuple(columns[i] for i in active_dims.tolist())  # pyrefly: ignore[missing-attribute]
        yield kernel, columns
        for child in kernel.children():
            for sub in child if isinstance(child, ModuleList) else (child,):
                if isinstance(sub, Kernel):
                    yield from _iterate(
                        sub, columns, not isinstance(kernel, ScaleKernel)
                    )

    yield from _iterate(kernel, tuple(range(n_columns)), True)


def get_active_dimensions(kernel: GPyTorchKernel, searchspace: SearchSpace) -> set[int]:
    """Get the input columns used by a kernel, i.e. those of its leaf kernels.

    Args:
        kernel: The resolved kernel to inspect.
        searchspace: The search space defining the full model input.

    Returns:
        The active input column indices.
    """
    from gpytorch.kernels import Kernel

    return {
        column
        for k, columns in iter_gpytorch_kernel_tree(
            kernel, len(searchspace.comp_rep_columns)
        )
        if not any(isinstance(m, Kernel) for m in k.modules() if m is not k)
        for column in columns
    }


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
