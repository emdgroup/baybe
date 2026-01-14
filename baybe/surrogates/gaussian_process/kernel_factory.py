"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import assert_never, override

from baybe.kernels.base import Kernel
from baybe.kernels.composite import AdditiveKernel, ProjectionKernel
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor


def to_kernel_factory(x: Kernel | KernelFactory, /) -> KernelFactory:
    """Wrap a kernel into a plain kernel factory (with factory passthrough)."""
    return x.to_factory() if isinstance(x, Kernel) else x


class ProjectionMatrixInitialization(Enum):
    """Initialization strategies for kernel projection matrices."""

    MASKING = "MASKING"
    """Axis-aligned masking (random selection of input dimensions)."""

    ORTHONORMAL = "ORTHONORMAL"
    """Random orthonormal basis."""

    PLS = "PLS"
    """Partial Least Squares (PLS) directions."""

    SPHERICAL = "SPHERICAL"
    """Uniform random sampling on the unit sphere."""


def _make_projection_matrices(
    n_projections: int,
    n_matrices: int,
    initialization: ProjectionMatrixInitialization,
    train_x: np.ndarray,
    train_y: np.ndarray,
) -> np.ndarray:
    """Create a collection of projection matrices.

    Args:
        n_projections: The number of projections in each matrix.
        n_matrices: The number of projection matrices to create.
        initialization: The initialization strategy to use.
        train_x: The training inputs.
        train_y: The training outputs.

    Returns:
        An array of shape ``(n_matrices, n_input_dims, n_projections)`` containing the
        created matrices.
    """
    n_input_dims = train_x.shape[-1]

    if n_matrices == 0:
        return np.empty((0, n_input_dims, n_projections))

    if initialization is ProjectionMatrixInitialization.MASKING:
        matrices = []
        for _ in range(n_matrices):
            matrix = np.eye(n_input_dims)
            matrix = matrix[
                :, np.random.choice(n_input_dims, n_projections, replace=False)
            ]
            matrices.append(matrix)

    elif initialization is ProjectionMatrixInitialization.ORTHONORMAL:
        matrices = []
        for _ in range(n_matrices):
            random_matrix = np.random.randn(n_input_dims, n_projections)
            q, _ = np.linalg.qr(random_matrix)
            matrices.append(q[:, :n_projections])

    elif initialization is ProjectionMatrixInitialization.PLS:
        from sklearn.cross_decomposition import PLSRegression

        pls = PLSRegression(n_components=n_projections)
        pls.fit(train_x, train_y)
        M = pls.x_rotations_

        # IMPROVE: One could use the remaining PLS directions for the next matrices
        #   until they are exhausted, then switch to orthonormal.
        matrices = [
            M,
            *_make_projection_matrices(
                n_projections=n_projections,
                n_matrices=n_matrices - 1,
                initialization=ProjectionMatrixInitialization.ORTHONORMAL,
                train_x=train_x,
                train_y=train_y,
            ),
        ]

    elif initialization is ProjectionMatrixInitialization.SPHERICAL:
        matrices = []
        for _ in range(n_matrices):
            matrix = np.random.randn(n_input_dims, n_projections)
            matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
            matrices.append(matrix)

    else:
        assert_never(initialization)

    return np.stack(matrices) if n_matrices > 1 else matrices[0][None, ...]


class KernelFactory(Protocol):
    """A protocol defining the interface expected for kernel factories."""

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        """Create a :class:`baybe.kernels.base.Kernel` for the given DOE context."""
        ...


@define(frozen=True)
class PlainKernelFactory(KernelFactory, SerialMixin):
    """A trivial factory that returns a fixed pre-defined kernel upon request."""

    kernel: Kernel = field(validator=instance_of(Kernel))
    """The fixed kernel to be returned by the factory."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return self.kernel


@define(frozen=True)
class ProjectionKernelFactory(KernelFactory, SerialMixin):
    """A factory producing projected kernels."""

    n_projections: int = field(validator=instance_of(int))
    """The number of projections to be used in each projection matrix."""

    n_matrices: int = field(validator=instance_of(int))
    """The number of projection matrices to be used."""

    initialization: ProjectionMatrixInitialization = field(
        converter=ProjectionMatrixInitialization
    )
    """The initialization strategy for the projection matrices."""

    base_kernel_factory: KernelFactory = field(
        alias="kernel_or_factory",
        converter=to_kernel_factory,
    )
    """The factory creating the base kernel to be applied to the projected inputs."""

    learn_projection: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """See :attr:`baybe.kernels.composite.ProjectionKernel.learn_projection`."""

    @base_kernel_factory.default
    def _default_base_kernel_factory(self) -> KernelFactory:
        from baybe.surrogates.gaussian_process.presets.default import (
            DefaultKernelFactory,
        )

        return DefaultKernelFactory()

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        base_kernel = self.base_kernel_factory(searchspace, train_x, train_y)
        projection_matrices = _make_projection_matrices(
            n_projections=self.n_projections,
            n_matrices=self.n_matrices,
            initialization=self.initialization,
            train_x=train_x.numpy(),
            train_y=train_y.numpy(),
        )
        kernels = [
            ProjectionKernel(
                base_kernel=base_kernel,
                projection_matrix=m,
                learn_projection=self.learn_projection,
            )
            for m in projection_matrices
        ]
        return AdditiveKernel(kernels) if self.n_matrices > 1 else kernels[0]


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
