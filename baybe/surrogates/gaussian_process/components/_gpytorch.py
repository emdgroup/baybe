"""Custom GPyTorch components."""

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from botorch.models.multitask import _compute_multitask_mean
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood
from gpytorch.means import MultitaskMean
from gpytorch.means.multitask_mean import Mean
from gpytorch.priors import LogNormalPrior
from torch import Tensor
from torch.nn import Module
from typing_extensions import override


class HadamardConstantMean(Mean):
    """A GPyTorch mean function implementing BoTorch's multitask mean logic.

    While GPyTorch already provides a :class:`~gpytorch.means.MultitaskMean` class, it
    computes mean values for all (input, task)-pairs (where input means all parameters
    except the task parameter), i.e. it intrinsically applies a Cartesian expansion.
    However, for the regular transfer learning setting, we only need the means for the
    pairs that are actually observed/requested. BoTorch subselects the relevant means
    from the GPyTorch output in `MultiTaskGP.forward`, i.e. it uses a class-based
    approach to define its special logic for the multitask case. In contrast, BayBE uses
    a composition approach, which is more flexible but requires that the logic is
    injected via a self-contained `Mean` object, which is what this class provides.

    Note:
        Analogous to GPyTorch's
        https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/likelihoods/hadamard_gaussian_likelihood.py
        but where the logic is applied to the mean function, i.e. we learn a different
        (constant) mean for each task.
    """

    def __init__(self, mean_module: Module, num_tasks: int, task_feature: int):
        super().__init__()
        self.multitask_mean = MultitaskMean(mean_module, num_tasks=num_tasks)
        self.task_feature = task_feature

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Adapted from https://github.com/meta-pytorch/botorch/blob/e0f4f5b941b5949a4a1171bf8d4ee9f74f146f3a/botorch/models/multitask.py#L397

        # Convert task feature to positive index
        task_feature = self.task_feature % x.shape[-1]

        # Split input into task and non-task components
        x_before = x[..., :task_feature]
        task_idcs = x[..., task_feature : task_feature + 1]
        x_after = x[..., task_feature + 1 :]

        return _compute_multitask_mean(
            self.multitask_mean, x_before, task_idcs, x_after
        )


def make_botorch_multitask_likelihood(
    num_tasks: int, task_feature: int
) -> HadamardGaussianLikelihood:
    """Adapted from :class:`botorch.models.multitask.MultiTaskGP`."""
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    return HadamardGaussianLikelihood(
        num_tasks=num_tasks,
        batch_shape=torch.Size(),
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior.mode,
        ),
        task_feature_index=task_feature,
    )


def _sum(kernel_values: Iterable[Any]) -> Any:
    """Sum kernel evaluations (tensors or lazy operators) without evaluating them.

    In contrast to the builtin :func:`sum`, no start value is added, so that lazily
    evaluated kernel tensors are combined via their own addition and remain lazy.

    Args:
        kernel_values: The kernel evaluations to sum (at least one).

    Returns:
        The sum of the kernel evaluations.
    """
    values = iter(kernel_values)
    total = next(values)
    for value in values:
        total = total + value
    return total


class PermutationInvariantKernel(Kernel):
    """A kernel summing a base kernel over permutations of its second input.

    The result is invariant under the permutations if the base kernel is invariant
    under jointly permuting both of its inputs.

    Args:
        base_kernel: The kernel to be symmetrized.
        index_maps: For each permutation, the input column used for each column.
    """

    index_maps: Tensor
    """For each permutation, the input column used for each column."""

    def __init__(self, base_kernel: Kernel, index_maps: Sequence[Sequence[int]]):
        super().__init__()
        self.base_kernel = base_kernel
        self.register_buffer("index_maps", torch.tensor(index_maps))

    @override
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        return _sum(
            self.base_kernel(x1, x2[..., idx], diag, last_dim_is_batch, **params)
            for idx in self.index_maps
        )


class MirrorInvariantKernel(Kernel):
    """A kernel summing a base kernel over reflections of both of its inputs.

    Args:
        base_kernel: The kernel to be symmetrized.
        column: The input column to be reflected.
        center: The mirror point in the input space of the kernel.
    """

    def __init__(self, base_kernel: Kernel, column: int, center: float):
        super().__init__()
        self.base_kernel = base_kernel
        self.column = column
        self.center = center

    def _reflect(self, x: Tensor) -> Tensor:
        """Reflect the mirrored column of the input at the center.

        Example:
            With center ``0.3``, the value ``0.1`` of the mirrored column becomes
            ``0.5``, while all other columns are unchanged.

        Args:
            x: The input.

        Returns:
            A copy of the input with the mirrored column reflected.
        """
        reflected = x.clone()
        reflected[..., self.column] = 2 * self.center - x[..., self.column]
        return reflected

    @override
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        return _sum(
            self.base_kernel(a, b, diag, last_dim_is_batch, **params)
            for a in (x1, self._reflect(x1))
            for b in (x2, self._reflect(x2))
        )


class DependencyGatedKernel(Kernel):
    """A kernel ignoring the affected parameters of inactive dependencies.

    Each input is assigned an activity pattern, i.e. the combination of active and
    inactive dependencies. Inputs with equal patterns are compared with the base kernel
    after fixing the affected columns of all inactive dependencies to a constant,
    inputs with different patterns are uncorrelated.

    The activity of a dependency is determined by matching the input columns of its
    causing parameter against known encodings.

    Args:
        base_kernel: The kernel to be gated.
        affected_columns: The input columns of the affected parameters of each
            dependency.
        causing_columns: The input columns of the causing parameter of each
            dependency.
        encodings: The known encodings of each causing parameter, one per row.
        activities: Whether each known encoding makes its dependency active.
    """

    atol: float = 1e-6
    """The absolute tolerance for matching encodings."""

    fill_value: float = 0.0
    """The constant replacing the affected columns of inactive dependencies."""

    def __init__(
        self,
        base_kernel: Kernel,
        affected_columns: Sequence[Sequence[int]],
        causing_columns: Sequence[Sequence[int]],
        encodings: Sequence[Tensor],
        activities: Sequence[Tensor],
    ):
        super().__init__()
        self.base_kernel = base_kernel
        self.affected_columns = [list(c) for c in affected_columns]
        self.causing_columns = [list(c) for c in causing_columns]
        for d, (encoding, activity) in enumerate(zip(encodings, activities)):
            self.register_buffer(f"encodings_{d}", encoding)
            self.register_buffer(f"activities_{d}", activity)

    def _get_patterns(self, x: Tensor) -> Tensor:
        """Get the activity pattern index of each input.

        Args:
            x: The inputs.

        Returns:
            The pattern index of each input, where bit ``d`` indicates whether
            dependency ``d`` is active.

        Example:
            With two dependencies of which only the second one is active for an
            input, the pattern index of that input is ``0b10``.

        Raises:
            ValueError: If an input contains an unknown encoding of a causing
                parameter.
        """
        patterns = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        for d, columns in enumerate(self.causing_columns):
            encodings = getattr(self, f"encodings_{d}").to(x)
            matches = torch.isclose(
                x[..., None, columns], encodings, rtol=0.0, atol=self.atol
            ).all(-1)
            if not matches.any(-1).all():
                raise ValueError(
                    "The input contains a value of a causing parameter of a dependency "
                    "symmetry that does not match any of the parameter's values."
                )
            active = (matches & getattr(self, f"activities_{d}")).any(-1)
            patterns = patterns + (active.long() << d)
        return patterns

    def _fix_inactive(self, x: Tensor, pattern: int) -> Tensor:
        """Fix the affected columns of the inactive dependencies of a pattern.

        Example:
            For pattern ``0b01`` (dependency 0 active, dependency 1 inactive), the
            affected columns of dependency 1 are set to :attr:`fill_value`, while those
            of dependency 0 keep their values.

        Args:
            x: The input.
            pattern: The activity pattern index.

        Returns:
            A copy of the input with the affected columns of the inactive dependencies
            replaced.
        """
        x = x.clone()
        for d, columns in enumerate(self.affected_columns):
            if not pattern >> d & 1:
                x[..., columns] = self.fill_value
        return x

    @override
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        p1, p2 = self._get_patterns(x1), self._get_patterns(x2)
        values = (
            (
                self.base_kernel(
                    self._fix_inactive(x1, p),
                    self._fix_inactive(x2, p),
                    diag,
                    last_dim_is_batch,
                    **params,
                ),
                p,
            )
            for p in range(2 ** len(self.affected_columns))
        )
        if diag:
            return _sum(k * ((p1 == p) & (p2 == p)) for k, p in values)
        return _sum(
            k.to_dense() * ((p1 == p)[..., :, None] & (p2 == p)[..., None, :])
            for k, p in values
        )


class TiedEntries(Module):
    """A parametrization letting several entries of a parameter share one value.

    This is not a kernel. It is a :mod:`torch.nn.utils.parametrize` parametrization
    that can be registered on an existing tensor parameter of a GPyTorch module (e.g.
    the ``raw_lengthscale`` of a kernel) via
    :func:`torch.nn.utils.parametrize.register_parametrization`. Afterward, the module
    stores and optimizes only a reduced tensor, which is expanded to the original shape
    on every access. Entries mapped to the same reduced entry thus always hold
    identical values, including during model fitting.

    BayBE uses it to share the lengthscales of permuted parameters within a joint ARD
    kernel, which is required for :class:`PermutationInvariantKernel` to yield a valid
    kernel. More generally, it can tie any per-dimension parameter whose entries should
    be learned jointly.

    Example:
        With ``index=[0, 0, 1]``, the reduced tensor ``[a, b]`` is expanded to
        ``[a, a, b]``, i.e. the first two entries are tied.

    Note:
        Registering a parametrization renames the parameter to
        ``parametrizations.<name>.original``, so that GPyTorch no longer finds
        constraints registered under the original name. They must thus be registered
        again on the parametrization.

    Args:
        index: For each entry of the expanded tensor, the entry of the reduced tensor
            holding its value.
    """

    index: Tensor
    """For each entry of the expanded tensor, the entry of the reduced tensor."""

    def __init__(self, index: Sequence[int]):
        super().__init__()
        self.register_buffer("index", torch.tensor(index))

    @override
    def forward(self, reduced: Tensor) -> Tensor:
        return reduced[..., self.index]

    def right_inverse(self, expanded: Tensor) -> Tensor:
        """Reduce a tensor by averaging the entries that share a value."""
        return torch.stack(
            [
                expanded[..., self.index == j].mean(-1)
                for j in range(int(self.index.max()) + 1)
            ],
            -1,
        )
