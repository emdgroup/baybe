"""Low-level optimizers of acquisition functions."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING
from typing_extensions import override

from attrs import define, field
from attrs.validators import gt, instance_of

from baybe.recommenders.pure.bayesian.botorch.optimizers.base import OptimizerProtocol
from baybe.searchspace import SearchSpace
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
    from torch import Tensor


@define(kw_only=True)
class GradientOptimizer(OptimizerProtocol):
    """Acquisition function optimizer using gradient-based optimization."""

    n_restarts: int = field(validator=[instance_of(int), gt(0)], default=10)
    """Number of times gradient-based optimization is restarted from different initial
    points. **Does not affect purely discrete optimization**.
    """

    n_raw_samples: int = field(validator=[instance_of(int), gt(0)], default=64)
    """Number of raw samples drawn for the initialization heuristic in gradient-based
    optimization. **Does not affect purely discrete optimization**.
    """

    sequential_continuous: bool = field(default=True)
    """Flag defining whether to apply sequential greedy or batch optimization in
    **continuous** search spaces. In discrete/hybrid spaces, sequential greedy
    optimization is applied automatically.
    """

    @override
    def __call__(
        self,
        batch_size: int,
        acquisition_function: BoAcquisitionFunction,
        searchspace: SearchSpace,
        fixed_parameters: dict[int, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Recommend from a search space using gradient-based optimization.

        Args:
            batch_size: The size of the recommendation batch.
            acquisition_function: The acquisition function to be optimized.
            searchspace: The search space from which to generate recommendations.
            fixed_parameters: A dictionary mapping parameter indices to fixed values.

        Returns:
            The recommendations and corresponding acquisition values.

        Raises:
            NotImplementedError: If the search space has a discrete component.
            ValueError: If the search space has cardinality constraints.
        """
        import torch
        from botorch.optim import optimize_acqf

        if not searchspace.discrete.is_empty:
            raise NotImplementedError(
                "Gradient-based optimization is not implemented "
                "for non-empty discrete search spaces."
            )

        if searchspace.continuous.n_subsets > 0:
            raise ValueError(
                f"'{self.__class__.__name__}' "
                f"expects a continuous subspace without cardinality constraints."
            )

        points, acqf_values = optimize_acqf(
            acq_function=acquisition_function,
            bounds=torch.from_numpy(
                searchspace.continuous.comp_rep_bounds.to_numpy(copy=True)
            ),
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            fixed_features=fixed_parameters or None,
            equality_constraints=flatten(
                c.to_botorch(
                    searchspace.continuous.parameters,
                    batch_size=batch_size if c.is_interpoint else None,
                )
                for c in searchspace.continuous.constraints_lin_eq
            )
            or None,
            inequality_constraints=flatten(
                c.to_botorch(
                    searchspace.continuous.parameters,
                    batch_size=batch_size if c.is_interpoint else None,
                )
                for c in searchspace.continuous.constraints_lin_ineq
            )
            or None,
            sequential=self.sequential_continuous,
        )

        return points, acqf_values


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
