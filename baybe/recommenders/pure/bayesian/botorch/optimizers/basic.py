"""Low-level optimizers of acquisition functions."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar, cast

from attrs import define, field, fields
from attrs.validators import gt, instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, IncompatibleSearchSpaceError
from baybe.recommenders.pure.bayesian.botorch.optimizers.base import OptimizerProtocol
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.settings import AutoBool
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.optimizers.base import Optimand


@define(kw_only=True)
class GradientOptimizer(OptimizerProtocol):
    """Acquisition function optimizer using gradient-based optimization."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.CONTINUOUS
    # See base class.

    n_restarts: int = field(validator=[instance_of(int), gt(0)], default=10)
    """Number of times gradient-based optimization is restarted from different initial
    points.
    """

    n_raw_samples: int = field(validator=[instance_of(int), gt(0)], default=64)
    """Number of raw samples drawn for the initialization heuristic in gradient-based
    optimization.
    """

    sequential_continuous: AutoBool = field(
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
    )
    """Flag defining whether to apply sequential greedy or batch optimization.
    """

    @override
    def __call__(
        self,
        batch_size: int,
        acquisition_function: Optimand,
        searchspace: SearchSpace,
        fixed_parameters: dict[str, float] | None = None,
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
            IncompatibilityError: If the search space has interpoint constraints and the
                ``sequential_continuous`` flag is set to ``True``.
            IncompatibleSearchSpaceError: If search space has a discrete component.
            ValueError: If the search space has cardinality constraints.
        """
        import torch
        from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
        from botorch.optim import optimize_acqf

        if searchspace.type is not self.compatibility:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' only supports continuous search spaces."
            )

        # TODO: Add option for automatic choice once the "settings" PR is merged,
        #   which ships the necessary machinery
        if (
            self.sequential_continuous is not AutoBool.FALSE
            and searchspace.continuous.has_interpoint_constraints
        ):
            raise IncompatibilityError(
                f"Setting the "
                f"'{fields(self.__class__).sequential_continuous.name}' "
                f"flag to ``True`` while interpoint constraints are present in the "
                f"continuous subspace is not supported. "
            )

        if searchspace.continuous.n_subsets > 0:
            raise ValueError(
                f"'{self.__class__.__name__}' "
                f"expects a continuous subspace without cardinality constraints."
            )

        if fixed_parameters:
            param_names = [p.name for p in searchspace.continuous.parameters]
            fixed_features = {
                param_names.index(name): val for name, val in fixed_parameters.items()
            }
        else:
            fixed_features = None

        points, acqf_values = optimize_acqf(
            acq_function=cast(BoAcquisitionFunction, acquisition_function),
            bounds=torch.from_numpy(
                searchspace.continuous.comp_rep_bounds.to_numpy(copy=True)
            ),
            q=batch_size,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            fixed_features=fixed_features or None,
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
            sequential=self.sequential_continuous is not AutoBool.FALSE,
        )

        return points, acqf_values


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
