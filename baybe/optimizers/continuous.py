"""Low-level optimizers."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar, cast

from attrs import define, field, fields
from attrs.validators import gt, instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, IncompatibleSearchSpaceError
from baybe.optimizers.base import OptimizerProtocol
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.settings import AutoBool
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.optimizers.base import ScoreFunction


@define(kw_only=True)
class GradientOptimizer(OptimizerProtocol):
    """Optimizer using gradient-based optimization."""

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
        score_function: ScoreFunction,
        searchspace: SearchSpace,
    ) -> tuple[Tensor, Tensor]:
        import torch
        from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
        from botorch.optim import optimize_acqf

        if searchspace.type is not self.compatibility:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' only supports continuous search spaces."
            )

        sequential_continuous = self.sequential_continuous.evaluate(
            lambda: not searchspace.continuous.has_interpoint_constraints
        )

        if sequential_continuous and searchspace.continuous.has_interpoint_constraints:
            raise IncompatibilityError(
                f"Setting the "
                f"'{fields(self.__class__).sequential_continuous.alias}' "
                f"flag to 'True' while interpoint constraints are present is not "
                f"supported. Set it to either 'False'/'Auto'."
            )

        if searchspace.continuous.n_subsets > 0:
            raise ValueError(
                f"'{self.__class__.__name__}' "
                f"expects a continuous subspace without cardinality constraints."
            )

        fixed_features = {
            i: p.value
            for i, p in enumerate(searchspace.continuous.parameters)
            if isinstance(p, _FixedNumericalContinuousParameter)
        }

        # NOTE: The explicit `or None` conversions are added as an additional safety net
        #   because it is unclear if the corresponding presence checks for these
        #   arguments is correctly implemented in all invoked BoTorch subroutines.
        #   For details: https://github.com/pytorch/botorch/issues/2042
        points, acqf_values = optimize_acqf(
            acq_function=cast(BoAcquisitionFunction, score_function),
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
            sequential=sequential_continuous,
        )

        return points, acqf_values


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
