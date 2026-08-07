"""Continuous optimizers."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define, field, fields
from attrs.validators import gt, instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, IncompatibleSearchSpaceError
from baybe.optimizers.base import OptimizerProtocol
from baybe.searchspace import SearchSpace
from baybe.settings import AutoBool
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from baybe.optimizers.base import OptimizationResult, ScoreFunction


@define(kw_only=True)
class ContinuousOptimizer(OptimizerProtocol):
    """Optimizer wrapping BoTorch's :func:`botorch.optim.optimize_acqf`."""

    n_starts: int = field(validator=[instance_of(int), gt(0)], default=10)
    """The number of starting points used for the optimization."""

    n_initial_samples: int = field(validator=[instance_of(int), gt(0)], default=64)
    """The number of samples drawn for the starting point selection heuristic."""

    sequential: AutoBool = field(
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
    )
    """Flag defining whether to apply sequential greedy or joint optimization."""

    @override
    def __call__(
        self,
        batch_size: int,
        score_function: ScoreFunction,
        searchspace: SearchSpace,
    ) -> OptimizationResult:
        import torch
        from botorch.optim import optimize_acqf

        subspace = searchspace.continuous

        sequential = self.sequential.evaluate(
            lambda: not subspace.has_interpoint_constraints
        )

        if sequential and subspace.has_interpoint_constraints:
            raise IncompatibilityError(
                f"Setting the "
                f"'{fields(self.__class__).sequential.alias}' "
                f"flag to 'True' while interpoint constraints are present is not "
                f"supported. Set it to either 'False'/'Auto'."
            )

        if subspace.n_subsets > 0:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' "
                f"expects single continuous space, i.e., containing no subsets."
            )

        bounds_df = searchspace.comp_rep_bounds
        fixed_features = searchspace._fixed_values or None

        # NOTE: The explicit `or None` conversions are added as an additional safety net
        #   because it is unclear if the corresponding presence checks for these
        #   arguments is correctly implemented in all invoked BoTorch subroutines.
        #   For details: https://github.com/pytorch/botorch/issues/2042
        points, acqf_values = optimize_acqf(
            acq_function=score_function,
            bounds=torch.from_numpy(bounds_df.to_numpy(copy=True)),
            q=batch_size,
            num_restarts=self.n_starts,
            raw_samples=self.n_initial_samples,
            fixed_features=fixed_features,
            equality_constraints=flatten(
                c.to_botorch(
                    subspace.parameters,
                    idx_offset=len(searchspace.discrete.comp_rep_columns),
                    batch_size=batch_size if c.is_interpoint else None,
                )
                for c in subspace.constraints_lin_eq
            )
            or None,
            inequality_constraints=flatten(
                c.to_botorch(
                    subspace.parameters,
                    idx_offset=len(searchspace.discrete.comp_rep_columns),
                    batch_size=batch_size if c.is_interpoint else None,
                )
                for c in subspace.constraints_lin_ineq
            )
            or None,
            sequential=sequential,
        )

        assert acqf_values is not None  # handle missing BoTorch function overload
        return points, acqf_values


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
