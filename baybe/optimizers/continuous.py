"""Continuous optimizers."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from attrs import define, field, fields
from attrs.validators import gt, instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, IncompatibleSearchSpaceError
from baybe.optimizers.base import OptimizerProtocol
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.settings import AutoBool
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.optimizers.base import ScoreFunction


@define(kw_only=True)
class ContinuousOptimizer(OptimizerProtocol[SearchSpace]):
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
        space: SearchSpace,
    ) -> tuple[Tensor, Tensor]:
        import torch
        from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
        from botorch.optim import optimize_acqf

        cont = space.continuous

        sequential = self.sequential.evaluate(
            lambda: not cont.has_interpoint_constraints
        )

        if sequential and cont.has_interpoint_constraints:
            raise IncompatibilityError(
                f"Setting the "
                f"'{fields(self.__class__).sequential.alias}' "
                f"flag to 'True' while interpoint constraints are present is not "
                f"supported. Set it to either 'False'/'Auto'."
            )

        if cont.n_subsets > 0:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' "
                f"expects single continuous space, i.e., containing no subsets."
            )

        # TODO: Unify with fixed_values depending on SearchSpace refactor
        bounds_df = space.comp_rep_bounds
        internal_fixed = {
            space.comp_rep_columns.index(p.name): p.value
            for p in cont.parameters
            if isinstance(p, _FixedNumericalContinuousParameter)
        }
        external_fixed = {
            space.comp_rep_columns.index(col): val
            for col, val in space.fixed_values.items()
        }
        overlap = internal_fixed.keys() & external_fixed.keys()
        if overlap and any(internal_fixed[k] != external_fixed[k] for k in overlap):
            conflicting = [space.comp_rep_columns[i] for i in sorted(overlap)]
            raise ValueError(
                f"Conflicting values for fixed parameters {conflicting}: "
                f"cardinality constraints and external fixed_values disagree."
            )
        merged_fixed = {**internal_fixed, **external_fixed}

        # NOTE: The explicit `or None` conversions are added as an additional safety net
        #   because it is unclear if the corresponding presence checks for these
        #   arguments is correctly implemented in all invoked BoTorch subroutines.
        #   For details: https://github.com/pytorch/botorch/issues/2042
        points, acqf_values = optimize_acqf(
            acq_function=cast(BoAcquisitionFunction, score_function),
            bounds=torch.from_numpy(bounds_df.to_numpy(copy=True)),
            q=batch_size,
            num_restarts=self.n_starts,
            raw_samples=self.n_initial_samples,
            fixed_features=merged_fixed or None,
            equality_constraints=flatten(
                c.to_botorch(
                    cont.parameters,
                    idx_offset=len(space.discrete.comp_rep_columns),
                    batch_size=batch_size if c.is_interpoint else None,
                )
                for c in cont.constraints_lin_eq
            )
            or None,
            inequality_constraints=flatten(
                c.to_botorch(
                    cont.parameters,
                    idx_offset=len(space.discrete.comp_rep_columns),
                    batch_size=batch_size if c.is_interpoint else None,
                )
                for c in cont.constraints_lin_ineq
            )
            or None,
            sequential=sequential,
        )

        assert acqf_values is not None  # handle missing BoTorch function overload
        return points, acqf_values


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
