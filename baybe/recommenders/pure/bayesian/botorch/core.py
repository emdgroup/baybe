"""Botorch recommender core."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import ge, gt, instance_of
from typing_extensions import override

from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    InfeasibilityError,
)
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.botorch.continuous import (
    recommend_continuous_torch,
)
from baybe.recommenders.pure.bayesian.botorch.discrete import (
    recommend_discrete_with_subsets,
    recommend_discrete_without_subsets,
)
from baybe.recommenders.pure.bayesian.botorch.hybrid import (
    recommend_hybrid_with_subsets,
    recommend_hybrid_without_subsets,
)
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.utils.conversion import to_string
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

if TYPE_CHECKING:
    from torch import Tensor


@define(kw_only=True)
class BotorchRecommender(BayesianRecommender):
    """A pure recommender utilizing Botorch's optimization machinery.

    This recommender makes use of Botorch's ``optimize_acqf_discrete``,
    ``optimize_acqf`` and ``optimize_acqf_mixed`` functions to optimize discrete,
    continuous and hybrid search spaces, respectively. Accordingly, it can be applied to
    all kinds of search spaces.

    Note:
        In hybrid search spaces, the used algorithm performs a brute-force optimization
        that can be computationally expensive. Thus, the behavior of the algorithm in
        hybrid search spaces can be controlled via two additional parameters.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    supports_discrete_subset_generating_constraints: ClassVar[bool] = True
    # See base class.

    # Object variables
    sequential_continuous: bool = field(default=True)
    """Flag defining whether to apply sequential greedy or batch optimization in
    **continuous** search spaces. In discrete/hybrid spaces, sequential greedy
    optimization is applied automatically.
    """

    hybrid_sampler: DiscreteSamplingMethod | None = field(
        converter=optional_c(DiscreteSamplingMethod), default=None
    )
    """Strategy used for sampling the discrete subspace when performing hybrid search
    space optimization."""

    sampling_percentage: float = field(default=1.0)
    """Percentage of discrete search space that is sampled when performing hybrid search
    space optimization. Ignored when ``hybrid_sampler="None"``."""

    n_restarts: int = field(validator=[instance_of(int), gt(0)], default=10)
    """Number of times gradient-based optimization is restarted from different initial
    points. **Does not affect purely discrete optimization**.
    """

    n_raw_samples: int = field(validator=[instance_of(int), gt(0)], default=64)
    """Number of raw samples drawn for the initialization heuristic in gradient-based
    optimization. **Does not affect purely discrete optimization**.
    """

    max_n_subsets: int = field(default=10, validator=[instance_of(int), ge(1)])
    """Maximum number of subsets to evaluate when subset-generating constraints are
    present (e.g., continuous cardinality constraints). If the total number of
    subsets exceeds this limit, a random subset of that size is sampled for
    optimization instead of performing an exhaustive search."""

    @property
    def max_n_subspaces(self) -> int:
        """Deprecated! Use ``max_n_subsets`` instead."""
        warnings.warn(
            "'max_n_subspaces' has been renamed to 'max_n_subsets' and will "
            "be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.max_n_subsets

    @max_n_subspaces.setter
    def max_n_subspaces(self, value: int) -> None:
        """Deprecated! Use ``max_n_subsets`` instead."""  # noqa: D401
        warnings.warn(
            "'max_n_subspaces' has been renamed to 'max_n_subsets' and will "
            "be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.max_n_subsets = value

    @sampling_percentage.validator
    def _validate_percentage(  # noqa: DOC101, DOC103
        self, _: Any, value: float
    ) -> None:
        """Validate that the given value is in fact a percentage.

        Raises:
            ValueError: If ``value`` is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError(
                f"Hybrid sampling percentage needs to be between 0 and 1 but is {value}"
            )

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Surrogate", self._surrogate_model),
            to_string(
                "Acquisition function", self.acquisition_function, single_line=True
            ),
            to_string("Compatibility", self.compatibility, single_line=True),
            to_string(
                "Sequential continuous", self.sequential_continuous, single_line=True
            ),
            to_string("Hybrid sampler", self.hybrid_sampler, single_line=True),
            to_string(
                "Sampling percentage", self.sampling_percentage, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)

    @override
    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Generate recommendations from a discrete search space.

        Dispatches to the appropriate optimization routine depending on whether
        subset constraints are present.

        Args:
            subspace_discrete: The discrete subspace from which to generate
                recommendations.
            candidates_exp: The experimental representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Returns:
            The dataframe indices of the recommended points in the provided
            experimental representation.
        """
        if subspace_discrete.n_subsets > 0:
            return recommend_discrete_with_subsets(
                self, subspace_discrete, candidates_exp, batch_size
            )
        return recommend_discrete_without_subsets(
            self, subspace_discrete, candidates_exp, batch_size
        )

    @override
    def _recommend_continuous(
        self,
        subspace_continuous: SubspaceContinuous,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate recommendations from a continuous search space.

        Args:
            subspace_continuous: The continuous subspace from which to generate
                recommendations.
            batch_size: The size of the recommendation batch.

        Raises:
            IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
                function is used with a batch size > 1.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        assert self._objective is not None
        if (
            batch_size > 1
            and not self._get_acquisition_function(self._objective).supports_batching
        ):
            raise IncompatibleAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions for batch sizes > 1."
            )

        points, _ = recommend_continuous_torch(self, subspace_continuous, batch_size)

        return pd.DataFrame(points, columns=subspace_continuous.parameter_names)

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate recommendations from a hybrid search space.

        Dispatches to the appropriate optimization routine depending on whether
        subset constraints are present.

        Args:
            searchspace: The search space in which the recommendations should be made.
            candidates_exp: The experimental representation of the candidates
                of the discrete subspace.
            batch_size: The size of the calculated batch.

        Returns:
            The recommended points.
        """
        if searchspace.n_subsets > 0:
            return recommend_hybrid_with_subsets(
                self, searchspace, candidates_exp, batch_size
            )
        return recommend_hybrid_without_subsets(
            self, searchspace, candidates_exp, batch_size
        )

    def _optimize_over_subsets(
        self,
        subset_callables: Iterable[Callable[[], tuple[Any, Tensor]]],
    ) -> tuple[Any, Tensor]:
        """Optimize across subsets and return the result with the best acqf value.

        Each callable performs optimization for one subset configuration and returns
        a ``(result, acquisition_value)`` tuple. Subsets that raise
        ``InfeasibilityError`` are silently skipped.

        Args:
            subset_callables: An iterable of zero-argument callables. Each callable
                runs the optimization for one subset and returns
                ``(result, acqf_value)``. It may raise ``InfeasibilityError`` if the
                subset is infeasible.

        Raises:
            InfeasibilityError: If none of the subsets has a feasible solution.

        Returns:
            The result and acquisition value of the best subset.
        """
        from botorch.exceptions.errors import InfeasibilityError as BoInfeasibilityError

        results_all: list = []
        acqf_values_all: list[Tensor] = []

        for optimize_fn in subset_callables:
            try:
                result, acqf_value = optimize_fn()
                results_all.append(result)
                acqf_values_all.append(acqf_value)
            except (BoInfeasibilityError, InfeasibilityError):
                pass

        if not results_all:
            raise InfeasibilityError(
                "No feasible solution could be found. Potentially the specified "
                "constraints are too restrictive, i.e. there may be too many "
                "constraints or thresholds may have been set too tightly. "
                "Consider relaxing the constraints to improve the chances "
                "of finding a feasible solution."
            )

        best_idx = np.argmax(acqf_values_all)
        return results_all[best_idx], acqf_values_all[best_idx]


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
