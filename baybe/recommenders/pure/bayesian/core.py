"""Base class for all Bayesian recommenders."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional
from typing_extensions import override

from baybe.acquisition import qLogEI, qLogNEHVI
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    InfeasibilityError,
)
from baybe.objectives.base import Objective
from baybe.optimizers import ContinuousOptimizer, OptimizerProtocol
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.continuous import (
    recommend_continuous_torch,
)
from baybe.recommenders.pure.bayesian.discrete import (
    recommend_discrete_with_subsets,
    recommend_discrete_without_subsets,
)
from baybe.recommenders.pure.bayesian.hybrid import (
    recommend_hybrid_with_subsets,
    recommend_hybrid_without_subsets,
)
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)
from baybe.settings import Settings
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import (
    Surrogate,
    SurrogateProtocol,
)
from baybe.utils.validation import preprocess_dataframe, validate_object_names

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
    from torch import Tensor


def _autoreplicate(surrogate: SurrogateProtocol, /) -> SurrogateProtocol:
    """Replicates single-output surrogate models and passes through everything else."""
    if isinstance(surrogate, Surrogate) and not surrogate.supports_multi_output:
        return surrogate.replicate()
    return surrogate


@define
class BayesianRecommender(PureRecommender):
    """A recommender utilizing Bayesian optimization machinery.

    This recommender makes use of a given optimizer to optimize acquisition
    functions. It can be applied to all kinds of search spaces.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    supports_discrete_subset_generating_constraints: ClassVar[bool] = True
    # See base class.

    _surrogate_model: SurrogateProtocol = field(
        alias="surrogate_model",
        factory=GaussianProcessSurrogate,
        converter=_autoreplicate,
    )
    """The surrogate model."""

    acquisition_function: AcquisitionFunction | None = field(
        default=None, converter=optional(convert_acqf)
    )
    """The acquisition function. When omitted, a default is used."""

    optimizer: OptimizerProtocol = field(
        alias="optimizer",
        factory=ContinuousOptimizer,
    )
    """The optimizer used to optimize the acquisition function."""

    # TODO: The objective is currently only required for validating the recommendation
    #   context. Once multi-target support is complete, we might want to refactor
    #   the validation mechanism, e.g. by
    #   * storing only the minimal low-level information required
    #   * switching to a strategy where we catch the BoTorch exceptions
    #   * ...
    _objective: Objective | None = field(default=None, init=False, eq=False)
    """The encountered objective to be optimized."""

    _botorch_acqf = field(default=None, init=False, eq=False)
    """The induced BoTorch acquisition function."""

    def _get_acquisition_function(self, objective: Objective) -> AcquisitionFunction:
        """Select the appropriate default acquisition function for the given context."""
        if self.acquisition_function is None:
            return qLogNEHVI() if objective.is_multi_output else qLogEI()
        return self.acquisition_function

    def get_surrogate(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> SurrogateProtocol:
        """Get the trained surrogate model."""
        # This fit applies internal caching and does not necessarily involve computation
        self._surrogate_model.fit(searchspace, objective, measurements)
        return self._surrogate_model

    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> None:
        """Create the acquisition function for the current training data."""  # noqa: E501
        self._objective = objective
        acqf = self._get_acquisition_function(objective)

        if objective.is_multi_output and not acqf.supports_multi_output:
            raise IncompatibleAcquisitionFunctionError(
                f"You attempted to use a single-output acquisition function in a "
                f"{len(objective.targets)}-target multi-output context."
            )

        surrogate = self.get_surrogate(searchspace, objective, measurements)
        self._botorch_acqf = acqf.to_botorch(
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
        )

    def get_acquisition_function(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> BoAcquisitionFunction:
        """Get the BoTorch acquisition function for the given recommendation context.

        For details on the method arguments, see :meth:`recommend`.
        """
        self._setup_botorch_acqf(
            searchspace, objective, measurements, pending_experiments
        )
        return self._botorch_acqf

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if objective is None:
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' require "
                f"that an objective is specified."
            )

        validate_object_names(searchspace.parameters + objective.targets)

        if (measurements is None) or measurements.empty:
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' do not support "
                f"empty training data."
            )

        measurements = preprocess_dataframe(
            measurements,
            searchspace,
            objective,
            numerical_measurements_must_be_within_tolerance=False,
        )

        if pending_experiments is not None:
            pending_experiments = preprocess_dataframe(
                pending_experiments,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        self._setup_botorch_acqf(
            searchspace, objective, measurements, pending_experiments
        )

        try:
            with Settings(preprocess_dataframes=False):
                return super().recommend(
                    batch_size=batch_size,
                    searchspace=searchspace,
                    objective=objective,
                    measurements=measurements,
                    pending_experiments=pending_experiments,
                )
        except RuntimeError as ex:
            # Search spaces with continuous components are incompatible with surrogates
            # that do not support gradient computation
            if (
                "does not have a grad_fn" in str(ex)
                and not searchspace.continuous.is_empty
            ):
                from baybe.exceptions import IncompatibleSurrogateError
                from baybe.surrogates import GaussianProcessSurrogate

                raise IncompatibleSurrogateError(
                    f"The search space contains continuous parameters, but the applied "
                    f"surrogate of type '{self._surrogate_model.__class__.__name__}' "
                    f"does not support the required gradient computation. Choose a "
                    f"surrogate that supports gradients, e.g. the "
                    f"'{GaussianProcessSurrogate.__name__}'."
                ) from ex
            else:
                raise

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

    def acquisition_values(
        self,
        candidates: pd.DataFrame,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        acquisition_function: AcquisitionFunction | None = None,
    ) -> pd.Series:
        """Compute the acquisition values for the given candidates.

        Args:
            candidates: The candidate points in experimental representation.
                For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
            searchspace:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            objective:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            measurements:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            pending_experiments:
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            acquisition_function: The acquisition function to be evaluated.
                If not provided, the acquisition function of the recommender is used.

        Returns:
            A series of individual acquisition values, one for each candidate.
        """
        surrogate = self.get_surrogate(searchspace, objective, measurements)
        acqf = acquisition_function or self._get_acquisition_function(objective)
        return acqf.evaluate(
            candidates,
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
            jointly=False,
        )

    def joint_acquisition_value(  # noqa: DOC101, DOC103
        self,
        candidates: pd.DataFrame,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        acquisition_function: AcquisitionFunction | None = None,
    ) -> float:
        """Compute the joint acquisition value for the given candidate batch.

        For details on the method arguments, see :meth:`acquisition_values`.

        Returns:
            The joint acquisition value of the batch.
        """
        surrogate = self.get_surrogate(searchspace, objective, measurements)
        acqf = acquisition_function or self._get_acquisition_function(objective)
        return acqf.evaluate(
            candidates,
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
            jointly=True,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
