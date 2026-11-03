"""Base class for all Bayesian recommenders."""

from __future__ import annotations

import gc
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional
from attrs.validators import deep_iterable, instance_of
from typing_extensions import override

from baybe.acquisition import qLogEI, qLogNEHVI
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
    IncompatibleArgumentError,
)
from baybe.objectives.base import Objective
from baybe.objectives.tfpr import TFPRObjective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.settings import Settings
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import (
    Surrogate,
    SurrogateProtocol,
)
from baybe.symmetries.base import Symmetry
from baybe.utils.validation import preprocess_dataframe, validate_object_names

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction


def _autoreplicate(surrogate: SurrogateProtocol, /) -> SurrogateProtocol:
    """Replicates single-output surrogate models and passes through everything else."""
    if isinstance(surrogate, Surrogate) and not surrogate.supports_multi_output:
        return surrogate.replicate()
    return surrogate


@define
class BayesianRecommender(PureRecommender, ABC):
    """An abstract class for Bayesian Recommenders."""

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

    symmetries: tuple[Symmetry, ...] = field(
        factory=tuple,
        converter=tuple,
        validator=deep_iterable(member_validator=instance_of(Symmetry)),
        kw_only=True,
    )
    """Symmetries triggering data augmentation during model fitting."""

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
        if isinstance(objective, TFPRObjective):
            raise IncompatibilityError(
                f"Objectives of type '{TFPRObjective.__name__}' rank candidates "
                f"directly and do not use an acquisition function."
            )
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

        # Perform data augmentation
        for s in self.symmetries:
            measurements = s.augment_measurements(measurements, searchspace)

        surrogate = self.get_surrogate(searchspace, objective, measurements)
        self._botorch_acqf = acqf.to_botorch(
            surrogate,
            searchspace,
            objective,
            measurements,
            pending_experiments,
        )

    def _setup_tfpr(
        self,
        searchspace: SearchSpace,
        objective: TFPRObjective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None,
    ) -> None:
        """Validate the TFPR recommendation context and fit the surrogate."""
        name = TFPRObjective.__name__
        if self.acquisition_function is not None:
            raise IncompatibilityError(
                f"Objectives of type '{name}' rank candidates directly and do not use "
                f"an acquisition function, but '{self.__class__.__name__}' was "
                f"configured with '{type(self.acquisition_function).__name__}'."
            )
        if searchspace.type is not SearchSpaceType.DISCRETE:
            raise IncompatibilityError(
                f"Objectives of type '{name}' require a discrete search space."
            )
        if searchspace.discrete.n_subsets > 0:
            raise IncompatibilityError(
                f"Objectives of type '{name}' do not support discrete "
                f"subset-generating constraints."
            )
        if pending_experiments is not None:
            raise IncompatibleArgumentError(
                f"Pending experiments were passed to '{self.__class__.__name__}"
                f".{self.recommend.__name__}' but objectives of type '{name}' cannot "
                f"use this information. If you want to exclude the pending "
                f"experiments from the candidate set, adjust the search space "
                f"accordingly."
            )
        if not hasattr(self._surrogate_model, "posterior_stats"):
            raise IncompatibilityError(
                f"Objectives of type '{name}' require a surrogate providing a "
                f"'posterior_stats' method, which the used surrogate of type "
                f"'{self._surrogate_model.__class__.__name__}' does not."
            )

        self._objective = objective
        self._botorch_acqf = None

        # Perform data augmentation
        for s in self.symmetries:
            measurements = s.augment_measurements(measurements, searchspace)

        self.get_surrogate(searchspace, objective, measurements)

    def _recommend_discrete_tfpr(
        self, candidates_exp: pd.DataFrame, batch_size: int
    ) -> pd.Index:
        """Rank the discrete candidates via the encountered TFPR objective.

        Args:
            candidates_exp: The experimental representation of all discrete candidate
                points to be considered.
            batch_size: The size of the recommendation batch.

        Returns:
            The dataframe indices of the top-ranked candidates.
        """
        assert isinstance(self._objective, TFPRObjective)
        posterior_stats = getattr(self._surrogate_model, "posterior_stats")
        stats = posterior_stats(candidates_exp, stats=("mean", "std"))
        fitness = self._objective.compute_fitness(stats)
        order = np.argsort(-fitness.to_numpy(), kind="stable")[:batch_size]
        return candidates_exp.index[order]

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

        # Experimental input validation
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

        if isinstance(objective, TFPRObjective):
            self._setup_tfpr(searchspace, objective, measurements, pending_experiments)
        else:
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
