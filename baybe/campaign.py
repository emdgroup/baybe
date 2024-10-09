"""Functionality for managing DOE campaigns. Main point of interaction via Python."""

from __future__ import annotations

import gc
import json
from typing import TYPE_CHECKING

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional
from attrs.validators import instance_of

from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective, to_objective
from baybe.parameters.base import Parameter
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.base import MetaRecommender
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace.core import (
    SearchSpace,
    SearchSpaceType,
    to_searchspace,
    validate_searchspace_from_config,
)
from baybe.serialization import SerialMixin, converter
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.base import Target
from baybe.telemetry import (
    TELEM_LABELS,
    telemetry_record_recommended_measurement_percentage,
    telemetry_record_value,
)
from baybe.utils.boolean import eq_dataframe
from baybe.utils.plotting import to_string

if TYPE_CHECKING:
    from botorch.posteriors import Posterior


@define
class Campaign(SerialMixin):
    """Main class for interaction with BayBE.

    Campaigns define and record an experimentation process, i.e. the execution of a
    series of measurements and the iterative sequence of events involved.

    In particular, a campaign:
        * Defines the objective of an experimentation process.
        * Defines the search space over which the experimental parameter may vary.
        * Defines a recommender for exploring the search space.
        * Records the measurement data collected during the process.
        * Records metadata about the progress of the experimentation process.
    """

    # DOE specifications
    searchspace: SearchSpace = field(converter=to_searchspace)
    """The search space in which the experiments are conducted.
    When passing a :class:`baybe.parameters.base.Parameter`,
    a :class:`baybe.searchspace.discrete.SubspaceDiscrete`, or a
    a :class:`baybe.searchspace.continuous.SubspaceContinuous`, conversion to
    :class:`baybe.searchspace.core.SearchSpace` is automatically applied."""

    objective: Objective | None = field(default=None, converter=optional(to_objective))
    """The optimization objective.
    When passing a :class:`baybe.targets.base.Target`, conversion to
    :class:`baybe.objectives.single.SingleTargetObjective` is automatically applied."""

    recommender: RecommenderProtocol = field(
        factory=TwoPhaseMetaRecommender,
        validator=instance_of(RecommenderProtocol),
    )
    """The employed recommender"""

    # Metadata
    n_batches_done: int = field(default=0, init=False)
    """The number of already processed batches."""

    n_fits_done: int = field(default=0, init=False)
    """The number of fits already done."""

    # Private
    _measurements_exp: pd.DataFrame = field(
        factory=pd.DataFrame, eq=eq_dataframe, init=False
    )
    """The experimental representation of the conducted experiments."""

    _cached_recommendation: pd.DataFrame = field(
        factory=pd.DataFrame, eq=eq_dataframe, init=False
    )
    """The cached recommendations."""

    def __str__(self) -> str:
        metadata_fields = [
            to_string("Batches done", self.n_batches_done, single_line=True),
            to_string("Fits done", self.n_fits_done, single_line=True),
        ]
        metadata = to_string("Meta Data", *metadata_fields)
        fields = [metadata, self.searchspace, self.objective, self.recommender]

        return to_string(self.__class__.__name__, *fields)

    @property
    def measurements(self) -> pd.DataFrame:
        """The experimental data added to the Campaign."""
        return self._measurements_exp

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """The parameters of the underlying search space."""
        return self.searchspace.parameters

    @property
    def targets(self) -> tuple[Target, ...]:
        """The targets of the underlying objective."""
        return self.objective.targets if self.objective is not None else ()

    @classmethod
    def from_config(cls, config_json: str) -> Campaign:
        """Create a campaign from a configuration JSON.

        Args:
            config_json: The string with the configuration JSON.

        Returns:
            The constructed campaign.
        """
        config = json.loads(config_json)
        return converter.structure(config, Campaign)

    @classmethod
    def validate_config(cls, config_json: str) -> None:
        """Validate a given campaign configuration JSON.

        Args:
            config_json: The JSON that should be validated.
        """
        config = json.loads(config_json)
        _validation_converter.structure(config, Campaign)

    def add_measurements(
        self,
        data: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Add results from a dataframe to the internal database.

        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values,
        a campaign flag determines if values that lie outside a specified tolerance
        are accepted.
        Note that this modifies the provided data in-place.

        Args:
            data: The data to be added (with filled values for targets). Preferably
                created via :func:`baybe.campaign.Campaign.recommend`.
            numerical_measurements_must_be_within_tolerance: Flag indicating if
                numerical parameters need to be within their tolerances.

        Raises:
            ValueError: If one of the targets has missing values or NaNs in the provided
                dataframe.
            TypeError: If the target has non-numeric entries in the provided dataframe.
        """
        # Invalidate recommendation cache first (in case of uncaught exceptions below)
        self._cached_recommendation = pd.DataFrame()

        # Check if all targets have valid values
        for target in self.targets:
            if data[target.name].isna().any():
                raise ValueError(
                    f"The target '{target.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing target values are not supported."
                )
            if data[target.name].dtype.kind not in "iufb":
                raise TypeError(
                    f"The target '{target.name}' has non-numeric entries in the "
                    f"provided dataframe. Non-numeric target values are not supported."
                )

        # Check if all targets have valid values
        for param in self.parameters:
            if data[param.name].isna().any():
                raise ValueError(
                    f"The parameter '{param.name}' has missing values or NaNs in the "
                    f"provided dataframe. Missing parameter values are not supported."
                )
            if param.is_numerical and (data[param.name].dtype.kind not in "iufb"):
                raise TypeError(
                    f"The numerical parameter '{param.name}' has non-numeric entries in"
                    f" the provided dataframe."
                )

        # Update meta data
        # TODO: refactor responsibilities
        if self.searchspace.type in (SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID):
            self.searchspace.discrete.mark_as_measured(
                data, numerical_measurements_must_be_within_tolerance
            )

        # Read in measurements and add them to the database
        self.n_batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.n_batches_done
        to_insert["FitNr"] = np.nan

        self._measurements_exp = pd.concat(
            [self._measurements_exp, to_insert], axis=0, ignore_index=True
        )

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_ADD_RESULTS"], 1)
        telemetry_record_recommended_measurement_percentage(
            self._cached_recommendation,
            data,
            self.parameters,
            numerical_measurements_must_be_within_tolerance,
        )

    def recommend(
        self,
        batch_size: int,
        pending_experiments: pd.DataFrame | None = None,
        batch_quantity: int = None,  # type: ignore[assignment]
    ) -> pd.DataFrame:
        """Provide the recommendations for the next batch of experiments.

        Args:
            batch_size: Number of requested recommendations.
            pending_experiments: Parameter configurations specifying experiments
                that are currently pending.
            batch_quantity: Deprecated! Use ``batch_size`` instead.

        Returns:
            Dataframe containing the recommendations in experimental representation.

        Raises:
            ValueError: If ``batch_size`` is smaller than 1.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but provided "
                f"{batch_size=}."
            )

        # Invalidate cached recommendation if pending experiments are provided
        if (pending_experiments is not None) and (len(pending_experiments) > 0):
            self._cached_recommendation = pd.DataFrame()

        # If there are cached recommendations and the batch size of those is equal to
        # the previously requested one, we just return those
        if len(self._cached_recommendation) == batch_size:
            return self._cached_recommendation

        # Update recommendation meta data
        if len(self._measurements_exp) > 0:
            self.n_fits_done += 1
            self._measurements_exp.fillna({"FitNr": self.n_fits_done}, inplace=True)

        # Get the recommended search space entries
        rec = self.recommender.recommend(
            batch_size,
            self.searchspace,
            self.objective,
            self._measurements_exp,
            pending_experiments,
        )

        # Cache the recommendations
        self._cached_recommendation = rec.copy()

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_RECOMMEND"], 1)
        telemetry_record_value(TELEM_LABELS["BATCH_SIZE"], batch_size)

        return rec

    def posterior(self, candidates: pd.DataFrame) -> Posterior:
        """Get the posterior predictive distribution for the given candidates.

        The predictive distribution is based on the surrogate model of the last used
        recommender.

        Args:
            candidates: The candidate points in experimental recommendations.
                For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.

        Raises:
            IncompatibilityError: If the underlying surrogate model exposes no
                method for computing the posterior distribution.

        Returns:
            Posterior: The corresponding posterior object.
            For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        surrogate = self.get_surrogate()
        if not hasattr(surrogate, method_name := "posterior"):
            raise IncompatibilityError(
                f"The used surrogate type '{surrogate.__class__.__name__}' does not "
                f"provide a '{method_name}' method."
            )

        import torch

        with torch.no_grad():
            return surrogate.posterior(candidates)

    def get_surrogate(self) -> SurrogateProtocol:
        """Get the current surrogate model.

        Raises:
            RuntimeError: If the current recommender does not provide a surrogate model.

        Returns:
            Surrogate: The surrogate of the current recommender.

        Note:
            Currently, this method always returns the surrogate model with respect to
            the transformed target(s) / objective. This means that if you are using a
            ``SingleTargetObjective`` with a transformed target or a
            ``DesirabilityObjective``, the model's output will correspond to the
            transformed quantities and not the original untransformed target(s).
        """
        if self.objective is None:
            raise IncompatibilityError(
                f"No surrogate is available since no '{Objective.__name__}' is defined."
            )

        pure_recommender: RecommenderProtocol
        if isinstance(self.recommender, MetaRecommender):
            pure_recommender = self.recommender.get_current_recommender()
        else:
            pure_recommender = self.recommender

        if isinstance(pure_recommender, BayesianRecommender):
            return pure_recommender.get_surrogate(
                self.searchspace, self.objective, self.measurements
            )
        else:
            raise RuntimeError(
                f"The current recommender is of type "
                f"'{pure_recommender.__class__.__name__}', which does not provide "
                f"a surrogate model. Surrogate models are only available for "
                f"recommender subclasses of '{BayesianRecommender.__name__}'."
            )


def _add_version(dict_: dict) -> dict:
    """Add the package version to the given dictionary."""
    from baybe import __version__

    return {**dict_, "version": __version__}


def _drop_version(dict_: dict) -> dict:
    """Drop the package version from the given dictionary."""
    dict_.pop("version", None)
    return dict_


# Register de-/serialization hooks
unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    Campaign, converter, _cattrs_include_init_false=True
)
structure_hook = cattrs.gen.make_dict_structure_fn(
    Campaign, converter, _cattrs_include_init_false=True, _cattrs_forbid_extra_keys=True
)
converter.register_unstructure_hook(
    Campaign, lambda x: _add_version(unstructure_hook(x))
)
converter.register_structure_hook(
    Campaign, lambda d, cl: structure_hook(_drop_version(d), cl)
)


# Converter for config validation
_validation_converter = converter.copy()
_validation_converter.register_structure_hook(
    SearchSpace, validate_searchspace_from_config
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
