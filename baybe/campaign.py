"""Functionality for managing DOE campaigns. Main point of interaction via Python."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Collection, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar

import cattrs
import numpy as np
import pandas as pd
from attrs import Attribute, Factory, define, evolve, field, fields
from attrs.converters import optional
from attrs.validators import instance_of
from typing_extensions import override

from baybe.constraints.base import DiscreteConstraint
from baybe.exceptions import (
    IncompatibilityError,
    NoMeasurementsError,
    NotEnoughPointsLeftError,
)
from baybe.objectives.base import Objective, to_objective
from baybe.parameters.base import Parameter
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.base import MetaRecommender
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace._filtered import FilteredSubspaceDiscrete
from baybe.searchspace.core import (
    SearchSpace,
    SearchSpaceType,
    to_searchspace,
    validate_searchspace_from_config,
)
from baybe.serialization import SerialMixin, converter
from baybe.surrogates.base import PosteriorStatistic, SurrogateProtocol
from baybe.targets.base import Target
from baybe.utils.basic import UNSPECIFIED, UnspecifiedType, is_all_instance
from baybe.utils.boolean import eq_dataframe
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import (
    _ValidatedDataFrame,
    filter_df,
    fuzzy_row_match,
    normalize_input_dtypes,
)
from baybe.utils.validation import (
    validate_object_names,
    validate_objective_input,
    validate_parameter_input,
    validate_target_input,
)

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction
    from botorch.posteriors import Posterior

    _T = TypeVar("_T")

# Metadata columns
_RECOMMENDED = "recommended"
_MEASURED = "measured"
_EXCLUDED = "excluded"
_METADATA_COLUMNS = [_RECOMMENDED, _MEASURED, _EXCLUDED]


def _make_allow_flag_default_factory(
    default: bool,
) -> Callable[[Campaign], bool | UnspecifiedType]:
    """Make a default factory for allow_* flags."""

    def default_allow_flag(campaign: Campaign) -> bool | UnspecifiedType:
        """Attrs-compatible default factory for allow_* flags."""
        if campaign.searchspace.type is SearchSpaceType.DISCRETE:
            return default
        return UNSPECIFIED

    return default_allow_flag


def _set_with_cache_cleared(instance: Campaign, attribute: Attribute, value: _T) -> _T:
    """Attrs-compatible hook to clear the cache when changing an attribute."""
    if value != getattr(instance, attribute.name):
        instance.clear_cache()
    return value


def _validate_allow_flag(campaign: Campaign, attribute: Attribute, value: Any) -> None:
    """Attrs-compatible validator for context-aware validation of allow_* flags."""
    match campaign.searchspace.type:
        case SearchSpaceType.DISCRETE:
            if not isinstance(value, bool):
                raise ValueError(
                    f"For search spaces of '{SearchSpaceType.DISCRETE}', "
                    f"'{attribute.name}' must be a Boolean."
                )
        case _:
            if value is not UNSPECIFIED:
                raise ValueError(
                    f"For search spaces of type other than "
                    f"'{SearchSpaceType.DISCRETE}', '{attribute.name}' cannot be set "
                    f"since the flag is meaningless in such contexts.",
                )


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

    @objective.validator
    def _validate_objective(  # noqa: DOC101, DOC103
        self, _: Any, value: Objective | None
    ) -> None:
        """Validate no overlapping names between targets and parameters."""
        if value is None:
            return

        validate_object_names(self.searchspace.parameters + value.targets)

    recommender: RecommenderProtocol = field(
        factory=TwoPhaseMetaRecommender,
        validator=instance_of(RecommenderProtocol),
        on_setattr=_set_with_cache_cleared,
    )
    """The employed recommender"""

    allow_recommending_already_measured: bool | UnspecifiedType = field(
        default=Factory(
            _make_allow_flag_default_factory(default=True), takes_self=True
        ),
        validator=_validate_allow_flag,
        on_setattr=_set_with_cache_cleared,
        kw_only=True,
    )
    """Allow to recommend experiments that were already measured earlier.
    Can only be set for discrete search spaces."""

    allow_recommending_already_recommended: bool | UnspecifiedType = field(
        default=Factory(
            _make_allow_flag_default_factory(default=False), takes_self=True
        ),
        validator=_validate_allow_flag,
        on_setattr=_set_with_cache_cleared,
        kw_only=True,
    )
    """Allow to recommend experiments that were already recommended earlier.
    Can only be set for discrete search spaces."""

    allow_recommending_pending_experiments: bool | UnspecifiedType = field(
        default=Factory(
            _make_allow_flag_default_factory(default=False), takes_self=True
        ),
        validator=_validate_allow_flag,
        on_setattr=_set_with_cache_cleared,
        kw_only=True,
    )
    """Allow pending experiments to be part of the recommendations.
    Can only be set for discrete search spaces."""

    # Metadata
    _searchspace_metadata: pd.DataFrame = field(init=False, eq=eq_dataframe)
    """Metadata tracking the experimentation status of the search space."""

    n_batches_done: int = field(default=0, init=False)
    """The number of already processed batches."""

    n_fits_done: int = field(default=0, init=False)
    """The number of fits already done."""

    # Private
    _measurements_exp: pd.DataFrame = field(
        factory=pd.DataFrame, eq=eq_dataframe, init=False
    )
    """The experimental representation of the conducted experiments."""

    _cached_recommendation: pd.DataFrame | None = field(
        default=None, init=False, eq=False
    )
    """The cached recommendations."""

    @_searchspace_metadata.default
    def _default_searchspace_metadata(self) -> pd.DataFrame:
        """Create a fresh metadata object."""
        df = pd.DataFrame(
            False,
            index=self.searchspace.discrete.exp_rep.index,
            columns=_METADATA_COLUMNS,
        )

        return df

    @override
    def __str__(self) -> str:
        recommended_count = sum(self._searchspace_metadata[_RECOMMENDED])
        measured_count = sum(self._searchspace_metadata[_MEASURED])
        excluded_count = sum(self._searchspace_metadata[_EXCLUDED])
        n_elements = len(self._searchspace_metadata)
        searchspace_fields = [
            to_string(
                "Recommended:",
                f"{recommended_count}/{n_elements}",
                single_line=True,
            ),
            to_string(
                "Measured:",
                f"{measured_count}/{n_elements}",
                single_line=True,
            ),
            to_string(
                "Excluded:",
                f"{excluded_count}/{n_elements}",
                single_line=True,
            ),
        ]
        metadata_fields = [
            to_string("Batches done", self.n_batches_done, single_line=True),
            to_string("Fits done", self.n_fits_done, single_line=True),
            to_string("Discrete Subspace Meta Data", *searchspace_fields),
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

    def _cache_recommendation(self, df: pd.DataFrame, /) -> None:
        """Cache the given recommendation."""
        self._cached_recommendation = df.copy()

    def clear_cache(self) -> None:
        """Clear the internal recommendation cache."""
        self._cached_recommendation = None

    def add_measurements(
        self,
        data: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Add results from a dataframe to the internal database.

        Each addition of data is considered a new batch. Added results are checked for
        validity. Categorical values need to have an exact match. For numerical values,
        a campaign flag determines if values that lie outside a specified tolerance
        are accepted. Possible validation exceptions are documented in
        :func:`baybe.utils.validation.validate_target_input` and
        :func:`baybe.utils.validation.validate_parameter_input`.

        Args:
            data: The data to be added (with filled values for targets). Preferably
                created via :func:`baybe.campaign.Campaign.recommend`.
            numerical_measurements_must_be_within_tolerance: Flag indicating if
                numerical parameters need to be within their tolerances.
        """
        # Validate target and parameter input values
        validate_target_input(data, self.targets)
        if self.objective is not None:
            validate_objective_input(data, self.objective)
        validate_parameter_input(
            data, self.parameters, numerical_measurements_must_be_within_tolerance
        )
        data = normalize_input_dtypes(data, self.parameters + self.targets)
        data.__class__ = _ValidatedDataFrame

        # With new measurements, the recommendations must always be recomputed
        self.clear_cache()

        # Read in measurements and add them to the database
        self.n_batches_done += 1
        to_insert = data.copy()
        to_insert["BatchNr"] = self.n_batches_done
        to_insert["FitNr"] = np.nan

        self._measurements_exp = pd.concat(
            [self._measurements_exp, to_insert], axis=0, ignore_index=True
        )

        # Update metadata
        if self.searchspace.type in (SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID):
            idxs_matched = fuzzy_row_match(
                self.searchspace.discrete.exp_rep, data, self.parameters
            )
            self._searchspace_metadata.loc[idxs_matched, _MEASURED] = True

    def update_measurements(
        self,
        data: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Update previously added measurements.

        This can be useful to correct mistakes or update target measurements. The
        match to existing data entries is made based on the index. This will reset
        the `FitNr` of the corresponding measurement and reset cached recommendations.

        Args:
            data: The measurement data to be updated (with filled values for targets).
            numerical_measurements_must_be_within_tolerance: Flag indicating if
                numerical parameters need to be within their tolerances.

        Raises:
            ValueError: If the given data contains duplicated indices.
            ValueError: If the given data contains indices not present in existing
                measurements.
        """
        # Validate target and parameter input values
        validate_target_input(data, self.targets)
        if self.objective is not None:
            validate_objective_input(data, self.objective)
        validate_parameter_input(
            data, self.parameters, numerical_measurements_must_be_within_tolerance
        )
        data = normalize_input_dtypes(data, self.parameters + self.targets)
        data.__class__ = _ValidatedDataFrame

        # With changed measurements, the recommendations must always be recomputed
        self.clear_cache()

        # Block duplicate input indices
        if data.index.has_duplicates:
            raise ValueError(
                "The input dataframe containing the measurement updates has duplicated "
                "indices. Please ensure that all updates for a given measurement are "
                "made in a single combined entry."
            )

        # Allow only existing indices
        if nonmatching_idxs := set(data.index).difference(self._measurements_exp.index):
            raise ValueError(
                f"Updating measurements requires indices matching the "
                f"existing measurements. The following indices were in the input, but "
                f"are not found in the existing entries: {nonmatching_idxs}"
            )

        # Perform the update
        cols = [p.name for p in self.parameters] + [t.name for t in self.targets]
        self._measurements_exp.loc[data.index, cols] = data[cols]

        # Reset fit number
        self._measurements_exp.loc[data.index, "FitNr"] = np.nan

    def toggle_discrete_candidates(  # noqa: DOC501
        self,
        constraints: Collection[DiscreteConstraint] | pd.DataFrame,
        exclude: bool,
        complement: bool = False,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """In-/exclude certain discrete points in/from the candidate set.

        Args:
            constraints: A filtering mechanism determining the candidates subset to be
                in-/excluded. Can be either a collection of
                :class:`~baybe.constraints.base.DiscreteConstraint` or a dataframe.
                For the latter, see :func:`~baybe.utils.dataframe.filter_df`
                for details.
            exclude: If ``True``, the specified candidates are excluded.
                If ``False``, the candidates are considered for recommendation.
            complement: If ``True``, the filtering mechanism is inverted so that
                the complement of the candidate subset specified by the filter is
                toggled. For details, see :func:`~baybe.utils.dataframe.filter_df`.
            dry_run: If ``True``, the target subset is only extracted but not
                affected. If ``False``, the candidate set is updated correspondingly.
                Useful for setting up the correct filtering mechanism.

        Returns:
            A new dataframe containing the  discrete candidate set passing through the
            specified filter.
        """
        # IMPROVE: The cache invalidation could be made more fine-grained:
        #   * When including points, the cache only needs to be cleared if the active
        #    search space gets *actually* larger (i.e. including already included
        #    points does not change the situation).
        #  * When excluding points, the cache only needs to be cleared if the excluded
        #    points were part of the cached recommendations.
        #  * Additional shortcuts might be possible.
        self.clear_cache()

        df = self.searchspace.discrete.exp_rep

        if isinstance(constraints, pd.DataFrame):
            # Determine the candidate subset to be toggled
            points = filter_df(df, constraints, complement)

        elif isinstance(constraints, Collection) and is_all_instance(
            constraints, DiscreteConstraint
        ):
            # TODO: Should be taken over by upcoming `SubspaceDiscrete.filter` method,
            #   automatically choosing the appropriate backend (polars/pandas/...)

            # Filter the search space dataframe according to the given constraint
            idx = reduce(
                lambda x, y: x.intersection(y), (c.get_valid(df) for c in constraints)
            )

            # Determine the candidate subset to be toggled
            points = df.drop(index=idx) if complement else df.loc[idx].copy()

        else:
            raise TypeError(
                "Candidate toggling is not implemented for the given type of "
                "constraint specifications."
            )

        if not dry_run:
            self._searchspace_metadata.loc[points.index, _EXCLUDED] = exclude

        return points

    def recommend(
        self,
        batch_size: int,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Provide the recommendations for the next batch of experiments.

        Args:
            batch_size: Number of requested recommendations.
            pending_experiments: Parameter configurations specifying experiments
                that are currently pending.

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

        # IMPROVE: Currently, we simply invalidate the cache whenever pending
        #     experiments are provided, because in order to use it, we need to check if
        #     the previous call was done with the same pending experiments.

        if pending_experiments is not None:
            self.clear_cache()

            validate_parameter_input(pending_experiments, self.parameters)
            pending_experiments = normalize_input_dtypes(
                pending_experiments, self.parameters
            )
            pending_experiments.__class__ = _ValidatedDataFrame

        if (
            pending_experiments is None
            and (cache := self._cached_recommendation) is not None
            and self.allow_recommending_already_recommended
            and len(cache) == batch_size
        ):
            return cache

        # Update recommendation meta data
        if len(self._measurements_exp) > 0:
            self.n_fits_done += 1
            self._measurements_exp.fillna({"FitNr": self.n_fits_done}, inplace=True)

        # Prepare the search space according to the current campaign state
        if self.searchspace.type is SearchSpaceType.DISCRETE:
            # TODO: This implementation should at some point be hidden behind an
            #   appropriate public interface, like `SubspaceDiscrete.filter()`
            mask_todrop = self._searchspace_metadata[_EXCLUDED].astype(bool)
            if not self.allow_recommending_already_recommended:
                mask_todrop |= self._searchspace_metadata[_RECOMMENDED]
            if not self.allow_recommending_already_measured:
                mask_todrop |= self._searchspace_metadata[_MEASURED]
            if (
                not self.allow_recommending_pending_experiments
                and pending_experiments is not None
            ):
                mask_todrop |= pd.merge(
                    self.searchspace.discrete.exp_rep,
                    pending_experiments,
                    indicator=True,
                    how="left",
                )["_merge"].eq("both")
            searchspace = evolve(
                self.searchspace,
                discrete=FilteredSubspaceDiscrete.from_subspace(
                    self.searchspace.discrete, ~mask_todrop.to_numpy()
                ),
            )
        else:
            searchspace = self.searchspace

        # Pending experiments should not be passed to non-predictive recommenders
        # to avoid complaints about unused arguments, so we need to know of what
        # type the next recommender will be
        recommender = self.recommender
        if isinstance(recommender, MetaRecommender):
            recommender = recommender.get_non_meta_recommender(
                batch_size,
                searchspace,
                self.objective,
                self._measurements_exp,
                pending_experiments,
            )
        is_nonpredictive = isinstance(recommender, NonPredictiveRecommender)

        # Get the recommended search space entries
        try:
            # NOTE: The `recommend` call must happen on `self.recommender` to update
            #   potential inner states in case of meta recommenders!
            rec = self.recommender.recommend(
                batch_size,
                searchspace,
                self.objective,
                self._measurements_exp,
                None if is_nonpredictive else pending_experiments,
            )
        except NotEnoughPointsLeftError as ex:
            # Aliases for code compactness
            f = fields(Campaign)
            ok_m = self.allow_recommending_already_measured
            ok_r = self.allow_recommending_already_recommended
            ok_p = self.allow_recommending_pending_experiments
            ok_m_name = f.allow_recommending_already_measured.name
            ok_r_name = f.allow_recommending_already_recommended.name
            ok_p_name = f.allow_recommending_pending_experiments.name
            no_blocked_pending_points = ok_p or (pending_experiments is None)

            # If there are no candidate restrictions to be relaxed
            if ok_m and ok_r and no_blocked_pending_points:
                raise ex

            # Otherwise, extract possible relaxations
            solution = [
                f"'{name}=True'"
                for name, value in [
                    (ok_m_name, ok_m),
                    (ok_r_name, ok_r),
                    (ok_p_name, no_blocked_pending_points),
                ]
                if not value
            ]
            message = solution[0] if len(solution) == 1 else " and/or ".join(solution)
            raise NotEnoughPointsLeftError(
                f"{str(ex)} Consider setting {message}."
            ) from ex

        if pending_experiments is None:  # see IMPROVE comment above
            self._cache_recommendation(rec)

        # Update metadata
        if self.searchspace.type in (SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID):
            self._searchspace_metadata.loc[rec.index, _RECOMMENDED] = True

        return rec

    def posterior(self, candidates: pd.DataFrame | None = None) -> Posterior:
        """Get the posterior predictive distribution for the given candidates.

        Args:
            candidates: The candidate points in experimental recommendations. If not
                provided, the posterior for the existing campaign measurements is
                returned. For details, see
                :meth:`baybe.surrogates.base.Surrogate.posterior`.

        Raises:
            IncompatibilityError: If the underlying surrogate model exposes no
                method for computing the posterior distribution.

        Returns:
            Posterior: The corresponding posterior object.
            For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        if candidates is None:
            candidates = self.measurements[[p.name for p in self.parameters]]

        surrogate = self.get_surrogate()
        if not hasattr(surrogate, method_name := "posterior"):
            raise IncompatibilityError(
                f"The used surrogate type '{surrogate.__class__.__name__}' does not "
                f"provide a '{method_name}' method."
            )

        return surrogate.posterior(candidates)

    def posterior_stats(
        self,
        candidates: pd.DataFrame | None = None,
        stats: Sequence[PosteriorStatistic] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Return posterior statistics for each target.

        Args:
            candidates: The candidate points in experimental representation. If not
                provided, the statistics of the existing campaign measurements are
                calculated. For details, see
                :meth:`baybe.surrogates.base.Surrogate.posterior_stats`.
            stats: Sequence indicating which statistics to compute. Also accepts
                floats, for which the corresponding quantile point will be computed.

        Raises:
            ValueError: If a requested quantile is outside the open interval (0,1).
            TypeError: If the posterior utilized by the surrogate does not support
                a requested statistic.

        Returns:
            A dataframe with posterior statistics for each target and candidate.
        """
        if candidates is None:
            if self.measurements.empty:
                raise NoMeasurementsError(
                    f"No candidates were provided and the campaign has no measurements "
                    f"yet. '{self.posterior_stats.__name__}' has no candidates to "
                    f"compute statistics for in this case."
                )

            candidates = self.measurements[[p.name for p in self.parameters]]

        surrogate = self.get_surrogate()
        if not hasattr(surrogate, method_name := "posterior_stats"):
            raise IncompatibilityError(
                f"The used surrogate type '{surrogate.__class__.__name__}' does not "
                f"provide a '{method_name}' method."
            )

        return surrogate.posterior_stats(candidates, stats)

    def get_surrogate(
        self,
        batch_size: int | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> SurrogateProtocol:
        """Get the current surrogate model.

        Args:
            batch_size: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.
            pending_experiments: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.

        Raises:
            IncompatibilityError: If the current recommender does not provide a
                surrogate model.

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

        recommender = self._get_non_meta_recommender(batch_size, pending_experiments)
        if isinstance(recommender, BayesianRecommender):
            return recommender.get_surrogate(
                self.searchspace, self.objective, self.measurements
            )
        else:
            raise IncompatibilityError(
                f"The current recommender is of type "
                f"'{recommender.__class__.__name__}', which does not provide "
                f"a surrogate model. Surrogate models are only available for "
                f"recommender subclasses of '{BayesianRecommender.__name__}'."
            )

    def _get_non_meta_recommender(
        self,
        batch_size: int | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> RecommenderProtocol:
        """Get the current recommender.

        Args:
            batch_size: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.
            pending_experiments: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.

        Returns:
            The recommender for the current recommendation context.
        """
        if not isinstance(self.recommender, MetaRecommender):
            return self.recommender
        return self.recommender.get_non_meta_recommender(
            batch_size,
            self.searchspace,
            self.objective,
            self.measurements,
            pending_experiments,
        )

    def _get_bayesian_recommender(
        self,
        batch_size: int | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> BayesianRecommender:
        """Get the current Bayesian recommender (if available).

        For details on the method arguments, see :meth:`_get_non_meta_recommender`.
        """
        recommender = self._get_non_meta_recommender(batch_size, pending_experiments)
        if not isinstance(recommender, BayesianRecommender):
            raise IncompatibilityError(
                f"The current recommender is of type "
                f"'{recommender.__class__.__name__}', which does not provide "
                f"a surrogate model or acquisition values. Both objects are "
                f"only available for recommender subclasses of "
                f"'{BayesianRecommender.__name__}'."
            )
        return recommender

    def get_acquisition_function(
        self,
        batch_size: int | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> AcquisitionFunction:
        """Get the current acquisition function.

        Args:
            batch_size: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.
            pending_experiments: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.

        Raises:
            IncompatibilityError: If no objective has been specified.
            IncompatibilityError: If the current recommender does not use an acquisition
                function.

        Returns:
            The acquisition function of the current recommender.
        """
        if self.objective is None:
            raise IncompatibilityError(
                "Acquisition values can only be computed if an objective has "
                "been defined."
            )

        recommender = self._get_bayesian_recommender(batch_size, pending_experiments)
        return recommender.get_acquisition_function(
            self.searchspace, self.objective, self.measurements, pending_experiments
        )

    def acquisition_values(
        self,
        candidates: pd.DataFrame,
        acquisition_function: AcquisitionFunction | None = None,
        *,
        batch_size: int | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Compute the acquisition values for the given candidates.

        Args:
            candidates: The candidate points in experimental representation.
                For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
            acquisition_function: The acquisition function to be evaluated.
                If not provided, the acquisition function of the recommender is used.
            batch_size: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.
            pending_experiments: See :meth:`recommend`.
                Only required when using meta recommenders that demand it.

        Returns:
            A series of individual acquisition values, one for each candidate.
        """
        recommender = self._get_bayesian_recommender(batch_size, pending_experiments)
        assert self.objective is not None
        return recommender.acquisition_values(
            candidates,
            self.searchspace,
            self.objective,
            self.measurements,
            pending_experiments,
            acquisition_function,
        )

    def joint_acquisition_value(  # noqa: DOC101, DOC103
        self,
        candidates: pd.DataFrame,
        acquisition_function: AcquisitionFunction | None = None,
        *,
        batch_size: int | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> float:
        """Compute the joint acquisition values for the given candidate batch.

        For details on the method arguments, see :meth:`acquisition_values`.

        Returns:
            The joint acquisition value of the batch.
        """
        recommender = self._get_bayesian_recommender(batch_size, pending_experiments)
        assert self.objective is not None
        return recommender.joint_acquisition_value(
            candidates,
            self.searchspace,
            self.objective,
            self.measurements,
            pending_experiments,
            acquisition_function,
        )


def _add_version(dict_: dict) -> dict:
    """Add the package version to the given dictionary."""
    from baybe import __version__

    return {**dict_, "version": __version__}


def _drop_version(dict_: dict) -> dict:
    """Drop the package version from the given dictionary."""
    dict_.pop("version", None)
    return dict_


# Register (un-)structure hooks
unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    Campaign, converter, _cattrs_include_init_false=True
)
structure_hook = cattrs.gen.make_dict_structure_fn(
    Campaign, converter, _cattrs_include_init_false=True
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
