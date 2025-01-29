"""Functionality for managing DOE campaigns. Main point of interaction via Python."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Collection
from functools import reduce
from typing import TYPE_CHECKING, Any

import cattrs
import numpy as np
import pandas as pd
from attrs import Attribute, Factory, define, evolve, field, fields
from attrs.converters import optional
from attrs.validators import instance_of
from typing_extensions import override

from baybe.constraints.base import DiscreteConstraint
from baybe.exceptions import IncompatibilityError, NotEnoughPointsLeftError
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
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.base import Target
from baybe.telemetry import (
    TELEM_LABELS,
    telemetry_record_recommended_measurement_percentage,
    telemetry_record_value,
)
from baybe.utils.basic import UNSPECIFIED, UnspecifiedType, is_all_instance
from baybe.utils.boolean import eq_dataframe
from baybe.utils.dataframe import filter_df, fuzzy_row_match
from baybe.utils.plotting import to_string

if TYPE_CHECKING:
    from botorch.posteriors import Posterior

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

    recommender: RecommenderProtocol = field(
        factory=TwoPhaseMetaRecommender,
        validator=instance_of(RecommenderProtocol),
    )
    """The employed recommender"""

    allow_recommending_already_measured: bool | UnspecifiedType = field(
        default=Factory(
            _make_allow_flag_default_factory(default=True), takes_self=True
        ),
        validator=_validate_allow_flag,
        kw_only=True,
    )
    """Allow to recommend experiments that were already measured earlier.
    Can only be set for discrete search spaces."""

    allow_recommending_already_recommended: bool | UnspecifiedType = field(
        default=Factory(
            _make_allow_flag_default_factory(default=False), takes_self=True
        ),
        validator=_validate_allow_flag,
        kw_only=True,
    )
    """Allow to recommend experiments that were already recommended earlier.
    Can only be set for discrete search spaces."""

    allow_recommending_pending_experiments: bool | UnspecifiedType = field(
        default=Factory(
            _make_allow_flag_default_factory(default=False), takes_self=True
        ),
        validator=_validate_allow_flag,
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

    _cached_recommendation: pd.DataFrame = field(
        factory=pd.DataFrame, eq=eq_dataframe, init=False
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
        df.loc[:, _EXCLUDED] = self.searchspace.discrete._excluded
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
                self.searchspace.discrete.exp_rep,
                data,
                self.parameters,
                numerical_measurements_must_be_within_tolerance,
            )
            self._searchspace_metadata.loc[idxs_matched, _MEASURED] = True

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_ADD_RESULTS"], 1)
        telemetry_record_recommended_measurement_percentage(
            self._cached_recommendation,
            data,
            self.parameters,
            numerical_measurements_must_be_within_tolerance,
        )

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
        # Clear cache
        self._cached_recommendation = pd.DataFrame()

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

        # Prepare the search space according to the current campaign state
        if self.searchspace.type is SearchSpaceType.DISCRETE:
            # TODO: This implementation should at some point be hidden behind an
            #   appropriate public interface, like `SubspaceDiscrete.filter()`
            mask_todrop = self._searchspace_metadata[_EXCLUDED].copy()
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

        # Cache the recommendations
        self._cached_recommendation = rec.copy()

        # Update metadata
        if self.searchspace.type in (SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID):
            self._searchspace_metadata.loc[rec.index, _RECOMMENDED] = True

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_RECOMMEND"], 1)
        telemetry_record_value(TELEM_LABELS["BATCH_SIZE"], batch_size)

        return rec

    def posterior(self, candidates: pd.DataFrame) -> Posterior:
        """Get the posterior predictive distribution for the given candidates.

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

        recommender: RecommenderProtocol
        if isinstance(self.recommender, MetaRecommender):
            recommender = self.recommender.get_non_meta_recommender(
                batch_size,
                self.searchspace,
                self.objective,
                self.measurements,
                pending_experiments,
            )
        else:
            recommender = self.recommender

        if isinstance(recommender, BayesianRecommender):
            return recommender.get_surrogate(
                self.searchspace, self.objective, self.measurements
            )
        else:
            raise RuntimeError(
                f"The current recommender is of type "
                f"'{recommender.__class__.__name__}', which does not provide "
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


# Register (un-)structure hooks
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
