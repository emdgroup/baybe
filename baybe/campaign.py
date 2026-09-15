"""Functionality for managing DOE campaigns. Main point of interaction via Python."""

from __future__ import annotations

import gc
import json
import warnings
from collections.abc import Collection, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast

import cattrs
import narwhals.stable.v2 as nw
import pandas as pd
from attrs import Attribute, cmp_using, define, evolve, field, fields, setters
from attrs.converters import optional
from attrs.validators import instance_of
from narwhals.stable.v2.dependencies import is_into_dataframe
from narwhals.stable.v2.typing import IntoDataFrame, IntoDataFrameT, IntoSeries
from typing_extensions import override

from baybe.constraints.base import DiscreteConstraint, DiscreteFilteringConstraint
from baybe.exceptions import (
    DeprecationError,
    IncompatibilityError,
    NoMeasurementsError,
    NotEnoughPointsLeftError,
    NothingToComputeError,
)
from baybe.objectives.base import Objective, to_objective
from baybe.parameters.base import Parameter
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.base import MetaRecommender
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace.candidates import TableCandidates
from baybe.searchspace.core import (
    SearchSpace,
    SearchSpaceType,
    to_searchspace,
    validate_searchspace_from_config,
)
from baybe.serialization import SerialMixin, converter
from baybe.settings import Settings, active_settings
from baybe.surrogates.base import PosteriorStatistic, SurrogateProtocol
from baybe.targets.base import Target
from baybe.utils.basic import is_all_instance
from baybe.utils.boolean import AutoBool
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import (
    _df_equals,
    _df_with_backend,
    _infer_backend,
    filter_df,
    fuzzy_row_match,
)
from baybe.utils.validation import (
    preprocess_dataframe,
    validate_object_names,
    validate_target_input,
)

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BoAcquisitionFunction
    from botorch.posteriors import Posterior

    from baybe.acquisition.base import AcquisitionFunction

    _T = TypeVar("_T")

# Legacy constants kept for deserialization migration only
_EXCLUDED = "excluded"
_MEASURED = "measured"
_RECOMMENDED = "recommended"


def _set_with_cache_cleared(instance: Campaign, attribute: Attribute, value: _T) -> _T:
    """Attrs-compatible hook to clear the cache when changing an attribute."""
    if value != getattr(instance, attribute.name):
        instance.clear_cache()
    return value


_convert_validate_and_clear_cache = setters.pipe(
    setters.convert, setters.validate, _set_with_cache_cleared
)
"""Attrs on_setattr hook that converts, validates, and clears the cache on changes."""


def _validate_allow_flag(
    campaign: Campaign, attribute: Attribute, value: AutoBool
) -> None:
    """Attrs-compatible validator for context-aware validation of allow_* flags."""
    if campaign.searchspace.type is SearchSpaceType.DISCRETE:
        return

    if value is AutoBool.FALSE:
        raise IncompatibilityError(
            f"For search spaces involving a continuous subspace, the flag "
            f"'{attribute.alias}' cannot be set to 'False' for algorithmic reasons. "
            f"Either let the value be automatically determined by not setting it "
            f"explicitly / setting it to 'auto' or explicitly set it to 'True'."
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
        on_setattr=_convert_validate_and_clear_cache,
    )
    """The employed recommender"""

    _allow_recommending_already_measured: AutoBool = field(
        alias="allow_recommending_already_measured",
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
        validator=_validate_allow_flag,
        on_setattr=_convert_validate_and_clear_cache,
        kw_only=True,
    )
    """Allow recommending experiments that have already been measured."""

    _allow_recommending_already_recommended: AutoBool = field(
        alias="allow_recommending_already_recommended",
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
        validator=_validate_allow_flag,
        on_setattr=_convert_validate_and_clear_cache,
        kw_only=True,
    )
    """Allow recommending experiments that have already been recommended."""

    _allow_recommending_pending_experiments: AutoBool = field(
        alias="allow_recommending_pending_experiments",
        default=AutoBool.AUTO,
        converter=AutoBool.from_unstructured,  # type: ignore[misc]
        validator=_validate_allow_flag,
        on_setattr=_convert_validate_and_clear_cache,
        kw_only=True,
    )
    """Allow recommending pending experiments."""

    # Private
    _excluded_experiments: nw.DataFrame = field(eq=cmp_using(_df_equals), init=False)
    """The parameter configurations that have been excluded from recommendations."""

    _measurements: list[nw.DataFrame] = field(
        factory=list,
        eq=cmp_using(lambda a, b: len(a) == len(b) and all(map(_df_equals, a, b))),
        init=False,
    )
    """The measurements added to the campaign, one frame per batch."""

    _recommended_experiments: nw.DataFrame = field(eq=cmp_using(_df_equals), init=False)
    """The (deduplicated) parameter configurations that have been recommended."""

    _cached_recommendation: nw.DataFrame | None = field(
        default=None, init=False, eq=False
    )
    """The cached recommendations."""

    @_excluded_experiments.default
    def _default_excluded_experiments(self) -> nw.DataFrame:
        """Create an empty excluded experiments DataFrame with correct schema."""
        cols = [p.name for p in self.searchspace.parameters]
        return nw.from_dict(
            {c: [] for c in cols},
            backend=active_settings.default_dataframe_backend,
        )

    @_recommended_experiments.default
    def _default_recommended_experiments(self) -> nw.DataFrame:
        """Create an empty recommended experiments DataFrame with correct schema."""
        cols = [p.name for p in self.searchspace.parameters]
        return nw.from_dict(
            {c: [] for c in cols},
            backend=active_settings.default_dataframe_backend,
        )

    @override
    def __str__(self) -> str:
        fields = [self.searchspace, self.objective, self.recommender]
        return to_string(self.__class__.__name__, *fields)

    @property
    def measurements(self) -> IntoDataFrame:
        """The experimental data added to the Campaign."""
        if not self._measurements:
            cols = [p.name for p in self.searchspace.parameters] + [
                t.name for t in (self.objective.targets if self.objective else ())
            ]
            return nw.from_dict(
                {c: [] for c in cols},
                backend=active_settings.default_dataframe_backend,
            ).to_native()

        return nw.concat(self._measurements, how="vertical").to_native()

    @property
    def n_batches_done(self) -> NoReturn:
        """Deprecated!"""
        raise DeprecationError("'n_batches_done' is no longer available.")

    @property
    def n_fits_done(self) -> NoReturn:
        """Deprecated!"""
        raise DeprecationError("'n_fits_done' is no longer available.")

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """The parameters of the underlying search space."""
        return self.searchspace.parameters

    @property
    def targets(self) -> tuple[Target, ...]:
        """The targets of the underlying objective."""
        return self.objective.targets if self.objective is not None else ()

    @property
    def allow_recommending_already_measured(self) -> bool:
        """Allow recommending experiments that have already been measured."""
        if self._allow_recommending_already_measured is AutoBool.AUTO:
            return True
        return bool(self._allow_recommending_already_measured)

    @allow_recommending_already_measured.setter
    def allow_recommending_already_measured(self, value: bool) -> None:
        """Set candidate flag for already measured experiments."""
        # Note: uses attrs converter
        self._allow_recommending_already_measured = value  # type: ignore[assignment]

    @property
    def allow_recommending_already_recommended(self) -> bool:
        """Allow recommending experiments that have already been recommended."""
        if self._allow_recommending_already_recommended is AutoBool.AUTO:
            return self.searchspace.type is not SearchSpaceType.DISCRETE
        return bool(self._allow_recommending_already_recommended)

    @allow_recommending_already_recommended.setter
    def allow_recommending_already_recommended(self, value: bool) -> None:
        """Set candidate flag for already recommended experiments."""
        # Note: uses attrs converter
        self._allow_recommending_already_recommended = value  # type: ignore[assignment]

    @property
    def allow_recommending_pending_experiments(self) -> bool:
        """Allow recommending pending experiments."""
        if self._allow_recommending_pending_experiments is AutoBool.AUTO:
            return self.searchspace.type is not SearchSpaceType.DISCRETE
        return bool(self._allow_recommending_pending_experiments)

    @allow_recommending_pending_experiments.setter
    def allow_recommending_pending_experiments(self, value: bool) -> None:
        """Set candidate flag for pending experiments."""
        # Note: uses attrs converter
        self._allow_recommending_pending_experiments = value  # type: ignore[assignment]

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

    def _cache_recommendation(self, df: nw.DataFrame, /) -> None:
        """Cache the given recommendation."""
        self._cached_recommendation = df

    def clear_cache(self) -> None:
        """Clear the internal recommendation cache."""
        self._cached_recommendation = None

    def add_measurements(
        self,
        data: IntoDataFrame,
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
        # Preprocess incoming data
        data_nw = nw.from_native(
            preprocess_dataframe(
                data,
                self.searchspace,
                self.objective,
                numerical_measurements_must_be_within_tolerance,
            ),
            eager_only=True,
        )

        # With new measurements, the recommendations must always be recomputed
        self.clear_cache()

        # Append the new batch
        self._measurements.append(data_nw)

    def update_measurements(
        self,
        data: IntoDataFrame,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Update previously added measurements."""
        raise NotImplementedError(
            f"'{self.update_measurements.__name__}' is temporarily unavailable."
        )

    def toggle_discrete_candidates(  # noqa: DOC501
        self,
        constraints: Collection[DiscreteConstraint] | IntoDataFrame,
        exclude: bool,
        complement: bool = False,
        dry_run: bool = False,
    ) -> IntoDataFrame:
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
            A new dataframe containing the discrete candidate set passing through the
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

        df = self.searchspace.discrete._get_candidates().collect().to_pandas()

        if is_into_dataframe(constraints):
            # Determine the candidate subset to be toggled
            points = filter_df(
                df,
                # TODO[typing]: https://github.com/facebook/pyrefly/issues/4849
                nw.from_native(constraints, eager_only=True).to_pandas(),  # pyrefly: ignore[no-matching-overload]
                complement,
            )

        elif isinstance(constraints, Collection) and is_all_instance(
            constraints, DiscreteFilteringConstraint
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

        points_nw = nw.from_native(points, eager_only=True)

        if not dry_run:
            if exclude and not points_nw.is_empty():
                # Add the toggled points (avoid duplicates)
                concatenated = nw.concat(
                    [self._excluded_experiments, points_nw], how="vertical"
                )
                self._excluded_experiments = concatenated.unique()
            elif not exclude and not self._excluded_experiments.is_empty():
                # Remove the re-included points
                self._excluded_experiments = self._excluded_experiments.join(
                    points_nw, on=points_nw.columns, how="anti"
                )

        return points_nw.to_native()

    def recommend(
        self,
        batch_size: int,
        pending_experiments: IntoDataFrameT | None = None,
    ) -> IntoDataFrameT:
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

        backend = _infer_backend(pending_experiments)

        # IMPROVE: Currently, we simply invalidate the cache whenever pending
        #     experiments are provided, because in order to use it, we need to check if
        #     the previous call was done with the same pending experiments.
        if pending_experiments is not None:
            self.clear_cache()

        # Preprocess pending experiments
        pending_experiments_pd: pd.DataFrame | None = (
            nw.from_native(
                preprocess_dataframe(
                    pending_experiments,
                    self.searchspace,
                    numerical_measurements_must_be_within_tolerance=False,
                ),
                eager_only=True,
            ).to_pandas()
            if pending_experiments is not None
            else None
        )

        # TODO: Proper fix for the allow_* flags required
        if (
            active_settings.cache_campaign_recommendations
            and (cache := self._cached_recommendation) is not None
            and pending_experiments_pd is None
            and self.allow_recommending_already_recommended
            and len(cache) == batch_size
        ):
            # TODO: Potentially the cast becomes obsolete once Campaign is generic
            return cast(IntoDataFrameT, cache)

        # Prepare the search space according to the current campaign state
        if self.searchspace.type is SearchSpaceType.DISCRETE:
            # TODO: This implementation should at some point be hidden behind an
            #   appropriate public interface, like `SubspaceDiscrete.filter()`
            candidates = (
                self.searchspace.discrete._get_candidates().collect().to_pandas()
            )
            mask_todrop = pd.Series(False, index=candidates.index)
            if not self._excluded_experiments.is_empty():
                mask_todrop |= (
                    pd.merge(
                        candidates,
                        self._excluded_experiments.to_pandas(),
                        indicator=True,
                        how="left",
                    )["_merge"]
                    .eq("both")
                    .to_numpy()
                )
            if (
                not self.allow_recommending_already_recommended
                and not self._recommended_experiments.is_empty()
            ):
                mask_todrop |= (
                    pd.merge(
                        candidates,
                        self._recommended_experiments.to_pandas(),
                        indicator=True,
                        how="left",
                    )["_merge"]
                    .eq("both")
                    .to_numpy()
                )
            if not self.allow_recommending_already_measured and self._measurements:
                measured_idxs = fuzzy_row_match(
                    candidates,
                    nw.from_native(self.measurements, eager_only=True).to_pandas(),
                    self.parameters,
                )
                mask_todrop.loc[measured_idxs] = True
            if (
                not self.allow_recommending_pending_experiments
                and pending_experiments_pd is not None
            ):
                mask_todrop |= (
                    pd.merge(
                        candidates,
                        pending_experiments_pd,
                        indicator=True,
                        how="left",
                    )["_merge"]
                    .eq("both")
                    .to_numpy()
                )
            # TODO: Replace index-based selection and explicit TableCandidates
            #   instantiation with .filter() method to avoid materialization
            searchspace = evolve(
                self.searchspace,
                discrete=evolve(
                    self.searchspace.discrete,
                    candidates=TableCandidates(
                        self.searchspace.discrete.parameters,
                        candidates.loc[~mask_todrop],
                    ),
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
                self.measurements,
                pending_experiments_pd,
            )
        is_nonpredictive = isinstance(recommender, NonPredictiveRecommender)

        # Get the recommended search space entries
        try:
            with Settings(preprocess_dataframes=False):
                # NOTE: The `recommend` call must happen on `self.recommender` to update
                #   potential inner states in case of meta recommenders!
                rec = self.recommender.recommend(
                    batch_size,
                    searchspace,
                    self.objective,
                    self.measurements,
                    None if is_nonpredictive else pending_experiments_pd,
                )
        except NotEnoughPointsLeftError as ex:
            # Aliases for code compactness
            f = fields(Campaign)
            ok_m = self.allow_recommending_already_measured
            ok_r = self.allow_recommending_already_recommended
            ok_p = self.allow_recommending_pending_experiments
            ok_m_name = f._allow_recommending_already_measured.alias
            ok_r_name = f._allow_recommending_already_recommended.alias
            ok_p_name = f._allow_recommending_pending_experiments.alias
            no_blocked_pending_points = ok_p or (pending_experiments_pd is None)

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

        rec_nw = nw.from_native(rec, eager_only=True)

        if (
            active_settings.cache_campaign_recommendations
            and pending_experiments_pd is None  # see IMPROVE comment above
        ):
            self._cache_recommendation(rec_nw)

        # Track recommended experiments (deduplicated)
        if self.searchspace.type in (SearchSpaceType.DISCRETE, SearchSpaceType.HYBRID):
            param_cols = [p.name for p in self.parameters]
            rec_params_pd = rec_nw.select(param_cols).to_pandas()
            frames = [
                f
                for f in (self._recommended_experiments.to_pandas(), rec_params_pd)
                if not f.empty
            ]
            self._recommended_experiments = nw.from_native(
                pd.concat(frames, axis=0, ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True),
                eager_only=True,
            )

        return cast(
            IntoDataFrameT,
            _df_with_backend(rec_nw, backend).to_native(),
        )

    def posterior(
        self, candidates: IntoDataFrame | None = None, *, joint: bool = True
    ) -> Posterior:
        """Get the posterior predictive distribution for the given candidates.

        Args:
            candidates: The candidate points in experimental recommendations. If not
                provided, the posterior for the existing campaign measurements is
                returned. For details, see
                :meth:`baybe.surrogates.base.Surrogate.posterior`.
            joint: See :meth:`baybe.surrogates.base.Surrogate.posterior`.

        Raises:
            IncompatibilityError: If the underlying surrogate model exposes no
                method for computing the posterior distribution.

        Returns:
            Posterior: The corresponding posterior object.
            For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        if candidates is None:
            candidates = nw.from_native(self.measurements, eager_only=True).to_pandas()[
                [p.name for p in self.parameters]
            ]

        surrogate = self.get_surrogate()
        if not hasattr(surrogate, method_name := "posterior"):
            raise IncompatibilityError(
                f"The used surrogate type '{surrogate.__class__.__name__}' does not "
                f"provide a '{method_name}' method."
            )

        # pyrefly: ignore[missing-attribute]
        return surrogate.posterior(candidates, joint=joint)

    def posterior_stats(
        self,
        candidates: IntoDataFrame | None = None,
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
            if not self._measurements:
                raise NoMeasurementsError(
                    f"No candidates were provided and the campaign has no measurements "
                    f"yet. '{self.posterior_stats.__name__}' has no candidates to "
                    f"compute statistics for in this case."
                )

            candidates = nw.from_native(self.measurements, eager_only=True).to_pandas()[
                [p.name for p in self.parameters]
            ]

        surrogate = self.get_surrogate()
        if not hasattr(surrogate, method_name := "posterior_stats"):
            raise IncompatibilityError(
                f"The used surrogate type '{surrogate.__class__.__name__}' does not "
                f"provide a '{method_name}' method."
            )

        # pyrefly: ignore[missing-attribute]
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
        pending_experiments: IntoDataFrame | None = None,
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
        pending_experiments: IntoDataFrame | None = None,
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
        pending_experiments: IntoDataFrame | None = None,
    ) -> BoAcquisitionFunction:
        """Get the current BoTorch acquisition function.

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
            The BoTorch acquisition function of the current recommender.
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
        candidates: IntoDataFrameT,
        acquisition_function: AcquisitionFunction | None = None,
        *,
        batch_size: int | None = None,
        pending_experiments: IntoDataFrameT | None = None,
    ) -> IntoSeries:
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
        candidates: IntoDataFrameT,
        acquisition_function: AcquisitionFunction | None = None,
        *,
        batch_size: int | None = None,
        pending_experiments: IntoDataFrameT | None = None,
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

    def identify_non_dominated_configurations(
        self,
        configurations: IntoDataFrame | None = None,
        /,
        *,
        consider_campaign_measurements: bool = True,
    ) -> pd.Series:
        """Create a Boolean mask indicating non-dominated configurations.

        Args:
            configurations: A dataframe carrying values for all targets tracked by the
                campaign's objective. If ``None``, uses the campaign's measurements.
            consider_campaign_measurements: If ``True``, the campaign's measurements are
                considered for identifying the non-dominated configurations, but will
                not be reflected in the returned mask themselves.

        Raises:
            IncompatibilityError: If no objective is defined for the campaign.
            NothingToComputeError: If no configurations are provided as argument and no
                measurements are added to the campaign yet.

        Returns:
            A Boolean series indicating which configurations are non-dominated.
        """
        if self.objective is None:
            raise IncompatibilityError(
                "Cannot get the non-dominated configurations since no "
                f"'{Objective.__name__}' is defined."
            )

        if not self._measurements:
            if configurations is None:
                raise NothingToComputeError(
                    "The calculation of non-dominated points was requested, but "
                    "neither are configurations provided nor does the campaign have "
                    "any measurements added yet. Therefore, there is nothing to "
                    "compute."
                )
            if consider_campaign_measurements:
                warnings.warn(
                    "No measurements have been added to the campaign yet, but the flag "
                    "`consider_campaign_measurements` is set to "
                    "'True'. Therefore, the non-dominated configurations will be "
                    "determined without taking any measurements into account.",
                    UserWarning,
                )

        measurements_pd = nw.from_native(self.measurements, eager_only=True).to_pandas()
        if configurations is None:
            configurations = measurements_pd
        else:
            configurations = nw.from_native(configurations, eager_only=True).to_pandas()
            validate_target_input(configurations, self.objective.targets)

        if consider_campaign_measurements and self._measurements:
            configurations = pd.concat([configurations, measurements_pd])

        non_dominated = self.objective.identify_non_dominated_configurations(
            configurations
        )

        if consider_campaign_measurements:
            non_dominated = non_dominated.iloc[: -len(measurements_pd)]

        return non_dominated


def _add_version(dict_: dict) -> dict:
    """Add the package version to the given dictionary."""
    from baybe import __version__

    return {**dict_, "version": __version__}


def _drop_version(dict_: dict) -> dict:
    """Drop the package version from the given dictionary."""
    dict_.pop("version", None)
    return dict_


# >>>>>>>>>> Deprecation
_EXCLUDED = "excluded"
_MEASURED = "measured"
_RECOMMENDED = "recommended"


def _unpickle_dataframe(encoded: str, /) -> pd.DataFrame:
    """Deserialize a legacy pickle/base64-encoded pandas DataFrame."""
    import base64
    import pickle

    return pickle.loads(base64.b64decode(encoded.encode("utf-8")))


def _migrate_legacy_campaign(dict_: dict, /) -> dict:
    """Migrate a legacy Campaign dictionary to the current format.

    Handles all structural changes made after the 0.15.0 release:
    - ``n_fits_done`` / ``n_batches_done`` fields (discarded)
    - ``measurements_exp`` key rename to ``measurements``
    - ``FitNr`` / ``BatchNr`` columns in measurements (stripped)
    - Single ``pd.DataFrame`` ``measurements`` (pickle/base64) converted to
      ``list[nw.DataFrame]`` (parquet/base64)
    - ``searchspace_metadata`` with ``_RECOMMENDED`` / ``_EXCLUDED`` columns
      migrated to ``recommended_experiments`` / ``excluded_experiments``
    - ``cached_recommendation`` (any format) discarded
    """
    dict_.pop("n_fits_done", None)
    dict_.pop("n_batches_done", None)

    # Rename legacy measurements key
    if "measurements_exp" in dict_:
        dict_["measurements"] = dict_.pop("measurements_exp")

    # Convert single pd.DataFrame measurements (old pickle/base64 string) to
    # list[nw.DataFrame] (parquet/base64 list), stripping legacy columns along the way
    if "measurements" in dict_ and isinstance(dict_["measurements"], str):
        meas = _unpickle_dataframe(dict_["measurements"])
        meas = meas.drop(columns=[c for c in ("FitNr", "BatchNr") if c in meas.columns])
        batches = [] if meas.empty else [nw.from_native(meas, eager_only=True)]
        dict_["measurements"] = converter.unstructure(
            batches,
            unstructure_as=list[nw.DataFrame],
        )

    # Migrate legacy searchspace metadata to new fields
    if "searchspace_metadata" in dict_:
        metadata = _unpickle_dataframe(dict_.pop("searchspace_metadata"))
        if _RECOMMENDED in metadata.columns and "recommended_experiments" not in dict_:
            recommended_idxs = metadata.index[metadata[_RECOMMENDED]]
            dict_["_legacy_recommended_idxs"] = recommended_idxs.tolist()
        if _EXCLUDED in metadata.columns and "excluded_experiments" not in dict_:
            excluded_idxs = metadata.index[metadata[_EXCLUDED]]
            dict_["_legacy_excluded_idxs"] = excluded_idxs.tolist()

    # Drop legacy ``comp_rep`` from the discrete subspace (was never meaningful
    # outside of the old serialization)
    try:
        dict_["searchspace"]["discrete"].pop("comp_rep", None)
    except (KeyError, AttributeError):
        pass

    # Drop cache
    dict_.pop("cached_recommendation", None)

    return dict_


# <<<<<<<<<< Deprecation


def _prepare_for_structuring(dict_: dict, /) -> dict:
    """Prepare a Campaign dictionary for structuring."""
    dict_ = dict_.copy()
    _drop_version(dict_)
    _migrate_legacy_campaign(dict_)
    return dict_


unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    Campaign,
    converter,
    _cattrs_include_init_false=True,
    _cached_recommendation=cattrs.gen.override(omit=True),
)
structure_hook = cattrs.gen.make_dict_structure_fn(
    Campaign,
    converter,
    _cattrs_include_init_false=True,
    _cached_recommendation=cattrs.gen.override(omit=True),
)
converter.register_unstructure_hook(
    Campaign, lambda x: _add_version(unstructure_hook(x))
)


def _structure_campaign(d: dict, cl: type) -> Campaign:
    """Structure a Campaign from a dictionary, handling legacy migrations."""
    prepared = _prepare_for_structuring(d)
    legacy_recommended_idxs = prepared.pop("_legacy_recommended_idxs", None)
    legacy_excluded_idxs = prepared.pop("_legacy_excluded_idxs", None)
    campaign = structure_hook(prepared, cl)

    # >>>>>>>>>> Deprecation
    # Post-structure reconstruction from legacy metadata indices
    if legacy_recommended_idxs is not None or legacy_excluded_idxs is not None:
        candidates = (
            campaign.searchspace.discrete._get_candidates().collect().to_pandas()
        )
        if legacy_recommended_idxs is not None:
            campaign._recommended_experiments = nw.from_native(
                candidates.loc[legacy_recommended_idxs].reset_index(drop=True),
                eager_only=True,
            )
        if legacy_excluded_idxs is not None:
            campaign._excluded_experiments = nw.from_native(
                candidates.loc[legacy_excluded_idxs].reset_index(drop=True),
                eager_only=True,
            )

    # <<<<<<<<<< Deprecation

    return campaign


converter.register_structure_hook(Campaign, _structure_campaign)


# Converter for config validation
_validation_converter = converter.copy()
_validation_converter.register_structure_hook(
    SearchSpace, validate_searchspace_from_config
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
