"""Feasibility-based recommender wrapper."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import pandas as pd
from attrs import define, evolve, field
from attrs.validators import ge, instance_of, le
from typing_extensions import override

from baybe.constraints.base import ContinuousNonlinearConstraint
from baybe.exceptions import (
    IncompatibilityError,
    IncompleteMeasurementsError,
    InfeasibilityError,
    NoFeasibleObservationWarning,
)
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.searchspace._filtered import FilteredSubspaceDiscrete
from baybe.serialization import (
    SerialMixin,
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
from baybe.surrogates.gaussian_process.core import (
    GaussianProcessSurrogate,
    _ModelContext,
)
from baybe.targets.binary import BinaryTarget
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from botorch.posteriors import Posterior
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.parameters import NumericalContinuousParameter


@define
class _GaussianProcessClassifier(GaussianProcessSurrogate):
    """Gaussian process classifier tailored to the feasibility filtering.

    The model is a variational Gaussian process with Bernoulli likelihood, predicting
    feasibility probabilities instead of Gaussian-distributed target values. Since the
    remaining Bayesian optimization machinery requires Gaussian posteriors, the model is
    an implementation detail of the feasibility recommender and not part of the public
    surrogate API.
    """

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the variational Gaussian process using the evidence lower bound."""
        assert self._searchspace is not None  # ensured by base class
        assert self._objective is not None  # ensured by base class
        assert self._measurements is not None  # ensured by base class

        context = _ModelContext(self._searchspace, self._objective, self._measurements)
        kernel, mean, _, _ = self._resolve_components(context)
        input_transform = self._make_input_transform(context)

        from botorch.exceptions import InputDataWarning, UserInputWarning
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskVariationalGP
        from gpytorch.likelihoods import BernoulliLikelihood
        from gpytorch.mlls import VariationalELBO

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InputDataWarning)
            # The minibatch caveat of the input transform does not apply since the
            # model is always fitted on the full data set.
            warnings.simplefilter("ignore", UserInputWarning)

            model = SingleTaskVariationalGP(
                train_x,
                train_y,
                likelihood=BernoulliLikelihood(),
                inducing_points=len(train_x),
                covar_module=kernel,
                mean_module=mean,
                input_transform=input_transform,
            )
        mll = VariationalELBO(model.likelihood, model.model, num_data=len(train_x))
        fit_gpytorch_mll(mll)
        self._model = model

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        # Model being fit is guaranteed by the call in `posterior`
        assert self._model is not None

        # Enabling observation noise applies the Bernoulli likelihood, turning the
        # latent Gaussian distribution into class probabilities.
        return self._model.posterior(candidates_comp_scaled, observation_noise=True)

    def posterior_probabilities(self, candidates_comp: Tensor, /) -> Tensor:
        """Predict the feasibility probabilities for computationally represented input.

        Args:
            candidates_comp: The candidates in computational representation.

        Returns:
            The predicted feasibility probabilities as a flat tensor.
        """
        posterior = cast("GPyTorchPosterior", self._posterior_comp(candidates_comp))
        return posterior.mean.reshape(-1)


@define(eq=False)
class _FeasibilityConstraint(ContinuousNonlinearConstraint):
    """Constraint restricting a continuous space to predicted-feasible configurations.

    The constraint encodes the decision boundary of the feasibility classifier as
    ``p(x) - threshold >= 0``, where ``p(x)`` is the predicted feasibility
    probability. The constraint operates on the full computational representation,
    i.e. it also accesses the discrete columns when applied to a hybrid space.
    """

    classifier: _GaussianProcessClassifier = field(repr=False)
    """The fitted classifier predicting the feasibility probability."""

    threshold: float = field(validator=[instance_of(float), ge(0.0), le(1.0)])
    """Minimum predicted feasibility probability required for a configuration."""

    @override
    def to_botorch(
        self,
        parameters: Iterable[NumericalContinuousParameter],
        idx_offset: int = 0,
        *,
        batch_size: int | None = None,
    ) -> list[tuple[Callable[[Tensor], Tensor], bool]]:
        """Translate the predicted feasible region into BoTorch format.

        The returned callable supports both individual points and batches, so that it
        can be used by BoTorch's optimizer as well as by the initial condition
        generator.

        Args:
            parameters: The parameter objects of the continuous space. Unused because
                the underlying classifier operates on the full computational
                representation.
            idx_offset: The index offset of the continuous parameters within the full
                computational representation. Unused for the same reason.
            batch_size: The batch size used for the recommendation. Unused because
                the constraint is an intra-point constraint.

        Returns:
            A single intra-point constraint encoding ``p(x) - threshold >= 0``.
        """
        del parameters, idx_offset, batch_size  # unused

        def constraint(x: Tensor) -> Tensor:
            flat = x.reshape(-1, x.shape[-1])
            probabilities = self.classifier.posterior_probabilities(flat)
            return (probabilities - self.threshold).reshape(x.shape[:-1])

        return [(constraint, True)]


@define(kw_only=True)
class ExperimentFeasibilityRecommender(PureRecommender, SerialMixin):
    """Recommender that restricts the search space and delegates to another recommender.

    A probabilistic classifier is fitted on the feasibility labels contained in the
    measurements. The search space is then restricted to the configurations predicted
    feasible before the wrapped recommender is executed:

    * For purely discrete spaces, the candidate set is filtered.
    * For spaces with a continuous part, a nonlinear constraint encoding the predicted
      feasible region is added to the continuous subspace. The wrapped recommender
      must declare support for continuous nonlinear constraints. This keeps the
      continuous space fully continuous instead of discretizing it.

    As long as all measured experiments are infeasible, no decision boundary can be
    learned and the wrapped recommender is executed without filtering, since failed
    experiments do not rule out feasible configurations elsewhere.
    """

    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    _classifier: _GaussianProcessClassifier = field(
        factory=_GaussianProcessClassifier, init=False, eq=False, repr=False
    )
    """The classifier predicting the feasibility probability."""

    feasibility_threshold: float = field(
        default=0.5, validator=[instance_of(float), ge(0.0), le(1.0)]
    )
    """Minimum predicted feasibility probability required for a candidate."""

    include_infeasible_measurements: bool = field(
        default=True, validator=instance_of(bool)
    )
    """Whether measurements of infeasible experiments are used for model training.

    Only experiments containing values for all objective targets are affected.
    Experiments without such values are left unchanged, i.e. no dummy values are
    introduced.
    """

    recommender: PureRecommender = field(
        factory=BotorchRecommender, validator=instance_of(PureRecommender)
    )
    """The recommender that is executed on the restricted search space."""

    target: BinaryTarget = field(validator=instance_of(BinaryTarget))
    """The binary target encoding feasibility."""

    @override
    def __str__(self) -> str:
        return to_string(
            self.__class__.__name__,
            to_string("Recommender", self.recommender),
            to_string("Feasibility target", self.target.name, single_line=True),
            to_string(
                "Feasibility threshold", self.feasibility_threshold, single_line=True
            ),
            to_string(
                "Include infeasible measurements",
                self.include_infeasible_measurements,
                single_line=True,
            ),
        )

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Recommend from the configurations predicted to be feasible.

        If all measured feasibility labels are identical, no decision boundary can be
        learned and the wrapped recommender is executed without filtering. If all
        labels are infeasible, a
        :class:`~baybe.exceptions.NoFeasibleObservationWarning` is emitted.

        Args:
            batch_size: The number of recommendations to generate.
            searchspace: The search space to generate recommendations from.
            objective: An optional objective passed on to the wrapped recommender.
            measurements: The measurements containing the feasibility labels.
            pending_experiments: Optional pending experiments passed on to the
                wrapped recommender.

        Raises:
            IncompatibilityError: If the search space has a continuous part and the
                wrapped recommender does not support continuous nonlinear constraints.
            IncompleteMeasurementsError: If no measurements or no feasibility labels
                are provided, or if no measurements remain for model training.
            InfeasibilityError: If no configuration reaches the feasibility threshold.

        Returns:
            A dataframe containing the recommendations as individual rows.
        """
        if measurements is None or measurements.empty:
            raise IncompleteMeasurementsError(
                f"'{self.__class__.__name__}' requires measurements containing "
                f"the feasibility target '{self.target.name}'."
            )

        labels = self._extract_labels(measurements)
        objective_measurements = self._objective_measurements(
            measurements, labels, objective
        )
        if labels.nunique() == 1:
            # A single observed class yields no decision boundary, so filtering
            # would either keep everything or nothing. Delegate unfiltered instead.
            if labels.iloc[0] < self.feasibility_threshold:
                warnings.warn(
                    f"None of the measured experiments has been observed as "
                    f"feasible for the target '{self.target.name}'. Failed "
                    f"experiments do not imply that no feasible configuration "
                    f"exists, so the feasibility filter is skipped until a "
                    f"feasible experiment is available.",
                    NoFeasibleObservationWarning,
                )
            prepared = searchspace
        else:
            if (
                searchspace.type is not SearchSpaceType.DISCRETE
                and not self.recommender.supports_continuous_nonlinear_constraints
            ):
                raise IncompatibilityError(
                    f"The '{self.__class__.__name__}' requires a wrapped recommender "
                    f"supporting continuous nonlinear constraints for search spaces "
                    f"with a continuous part, but "
                    f"'{self.recommender.__class__.__name__}' does not. Pass a "
                    f"compatible recommender via the 'recommender' field."
                )
            feasibility_measurements = measurements.copy()
            feasibility_measurements[self.target.name] = labels
            # The fit is cached by the surrogate and only repeated if the search
            # space or the measurements changed.
            self._classifier.fit(
                searchspace, self.target.to_objective(), feasibility_measurements
            )
            prepared = self._restrict_searchspace(searchspace, self._classifier)

        return self.recommender.recommend(
            batch_size,
            prepared,
            objective,
            objective_measurements,
            pending_experiments,
        )

    def _extract_labels(self, measurements: pd.DataFrame, /) -> pd.Series:
        """Extract the feasibility labels from the given measurements.

        Args:
            measurements: The measurements containing the feasibility target.

        Raises:
            IncompleteMeasurementsError: If the feasibility column is missing or
                contains missing values.

        Returns:
            The feasibility labels in computational representation.
        """
        if (name := self.target.name) not in measurements.columns:
            raise IncompleteMeasurementsError(
                f"The measurements do not contain the feasibility target "
                f"'{name}'. Available columns: {list(measurements.columns)}."
            )

        series = measurements[name]
        missing = series.isna()
        if missing.any():
            raise IncompleteMeasurementsError(
                f"The feasibility target '{name}' contains missing values at row "
                f"indices {list(series.index[missing])}."
            )

        return self.target.transform(series)

    def _filter_searchspace(
        self, searchspace: SearchSpace, classifier: _GaussianProcessClassifier, /
    ) -> SearchSpace:
        """Restrict the discrete candidate set to the points predicted feasible.

        Args:
            searchspace: The search space to filter.
            classifier: The fitted feasibility classifier.

        Raises:
            InfeasibilityError: If no candidate reaches the feasibility threshold.

        Returns:
            The search space with a filtered discrete candidate set.
        """
        subspace = searchspace.discrete
        candidates_exp, _ = subspace.get_candidates()
        probabilities = (
            classifier.posterior_probabilities(
                to_tensor(subspace.transform(candidates_exp))
            )
            .detach()
            .numpy()
        )
        feasible = probabilities >= self.feasibility_threshold

        if not feasible.any():
            raise InfeasibilityError(
                f"No candidate reaches the feasibility threshold "
                f"{self.feasibility_threshold}."
            )

        # Derive a mask over the full candidate set, which also works when the
        # incoming subspace has already been filtered.
        mask_keep = np.isin(
            subspace.exp_rep.index.to_numpy(),
            candidates_exp.index[feasible].to_numpy(),
        )
        filtered = (
            evolve(subspace, mask_keep=mask_keep)
            if isinstance(subspace, FilteredSubspaceDiscrete)
            else FilteredSubspaceDiscrete.from_subspace(subspace, mask_keep)
        )
        return evolve(searchspace, discrete=filtered)

    def _objective_measurements(
        self,
        measurements: pd.DataFrame,
        labels: pd.Series,
        objective: Objective | None,
        /,
    ) -> pd.DataFrame:
        """Get the measurements passed on to the wrapped recommender.

        Measurements of infeasible experiments are only removed if they contain
        values for all objective targets. Experiments without such values are left
        unchanged, i.e. no dummy values are introduced.

        Args:
            measurements: The available experimentation data.
            labels: The feasibility labels matching the rows of ``measurements``.
            objective: The objective passed on to the wrapped recommender.

        Raises:
            IncompleteMeasurementsError: If no measurements remain for model
                training after removing the infeasible ones.

        Returns:
            The measurements to be passed on to the wrapped recommender.
        """
        if (
            self.include_infeasible_measurements
            or objective is None
            or isinstance(self.recommender, NonPredictiveRecommender)
        ):
            return measurements

        values = [t.name for t in objective.targets if t.name in measurements.columns]
        if not values:
            return measurements

        has_values = measurements[values].notna().all(axis=1).to_numpy()
        is_infeasible = labels.to_numpy() == 0
        filtered = measurements.loc[~(is_infeasible & has_values)]

        if filtered.empty:
            raise IncompleteMeasurementsError(
                "No measurements remain for model training because all measured "
                "experiments are infeasible and 'include_infeasible_measurements' "
                "is disabled."
            )

        return filtered

    def _restrict_searchspace(
        self, searchspace: SearchSpace, classifier: _GaussianProcessClassifier, /
    ) -> SearchSpace:
        """Restrict the search space to the configurations predicted feasible.

        In spaces with a continuous part, the predicted feasible region is encoded as
        a nonlinear constraint added to the continuous subspace. In purely discrete
        spaces, the candidate set is filtered instead.

        Args:
            searchspace: The search space to restrict.
            classifier: The fitted feasibility classifier.

        Returns:
            The restricted search space.
        """
        if searchspace.type is SearchSpaceType.DISCRETE:
            return self._filter_searchspace(searchspace, classifier)

        continuous = searchspace.continuous
        constraint = _FeasibilityConstraint(
            parameters=list(continuous.parameter_names),
            classifier=classifier,
            threshold=self.feasibility_threshold,
        )
        return evolve(
            searchspace,
            continuous=evolve(
                continuous,
                constraints_nonlin=(*continuous.constraints_nonlin, constraint),
            ),
        )


# Prevent (de-)serialization because the constraint contains a fitted model
converter.register_unstructure_hook(_FeasibilityConstraint, block_serialization_hook)
converter.register_structure_hook(_FeasibilityConstraint, block_deserialization_hook)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
