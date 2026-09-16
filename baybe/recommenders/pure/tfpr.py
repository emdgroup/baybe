"""Top-Fraction Pareto Ranking recommender.

The dominance and fitness approach builds on the POEM method described by Brereton
et al. in "Predicting drug properties with parameter-free machine learning:
pareto-optimal embedded modeling" (https://doi.org/10.1088/2632-2153/ab891b).
"""

from __future__ import annotations

import gc
import math
from typing import Any, ClassVar

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from cattrs.gen import make_dict_structure_fn
from typing_extensions import override

from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.objectives.pareto import ParetoObjective
from baybe.recommenders.pure.surrogate import SurrogateRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.serialization.core import converter
from baybe.settings import Settings
from baybe.transformations import IdentityTransformation
from baybe.utils.conversion import to_string
from baybe.utils.validation import preprocess_dataframe, validate_object_names

_EPSILON = 0.05
"""Small stabilizer used by the original TFPR fitness formula."""


def _auto_top_fraction(n_candidates: int) -> float:
    """Return the original TFPR size-dependent top-fraction rule."""
    if n_candidates <= 5000:
        return 1.0
    if n_candidates > 20000:
        return 0.2
    return 0.2 + (1.0 - 0.2) * (
        1 - 1 / (1 + math.exp(-0.0001 * (n_candidates - 27500)))
    )


def _make_float(value: Any, /) -> float:
    """Convert a non-Boolean finite-like numeric value to float."""
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid floating-point inputs.")
    return float(value)


def _make_optional_float(value: Any, /) -> float | None:
    """Convert ``None`` or a non-Boolean finite-like numeric value."""
    return None if value is None else _make_float(value)


def _make_tolerances(value: Any, /) -> dict[str, float]:
    """Convert a tolerance mapping to a copied plain dict."""
    return {key: _make_float(val) for key, val in dict(value).items()}


def _pass_through_structure_hook(value: Any, _: Any, /) -> Any:
    """Pass through values so attrs converters perform strict validation."""
    return value


def _tie_mask(value: float, others: np.ndarray, tolerance: float, /) -> np.ndarray:
    """Identify exact and relative-tolerance ties against one value."""
    exact_tie = np.equal(others, value)
    if tolerance == 0.0:
        return exact_tie
    difference = np.abs(value - others)
    scale = np.maximum(abs(value), np.abs(others))
    return exact_tie | (difference <= tolerance * scale)


def _tfpr_fitness(
    values: np.ndarray,
    weights: np.ndarray,
    tolerances: np.ndarray,
    top_fraction: float | None,
    /,
) -> np.ndarray:
    """Compute TFPR fitness scores without materializing a dense pair matrix."""
    n_candidates, _ = values.shape
    fitness = np.zeros(n_candidates, dtype=float)
    total_weight = int(weights.sum())
    if n_candidates <= 1 or total_weight == 0:
        return fitness

    fraction = (
        _auto_top_fraction(n_candidates) if top_fraction is None else top_fraction
    )
    top_k = min(n_candidates, max(2, math.ceil(n_candidates * fraction)))
    threshold = 0.5 * total_weight

    top_indices: list[np.ndarray] = []
    top_masks: list[np.ndarray] = []
    for objective_index in range(values.shape[1]):
        order = np.argsort(-values[:, objective_index], kind="stable")[:top_k]
        top_indices.append(order)
        mask = np.zeros(n_candidates, dtype=bool)
        mask[order] = True
        top_masks.append(mask)

    for candidate_index in range(n_candidates):
        dominance = np.zeros(n_candidates, dtype=float)
        for objective_index, weight in enumerate(weights):
            if weight == 0 or not top_masks[objective_index][candidate_index]:
                continue

            indices = top_indices[objective_index]
            candidate_value = values[candidate_index, objective_index]
            other_values = values[indices, objective_index]
            not_self = indices != candidate_index
            ties = _tie_mask(candidate_value, other_values, tolerances[objective_index])
            wins = (candidate_value > other_values) & ~ties

            dominance[indices[wins & not_self]] += weight
            dominance[indices[ties & not_self]] += weight / 2

        others = np.arange(n_candidates) != candidate_index
        mean_dominance = dominance.sum() / ((n_candidates - 1) * total_weight)
        n_dominating = np.count_nonzero(dominance[others] > threshold)
        n_submitting = np.count_nonzero(dominance[others] < threshold)
        fitness[candidate_index] = (
            mean_dominance * (n_dominating + _EPSILON) / (n_submitting + _EPSILON)
        )

    return fitness


@define(kw_only=True)
class TFPRRecommender(SurrogateRecommender):
    """Recommend discrete candidates by posterior optimism and TFPR ranking."""

    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    weights: dict[str, int] = field(factory=dict, converter=dict)
    """Target-name weights used by TFPR, where unspecified targets receive weight 1."""

    tolerances: dict[str, float] = field(factory=dict, converter=_make_tolerances)
    """Target-name relative tie tolerances, where unspecified targets receive 0."""

    optimism_lambda: float = field(default=0.0, converter=_make_float)
    """Nonnegative multiplier for posterior standard-deviation optimism."""

    top_fraction: float | None = field(default=None, converter=_make_optional_float)
    """Fraction of per-target top candidates considered by TFPR.

    ``None`` activates the original automatic rule.
    """

    _objective: ParetoObjective | None = field(default=None, init=False, eq=False)
    """The encountered objective to be optimized."""

    @optimism_lambda.validator
    def _validate_optimism_lambda(  # noqa: DOC101, DOC103
        self, _: Any, value: float
    ) -> None:
        """Validate the optimism multiplier.

        Raises:
            ValueError: If the value is not finite and nonnegative.
        """
        if not math.isfinite(value) or value < 0:
            raise ValueError("The optimism multiplier must be finite and nonnegative.")

    @tolerances.validator
    def _validate_tolerances(  # noqa: DOC101, DOC103
        self, _: Any, value: dict[str, float]
    ) -> None:
        """Validate tolerance values.

        Raises:
            TypeError: If a key is not a string.
            ValueError: If a value is not finite and nonnegative.
        """
        for target_name, tolerance in value.items():
            if not isinstance(target_name, str):
                raise TypeError("Tolerance mappings must use target-name string keys.")
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("TFPR tolerances must be finite and nonnegative.")

    @top_fraction.validator
    def _validate_top_fraction(  # noqa: DOC101, DOC103
        self, _: Any, value: float | None
    ) -> None:
        """Validate the top-fraction override.

        Raises:
            ValueError: If the value is not ``None`` or in the interval ``(0, 1]``.
        """
        if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError("TFPR top_fraction must be None or satisfy 0 < f <= 1.")

    @weights.validator
    def _validate_weights(  # noqa: DOC101, DOC103
        self, _: Any, value: dict[str, int]
    ) -> None:
        """Validate weight values.

        Raises:
            TypeError: If a key is not a string or a value is not an integer.
            ValueError: If a value is outside the interval ``[0, 10]``.
        """
        for target_name, weight in value.items():
            if not isinstance(target_name, str):
                raise TypeError("Weight mappings must use target-name string keys.")
            if not isinstance(weight, int) or isinstance(weight, bool):
                raise TypeError("TFPR weights must be integer values.")
            if not 0 <= weight <= 10:
                raise ValueError("TFPR weights must be between 0 and 10.")

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Surrogate", self._surrogate_model),
            to_string("Compatibility", self.compatibility, single_line=True),
            to_string("Weights", self.weights, single_line=True),
            to_string("Tolerances", self.tolerances, single_line=True),
            to_string("Optimism lambda", self.optimism_lambda, single_line=True),
            to_string("Top fraction", self.top_fraction, single_line=True),
        ]
        return to_string(self.__class__.__name__, *fields)

    def _make_target_options(
        self, objective: ParetoObjective, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Make ordered TFPR options for the objective targets."""
        target_names = [target.name for target in objective.targets]
        target_name_set = set(target_names)

        transformed_targets = [
            target.name
            for target in objective.targets
            if not isinstance(target.transformation, IdentityTransformation)
        ]
        if transformed_targets:
            raise IncompatibilityError(
                "TFPR currently supports only identity target transformations. "
                f"Transformed targets: {transformed_targets}."
            )

        unknown_weight_targets = set(self.weights) - target_name_set
        if unknown_weight_targets:
            raise ValueError(
                f"The TFPR weight mapping contains unknown targets: "
                f"{unknown_weight_targets}."
            )

        unknown_tolerance_targets = set(self.tolerances) - target_name_set
        if unknown_tolerance_targets:
            raise ValueError(
                f"The TFPR tolerance mapping contains unknown targets: "
                f"{unknown_tolerance_targets}."
            )

        weights = np.array([self.weights.get(name, 1) for name in target_names])
        if not weights.any():
            raise ValueError("At least one TFPR target weight must be positive.")

        tolerances = np.array([self.tolerances.get(name, 0.0) for name in target_names])
        directions = np.array(
            [-1.0 if target.minimize else 1.0 for target in objective.targets]
        )
        return weights, tolerances, directions, target_names

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not isinstance(objective, ParetoObjective):
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' require a "
                f"'{ParetoObjective.__name__}'."
            )

        if searchspace.type is not SearchSpaceType.DISCRETE:
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' require a "
                "discrete search space."
            )

        validate_object_names(searchspace.parameters + objective.targets)
        self._make_target_options(objective)

        if (measurements is None) or measurements.empty:
            raise NotImplementedError(
                f"Recommenders of type '{self.__class__.__name__}' do not support "
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

        surrogate = self.get_surrogate(searchspace, objective, measurements)
        if not hasattr(surrogate, "posterior_stats"):
            raise IncompatibilityError(
                f"The used surrogate type '{surrogate.__class__.__name__}' does not "
                f"provide a 'posterior_stats' method."
            )

        self._objective = objective
        with Settings(preprocess_dataframes=False):
            return super().recommend(
                batch_size=batch_size,
                searchspace=searchspace,
                objective=objective,
                measurements=measurements,
                pending_experiments=pending_experiments,
            )

    @override
    def _recommend_discrete(
        self,
        subspace_discrete: SubspaceDiscrete,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.Index:
        """Generate recommendations from a discrete search space.

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
        if self._objective is None:
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' require a "
                f"'{ParetoObjective.__name__}'."
            )

        weights, tolerances, directions, target_names = self._make_target_options(
            self._objective
        )
        posterior_stats = getattr(self._surrogate_model, "posterior_stats")
        stats = posterior_stats(candidates_exp, stats=("mean", "std"))
        mean_columns = [f"{name}_mean" for name in target_names]
        std_columns = [f"{name}_std" for name in target_names]
        means = stats[mean_columns].to_numpy(dtype=float)
        stds = stats[std_columns].to_numpy(dtype=float)
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError("TFPR posterior mean/std values must be finite.")
        if (stds < 0).any():
            raise ValueError("TFPR posterior standard deviations must be nonnegative.")

        values = directions * means + self.optimism_lambda * stds
        if not np.isfinite(values).all():
            raise ValueError("TFPR optimistic posterior values must be finite.")

        fitness = _tfpr_fitness(values, weights, tolerances, self.top_fraction)
        order = np.argsort(-fitness, kind="stable")[:batch_size]
        return candidates_exp.index[order]


_structure_tfpr_recommender_inner = make_dict_structure_fn(
    TFPRRecommender,
    converter,
    weights=cattrs.override(struct_hook=_pass_through_structure_hook),
    tolerances=cattrs.override(struct_hook=_pass_through_structure_hook),
    optimism_lambda=cattrs.override(struct_hook=_pass_through_structure_hook),
    top_fraction=cattrs.override(struct_hook=_pass_through_structure_hook),
)


@converter.register_structure_hook
def _structure_tfpr_recommender(
    value: dict[str, Any], cls: type[TFPRRecommender]
) -> TFPRRecommender:
    """Structure TFPR recommenders while preserving strict field validation."""
    value = value.copy()
    if (type_ := value.pop("type", None)) and type_ != cls.__name__:
        raise TypeError(
            f"The type field '{type_}' does not match the target class "
            f"'{cls.__name__}'."
        )
    return _structure_tfpr_recommender_inner(value, cls)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
