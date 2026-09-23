"""Top-Fraction Pareto Ranking recommender.

The dominance and fitness approach builds on the POEM method described by Brereton
et al. in "Predicting drug properties with parameter-free machine learning:
pareto-optimal embedded modeling" (https://doi.org/10.1088/2632-2153/ab891b).
"""

from __future__ import annotations

import gc
import math
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_mapping, ge, gt, instance_of, le
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.objectives.pareto import ParetoObjective
from baybe.recommenders.pure.surrogate import SurrogateRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.transformations import IdentityTransformation
from baybe.utils.conversion import to_string
from baybe.utils.validation import finite_float

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


def _make_tolerances(value: Any, /) -> dict[str, float]:
    """Convert a tolerance mapping to floating-point values."""
    return {key: float(val) for key, val in dict(value).items()}


def _make_weights(value: Any, /) -> dict[str, int]:
    """Convert a weight mapping to integer values by truncating toward zero."""
    return {key: int(val) for key, val in dict(value).items()}


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
    """Compute exact TFPR fitness with vectorized pairwise comparisons."""
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
    top_values: list[np.ndarray] = []
    for objective_index in range(values.shape[1]):
        order = np.argsort(-values[:, objective_index], kind="stable")[:top_k]
        top_indices.append(order)
        top_values.append(values[order, objective_index])
        mask = np.zeros(n_candidates, dtype=bool)
        mask[order] = True
        top_masks.append(mask)

    active_indices = np.flatnonzero(np.any(top_masks, axis=0))
    active_positions = np.full(n_candidates, -1, dtype=int)
    active_positions[active_indices] = np.arange(len(active_indices))
    top_positions = [active_positions[indices] for indices in top_indices]
    n_inactive = n_candidates - len(active_indices)

    # Reuse one row buffer while vectorizing each candidate's pairwise comparisons.
    dominance = np.zeros(len(active_indices), dtype=float)
    for candidate_position, candidate_index in enumerate(active_indices):
        dominance.fill(0.0)
        for objective_index, weight in enumerate(weights):
            if weight == 0 or not top_masks[objective_index][candidate_index]:
                continue

            positions = top_positions[objective_index]
            candidate_value = values[candidate_index, objective_index]
            other_values = top_values[objective_index]
            not_self = positions != candidate_position
            ties = _tie_mask(candidate_value, other_values, tolerances[objective_index])
            wins = (candidate_value > other_values) & ~ties

            dominance[positions[wins & not_self]] += weight
            dominance[positions[ties & not_self]] += weight / 2

        mean_dominance = dominance.sum() / ((n_candidates - 1) * total_weight)
        n_dominating = np.count_nonzero(dominance > threshold)
        # Self-comparisons remain zero and must not count as submissions.
        n_submitting = np.count_nonzero(dominance < threshold) - 1 + n_inactive
        fitness[candidate_index] = (
            mean_dominance * (n_dominating + _EPSILON) / (n_submitting + _EPSILON)
        )

    return fitness


@define(kw_only=True)
class TFPRRecommender(SurrogateRecommender):
    """Recommend discrete candidates by posterior optimism and TFPR ranking."""

    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    weights: dict[str, int] = field(
        factory=dict,
        converter=_make_weights,
        validator=deep_mapping(
            key_validator=instance_of(str), value_validator=[ge(0), le(10)]
        ),
    )
    """Target-name weights used by TFPR, where unspecified targets receive weight 1.

    Values are converted with :class:`int`, so fractional values are truncated toward
    zero and Boolean values become ``1`` or ``0``.
    """

    tolerances: dict[str, float] = field(
        factory=dict,
        converter=_make_tolerances,
        validator=deep_mapping(
            key_validator=instance_of(str), value_validator=[finite_float, ge(0)]
        ),
    )
    """Target-name relative tie tolerances, where unspecified targets receive 0."""

    optimism_lambda: float = field(
        default=0.0, converter=float, validator=[finite_float, ge(0)]
    )
    """Nonnegative multiplier for posterior standard-deviation optimism."""

    top_fraction: float | None = field(
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0), le(1)]),
    )
    """Fraction of per-target top candidates considered by TFPR.

    ``None`` activates the original automatic rule.
    """

    _objective: ParetoObjective | None = field(default=None, init=False, eq=False)
    """The encountered objective to be optimized."""

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
    def _prepare_recommendation(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None,
    ) -> None:
        if not hasattr(self._surrogate_model, "posterior_stats"):
            raise IncompatibilityError(
                f"The used surrogate type '{self._surrogate_model.__class__.__name__}' "
                f"does not provide a 'posterior_stats' method."
            )

        self.get_surrogate(
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
        )
        self._objective = cast(ParetoObjective, objective)

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

        self._make_target_options(objective)

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

        Raises:
            IncompatibilityError: If no Pareto objective is available.
            ValueError: If the surrogate returns invalid posterior statistics.

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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
