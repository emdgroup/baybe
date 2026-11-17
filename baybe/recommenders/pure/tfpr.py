"""Top-Fraction Pareto Ranking (TFPR) recommender.

The dominance and fitness approach builds on the POEM method described by Brereton
et al. in "Predicting drug properties with parameter-free machine learning:
pareto-optimal embedded modeling" (https://doi.org/10.1088/2632-2153/ab891b).
"""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field, fields
from attrs.converters import optional as optional_c
from attrs.validators import deep_mapping, ge, gt, instance_of, le, lt
from attrs.validators import optional as optional_v
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
from typing_extensions import override

from baybe.exceptions import IncompatibilityError, IncompatibleArgumentError
from baybe.objectives.base import Objective
from baybe.objectives.pareto import ParetoObjective
from baybe.recommenders.pure.surrogate import SurrogateRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
from baybe.serialization import converter
from baybe.transformations import IdentityTransformation
from baybe.utils.conversion import to_string
from baybe.utils.validation import finite_float

_EPSILON = 0.05
"""Small stabilizer used by the original TFPR fitness formula."""


def _auto_top_fraction(n_candidates: int, /) -> float:
    """Return the original TFPR size-dependent top-fraction rule."""
    if n_candidates <= 5000:
        return 1.0
    if n_candidates > 20000:
        return 0.2
    return 0.2 + (1.0 - 0.2) * (
        1 - 1 / (1 + math.exp(-0.0001 * (n_candidates - 27500)))
    )


def _make_float_mapping(value: Mapping[str, float], /) -> dict[str, float]:
    """Copy a mapping while converting its values to floating-point numbers."""
    return {key: float(val) for key, val in dict(value).items()}


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
    dominance_threshold: float,
    /,
) -> np.ndarray:
    """Compute exact TFPR fitness with vectorized pairwise comparisons."""
    n_candidates, n_targets = values.shape
    fitness = np.zeros(n_candidates, dtype=float)
    total_weight = float(weights.sum())
    if n_candidates <= 1 or total_weight == 0.0:
        return fitness

    fraction = (
        _auto_top_fraction(n_candidates) if top_fraction is None else top_fraction
    )
    top_k = min(n_candidates, max(2, math.ceil(n_candidates * fraction)))
    # A candidate dominates another one if it secures more than the threshold share
    # of the total weight and submits to it if it secures less than the complementary
    # share. Both thresholds coincide at the original TFPR value of 0.5.
    dominating_threshold = dominance_threshold * total_weight
    submitting_threshold = (1.0 - dominance_threshold) * total_weight
    # Sums of non-integer weights carry rounding errors, so secured weights within
    # floating-point precision of a threshold are treated as reaching it exactly.
    margin = 2 * n_targets * np.finfo(float).eps * total_weight

    top_indices: list[np.ndarray] = []
    top_masks: list[np.ndarray] = []
    top_values: list[np.ndarray] = []
    for target_index in range(n_targets):
        order = np.argsort(-values[:, target_index], kind="stable")[:top_k]
        top_indices.append(order)
        top_values.append(values[order, target_index])
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
        for target_index, weight in enumerate(weights):
            if weight == 0 or not top_masks[target_index][candidate_index]:
                continue

            positions = top_positions[target_index]
            candidate_value = values[candidate_index, target_index]
            other_values = top_values[target_index]
            not_self = positions != candidate_position
            ties = _tie_mask(candidate_value, other_values, tolerances[target_index])
            wins = (candidate_value > other_values) & ~ties

            dominance[positions[wins & not_self]] += weight
            dominance[positions[ties & not_self]] += weight / 2

        mean_dominance = dominance.sum() / ((n_candidates - 1) * total_weight)
        is_dominating = dominance > dominating_threshold + margin
        is_submitting = dominance < submitting_threshold - margin
        # The candidate is not compared against itself
        is_dominating[candidate_position] = is_submitting[candidate_position] = False
        n_dominating = np.count_nonzero(is_dominating)
        # Inactive candidates are outside all compared top sets, so the candidate
        # secures no weight against them
        n_submitting = np.count_nonzero(is_submitting) + n_inactive
        fitness[candidate_index] = (
            mean_dominance * (n_dominating + _EPSILON) / (n_submitting + _EPSILON)
        )

    return fitness


@define
class TFPRRecommender(SurrogateRecommender):
    """A recommender ranking discrete candidates via Top-Fraction Pareto Ranking.

    Instead of optimizing an acquisition function, the recommender ranks all discrete
    candidates by pairwise dominance of their optimistic posterior predictions, as
    obtained from the surrogate model.
    """

    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.

    optimism_lambda: float = field(
        default=0.0, converter=float, validator=[finite_float, ge(0)], kw_only=True
    )
    """Multiplier for the posterior standard deviation added in favorable direction."""

    top_fraction: float | None = field(
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0), le(1)]),
        kw_only=True,
    )
    """Fraction of per-target top candidates considered in the ranking.

    ``None`` activates an automatic rule based on the number of candidates.
    """

    dominance_threshold: float = field(
        default=0.5,
        converter=float,
        validator=[finite_float, ge(0.5), lt(1)],
        kw_only=True,
    )
    """Share of the total target weight required for dominating another candidate.

    In each pairwise comparison, a candidate secures the weights of the targets it
    wins and half the weights of the targets it ties. It dominates the other candidate
    if it secures more than this share of the total weight and is dominated if it
    secures less than the complementary share. The default of ``0.5`` corresponds to
    a weighted majority. Higher values make dominance stricter, with values close
    to ``1`` requiring a candidate to win (nearly) all weighted targets.
    """

    _weights: dict[str, float] | None = field(
        default=None,
        init=False,
        alias="_weights",
        converter=optional_c(_make_float_mapping),
        validator=optional_v(
            deep_mapping(
                key_validator=instance_of(str),
                value_validator=[finite_float, ge(0.0)],
                mapping_validator=instance_of(dict),
            )
        ),
    )
    """Target weights controlling how strongly each target counts in comparisons.

    Private and subject to change. When set, must contain exactly one weight for each
    target of the objective. ``None`` gives all targets equal weight.
    """

    _tolerances: dict[str, float] | None = field(
        default=None,
        init=False,
        alias="_tolerances",
        converter=optional_c(_make_float_mapping),
        validator=optional_v(
            deep_mapping(
                key_validator=instance_of(str),
                value_validator=[finite_float, ge(0.0)],
                mapping_validator=instance_of(dict),
            )
        ),
    )
    """Relative tolerances within which target values are considered tied.

    Private and subject to change. When set, must contain exactly one tolerance for
    each target of the objective. ``None`` considers only exactly equal values tied.
    """

    _objective: ParetoObjective | None = field(default=None, init=False, eq=False)
    """The encountered objective to be optimized."""

    @_weights.validator
    def _validate_weights(self, _, weights) -> None:  # noqa: DOC101, DOC103
        if weights is not None and not any(w > 0 for w in weights.values()):
            raise ValueError("At least one weight must be positive.")

    @override
    def __str__(self) -> str:
        lines = [
            to_string("Surrogate", self._surrogate_model),
            to_string("Compatibility", self.compatibility, single_line=True),
            to_string("Optimism lambda", self.optimism_lambda, single_line=True),
            to_string("Top fraction", self.top_fraction, single_line=True),
            to_string(
                "Dominance threshold", self.dominance_threshold, single_line=True
            ),
        ]
        if self._weights is not None:
            lines.append(to_string("Weights", self._weights, single_line=True))
        if self._tolerances is not None:
            lines.append(to_string("Tolerances", self._tolerances, single_line=True))
        return to_string(self.__class__.__name__, *lines)

    def _get_target_settings(
        self, attribute_name: str, target_names: Sequence[str], default: float, /
    ) -> np.ndarray:
        """Arrange a per-target setting in the order of the given target names.

        Args:
            attribute_name: The name of the attribute holding the setting.
            target_names: The names of the targets.
            default: The value used for all targets if the setting is not specified.

        Raises:
            IncompatibilityError: If the specified setting does not match the targets.

        Returns:
            The setting values, ordered like the target names.
        """
        settings: dict[str, float] | None = getattr(self, attribute_name)
        if settings is None:
            return np.full(len(target_names), default)

        missing = set(target_names) - set(settings)
        unknown = set(settings) - set(target_names)
        if missing or unknown:
            alias = getattr(fields(type(self)), attribute_name).alias
            raise IncompatibilityError(
                f"The '{alias}' of the '{self.__class__.__name__}' must contain "
                f"exactly one value for each target of the objective. Missing "
                f"targets: {sorted(missing)}. Unknown targets: {sorted(unknown)}."
            )
        return np.array([settings[name] for name in target_names], dtype=float)

    @override
    def _prepare_recommendation(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None,
    ) -> None:
        if not isinstance(objective, ParetoObjective):
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' require an "
                f"objective of type '{ParetoObjective.__name__}'."
            )
        if transformed := [
            t.name
            for t in objective.targets
            if not isinstance(t.transformation, IdentityTransformation)
        ]:
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' currently support "
                f"only targets without transformations. Transformed targets: "
                f"{transformed}."
            )
        if searchspace.type is not SearchSpaceType.DISCRETE:
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' require a discrete "
                f"search space."
            )
        if searchspace.discrete.n_subsets > 0:
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' do not support "
                f"discrete subset-generating constraints."
            )
        if pending_experiments is not None:
            raise IncompatibleArgumentError(
                f"Pending experiments were passed to '{self.__class__.__name__}"
                f".{self.recommend.__name__}' but recommenders of this type cannot "
                f"use this information. If you want to exclude the pending "
                f"experiments from the candidate set, adjust the search space "
                f"accordingly."
            )
        # Validate the private settings against the targets before fitting
        target_names = [t.name for t in objective.targets]
        self._get_target_settings("_weights", target_names, 1.0)
        self._get_target_settings("_tolerances", target_names, 0.0)
        if not hasattr(self._surrogate_model, "posterior_stats"):
            raise IncompatibilityError(
                f"Recommenders of type '{self.__class__.__name__}' require a "
                f"surrogate providing a 'posterior_stats' method, which the used "
                f"surrogate of type '{self._surrogate_model.__class__.__name__}' "
                f"does not."
            )

        self._objective = objective
        self.get_surrogate(searchspace, objective, measurements)

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
            ValueError: If the surrogate returns invalid posterior statistics.

        Returns:
            The dataframe indices of the recommended points in the provided
            experimental representation.
        """
        assert self._objective is not None
        targets = self._objective.targets
        names = [t.name for t in targets]
        weights = self._get_target_settings("_weights", names, 1.0)
        tolerances = self._get_target_settings("_tolerances", names, 0.0)

        posterior_stats = getattr(self._surrogate_model, "posterior_stats")
        stats = posterior_stats(candidates_exp, stats=("mean", "std"))
        means = stats[[f"{n}_mean" for n in names]].to_numpy(dtype=float)
        stds = stats[[f"{n}_std" for n in names]].to_numpy(dtype=float)
        if not (np.isfinite(means).all() and np.isfinite(stds).all()):
            raise ValueError("Posterior means and standard deviations must be finite.")
        if (stds < 0).any():
            raise ValueError("Posterior standard deviations must be non-negative.")

        directions = np.array([-1.0 if t.minimize else 1.0 for t in targets])
        values = directions * means + self.optimism_lambda * stds
        if not np.isfinite(values).all():
            raise ValueError("Optimistic posterior values must be finite.")

        fitness = _tfpr_fitness(
            values, weights, tolerances, self.top_fraction, self.dominance_threshold
        )
        order = np.argsort(-fitness, kind="stable")[:batch_size]
        return candidates_exp.index[order]


_PRIVATE_SETTINGS = ("_weights", "_tolerances")
"""The private ranking settings, which are no constructor arguments."""


def _unstructure_tfpr_recommender(recommender: TFPRRecommender, /) -> dict[str, Any]:
    """Unstructure a TFPR recommender including its private ranking settings."""
    # NOTE: Like for all other pure recommenders, the deprecated flags are dropped.
    #   The encountered objective is transient and thus not serialized either.
    fn = make_dict_unstructure_fn(
        TFPRRecommender,
        converter,
        _cattrs_include_init_false=True,
        _objective=cattrs.override(omit=True),
        _deprecated_allow_repeated_recommendations=cattrs.override(omit=True),
        _deprecated_allow_recommending_already_measured=cattrs.override(omit=True),
        _deprecated_allow_recommending_pending_experiments=cattrs.override(omit=True),
    )
    return fn(recommender)


def _structure_tfpr_recommender(
    dct: dict[str, Any], cls: type[TFPRRecommender], /
) -> TFPRRecommender:
    """Structure a TFPR recommender including its private ranking settings."""
    dct = dict(dct)
    settings = {name: dct.pop(name) for name in _PRIVATE_SETTINGS if name in dct}
    recommender = make_dict_structure_fn(cls, converter)(dct)
    # NOTE: The settings are assigned manually instead of via
    #   `_cattrs_include_init_false` since cattrs collects but never raises errors
    #   occurring for attributes excluded from the constructor. Assignment applies the
    #   attribute converters and validators.
    for name, value in settings.items():
        setattr(recommender, name, value)
    return recommender


converter.register_unstructure_hook(TFPRRecommender, _unstructure_tfpr_recommender)
converter.register_structure_hook(TFPRRecommender, _structure_tfpr_recommender)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
