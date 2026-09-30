"""Functionality for Top-Fraction Pareto Ranking (TFPR) objectives.

The dominance and fitness approach builds on the POEM method described by Brereton
et al. in "Predicting drug properties with parameter-free machine learning:
pareto-optimal embedded modeling" (https://doi.org/10.1088/2632-2153/ab891b).
"""

from __future__ import annotations

import gc
import math
from collections.abc import Iterable
from typing import ClassVar, NoReturn

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_iterable, ge, gt, instance_of, le, min_len
from attrs.validators import optional as optional_v
from cattrs.gen import make_dict_structure_fn
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.objectives.validation import validate_target_names
from baybe.serialization import converter
from baybe.targets.numerical import NumericalTarget
from baybe.transformations import IdentityTransformation
from baybe.utils.basic import to_tuple
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import pretty_print_df
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


def _make_weights(value: Iterable[float], /) -> tuple[int, ...]:
    """Convert weights to integers by rounding to the nearest integer."""
    return tuple(round(v) for v in value)


def _make_tolerances(value: Iterable[float], /) -> tuple[float, ...]:
    """Convert tolerances to floating-point values."""
    return tuple(float(v) for v in value)


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


@define(frozen=True, slots=False)
class TFPRObjective(Objective):
    """An objective ranking candidates via Top-Fraction Pareto Ranking (TFPR).

    Instead of optimizing an acquisition function, TFPR ranks the discrete candidates
    by pairwise dominance of their optimistic posterior predictions.
    """

    is_multi_output: ClassVar[bool] = True
    # See base class.

    _targets: tuple[NumericalTarget, ...] = field(
        converter=to_tuple,
        validator=[
            min_len(2),
            deep_iterable(member_validator=instance_of(NumericalTarget)),
            validate_target_names,
        ],
        alias="targets",
    )
    "The targets considered by the objective."

    weights: tuple[int, ...] = field(
        converter=_make_weights,
        validator=deep_iterable(member_validator=[ge(0), le(10)]),
    )
    """The integer weights controlling how strongly each target contributes.

    A weight of ``k`` counts the corresponding target ``k`` times in the pairwise
    dominance comparison. Values are rounded to the nearest integer.
    By default, all targets are considered equally important.
    """

    tolerances: tuple[float, ...] = field(
        converter=_make_tolerances,
        validator=deep_iterable(member_validator=[finite_float, ge(0)]),
    )
    """The relative tolerances within which target values are considered tied.

    By default, only exactly equal values are considered tied.
    """

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

    @weights.default
    def _default_weights(self) -> tuple[int, ...]:
        """Create unit weights for all targets."""
        return tuple(1 for _ in self.targets)

    @tolerances.default
    def _default_tolerances(self) -> tuple[float, ...]:
        """Create zero tolerances for all targets."""
        return tuple(0.0 for _ in self.targets)

    @_targets.validator
    def _validate_targets(self, _, targets) -> None:  # noqa: DOC101, DOC103
        if transformed := [
            t.name
            for t in targets
            if not isinstance(t.transformation, IdentityTransformation)
        ]:
            raise ValueError(
                f"'{self.__class__.__name__}' currently supports only targets without "
                f"transformations. Transformed targets: {transformed}."
            )

    @weights.validator
    def _validate_weights(self, _, weights) -> None:  # noqa: DOC101, DOC103
        if (lw := len(weights)) != (lt := len(self.targets)):
            raise ValueError(
                f"If custom weights are specified, there must be one for each target. "
                f"Specified number of targets: {lt}. Specified number of weights: {lw}."
            )
        if not any(weights):
            raise ValueError("At least one weight must be positive.")

    @tolerances.validator
    def _validate_tolerances(self, _, tolerances) -> None:  # noqa: DOC101, DOC103
        if (ltol := len(tolerances)) != (lt := len(self.targets)):
            raise ValueError(
                f"If custom tolerances are specified, there must be one for each "
                f"target. Specified number of targets: {lt}. Specified number of "
                f"tolerances: {ltol}."
            )

    @override
    @property
    def targets(self) -> tuple[NumericalTarget, ...]:
        return self._targets

    @override
    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @override
    @property
    def supports_partial_measurements(self) -> bool:
        return True

    @override
    def __str__(self) -> str:
        targets_df = pd.DataFrame([target.summary() for target in self.targets])
        targets_df["Weights"] = self.weights
        targets_df["Tolerances"] = self.tolerances

        fields = [
            to_string("Type", self.__class__.__name__, single_line=True),
            to_string("Targets", pretty_print_df(targets_df)),
            to_string("Optimism lambda", self.optimism_lambda, single_line=True),
            to_string("Top fraction", self.top_fraction, single_line=True),
        ]
        return to_string("Objective", *fields)

    def compute_fitness(self, posterior_stats: pd.DataFrame, /) -> pd.Series:
        """Compute the TFPR fitness from posterior statistics of the targets.

        Args:
            posterior_stats: A dataframe containing ``<target>_mean`` and
                ``<target>_std`` columns for all targets of the objective, as
                returned by :meth:`baybe.surrogates.base.Surrogate.posterior_stats`.

        Raises:
            ValueError: If the posterior statistics are not finite or contain negative
                standard deviations.

        Returns:
            A series containing the fitness value of each candidate, where larger
            values are better.
        """
        names = [t.name for t in self.targets]
        means = posterior_stats[[f"{n}_mean" for n in names]].to_numpy(dtype=float)
        stds = posterior_stats[[f"{n}_std" for n in names]].to_numpy(dtype=float)
        if not (np.isfinite(means).all() and np.isfinite(stds).all()):
            raise ValueError("Posterior means and standard deviations must be finite.")
        if (stds < 0).any():
            raise ValueError("Posterior standard deviations must be non-negative.")

        directions = np.array([-1.0 if t.minimize else 1.0 for t in self.targets])
        values = directions * means + self.optimism_lambda * stds
        if not np.isfinite(values).all():
            raise ValueError("Optimistic posterior values must be finite.")

        fitness = _tfpr_fitness(
            values,
            np.asarray(self.weights),
            np.asarray(self.tolerances),
            self.top_fraction,
        )
        return pd.Series(fitness, index=posterior_stats.index, name="Fitness")

    @override
    def to_botorch_posterior_transform(self) -> NoReturn:
        raise NotImplementedError(
            f"Objectives of type '{type(self).__name__}' do not support conversion "
            f"to BoTorch posterior transforms."
        )


# NOTE: The default cattrs hook would cast the weights via ``int`` (i.e. truncate)
#   before the attrs converter sees them, so the raw values are passed through instead.
converter.register_structure_hook(
    TFPRObjective,
    make_dict_structure_fn(
        TFPRObjective,
        converter,
        weights=cattrs.override(struct_hook=lambda x, _: x),
    ),
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
