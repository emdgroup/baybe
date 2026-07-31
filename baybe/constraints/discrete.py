"""Discrete constraints."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, cast

import cattrs
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, in_, min_len
from typing_extensions import override

from baybe.constraints.base import CardinalityConstraint, DiscreteConstraint
from baybe.constraints.conditions import (
    Condition,
    ThresholdCondition,
    _threshold_operators,
    _valid_logic_combiners,
)
from baybe.serialization import (
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    import polars as pl

    from baybe.symmetries.dependency import DependencySymmetry
    from baybe.symmetries.permutation import PermutationSymmetry


@define
class DiscreteExcludeConstraint(DiscreteConstraint):
    """Class for modelling exclusion constraints."""

    # object variables
    conditions: list[Condition] = field(validator=min_len(1))
    """List of individual conditions."""

    combiner: str = field(default="AND", validator=in_(_valid_logic_combiners))
    """Operator encoding how to combine the individual conditions."""

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # The OR combiner supports incremental filtering (a single true
        # condition suffices to mark a row as invalid), so at least one
        # parameter is enough. Other combiners need all parameters.
        present = available & set(self.parameters)
        if not present:
            return False
        if self.combiner != "OR" and present != set(self.parameters):
            return False
        return True

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        pairs = [(p, c) for p, c in zip(self.parameters, self.conditions) if p in df]
        satisfied = [cond.evaluate(df[p]) for p, cond in pairs]
        res = reduce(_valid_logic_combiners[self.combiner], satisfied)

        return df.index[res]

    @override
    def get_invalid_polars(self) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        satisfied = []
        for k, cond in enumerate(self.conditions):
            satisfied.append(cond.to_polars(pl.col(self.parameters[k])))

        expr = pl.reduce(_valid_logic_combiners[self.combiner], satisfied)

        return expr


@define
class DiscreteSumConstraint(DiscreteConstraint):
    """Class for modelling sum constraints.

    The constraint evaluates whether the (optionally weighted) sum of the specified
    parameters satisfies the given threshold condition.
    """

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # IMPROVE: Look-ahead filtering would be possible if parameter
    # value ranges (min/max) were available to the constraint, allowing
    # bound-based pruning of partial sums before all parameters are
    # present. This could be expressed via a _can_evaluate override.

    # class variables
    numerical_only: ClassVar[bool] = True
    # See base class.

    # object variables
    condition: ThresholdCondition = field()
    """The condition modeled by this constraint."""

    coefficients: tuple[float, ...] = field(
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=deep_iterable(member_validator=finite_float),
    )
    """The coefficients for the weighted sum, one per entry in ``parameters``.

    Defaults to all-ones, i.e. an unweighted sum."""

    @coefficients.default
    def _default_coefficients(self) -> tuple[float, ...]:
        """Return equal weight coefficients as default."""
        return (1.0,) * len(self.parameters)

    @coefficients.validator
    def _validate_coefficients(  # noqa: DOC101, DOC103
        self, _: Any, coefficients: Sequence[float]
    ) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the number of coefficients does not match the number of
                parameters.
        """
        if len(self.parameters) != len(coefficients):
            raise ValueError(
                "The given 'coefficients' list must have one floating point entry for "
                "each entry in 'parameters'."
            )
        if any(c == 0.0 for c in coefficients):
            raise ValueError("All entries in 'coefficients' must be non-zero.")

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        evaluate_df = pd.Series(
            sum(
                df[p].to_numpy() * c for p, c in zip(self.parameters, self.coefficients)
            ),
            index=df.index,
        )
        mask_bad = ~self.condition.evaluate(evaluate_df)

        return df.index[mask_bad]

    @override
    def get_invalid_polars(self) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        weighted = [pl.col(p) * c for p, c in zip(self.parameters, self.coefficients)]
        return self.condition.to_polars(pl.sum_horizontal(weighted)).not_()


@define
class DiscreteProductConstraint(DiscreteConstraint):
    """Class for modelling product constraints."""

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # class variables
    numerical_only: ClassVar[bool] = True
    # See base class.

    # object variables
    condition: ThresholdCondition = field()
    """The condition that is used for this constraint."""

    # IMPROVE: Look-ahead filtering would be possible if parameter
    # value ranges (min/max) were available to the constraint, allowing
    # bound-based pruning of partial products before all parameters are
    # present. This could be expressed via a _can_evaluate override.

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        evaluate_df = df[self.parameters].prod(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_df)

        return df.index[mask_bad]

    @override
    def get_invalid_polars(self) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        op = _threshold_operators[self.condition.operator]

        # Get the product of columns
        expr = pl.reduce(lambda acc, x: acc * x, pl.col(self.parameters))

        # Apply the threshold operator on expr and the condition threshold
        return op(expr, self.condition.threshold).not_()


class DiscreteNoLabelDuplicatesConstraint(DiscreteConstraint):
    """Constraint class for excluding entries where occurring labels are not unique.

    This can be useful to remove entries that arise from e.g. a permutation invariance
    as for instance here:

    - A,B,C,D would remain
    - A,A,B,C would be removed
    - A,A,B,B would be removed
    - A,A,B,A would be removed
    - A,C,A,C would be removed
    - A,C,B,C would be removed
    """

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # Duplicate detection is meaningful as soon as at least two of the
        # constraint's parameters are available: duplicates in a subset
        # will also be duplicates in the full set.
        return len(available & set(self.parameters)) >= 2

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        params = [p for p in self.parameters if p in df]
        mask_bad = df[params].nunique(axis=1) != len(params)

        return df.index[mask_bad]

    @override
    def get_invalid_polars(self) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        expr = pl.concat_list(pl.col(self.parameters)).list.n_unique() != len(
            self.parameters
        )

        return expr


@define
class DiscreteLinkedParametersConstraint(DiscreteConstraint):
    """Constraint class for linking the values of parameters.

    This constraint type effectively allows generating parameter sets that relate to
    the same underlying quantity, e.g. two parameters that represent the same molecule
    using different encodings. Linking the parameters removes all entries from the
    search space where the parameter values differ.
    """

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # Linked-parameter checking is meaningful as soon as at least two of
        # the constraint's parameters are available: if values differ in a
        # subset, they will also differ in the full set.
        return len(available & set(self.parameters)) >= 2

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        params = [p for p in self.parameters if p in set(df.columns)]
        mask_bad = df[params].nunique(axis=1) != 1

        return df.index[mask_bad]

    @override
    def get_invalid_polars(self) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        expr = pl.concat_list(pl.col(self.parameters)).list.n_unique() != 1

        return expr


@define
class DiscreteDependenciesConstraint(DiscreteConstraint):
    """Constraint that specifies dependencies between parameters.

    For instance some parameters might only be relevant when another parameter has a
    certain value (e.g. parameter switch is 'on'). All dependencies must be declared in
    a single constraint.
    """

    # object variables
    conditions: list[Condition] = field()
    """The list of individual conditions."""

    affected_parameters: list[list[str]] = field()
    """The parameters affected by the individual conditions."""

    # for internal use only
    permutation_invariant: bool = field(default=False, init=False)
    """Flag that indicates whether the affected parameters are permutation invariant.
    This should not be changed by the user but by other constraints using the class."""

    @affected_parameters.validator
    def _validate_affected_parameters(  # noqa: DOC101, DOC103
        self, _: Any, value: list[list[str]]
    ) -> None:
        """Validate the affected parameters.

        Raises:
            ValueError: If one set of affected parameters does not have exactly one
                condition.
        """
        if len(self.conditions) != len(value):
            raise ValueError(
                f"For the {self.__class__.__name__}, for each item in the "
                f"affected_parameters list you must provide exactly one condition in "
                f"the conditions list."
            )

    @property
    @override
    def _required_parameters(self) -> set[str]:
        """See base class."""
        params = set(self.parameters)
        for group in self.affected_parameters:
            params.update(group)
        return params

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        # Build an invariant indicator for each affected parameter: pair each value
        # with the value of the parameter it depends on. For rows where the dependency
        # condition is not met, use None as a sentinel so that all such rows with the
        # same dependency value appear identical, causing them to be detected as
        # duplicates. The indicator tuples are constructed directly without storing
        # any intermediate sentinel in the typed columns.
        censored_df = df.copy()
        for k, param in enumerate(self.parameters):
            invalid = ~self.conditions[k].evaluate(df[self.parameters[k]])
            for affected_param in self.affected_parameters[k]:
                censored_df[affected_param] = [
                    (None if inv else val, dep)
                    for val, dep, inv in zip(
                        censored_df[affected_param], censored_df[param], invalid
                    )
                ]

        # Merge the invariant indicator with all other parameters (i.e. neither the
        # affected nor the dependency-causing ones) and detect duplicates in that space.
        all_affected_params = [col for cols in self.affected_parameters for col in cols]
        other_params = (
            df.columns.drop(all_affected_params).drop(self.parameters).tolist()
        )
        invariant_indicator = censored_df[all_affected_params].apply(
            cast(Callable, frozenset)
            if self.permutation_invariant
            else cast(Callable, tuple),
            axis=1,
        )
        # Only include the other_params DataFrame if it is non-empty to avoid
        # pandas FutureWarning about concatenation with empty entries
        parts = [censored_df[other_params]] if other_params else []
        parts.append(invariant_indicator)
        df_eval = pd.concat(parts, axis=1)
        inds_bad = df.index[df_eval.duplicated(keep="first")]

        return inds_bad

    def to_symmetries(self) -> tuple[DependencySymmetry, ...]:
        """Convert to :class:`~baybe.symmetries.dependency.DependencySymmetry` objects.

        Create one symmetry object per dependency relationship, i.e., per
        (parameter, condition, affected_parameters) triple.

        Returns:
            A tuple of dependency symmetries, one for each dependency in the
            constraint.
        """
        from baybe.symmetries.dependency import DependencySymmetry

        return tuple(
            DependencySymmetry(
                parameter_name=p,
                condition=c,
                affected_parameter_names=aps,
            )
            for p, c, aps in zip(
                self.parameters, self.conditions, self.affected_parameters, strict=True
            )
        )


@define
class DiscretePermutationInvarianceConstraint(DiscreteConstraint):
    """Constraint class for declaring that a set of parameters is permutation invariant.

    More precisely, this means that, ``(val_from_param1, val_from_param2)`` is
    equivalent to ``(val_from_param2, val_from_param1)``.

    *Note:* This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.
    """

    # object variables
    dependencies: DiscreteDependenciesConstraint | None = field(default=None)
    """Dependencies connected with the invariant parameters."""

    @property
    @override
    def _required_parameters(self) -> set[str]:
        """See base class."""
        params = set(self.parameters)
        if self.dependencies:
            params.update(self.dependencies._required_parameters)
        return params

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # When dependencies are present, partial permutation dedup is unsafe:
        # the dependency logic changes which rows are permutation-equivalent
        # (inactive parameters become irrelevant), so removing permutation
        # duplicates before the dependency columns are available can discard
        # configurations that should have been kept as canonical representatives.
        if self.dependencies:
            return self._required_parameters <= available
        # Without dependencies, permutation dedup on a partial set is safe
        # during incremental construction: since new columns are added via
        # cross-product, rows that are permutation-equivalent on the available
        # subset will produce identical expansions.
        return len(available & set(self.parameters)) >= 2

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        cols = set(df.columns)
        params = [p for p in self.parameters if p in cols]

        # Merge a permutation invariant representation of all affected parameters with
        # the other parameters and indicate duplicates. This ensures that variation in
        # other parameters is also accounted for.
        other_params = df.columns.drop(params).tolist()
        frozen = df[params].apply(cast(Callable, frozenset), axis=1)
        parts = [df[other_params].copy(), frozen] if other_params else [frozen]
        df_eval = pd.concat(parts, axis=1)
        mask_duplicate_permutations = df_eval.duplicated(keep="first")

        # Indices of duplicate permutations
        inds_invalid = df_eval.index[mask_duplicate_permutations]

        # If there are dependencies connected to the invariant parameters evaluate them
        # here and remove resulting duplicates with a DependenciesConstraint
        if self.dependencies and self.dependencies._can_evaluate(set(df.columns)):
            self.dependencies.permutation_invariant = True
            inds_duplicate_independency_adjusted = self.dependencies.get_invalid(
                df.drop(index=inds_invalid)
            )
            inds_invalid = inds_invalid.union(inds_duplicate_independency_adjusted)

        return inds_invalid

    def to_symmetry(self) -> PermutationSymmetry:
        """Convert to a :class:`~baybe.symmetries.permutation.PermutationSymmetry`.

        The constraint's parameters form the primary permutation group. If
        dependencies are attached, their parameters are added as an additional
        group that is permuted in lockstep.

        Returns:
            The corresponding permutation symmetry.
        """
        from baybe.symmetries.permutation import PermutationSymmetry

        groups = [self.parameters]
        if self.dependencies:
            groups.append(list(self.dependencies.parameters))
        return PermutationSymmetry(permutation_groups=groups)


@define
class DiscreteCustomConstraint(DiscreteConstraint):
    """Class for user-defined custom constraints."""

    # object variables
    validator: Callable[[pd.DataFrame], pd.Series] = field()
    """A user-defined function modeling the validation of the constraint. The expected
    return is a pandas series with Boolean entries True/False for search space elements
    you want to keep/remove."""

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        mask_bad = ~self.validator(df[self.parameters])

        return df.index[mask_bad]


@define
class DiscreteBatchConstraint(DiscreteConstraint):
    """Constraint ensuring recommendations in a batch share certain parameter values.

    When this constraint is active, the recommender internally subsets the
    candidate set (one subset for each unique value of the constrained
    parameter), obtains a full batch recommendation from each subset, and
    returns the batch with the highest joint acquisition value.

    This constraint is not supported by all recommenders. It is not applied during
    search space creation (all parameter values remain in the search space).

    Example:
        If parameter ``Temperature`` has values ``[50, 100, 150]`` and a batch of
        10 is requested, the recommender will generate three candidate batches
        (one all-50, one all-100, one all-150) and return the best one.

    Notes:
        This constraint can lead to overhead in the computation since optimization
        results in individual optimizations over several subsets. If there are
        multiple subset-generating constraints active, this can drastically increase
        the computational cost due to the combinatorial explosion.
    """

    # Class variables
    eval_during_creation: ClassVar[bool] = False
    eval_during_modeling: ClassVar[bool] = True
    numerical_only: ClassVar[bool] = False

    def __attrs_post_init__(self):
        """Validate that exactly one parameter is specified."""
        if len(self.parameters) != 1:
            raise ValueError(
                f"'{self.__class__.__name__}' requires exactly one parameter, "
                f"but {len(self.parameters)} were provided: {self.parameters}."
            )

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        # Always returns an empty index because this constraint operates at the
        # batch level, not the row level. Individual rows are never invalid; the
        # constraint is enforced at recommendation time by subsetting candidates
        # into subsets.
        return pd.Index([])

    def subset_masks(
        self, candidates_exp: pd.DataFrame, /
    ) -> list[npt.NDArray[np.bool_]]:
        """Return Boolean masks defining the subsets for this constraint.

        Each mask selects the rows in ``candidates_exp`` that belong to one
        subset, i.e. share the same value for the constrained parameter.

        Args:
            candidates_exp: The experimental representation of candidate points.

        Returns:
            A list of Boolean masks, one per unique value of the constrained
            parameter.
        """
        param = self.parameters[0]
        return [
            (candidates_exp[param] == v).values for v in candidates_exp[param].unique()
        ]


@define
class DiscreteCardinalityConstraint(CardinalityConstraint, DiscreteConstraint):
    """Class for discrete cardinality constraints."""

    # Class variables
    numerical_only: ClassVar[bool] = True
    # See base class.

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # The max-cardinality check is safe on any non-empty subset: the
        # nonzero count can only increase as more parameters are added.
        return bool(available & set(self.parameters))

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        params = [p for p in self.parameters if p in set(df.columns)]
        all_present = len(params) == len(self.parameters)

        non_zeros = (df[params] != 0.0).sum(axis=1)
        # The max_cardinality check is safe on a partial subset: the nonzero
        # count can only increase as more parameters are added.
        mask_bad = non_zeros > self.max_cardinality
        # The min_cardinality check can only be applied when all parameters
        # are present, since missing parameters could still add nonzero values.
        if all_present:
            mask_bad |= non_zeros < self.min_cardinality
        return df.index[mask_bad]


# Constraints are approximately ordered according to increasing computational effort
# to minimize total time in their sequential application
DISCRETE_CONSTRAINTS_FILTERING_ORDER = (
    DiscreteExcludeConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteSumConstraint,
    DiscreteProductConstraint,
    DiscreteCardinalityConstraint,
    DiscreteCustomConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteDependenciesConstraint,
    DiscreteBatchConstraint,
)

# Prevent (de-)serialization of custom constraints
converter.register_unstructure_hook(DiscreteCustomConstraint, block_serialization_hook)
converter.register_structure_hook(DiscreteCustomConstraint, block_deserialization_hook)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
