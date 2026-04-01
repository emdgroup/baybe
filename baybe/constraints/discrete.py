"""Discrete constraints."""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field
from attrs.validators import in_, min_len
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
from baybe.utils.basic import Dummy

if TYPE_CHECKING:
    import polars as pl


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
    """Class for modelling sum constraints."""

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

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        evaluate_df = df[self.parameters].sum(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_df)

        return df.index[mask_bad]

    @override
    def get_invalid_polars(self) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        return self.condition.to_polars(pl.sum_horizontal(self.parameters)).not_()


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

        expr = (
            pl.concat_list(pl.col(self.parameters))
            .list.eval(pl.element().n_unique())
            .explode()
        ) != len(self.parameters)

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

        expr = (
            pl.concat_list(pl.col(self.parameters))
            .list.eval(pl.element().n_unique())
            .explode()
        ) != 1

        return expr


@define
class DiscreteDependenciesConstraint(DiscreteConstraint):
    """Constraint that specifies dependencies between parameters.

    For instance some parameters might only be relevant when another parameter has a
    certain value (e.g. parameter switch is 'on'). All dependencies must be declared in
    a single constraint.
    """

    # class variables
    eval_during_augmentation: ClassVar[bool] = True
    # See base class

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
        # Create df copy and mark entries where the dependency conditions are negative
        # with a dummy value to cause degeneracy.
        censored_df = df.copy()
        for k, _ in enumerate(self.parameters):
            # .loc assignments are not supported by mypy + pandas-stubs yet
            # See https://github.com/pandas-dev/pandas-stubs/issues/572
            censored_df.loc[  # type: ignore[call-overload]
                ~self.conditions[k].evaluate(df[self.parameters[k]]),
                self.affected_parameters[k],
            ] = Dummy()

        # Create an invariant indicator: pair each value of an affected parameter with
        # the corresponding value of the parameter it depends on. These indicators
        # will become invariant when frozenset is applied to them.
        for k, param in enumerate(self.parameters):
            for affected_param in self.affected_parameters[k]:
                censored_df[affected_param] = list(
                    zip(censored_df[affected_param], censored_df[param])
                )

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


@define
class DiscretePermutationInvarianceConstraint(DiscreteConstraint):
    """Constraint class for declaring that a set of parameters is permutation invariant.

    More precisely, this means that, ``(val_from_param1, val_from_param2)`` is
    equivalent to ``(val_from_param2, val_from_param1)``. Since it does not make sense
    to have this constraint with duplicated labels, this implementation also internally
    applies the :class:`baybe.constraints.discrete.DiscreteNoLabelDuplicatesConstraint`.

    *Note:* This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.
    """

    # class variables
    eval_during_augmentation: ClassVar[bool] = True
    # See base class

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
        # At least two parameters are needed for any deduplication. When only a
        # partial set is available, the constraint falls back to the always-safe
        # label-dedup logic.
        return len(available & set(self.parameters)) >= 2

    @override
    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        cols = set(df.columns)
        params = [p for p in self.parameters if p in cols]
        # When dependencies exist, permutation dedup on a partial set of
        # parameters is not safe because the dependency logic can change
        # which permutations are equivalent. In this case, only the
        # label-dedup part (which is always safe incrementally) is applied.
        if self.dependencies:
            if not self._required_parameters <= cols:
                return DiscreteNoLabelDuplicatesConstraint(
                    parameters=params
                ).get_invalid(df)

        # Get indices of entries with duplicate label entries. These will also be
        # dropped by this constraint.
        mask_duplicate_labels = pd.Series(False, index=df.index)
        mask_duplicate_labels[
            DiscreteNoLabelDuplicatesConstraint(parameters=params).get_invalid(df)
        ] = True

        # Merge a permutation invariant representation of all affected parameters with
        # the other parameters and indicate duplicates. This ensures that variation in
        # other parameters is also accounted for.
        other_params = df.columns.drop(params).tolist()
        frozen = df[params].apply(cast(Callable, frozenset), axis=1)
        parts = [df[other_params].copy(), frozen] if other_params else [frozen]
        df_eval = pd.concat(parts, axis=1).loc[
            ~mask_duplicate_labels  # only consider label-duplicate-free part
        ]
        mask_duplicate_permutations = df_eval.duplicated(keep="first")

        # Indices of entries with label-duplicates
        inds_duplicate_labels = df.index[mask_duplicate_labels]

        # Indices of duplicate permutations in the (already label-duplicate-free) df
        inds_duplicate_permutations = df_eval.index[mask_duplicate_permutations]

        # If there are dependencies connected to the invariant parameters evaluate them
        # here and remove resulting duplicates with a DependenciesConstraint
        inds_invalid = inds_duplicate_labels.union(inds_duplicate_permutations)
        if self.dependencies:
            self.dependencies.permutation_invariant = True
            inds_duplicate_independency_adjusted = self.dependencies.get_invalid(
                df.drop(index=inds_invalid)
            )
            inds_invalid = inds_invalid.union(inds_duplicate_independency_adjusted)

        return inds_invalid


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
    """Constraint ensuring all batch recommendations share the same parameter value.

    When this constraint is active, the recommender internally partitions the
    candidate set into subspaces — one for each unique value of the constrained
    parameter — obtains a full batch recommendation from each subspace, and
    returns the batch with the highest joint acquisition value.

    This constraint is not supported by all recommenders. It is not applied during
    search space creation (all parameter values remain in the search space).

    Example:
        If parameter ``Temperature`` has values ``[50, 100, 150]`` and a batch of
        10 is requested, the recommender will generate three candidate batches
        (one all-50, one all-100, one all-150) and return the best one.
    """

    # Class variables
    eval_during_creation: ClassVar[bool] = False
    eval_during_modeling: ClassVar[bool] = True

    numerical_only: ClassVar[bool] = False
    # See base class.

    def __attrs_post_init__(self):
        """Validate that exactly one parameter is specified."""
        if len(self.parameters) != 1:
            raise ValueError(
                f"'{self.__class__.__name__}' requires exactly one parameter, "
                f"but {len(self.parameters)} were provided: {self.parameters}."
            )

    @override
    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """Get the indices of invalid rows.

        Always returns an empty index because this constraint operates at the
        batch level, not the row level. Individual rows are never invalid; the
        constraint is enforced at recommendation time by partitioning candidates
        into subspaces.

        Args:
            data: A dataframe where each row represents a parameter configuration.

        Returns:
            An empty index.
        """
        return pd.Index([])

    def subspace_masks(
        self, candidates_exp: pd.DataFrame
    ) -> list[npt.NDArray[np.bool_]]:
        """Return boolean masks defining the subspaces for this constraint.

        Each mask selects the rows in ``candidates_exp`` that belong to one
        subspace, i.e. share the same value for the constrained parameter.

        Args:
            candidates_exp: The experimental representation of candidate points.

        Returns:
            A list of boolean masks, one per unique value of the constrained
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
)

# Prevent (de-)serialization of custom constraints
converter.register_unstructure_hook(DiscreteCustomConstraint, block_serialization_hook)
converter.register_structure_hook(DiscreteCustomConstraint, block_deserialization_hook)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
