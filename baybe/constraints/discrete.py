"""Discrete constraints."""

from collections.abc import Callable
from functools import reduce
from typing import Any, cast

import pandas as pd
from attr import define, field
from attr.validators import ge, in_, instance_of, min_len

from baybe.constraints.base import DiscreteConstraint
from baybe.constraints.conditions import (
    Condition,
    ThresholdCondition,
    _valid_logic_combiners,
)
from baybe.serialization import (
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
from baybe.utils.basic import Dummy


@define
class DiscreteExcludeConstraint(DiscreteConstraint):
    """Class for modelling exclusion constraints."""

    # object variables
    conditions: list[Condition] = field(validator=min_len(1))
    """List of individual conditions."""

    combiner: str = field(default="AND", validator=in_(_valid_logic_combiners))
    """Operator encoding how to combine the individual conditions."""

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        satisfied = [
            cond.evaluate(data[self.parameters[k]])
            for k, cond in enumerate(self.conditions)
        ]
        res = reduce(_valid_logic_combiners[self.combiner], satisfied)
        return data.index[res]


@define
class DiscreteSumConstraint(DiscreteConstraint):
    """Class for modelling sum constraints."""

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # object variables
    condition: ThresholdCondition = field()
    """The condition modeled by this constraint."""

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        evaluate_data = data[self.parameters].sum(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_data)

        return data.index[mask_bad]


@define
class DiscreteProductConstraint(DiscreteConstraint):
    """Class for modelling product constraints."""

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # object variables
    condition: ThresholdCondition = field()
    """The condition that is used for this constraint."""

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        evaluate_data = data[self.parameters].prod(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_data)

        return data.index[mask_bad]


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

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        mask_bad = data[self.parameters].nunique(axis=1) != len(self.parameters)

        return data.index[mask_bad]


class DiscreteLinkedParametersConstraint(DiscreteConstraint):
    """Constraint class for linking the values of parameters.

    This constraint type effectively allows generating parameter sets that relate to
    the same underlying quantity, e.g. two parameters that represent the same molecule
    using different encodings. Linking the parameters removes all entries from the
    search space where the parameter values differ.
    """

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        mask_bad = data[self.parameters].nunique(axis=1) != 1

        return data.index[mask_bad]


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

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        # Create data copy and mark entries where the dependency conditions are negative
        # with a dummy value to cause degeneracy.
        censored_data = data.copy()
        for k, _ in enumerate(self.parameters):
            # .loc assignments are not supported by mypy + pandas-stubs yet
            # See https://github.com/pandas-dev/pandas-stubs/issues/572
            censored_data.loc[  # type: ignore[call-overload]
                ~self.conditions[k].evaluate(data[self.parameters[k]]),
                self.affected_parameters[k],
            ] = Dummy()

        # Create an invariant indicator: pair each value of an affected parameter with
        # the corresponding value of the parameter it depends on. These indicators
        # will become invariant when frozenset is applied to them.
        for k, param in enumerate(self.parameters):
            for affected_param in self.affected_parameters[k]:
                censored_data[affected_param] = list(
                    zip(censored_data[affected_param], censored_data[param])
                )

        # Merge the invariant indicator with all other parameters (i.e. neither the
        # affected nor the dependency-causing ones) and detect duplicates in that space.
        all_affected_params = [col for cols in self.affected_parameters for col in cols]
        other_params = (
            data.columns.drop(all_affected_params).drop(self.parameters).tolist()
        )
        df_eval = pd.concat(
            [
                censored_data[other_params],
                censored_data[all_affected_params].apply(
                    cast(Callable, frozenset)
                    if self.permutation_invariant
                    else cast(Callable, tuple),
                    axis=1,
                ),
            ],
            axis=1,
        )
        inds_bad = data.index[df_eval.duplicated(keep="first")]

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

    # object variables
    dependencies: DiscreteDependenciesConstraint | None = field(default=None)
    """Dependencies connected with the invariant parameters."""

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        # Get indices of entries with duplicate label entries. These will also be
        # dropped by this constraint.
        mask_duplicate_labels = pd.Series(False, index=data.index)
        mask_duplicate_labels[
            DiscreteNoLabelDuplicatesConstraint(parameters=self.parameters).get_invalid(
                data
            )
        ] = True

        # Merge a permutation invariant representation of all affected parameters with
        # the other parameters and indicate duplicates. This ensures that variation in
        # other parameters is also accounted for.
        other_params = data.columns.drop(self.parameters).tolist()
        df_eval = pd.concat(
            [
                data[other_params].copy(),
                data[self.parameters].apply(cast(Callable, frozenset), axis=1),
            ],
            axis=1,
        ).loc[
            ~mask_duplicate_labels  # only consider label-duplicate-free part
        ]
        mask_duplicate_permutations = df_eval.duplicated(keep="first")

        # Indices of entries with label-duplicates
        inds_duplicate_labels = data.index[mask_duplicate_labels]

        # Indices of duplicate permutations in the (already label-duplicate-free) data
        inds_duplicate_permutations = df_eval.index[mask_duplicate_permutations]

        # If there are dependencies connected to the invariant parameters evaluate them
        # here and remove resulting duplicates with a DependenciesConstraint
        inds_invalid = inds_duplicate_labels.union(inds_duplicate_permutations)
        if self.dependencies:
            self.dependencies.permutation_invariant = True
            inds_duplicate_independency_adjusted = self.dependencies.get_invalid(
                data.drop(index=inds_invalid)
            )
            inds_invalid = inds_invalid.union(inds_duplicate_independency_adjusted)

        return inds_invalid


@define
class DiscreteCustomConstraint(DiscreteConstraint):
    """Class for user-defined custom constraints."""

    # object variables
    validator: Callable[[pd.DataFrame], pd.Series] = field()
    """A user-defined function modeling the validation of the constraint. The expected
    return is a pandas series with boolean entries True/False for search space elements
    you want to keep/remove."""

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        mask_bad = ~self.validator(data[self.parameters])

        return data.index[mask_bad]


@define
class DiscreteCardinalityConstraint(DiscreteConstraint):
    """Class for discrete cardinality constraints.

    TODO: delete the detailed explanation when PR of ContinuousCardinalityConstraint
    is merged.
    Places a constraint on the set of nonzero (i.e. "active") values among the
    specified parameter, bounding it between the two given integers,
        ``min_cardinality`` <= | {1(p_i != 0)}_i | <= ``max_cardinality``
    where ``1`` is the Kronecker delta function and ``{p_i}`` are the parameters
    specified for the constraint.

    Note that this can be equivalently regarded a L0-constraint on the vector containing
    the specified parameters.
    """

    min_cardinality: int = field(default=0, validator=[instance_of(int), ge(0)])
    "The minimum required cardinality."

    max_cardinality: int = field(validator=instance_of(int))
    "The maximum allowed cardinality."

    @max_cardinality.default
    def _default_max_cardinality(self):
        """Use the number of involved parameters as the upper limit by default."""
        return len(self.parameters)

    def __attrs_post_init__(self):
        """Validate the cardinality bounds.

        TODO: simplify the docstring when PR of ContinuousCardinalityConstraint is
        merged.

        Raises:
            ValueError: When the minimum allowed cardinality is larger that the maximum
                one.
            ValueError: When the maximum allowed cardinality is larger than the
                number of parameters.
            ValueError: When no cardinality constraint is needed.
        """
        # TODO: remove the duplicate when PR of ContinuousCardinalityConstraint is
        #  merged.
        if self.min_cardinality > self.max_cardinality:
            raise ValueError(
                f"The lower cardinality bound cannot be larger than the upper bound. "
                f"Provided values: {self.max_cardinality=}, {self.min_cardinality=}."
            )

        if self.max_cardinality > len(self.parameters):
            raise ValueError(
                f"The cardinality bound cannot exceed the number of parameters. "
                f"Provided values: {self.max_cardinality=}, {len(self.parameters)=}."
            )

        if self.min_cardinality == 0 and self.max_cardinality == len(self.parameters):
            raise ValueError(
                f"No constraint of type `{self.__class__.__name__}' is required "
                f"when 0 <= cardinality <= len(parameters)."
            )

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        non_zeros = (data[self.parameters] != 0.0).sum(axis=1)
        mask_bad = non_zeros > self.max_cardinality
        mask_bad |= non_zeros < self.min_cardinality
        return data.index[mask_bad]


# the order in which the constraint types need to be applied during discrete subspace
# filtering
DISCRETE_CONSTRAINTS_FILTERING_ORDER = (
    DiscreteCustomConstraint,
    DiscreteExcludeConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteSumConstraint,
    DiscreteProductConstraint,
    DiscreteCardinalityConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteDependenciesConstraint,
)

# Discrete constraints that are valid only for numeric parameters. It is needed for
# the purpose of validation.
# There are two options:
# Option A: A list containing such discrete constraints is maintained. It does
# not require any breaking change but requires some maintenance work. Whenever a
# new discrete constraint, which is valid only for numerical parameters,
# is introduced, it must be added to the list.
# Option B: Add an attribute, e.g. numerical_parameter_only,
# to the discrete constraints.
DISCRETE_CONSTRAINTS_ONLY_FOR_NUMERIC_PARAMETER = [
    DiscreteCardinalityConstraint,
    DiscreteSumConstraint,
    DiscreteProductConstraint,
]

# Prevent (de-)serialization of custom constraints
converter.register_unstructure_hook(DiscreteCustomConstraint, block_serialization_hook)
converter.register_structure_hook(DiscreteCustomConstraint, block_deserialization_hook)
