"""
Functionality for parameter constraints.
"""
# TODO: ForwardRefs via __future__ annotations are currently disabled due to this issue:
#  https://github.com/python-attrs/cattrs/issues/354

import logging
import operator as ops
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, ClassVar, List, Optional

import cattrs
import numpy as np
import pandas as pd
from attr import define, field
from attrs.validators import in_, min_len
from funcy import rpartial
from numpy.typing import ArrayLike

from baybe.utils import Dummy, get_base_unstructure_hook, unstructure_base
from baybe.utils.serialization import SerialMixin

log = logging.getLogger(__name__)


def _is_not_close(x: ArrayLike, y: ArrayLike, rtol: float, atol: float) -> np.ndarray:
    """The counterpart to `numpy.isclose`."""
    return np.logical_not(np.isclose(x, y, rtol=rtol, atol=atol))


class Condition(ABC, SerialMixin):
    """
    Abstract base class for all conditions. Conditions always evaluate an expression
    regarding a single parameter. Conditions are part of constraints, a constraint
    can have multiple conditions.
    """

    @abstractmethod
    def evaluate(self, data: pd.Series) -> pd.Series:
        """
        Evaluates the condition on a given data series.

        Parameters
        ----------
        data : pd.Series
            A series containing parameter values.

        Returns
        -------
        pd.Series
            A boolean series indicating which elements satisfy the condition.
        """


# provide threshold operators
_threshold_operators = {
    "<": ops.lt,
    "<=": ops.le,
    "=": rpartial(np.isclose, rtol=0.0),
    "==": rpartial(np.isclose, rtol=0.0),
    "!=": rpartial(_is_not_close, rtol=0.0),
    ">": ops.gt,
    ">=": ops.ge,
}

# define operators that are eligible for tolerance
_valid_tolerance_operators = ["=", "==", "!="]

_valid_logic_combiners = {
    "AND": ops.and_,
    "OR": ops.or_,
    "XOR": ops.xor,
}


@define
class ThresholdCondition(Condition):
    """
    Class for modelling threshold-based conditions.
    """

    # object variables
    threshold: float = field()
    operator: str = field(validator=[in_(_threshold_operators)])
    tolerance: Optional[float] = field()

    @tolerance.default
    def tolerance_default(self):
        """Default value for the tolerance"""
        return 1e-8 if self.operator in _valid_tolerance_operators else None

    @tolerance.validator
    def tolerance_validation(self, _, value):
        """Validates the threshold condition tolerance"""
        if (self.operator not in _valid_tolerance_operators) and (value is not None):
            raise ValueError(
                f"Setting the tolerance for a threshold condition is only valid "
                f"with the following operators: {_valid_tolerance_operators}."
            )

        if self.operator in _valid_tolerance_operators:
            if (value is None) or (value <= 0.0):
                raise ValueError(
                    f"When using a tolerance-enabled operator"
                    f" ({_valid_tolerance_operators}) the tolerance cannot be None "
                    f"or <= 0.0, but was {value}."
                )

    def evaluate(self, data: pd.Series) -> pd.Series:
        """See base class."""
        if data.dtype.kind not in "iufb":
            raise ValueError(
                "You tried to apply a threshold condition to non-numeric data. "
                "This operation is error-prone and not supported. Only use threshold "
                "conditions with numerical parameters."
            )
        func = rpartial(_threshold_operators[self.operator], self.threshold)
        if self.operator in _valid_tolerance_operators:
            func = rpartial(func, atol=self.tolerance)

        return data.apply(func)


@define
class SubSelectionCondition(Condition):
    """
    Class for defining valid parameter entries.
    """

    # object variables
    selection: List[Any] = field()

    def evaluate(self, data: pd.Series) -> pd.Series:
        """See base class."""
        return data.isin(self.selection)


@define
class Constraint(ABC, SerialMixin):
    """
    Abstract base class for all constraints. Constraints use conditions and chain them
    together to filter unwanted entries from the searchspace.
    """

    # class variables
    # TODO: it might turn out these are not needed at a later development stage
    eval_during_creation: ClassVar[bool]
    eval_during_modeling: ClassVar[bool]

    # Object variables
    parameters: List[str] = field(validator=min_len(1))

    @parameters.validator
    def validate_params(self, _, params):
        """Validates the parameter list."""
        if len(params) != len(set(params)):
            raise ValueError(
                f"The given 'parameters' list must have unique values "
                f"but was: {params}."
            )

    @abstractmethod
    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """
        Get the indices of dataframe entries that are invalid under the constraint.

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe where each row represents a particular parameter combination.

        Returns
        -------
        pd.Index
            The dataframe indices of rows where the constraint is violated.
        """


@define
class ExcludeConstraint(Constraint):
    """
    Class for modelling exclusion constraints.
    """

    # class variables
    eval_during_creation = True
    eval_during_modeling = False

    # object variables
    conditions: List[Condition] = field(validator=min_len(1))
    combiner: str = field(default="AND", validator=in_(_valid_logic_combiners))

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        satisfied = [
            cond.evaluate(data[self.parameters[k]])
            for k, cond in enumerate(self.conditions)
        ]
        res = reduce(_valid_logic_combiners[self.combiner], satisfied)
        return data.index[res]


@define
class SumConstraint(Constraint):
    """
    Class for modelling sum constraints.
    """

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # class variables
    eval_during_creation = True
    eval_during_modeling = False

    # object variables
    condition: ThresholdCondition = field()

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        evaluate_data = data[self.parameters].sum(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_data)

        return data.index[mask_bad]


@define
class ProductConstraint(Constraint):
    """
    Class for modelling product constraints.
    """

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # class variables
    eval_during_creation = True
    eval_during_modeling = False

    # object variables
    condition: ThresholdCondition = field()

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        evaluate_data = data[self.parameters].prod(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_data)

        return data.index[mask_bad]


class NoLabelDuplicatesConstraint(Constraint):
    """
    Constraint class for excluding entries where the occurring labels are not unique.
    This can be useful to remove entries that arise from e.g. a permutation invariance.
    Examples:
        - A,B,C,D would remain
        - A,A,B,C would be removed
        - A,A,B,B would be removed
        - A,A,B,A would be removed
        - A,C,A,C would be removed
        - A,C,B,C would be removed
    """

    # class variables
    eval_during_creation = True
    eval_during_modeling = False

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        mask_bad = data[self.parameters].nunique(axis=1) != len(self.parameters)

        return data.index[mask_bad]


class LinkedParametersConstraint(Constraint):
    """
    Constraint class for linking the values of parameters. This constraint type
    effectively allows generating parameter sets that relate to the same underlying
    quantity, e.g. two parameters that represent the same molecule using different
    encodings. Linking the parameters removes all entries from the searchspace where
    the parameter values differ.
    """

    # class variables
    eval_during_creation = True
    eval_during_modeling = False

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        mask_bad = data[self.parameters].nunique(axis=1) != 1

        return data.index[mask_bad]


@define
class DependenciesConstraint(Constraint):
    """
    Constraint that specifies dependencies between parameters. For instance some
    parameters might only be relevant when another parameter has a certain value
    (e.g. parameter switch is 'on'). All dependencies must be declared in a single
    constraint.
    """

    # class variables
    # TODO update usage in different evaluation stages once that is implemented in
    #  strategy and surrogate
    eval_during_creation = True
    eval_during_modeling = False

    # object variables
    conditions: List[Condition] = field()
    affected_parameters: List[List[str]] = field()
    # Flag that indicates whether the affected parameters are permutation invariant.
    # Not to be set by the user but by other constraints reusing this class.
    # TODO: This should be init=False, but would require changing the unstructuring
    #   logic. Let's wait for the next cattrs release with the following PR to be merged
    #   and then init=False attributes can be handled more elegantly:
    #   https://github.com/python-attrs/cattrs/pull/395/commits
    permutation_invariant: bool = field(default=False)

    @affected_parameters.validator
    def affected_parameters_validator(self, _, value):
        """Ensure each set of affected parameters has exactly one condition"""
        if len(self.conditions) != len(value):
            raise ValueError(
                f"For the {self.__class__.__name__}, for each item in the "
                f"affected_parameters list you must provide exactly one condition in "
                f"the conditions list."
            )

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        # Create data copy and mark entries where the dependency conditions are negative
        # with a dummy value to cause degeneracy.
        censored_data = data.copy()
        for k, _ in enumerate(self.parameters):
            censored_data.loc[
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
                    frozenset if self.permutation_invariant else tuple, axis=1
                ),
            ],
            axis=1,
        )
        inds_bad = data.index[df_eval.duplicated(keep="first")]

        return inds_bad


@define
class PermutationInvarianceConstraint(Constraint):
    """
    Constraint class for declaring that a set of parameters are permutation invariant,
    that is, (val_from_param1, val_from_param2) is equivalent to
    (val_from_param2, val_from_param1). Since it does not make sense to have this
    constraint with duplicated labels, this implementation also internally applies the
    `NoDuplicatesConstraint`.

    Note: This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.
    """

    # class variables
    # TODO update usage in different evaluation stages once that is implemented in
    #  strategy and surrogate
    eval_during_creation = True
    eval_during_modeling = False

    # object variables
    dependencies: Optional[DependenciesConstraint] = field(default=None)

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        # Get indices of entries with duplicate label entries. These will also be
        # dropped by this constraint.
        mask_duplicate_labels = pd.Series(False, index=data.index)
        mask_duplicate_labels[
            NoLabelDuplicatesConstraint(parameters=self.parameters).get_invalid(data)
        ] = True

        # Merge a permutation invariant representation of all affected parameters with
        # the other parameters and indicate duplicates. This ensures that variation in
        # other parameters is also accounted for.
        other_params = data.columns.drop(self.parameters).tolist()
        df_eval = pd.concat(
            [
                data[other_params].copy(),
                data[self.parameters].apply(frozenset, axis=1),
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
class CustomConstraint(Constraint):
    """
    Class for user-defined custom constraints.
    """

    # class variables
    eval_during_creation = True
    eval_during_modeling = False

    # object variables
    validator: Callable[[pd.Series], bool] = field()

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        mask_bad = ~data[self.parameters].apply(self.validator, axis=1)

        return data.index[mask_bad]


# the order in which the constraint types need to be applied
CONSTRAINTS_ORDER = (
    CustomConstraint,
    ExcludeConstraint,
    NoLabelDuplicatesConstraint,
    LinkedParametersConstraint,
    SumConstraint,
    ProductConstraint,
    PermutationInvarianceConstraint,
    DependenciesConstraint,
)


# Register structure / unstructure hooks
cattrs.register_unstructure_hook(Condition, unstructure_base)
cattrs.register_structure_hook(Condition, get_base_unstructure_hook(Condition))
cattrs.register_unstructure_hook(Constraint, unstructure_base)
cattrs.register_structure_hook(Constraint, get_base_unstructure_hook(Constraint))


def _custom_constraint_hook(*_):
    raise NotImplementedError("CustomConstraint does not support de-/serialization.")


cattrs.register_unstructure_hook(CustomConstraint, _custom_constraint_hook)
cattrs.register_structure_hook(CustomConstraint, _custom_constraint_hook)


def _validate_constraints(constraints: List[Constraint]) -> None:
    """
    Asserts that a given collection of constraints is valid.

    Parameters
    ----------
    constraints : list
        List of Constraint objects.

    Raises
    ------
    ValueError
        If the given list of constraints is invalid.
    """
    if sum(isinstance(itm, DependenciesConstraint) for itm in constraints) > 1:
        raise ValueError(
            f"There is only one {DependenciesConstraint.__name__} allowed. "
            f"Please specify all dependencies in one single constraint."
        )
