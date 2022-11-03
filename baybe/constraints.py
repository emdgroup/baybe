"""
Functionality for parameter constraints.
"""
from __future__ import annotations

import logging
import operator as ops
from abc import ABC, abstractmethod
from functools import reduce
from inspect import isabstract

from typing import ClassVar, Dict, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from funcy import rpartial
from numpy.typing import ArrayLike
from pydantic import BaseModel, conlist, Extra, validator

from .utils import check_if_in, StrictValidationError

log = logging.getLogger(__name__)

_constraints_order = [
    "EXCLUDE",
    "NO_LABEL_DUPLICATES",
    "LINKED_PARAMETERS",
    "SUM",
    "PRODUCT",
    "PERMUTATION_INVARIANCE",
    "DEPENDENCIES",
]


def _is_not_close(x: ArrayLike, y: ArrayLike, rtol: float, atol: float) -> np.ndarray:
    """The counterpart to `numpy.isclose`."""
    return np.logical_not(np.isclose(x, y, rtol=rtol, atol=atol))


class Condition(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all conditions. Conditions always evaluate an expression
    regarding a single parameter. Conditions are part of constraints, a constraint
    can have multiple conditions.
    """

    # class variables
    type: ClassVar[str]
    SUBCLASSES: ClassVar[Dict[str, Type[Constraint]]] = {}

    @classmethod
    def create(cls, config: dict) -> Condition:
        """Creates a new object matching the given specifications."""
        config = config.copy()
        condition_type = config.pop("type")
        check_if_in(condition_type, list(Condition.SUBCLASSES.keys()))
        return cls.SUBCLASSES[condition_type](**config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls

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


class ThresholdCondition(Condition):
    """
    Class for modelling threshold-based conditions.
    """

    # class variables
    type = "THRESHOLD"
    threshold: float
    operator: Literal["<", "<=", "=", "==", "!=", ">", ">="]
    tolerance: float = 1e-8

    # define the valid operators
    _operator_dict = {
        "<": ops.lt,
        "<=": ops.le,
        "=": rpartial(np.isclose, rtol=0.0),
        "==": rpartial(np.isclose, rtol=0.0),
        "!=": rpartial(_is_not_close, rtol=0.0),
        ">": ops.gt,
        ">=": ops.ge,
    }

    # define operators that are eligible for tolerance
    _tolerance_operators = ["=", "==", "!="]

    @validator("tolerance")
    def validate_tolerance(cls, tolerance, values):
        """Ensures tolerance can only be set with appropriate operators."""
        if values["operator"] not in cls._tolerance_operators:
            raise StrictValidationError(
                f"Tolerance for a threshold condition is only valid with the following "
                f"operators: {cls._tolerance_operators}."
            )
        return tolerance

    def evaluate(self, data: pd.Series) -> pd.Series:
        """See base class."""
        if data.dtype.kind not in "iufb":
            raise StrictValidationError(
                "You tried to apply a threshold condition to non-numeric data. "
                "This operation is error-prone and not supported. Only use threshold "
                "conditions with numerical parameters."
            )
        func = rpartial(self._operator_dict[self.operator], self.threshold)
        if self.operator in self._tolerance_operators:
            func = rpartial(func, atol=self.tolerance)

        return data.apply(func)


class SubSelectionCondition(Condition):
    """
    Class for defining valid parameter entries.
    """

    # class variables
    type = "SUBSELECTION"
    selection: list

    def evaluate(self, data: pd.Series) -> pd.Series:
        """See base class."""
        return data.isin(self.selection)


class Constraint(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all constraints. Constraints use conditions and chain them
    together to filter unwanted entries from the searchspace.
    """

    # class variables
    type: ClassVar[str]
    parameters: List[str]

    SUBCLASSES: ClassVar[Dict[str, Constraint]] = {}

    # TODO: it might turn out these are not needed at a later development stage
    eval_during_creation: ClassVar[bool]
    eval_during_modeling: ClassVar[bool]

    @validator("parameters")
    def validate_params(cls, parameters):
        """Validates the parameter list."""
        if len(parameters) != len(set(parameters)):
            raise AssertionError(
                f"The given 'parameter' list must have unique values "
                f"but was: {parameters}."
            )
        return parameters

    @classmethod
    def create(cls, config: dict) -> Constraint:
        """Creates a new object matching the given specifications."""
        config = config.copy()
        constraint_type = config.pop("type")
        check_if_in(constraint_type, list(Constraint.SUBCLASSES.keys()))
        return cls.SUBCLASSES[constraint_type](**config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls

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


class ExcludeConstraint(Constraint):
    """
    Class for modelling exclusion constraints.
    """

    # class variables
    type = "EXCLUDE"
    eval_during_creation = True
    eval_during_modeling = False
    conditions: conlist(Union[dict, Condition], min_items=1)
    combiner: Literal["AND", "OR", "XOR"] = "AND"

    @validator("conditions")
    def validate_conditions(cls, conditions):
        """Validates the conditions."""
        return [
            c if isinstance(c, Condition) else Condition.create(c) for c in conditions
        ]

    _combiner_dict = {
        "AND": ops.and_,
        "OR": ops.or_,
        "XOR": ops.xor,
    }

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        satisfied = [
            cond.evaluate(data[self.parameters[k]])
            for k, cond in enumerate(self.conditions)
        ]
        res = reduce(self._combiner_dict[self.combiner], satisfied)
        return data.index[res]


class SumConstraint(Constraint):
    """
    Class for modelling sum constraints.
    """

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # class variables
    type = "SUM"
    eval_during_creation = True
    eval_during_modeling = False
    condition: ThresholdCondition

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        evaluate_data = data[self.parameters].sum(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_data)

        return data.index[mask_bad]


class ProductConstraint(Constraint):
    """
    Class for modelling product constraints.
    """

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # class variables
    type = "PRODUCT"
    eval_during_creation = True
    eval_during_modeling = False
    condition: ThresholdCondition

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
    type = "NO_LABEL_DUPLICATES"
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
    type = "LINKED_PARAMETERS"
    eval_during_creation = True
    eval_during_modeling = False

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        mask_bad = data[self.parameters].nunique(axis=1) != 1

        return data.index[mask_bad]


class DependenciesConstraint(Constraint):
    """
    Constraint that specifies dependencies between parameters. For instance some
    parameters might only be relevant when another parameter has a certain value
    (e.g. 'on'). All dependencies must be declared in a single constraint.
    """

    type = "DEPENDENCIES"
    # TODO update usage in different evaluation stages once that is implemented in
    #  strategy and surrogate
    eval_during_creation = True
    eval_during_modeling = False
    conditions: List[Union[dict, Condition]]
    affected_parameters: List[List[str]]
    invariant = False

    @validator("conditions")
    def validate_conditions(cls, conditions):
        """Validates the conditions."""
        return [
            c if isinstance(c, Condition) else Condition.create(c) for c in conditions
        ]

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        # Create data copy and mark entries where the dependency conditions are negative
        # with a dummy value (None) to cause degeneracy.
        censored_data = data.copy()
        for k, _ in enumerate(self.parameters):
            censored_data.loc[
                ~self.conditions[k].evaluate(data[self.parameters[k]]),
                self.affected_parameters[k],
            ] = (
                # picking None as dummy value here does not work because it will be
                # converted to nan for float columns, evading comparison later
                np.finfo(float).eps
                * np.pi
            )

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
                    frozenset if self.invariant else tuple, axis=1
                ),
            ],
            axis=1,
        )
        inds_bad = data.index[df_eval.duplicated(keep="first")]

        return inds_bad


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
    type = "PERMUTATION_INVARIANCE"
    # TODO update usage in different evaluation stages once that is implemented in
    #  strategy and surrogate
    eval_during_creation = True
    eval_during_modeling = False
    dependencies: Optional[DependenciesConstraint]

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
            self.dependencies.invariant = True
            inds_duplicate_independency_adjusted = self.dependencies.get_invalid(
                data.drop(index=inds_invalid)
            )
            inds_invalid = inds_invalid.union(inds_duplicate_independency_adjusted)

        return inds_invalid
