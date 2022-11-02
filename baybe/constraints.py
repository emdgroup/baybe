"""
Functionality for parameter constraints.
"""
from __future__ import annotations

import logging
import operator as ops
from abc import ABC, abstractmethod
from functools import reduce
from inspect import isabstract

from typing import ClassVar, Dict, List, Literal, Type, Union

import pandas as pd
from funcy import rpartial
from pydantic import BaseModel, conlist, Extra, validator

from .utils import check_if_in

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


class Condition(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all conditions. Conditions are part of constraints,
    a constraint can have multiple conditions.
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

    # define the valid operators
    _operator_dict = {
        "<": ops.lt,
        "<=": ops.le,
        "=": ops.eq,
        "==": ops.eq,
        "!=": ops.ne,
        ">": ops.gt,
        ">=": ops.ge,
    }

    def evaluate(self, data: pd.Series) -> pd.Series:
        """See base class."""
        if data.dtype.kind not in "iufb":
            raise TypeError(
                "You tried to apply a threshold condition to non-numeric data. "
                "This operation is error-prone and not supported. Only use threshold "
                "conditions with numerical parameters."
            )

        return data.apply(rpartial(self._operator_dict[self.operator], self.threshold))


class SubSelectionCondition(Condition):
    """
    Class for defining valid parameter entries.
    """

    # class variables
    type = "SUBSELECTION"
    selection: list

    # TODO: Set up a validation that checks the sublist only contains valid entries.
    #  As this depends on the `Parameter`, it might be better to do it in `Constraint`.

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


class PermutationInvarianceConstraint(Constraint):
    """
    Constraint class for declaring that a set of parameters are permutation invariant,
    that is, (val_from_param1, val_from_param2) is equivalent to
    (val_from_param2, val_from_param1). Since it does not make sense to have this
    constraint with duplicated labels this implementation also drops duplicated labels
    similar to the NoLabelDuplicatesConstraint constraint.

    Note: This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.
    """

    # class variables
    type = "PERMUTATION_INVARIANCE"
    # TODO update usage in different evaluation stages once that is implemented in
    #  strategy and surrogate
    eval_during_creation = True
    eval_during_modeling = False

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """See base class."""
        # Get indices of entries with duplicate label entries. These will also be
        # dropped by this constraint.
        mask_duplicate_labels = data[self.parameters].nunique(axis=1) != len(
            self.parameters
        )

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
        ).loc[~mask_duplicate_labels]
        mask_duplicate_invariant = df_eval.duplicated(keep="first")

        inds_duplicate_labels = data.index[mask_duplicate_labels]
        inds_duplicate_invariant = df_eval.index[mask_duplicate_invariant]

        return inds_duplicate_labels.union(inds_duplicate_invariant)


class DependenciesConstraint(Constraint):
    """
    Constraint that specifies dependencies between parameters. For instance some
    parameters might only be relevant when another parameter has a certain value
    (e.g. 'on'). All dependencies must be declared in a single constraint.
    """

    # class variables
    type = "DEPENDENCIES"
    # TODO update usage in different evaluation stages once that is implemented in
    #  strategy and surrogate
    eval_during_creation = True
    eval_during_modeling = False
    conditions: List[Union[dict, Condition]]
    affected_parameters: List[List[str]]

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
        # TODO verify if also a censoring of the dependency-causing values might be
        #  necessary. If there is just one (e.g. 0.0) is should be fine but if there
        #  are multiple (e.g. "off" and "unpowered") it might not work as intended
        censored_data = data.copy()
        for k, _ in enumerate(self.parameters):
            censored_data.loc[
                ~self.conditions[k].evaluate(data[self.parameters[k]]),
                self.affected_parameters[k],
            ] = None

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
                censored_data[all_affected_params].apply(frozenset, axis=1),
            ],
            axis=1,
        )

        inds_bad = data.index[df_eval.duplicated(keep="first")]

        return inds_bad
