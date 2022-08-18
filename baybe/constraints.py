"""
Functionality to deal wth constraints between parameters
"""
from __future__ import annotations

import logging
import operator as ops
from abc import ABC, abstractmethod
from functools import partial

from typing import ClassVar, Dict, List, Literal, Union

import pandas as pd
from pydantic import BaseModel, conlist, Extra, validator

from .utils import check_if_in

log = logging.getLogger(__name__)


class Condition(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all conditions. Conditions are part of constraints,
    a constraint can have multiple conditions.
    """

    # class variables
    type: ClassVar[str]
    parameter: str
    SUBCLASSES: ClassVar[Dict[str, Constraint]] = {}

    @classmethod
    def create(cls, config: dict) -> Condition:
        """Creates a new parameter object matching the given specifications."""
        config = config.copy()
        condition_type = config.pop("type")
        check_if_in(condition_type, list(Condition.SUBCLASSES.keys()))
        return cls.SUBCLASSES[condition_type](**config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls

    @abstractmethod
    def evaluate(self, data: pd.Series) -> pd.Series:
        """
        Evaluates the condition on a series and returns True or False values as entries

        Parameters
        ----------
        data: pd.Series
            Series with the parameter values

        Returns
        -------
            pd.Series
        """


class ThresholdCondition(Condition):
    """
    Constraint class for threshold-based condition
    """

    # class variables
    type = "THRESHOLD"
    threshold: float
    operator: Literal["<", "<=", "=", "==", "!=", ">", ">="]

    # Define valid operators. The order of the operands had to be switched because
    # otherwise partial cannot define the second argument as partial always uses the
    # first argument first for non-keyword arguments
    _operator_dict = {
        # pylint: disable=arguments-out-of-order
        "<": lambda a, b: ops.lt(b, a),
        "<=": lambda a, b: ops.le(b, a),
        "=": lambda a, b: ops.eq(b, a),
        "==": lambda a, b: ops.eq(b, a),
        "!=": lambda a, b: ops.ne(b, a),
        ">": lambda a, b: ops.gt(b, a),
        ">=": lambda a, b: ops.ge(b, a),
    }

    def evaluate(self, data: pd.Series) -> pd.Series:
        """see base class"""
        return data.apply(partial(self._operator_dict[self.operator], self.threshold))


class SubselectionCondition(Condition):
    """
    Constraint class for defining a list of valid entries
    """

    # class variables
    type = "SUBSELECTION"
    selection: list

    # TODO set up a validation that checks the sublist only contains valid entries.
    #  As this depends on the Parameter it might be better to do that in Constraint

    def evaluate(self, data: pd.Series) -> pd.Series:
        """see base class"""
        return data.apply(lambda x: x in self.selection)


class Constraint(ABC, BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """
    Abstract base class for all constraints. Constraints use conidtions and chain them
    together to filter unwanted entries in the searchspace.
    """

    # class variables
    type: ClassVar[str]
    SUBCLASSES: ClassVar[Dict[str, Constraint]] = {}

    @classmethod
    def create(cls, config: dict) -> Constraint:
        """Creates a new parameter object matching the given specifications."""
        config = config.copy()
        constraint_type = config.pop("type")
        check_if_in(constraint_type, list(Constraint.SUBCLASSES.keys()))
        return cls.SUBCLASSES[constraint_type](**config)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.type] = cls

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Index:
        """
        Evaluates all condition associated with this constraint and combine them in
        the specified way

        Parameters
        ----------
        data: pd.DataFrame
            Contains datapoints in experimental representations

        Returns
        -------
        pd.Index
            Index of entries that fall udner the defined constraint and should thus
            be removed from the searchspace
        """


class ExcludeConstraint(Constraint):
    """
    Constraint class for exclusion-constraints
    """

    # class variables
    type = "EXCLUDE"
    conditions: conlist(Union[dict, Condition], min_items=1)
    combiner: Literal["AND", "OR", "XOR"] = "AND"

    @validator("conditions")
    def validate_conditions(cls, conditions):
        """Validate the conditions"""
        print("####### AAAAAAAAAAAAAAAAAA: ", conditions)
        return [Condition.create(c) for c in conditions]

    _combiner_dict = {
        "AND": ops.and_,
        "OR": ops.or_,
        "XOR": ops.xor,
    }

    _conditions_allowed_parameters = {
        "THRESHOLD": ["NUM_DISCRETE"],  # NUM_CONTINUOUS should also be here but that
        # requires additonal logic
        "SUBSELECTION": ["CAT", "NUM_DISCRETE", "SUBSTANCE", "CUSTOM"],
    }

    # TODO validate that condition types match with the allowed parameter types

    def evaluate(self, data: pd.DataFrame) -> pd.Index:
        """see base class"""

        res = self.conditions[0].evaluate(data[self.conditions[0].parameter])
        for cond in self.conditions:
            res = self._combiner_dict[self.combiner](
                res, cond.evaluate(data[cond.parameter])
            )

        return data.index[res]


class SumTargetConstraint(Constraint):
    """
    Constraint class for product-constraints
    """

    # class variables
    type = "SUM_TARGET"
    parameters: List[str]
    target_value: float
    tolerance: float = 0.0

    @validator("parameters")
    def validate_params(cls, parameters):
        """Validate parameter list"""
        if len(parameters) != len(set(parameters)):
            raise AssertionError(
                f"The 'parameter' list you gave for a constraint must have unique "
                f"values, but was: {parameters}"
            )
        return parameters

    def evaluate(self, data: pd.DataFrame) -> pd.Index:
        """see base class"""
        mask_bad = (
            data[self.parameters].sum(axis=1) - self.target_value
        ).abs() > self.tolerance

        return data.index[mask_bad]


class ProdTargetConstraint(Constraint):
    """
    Constraint class for sum-constraints
    """

    # class variables
    type = "PROD_TARGET"
    parameters: List[str]
    target_value: float
    tolerance: float = 0.0

    @validator("parameters")
    def validate_params(cls, parameters):
        """Validate parameter list"""
        if len(parameters) != len(set(parameters)):
            raise AssertionError(
                f"The 'parameter' list you gave for a constraint must have unique "
                f"values, but was: {parameters}"
            )
        return parameters

    def evaluate(self, data: pd.DataFrame) -> pd.Index:
        """see base class"""
        mask_bad = (
            data[self.parameters].prod(axis=1) - self.target_value
        ).abs() > self.tolerance

        return data.index[mask_bad]


class NoDuplicatesConstraint(Constraint):
    """
    Constraint class for excluding combinations where parameters have identical values
    """

    # class variables
    type = "NO_DUPLICATES"
    parameters: List[str]

    @validator("parameters")
    def validate_params(cls, parameters):
        """Validate parameter list"""
        if len(parameters) != len(set(parameters)):
            raise AssertionError(
                f"The 'parameter' list you gave for a constraint must have unique "
                f"values, but was: {parameters}"
            )
        return parameters

    def evaluate(self, data: pd.DataFrame) -> pd.Index:
        """see base class"""
        mask_bad = data[self.parameters].nunique(axis=1) != len(self.parameters)

        return data.index[mask_bad]
