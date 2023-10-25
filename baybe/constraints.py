"""Functionality for parameter constraints."""

import operator as ops
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, ClassVar, List, Optional, Tuple, Union

import cattrs
import numpy as np
import pandas as pd
import torch
from attr import define, field
from attrs.validators import in_, min_len
from funcy import rpartial
from numpy.typing import ArrayLike
from torch import Tensor

from baybe.parameters import NumericalContinuousParameter, Parameter
from baybe.utils import (
    DTypeFloatTorch,
    Dummy,
    get_base_structure_hook,
    SerialMixin,
    unstructure_base,
)


def _is_not_close(x: ArrayLike, y: ArrayLike, rtol: float, atol: float) -> np.ndarray:
    """The counterpart to ```numpy.isclose```.

    Args:
        x: First input array to compare.
        y: Second input array to compare.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        Returns a boolean array of where ```x``` and ```y``` are not equal within the
        given tolerances.

    """
    return np.logical_not(np.isclose(x, y, rtol=rtol, atol=atol))


class Condition(ABC, SerialMixin):
    """Abstract base class for all conditions.

    Conditions always evaluate an expression regarding a single parameter.
    Conditions are part of constraints, a constrain can have multiple conditions.
    """

    @abstractmethod
    def evaluate(self, data: pd.Series) -> pd.Series:
        """Evaluates the condition on a given data series.

        Args:
            data: A series containing parameter values.

        Returns:
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
    """Class for modelling threshold-based conditions.

    Args:
        threshold: The threshold value used in the condition.
        operator: The operator used in the condition.
        tolerance: A numerical tolerance. Set to a reasonable default tolerance.
    """

    # object variables
    threshold: float = field()
    operator: str = field(validator=[in_(_threshold_operators)])
    tolerance: Optional[float] = field()

    @tolerance.default
    def _tolerance_default(self) -> Union[float, None]:
        """Create the default value for the tolerance."""
        # Default value for the tolerance.
        return 1e-8 if self.operator in _valid_tolerance_operators else None

    @tolerance.validator
    def _validate_tolerance(self, _: Any, value: float) -> None:  # noqa: DOC101, DOC103
        """Validate the threshold condition tolerance.

        Raises:
            ValueError: If the operator does not allow for setting a tolerance.
            ValueError: If the operator allows for setting a tolerance, but the provided
                tolerance is either less than 0 or ```None```.
        """
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

    def evaluate(self, data: pd.Series) -> pd.Series:  # noqa: D102
        # See base class.
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
    """Class for defining valid parameter entries.

    Args:
        selection: The list of items which are considered valid.
    """

    # object variables
    selection: List[Any] = field()

    def evaluate(self, data: pd.Series) -> pd.Series:  # noqa: D102
        # See base class.
        return data.isin(self.selection)


@define
class Constraint(ABC, SerialMixin):
    """Abstract base class for all constraints.

    Constraints use conditions and chain them together to filter unwanted entries from
    the search space.

    Args:
        parameters: The list of parameters used for the constraint.
    """

    # class variables
    # TODO: it might turn out these are not needed at a later development stage
    eval_during_creation: ClassVar[bool]
    """Class variable encoding whether the condition is evaluated during creation."""
    eval_during_modeling: ClassVar[bool]
    """Class variable encoding whether the condition is evaluated during modeling."""

    # Object variables
    parameters: List[str] = field(validator=min_len(1))

    @parameters.validator
    def _validate_params(  # noqa: DOC101, DOC103
        self, _: Any, params: List[str]
    ) -> None:
        """Validate the parameter list.

        Raises:
            ValueError: If ```params``` contains duplicate values.
        """
        if len(params) != len(set(params)):
            raise ValueError(
                f"The given 'parameters' list must have unique values "
                f"but was: {params}."
            )

    @property
    def is_continuous(self) -> bool:
        """Boolean indicating if this is a constraint over continuous parameters."""
        return isinstance(self, ContinuousConstraint)

    @property
    def is_discrete(self) -> bool:
        """Boolean indicating if this is a constraint over discrete parameters."""
        return isinstance(self, DiscreteConstraint)


@define
class DiscreteConstraint(Constraint, ABC):
    """Abstract base class for discrete constraints.

    Discrete constraints use conditions and chain them together to filter unwanted
    entries from the search space.
    """

    # class variables
    eval_during_creation: ClassVar[bool] = True
    eval_during_modeling: ClassVar[bool] = False

    @abstractmethod
    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """Get the indices of dataframe entries that are invalid under the constraint.

        Args:
            data: A dataframe where each row represents a particular parameter
                combination.

        Returns:
            The dataframe indices of rows where the constraint is violated.
        """


@define
class DiscreteExcludeConstraint(DiscreteConstraint):
    """Class for modelling exclusion constraints.

    Args:
        conditions: List of individual conditions.
        combiner: Operator encoding how to combine the individual conditions.
    """

    # object variables
    conditions: List[Condition] = field(validator=min_len(1))
    combiner: str = field(default="AND", validator=in_(_valid_logic_combiners))

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

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        evaluate_data = data[self.parameters].sum(axis=1)
        mask_bad = ~self.condition.evaluate(evaluate_data)

        return data.index[mask_bad]


@define
class DiscreteProductConstraint(DiscreteConstraint):
    """Class for modelling product constraints.

    Args:
        condition: The condition that is used for this constraint.
    """

    # IMPROVE: refactor `SumConstraint` and `ProdConstraint` to avoid code copying

    # object variables
    condition: ThresholdCondition = field()

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

    Args:
        conditions: The list of individual conditions.
        affected_parameters: The parameters affected by the individual conditions.
    """

    # object variables
    conditions: List[Condition] = field()
    affected_parameters: List[List[str]] = field()

    # for internal use only
    permutation_invariant: bool = field(default=False, init=False)
    """Flag that indicates whether the affected parameters are permutation invariant.
    This should not be changed by the user but by other constraints using the class."""

    @affected_parameters.validator
    def _validate_affected_parameters(  # noqa: DOC101, DOC103
        self, _: Any, value: List[List[str]]
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
class DiscretePermutationInvarianceConstraint(DiscreteConstraint):
    """Constraint class for declaring that a set of parameters is permutation invariant.

    More precisely, this means that, ```(val_from_param1, val_from_param2)``` is
    equivalent to ```(val_from_param2, val_from_param1)```. Since it does not make sense
    to have this constraint with duplicated labels, this implementation also internally
    applies the :func:`baybe.constraints.DiscreteNoLabelDuplicatesConstraint`.

    Note: This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.

    Args:
        dependencies: Dependencies connected with the invariant parameters.
    """

    # object variables
    dependencies: Optional[DiscreteDependenciesConstraint] = field(default=None)

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
class DiscreteCustomConstraint(DiscreteConstraint):
    """Class for user-defined custom constraints.

    Args:
        validator: A user-defined function modeling the validation of the constraint.
    """

    # object variables
    validator: Callable[[pd.Series], bool] = field()

    def get_invalid(self, data: pd.DataFrame) -> pd.Index:  # noqa: D102
        # See base class.
        mask_bad = ~data[self.parameters].apply(self.validator, axis=1)

        return data.index[mask_bad]


@define
class ContinuousConstraint(Constraint, ABC):
    """Abstract base class for continuous constraints.

    Continuous constraints use parameter lists and coefficients to define in-/equality
    constraints over a continuous parameter space.

    Args:
        parameters: See base class.
        coefficients: In-/equality coefficient for each entry in ```parameters```.
        rhs: Right-hand side value of the in-/equality.
    """

    # class variables
    eval_during_creation: ClassVar[bool] = False
    eval_during_modeling: ClassVar[bool] = True

    # object variables
    coefficients: List[float] = field()
    rhs: float = field(default=0.0)

    @coefficients.validator
    def _validate_coefficients(  # noqa: DOC101, DOC103
        self, _: Any, coefficients: List[float]
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

    @coefficients.default
    def _default_coefficients(self):
        """Return equal weight coefficients as default."""
        return [1.0] * len(self.parameters)

    def to_botorch(
        self, parameters: List[NumericalContinuousParameter], idx_offset: int = 0
    ) -> Tuple[Tensor, Tensor, float]:
        """Cast the constraint in a format required by botorch.

        Used in calling ```optimize_acqf_*``` functions, for details see
        https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf

        Args:
            parameters: The parameter objects of the continuous space.
            idx_offset: Offset to the provided parameter indices.

        Returns:
            The tuple required by botorch.
        """
        param_names = [p.name for p in parameters]
        param_indices = [
            param_names.index(p) + idx_offset
            for p in self.parameters
            if p in param_names
        ]

        return (
            torch.tensor(param_indices),
            torch.tensor(self.coefficients, dtype=DTypeFloatTorch),
            self.rhs,
        )


@define
class ContinuousLinearEqualityConstraint(ContinuousConstraint):
    """Class for continuous equality constraints.

    The constraint is defined as `sum_i[ x_i * c_i ] == rhs`, where x_i are the
    parameter names from ```parameters``` and c_i are the entries of ```coefficients```.
    The constraint is typically fulfilled up to a small numerical tolerance.

    The class has no content as it only serves the purpose of distinguishing the
    constraints.
    """


@define
class ContinuousLinearInequalityConstraint(ContinuousConstraint):
    """Class for continuous inequality constraints.

    The constraint is defined as `sum_i[ x_i * c_i ] >= rhs`, where x_i are the
    parameter names from ```parameters``` and c_i are the entries of ```coefficients```.
    If you want to implement a constraint of the form `<=`, multiply ```rhs``` and
    ```coefficients``` by -1. The constraint is typically fulfilled up to a small
    numerical tolerance.

    The class has no content as it only serves the purpose of
    distinguishing the constraints.
    """


# the order in which the constraint types need to be applied during discrete subspace
# filtering
DISCRETE_CONSTRAINTS_FILTERING_ORDER = (
    DiscreteCustomConstraint,
    DiscreteExcludeConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteSumConstraint,
    DiscreteProductConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteDependenciesConstraint,
)


# Register structure / unstructure hooks
cattrs.register_unstructure_hook(Condition, unstructure_base)
cattrs.register_structure_hook(Condition, get_base_structure_hook(Condition))
cattrs.register_unstructure_hook(Constraint, unstructure_base)
cattrs.register_structure_hook(Constraint, get_base_structure_hook(Constraint))


def _custom_constraint_hook(*_) -> None:
    """Raises a NotImplementedError when trying to serialize a CustomConstraint."""
    raise NotImplementedError("CustomConstraint does not support de-/serialization.")


cattrs.register_unstructure_hook(DiscreteCustomConstraint, _custom_constraint_hook)
cattrs.register_structure_hook(DiscreteCustomConstraint, _custom_constraint_hook)


def _validate_constraints(  # noqa: DOC101, DOC103
    constraints: List[Constraint], parameters: List[Parameter]
) -> None:
    """Assert that a given collection of constraints is valid.

    Raises:
        ValueError: If there is more than one
            :class:`baybe.constraints.DiscreteDependenciesConstraint` declared.
        ValueError: If any constraint contains an invalid parameter name.
        ValueError: If any continuous constraint includes a discrete parameter.
        ValueError: If any discrete constraint includes a continuous parameter.
    """
    if sum(isinstance(itm, DiscreteDependenciesConstraint) for itm in constraints) > 1:
        raise ValueError(
            f"There is only one {DiscreteDependenciesConstraint.__name__} allowed. "
            f"Please specify all dependencies in one single constraint."
        )

    param_names_all = [p.name for p in parameters]
    param_names_discrete = [p.name for p in parameters if p.is_discrete]
    param_names_continuous = [p.name for p in parameters if not p.is_discrete]
    for constraint in constraints:
        if not all(p in param_names_all for p in constraint.parameters):
            raise ValueError(
                f"You are trying to create a constraint with at least one parameter "
                f"name that does not exist in the list of defined parameters. "
                f"Parameter list of the affected constraint: {constraint.parameters}"
            )

        if constraint.is_continuous and any(
            p in param_names_discrete for p in constraint.parameters
        ):
            raise ValueError(
                f"You are trying to initialize a continuous constraint over a "
                f"parameter that is discrete. Parameter list of the affected "
                f"constraint: {constraint.parameters}"
            )

        if constraint.is_discrete and any(
            p in param_names_continuous for p in constraint.parameters
        ):
            raise ValueError(
                f"You are trying to initialize a discrete constraint over a parameter "
                f"that is continuous. Parameter list of the affected constraint: "
                f"{constraint.parameters}"
            )
