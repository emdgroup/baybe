"""Functionality for managing search spaces."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, cast

import pandas as pd
import torch
from attr import define, field

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
    validate_constraints,
)
from baybe.constraints.base import Constraint, DiscreteConstraint
from baybe.parameters import (
    NumericalContinuousParameter,
    SubstanceEncoding,
    TaskParameter,
)
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.searchspace.validation import validate_parameters
from baybe.telemetry import TELEM_LABELS, telemetry_record_value
from baybe.utils import SerialMixin, converter


class SearchSpaceType(Enum):
    """Enum class for different types of search spaces and respective compatibility."""

    DISCRETE = "DISCRETE"
    """Flag for discrete search spaces resp. compatibility with discrete search
    spaces."""

    CONTINUOUS = "CONTINUOUS"
    """Flag for continuous search spaces resp. compatibility with continuous
    search spaces."""

    EITHER = "EITHER"
    """Flag compatibility with either discrete or continuous, but not hybrid
    search spaces."""

    HYBRID = "HYBRID"
    """Flag for hybrid search spaces resp. compatibility with hybrid search spaces."""


@define
class SearchSpace(SerialMixin):
    """Class for managing the overall search space.

    The search space might be purely discrete, purely continuous, or hybrid.
    Note that created objects related to the computational representations of parameters
    (e.g., parameter bounds, computational dataframes, etc.) may use a different
    parameter order than what is specified through the constructor: While the
    passed parameter list can contain parameters in arbitrary order, the
    aforementioned objects (by convention) list discrete parameters first, followed
    by continuous ones.
    """

    discrete: SubspaceDiscrete = field(factory=SubspaceDiscrete.empty)
    """The (potentially empty) discrete subspace of the overall search space."""

    continuous: SubspaceContinuous = field(factory=SubspaceContinuous.empty)
    """The (potentially empty) continuous subspace of the overall search space."""

    def __attrs_post_init__(self):
        """Perform validation and record telemetry values."""
        validate_parameters(self.parameters)
        validate_constraints(self.constraints, self.parameters)

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_SEARCHSPACE_CREATION"], 1)
        telemetry_record_value(TELEM_LABELS["NUM_PARAMETERS"], len(self.parameters))
        telemetry_record_value(
            TELEM_LABELS["NUM_CONSTRAINTS"],
            len(self.constraints) if self.constraints else 0,
        )

    @classmethod
    def from_product(
        cls,
        parameters: List[Parameter],
        constraints: Optional[List[Constraint]] = None,
        empty_encoding: bool = False,
    ) -> SearchSpace:
        """Create a search space from a cartesian product.

        In the search space, optional subsequent constraints are applied.
        That is, the discrete subspace becomes the (filtered) cartesian product
        containing all discrete parameter combinations while, analogously, the
        continuous subspace represents the (filtered) cartesian product of all
        continuous parameters.

        Args:
            parameters: The parameters spanning the search space.
            constraints: An optional set of constraints restricting the valid parameter
                space.
            empty_encoding: If ``True``, uses an "empty" encoding for all parameters.
                This is useful, for instance, in combination with random search
                strategies that do not read the actual parameter values, since it avoids
                the (potentially costly) transformation of the parameter values to their
                computational representation.

        Returns:
            The constructed search space.
        """
        # IMPROVE: The arguments get pre-validated here to avoid the potentially costly
        #   creation of the subspaces. Perhaps there is an elegant way to bypass the
        #   default validation in the initializer (which is required for other
        #   ways of object creation) in this particular case.
        validate_parameters(parameters)
        if constraints:
            validate_constraints(constraints, parameters)
        else:
            constraints = []

        discrete: SubspaceDiscrete = SubspaceDiscrete.from_product(
            parameters=[
                cast(DiscreteParameter, p) for p in parameters if p.is_discrete
            ],
            constraints=[
                cast(DiscreteConstraint, c) for c in constraints if c.is_discrete
            ],
            empty_encoding=empty_encoding,
        )
        continuous: SubspaceContinuous = SubspaceContinuous(
            parameters=[
                cast(NumericalContinuousParameter, p)
                for p in parameters
                if not p.is_discrete
            ],
            constraints_lin_eq=[
                cast(ContinuousLinearEqualityConstraint, c)
                for c in constraints
                if isinstance(c, ContinuousLinearEqualityConstraint)
            ],
            constraints_lin_ineq=[
                cast(ContinuousLinearInequalityConstraint, c)
                for c in constraints
                if isinstance(c, ContinuousLinearInequalityConstraint)
            ],
        )

        return SearchSpace(discrete=discrete, continuous=continuous)

    @property
    def parameters(self) -> List[Parameter]:
        """Return the list of parameters of the search space."""
        return self.discrete.parameters + self.continuous.parameters

    @property
    def constraints(self) -> List[Constraint]:
        """Return the constraints of the search space."""
        return (
            self.discrete.constraints
            + self.continuous.constraints_lin_eq
            + self.continuous.constraints_lin_ineq
        )

    @property
    def type(self) -> SearchSpaceType:
        """Return the type of the search space."""
        if self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.CONTINUOUS
        if not self.discrete.is_empty and self.continuous.is_empty:
            return SearchSpaceType.DISCRETE
        if not self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.HYBRID
        raise RuntimeError("This line should be impossible to reach.")

    @property
    def contains_mordred(self) -> bool:
        """Indicates if any of the discrete parameters uses ``MORDRED`` encoding."""
        return any(
            p.encoding is SubstanceEncoding.MORDRED for p in self.discrete.parameters
        )

    @property
    def contains_rdkit(self) -> bool:
        """Indicates if any of the discrete parameters uses ``RDKIT`` encoding."""
        return any(
            p.encoding is SubstanceEncoding.RDKIT for p in self.discrete.parameters
        )

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """Return bounds as tensor."""
        return torch.hstack(
            [self.discrete.param_bounds_comp, self.continuous.param_bounds_comp]
        )

    @property
    def task_idx(self) -> Optional[int]:
        """The column index of the task parameter in computational representation."""
        try:
            # TODO [16932]: Redesign metadata handling
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
        except StopIteration:
            return None
        # TODO[11611]: The current approach has two limitations:
        #   1.  It matches by column name and thus assumes that the parameter name
        #       is used as the column name.
        #   2.  It relies on the current implementation detail that discrete parameters
        #       appear first in the computational dataframe.
        #   --> Fix this when refactoring the data
        return self.discrete.comp_rep.columns.get_loc(task_param.name)

    @property
    def n_tasks(self) -> int:
        """The number of tasks encoded in the search space."""
        # TODO [16932]: This approach only works for a single task parameter. For
        #  multiple task parameters, we need to align what the output should even
        #  represent (e.g. number of combinatorial task combinations, number of
        #  tasks per task parameter, etc).
        try:
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
            return len(task_param.values)

        # When there are no task parameters, we effectively have a single task
        except StopIteration:
            return 1

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data from experimental to computational representation.

        This function can e.g. be used to transform data obtained from measurements.
        Continuous parameters are not transformed but included.

        Args:
            data: The data to be transformed. Must contain all specified parameters, can
                contain more columns.

        Returns:
            A dataframe with the parameters in computational representation.
        """
        # Transform subspaces separately
        df_discrete = self.discrete.transform(data)
        df_continuous = self.continuous.transform(data)

        # Combine Subspaces
        comp_rep = pd.concat([df_discrete, df_continuous], axis=1)

        return comp_rep


def structure_searchspace_from_config(specs: dict, _) -> SearchSpace:
    """Assemble the search space from "config" format.

    It uses the alternative :func:`baybe.searchspace.core.SearchSpace.from_product`
    constructor, which allows to deserialize search space specifications that are
    provided in a user-friendly format (i.e. via parameters and constraints).
    """
    parameters = converter.structure(specs["parameters"], List[Parameter])
    constraints = specs.get("constraints", None)
    if constraints:
        constraints = converter.structure(specs["constraints"], List[Constraint])
    return SearchSpace.from_product(parameters, constraints)


def validate_searchspace_from_config(specs: dict, _) -> None:
    """Validate but do not create the search space.

    Similar to :func:`baybe.searchspace.core.structure_searchspace_from_config` but
    without the actual search space creation step, thus intended for validation purposes
    only. It explicitly validates the given parameters and constraints since invalid
    specifications would be otherwise noticed only later during search space creation.
    """
    parameters = converter.structure(specs["parameters"], List[Parameter])
    validate_parameters(parameters)

    constraints = specs.get("constraints", None)
    if constraints:
        constraints = converter.structure(specs["constraints"], List[Constraint])
        validate_constraints(constraints, parameters)
