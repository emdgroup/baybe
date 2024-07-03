"""Continuous subspaces."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable, Sequence
from functools import reduce
from itertools import chain, product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attr import define, field

from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.base import ContinuousConstraint, ContinuousNonlinearConstraint
from baybe.constraints.validation import (
    validate_cardinality_constraints_are_nonoverlapping,
    validate_parameters_bounds_in_cardinality_constraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.base import ContinuousParameter
from baybe.parameters.utils import get_parameters_from_dataframe
from baybe.searchspace.validation import (
    get_transform_parameters,
    validate_parameter_names,
)
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.utils.basic import to_tuple
from baybe.utils.dataframe import pretty_print_df
from baybe.utils.numerical import DTypeFloatNumpy

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace

_MAX_CARDINALITY_SAMPLING_ATTEMPTS = 10_000
ZERO_THRESHOLD = 1e-5


@define
class SubspaceContinuous(SerialMixin):
    """Class for managing continuous subspaces.

    Builds the subspace from parameter definitions, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    parameters: tuple[NumericalContinuousParameter, ...] = field(
        converter=to_tuple, validator=lambda _, __, x: validate_parameter_names(x)
    )
    """The parameters of the subspace."""

    constraints_lin_eq: tuple[ContinuousLinearEqualityConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """Linear equality constraints."""

    constraints_lin_ineq: tuple[ContinuousLinearInequalityConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """Linear inequality constraints."""

    constraints_nonlin: tuple[ContinuousNonlinearConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """Nonlinear constraints."""

    def __str__(self) -> str:
        if self.is_empty:
            return ""

        start_bold = "\033[1m"
        end_bold = "\033[0m"

        # Convert the lists to dataFrames to be able to use pretty_printing
        param_list = [param.summary() for param in self.parameters]
        eq_constraints_list = [constr.summary() for constr in self.constraints_lin_eq]
        ineq_constraints_list = [
            constr.summary() for constr in self.constraints_lin_ineq
        ]
        nonlin_constraints_list = [
            constr.summary() for constr in self.constraints_nonlin
        ]
        param_df = pd.DataFrame(param_list)
        lin_eq_constr_df = pd.DataFrame(eq_constraints_list)
        lin_ineq_constr_df = pd.DataFrame(ineq_constraints_list)
        nonlinear_constr_df = pd.DataFrame(nonlin_constraints_list)

        # Put all attributes of the continuous class in one string
        continuous_str = f"""{start_bold}Continuous Search Space{end_bold}
            \n{start_bold}Continuous Parameters{end_bold}\n{pretty_print_df(param_df)}
            \n{start_bold}List of Linear Equality Constraints{end_bold}
            \r{pretty_print_df(lin_eq_constr_df)}
            \n{start_bold}List of Linear Inequality Constraints{end_bold}
            \r{pretty_print_df(lin_ineq_constr_df)}
            \n{start_bold}List of Nonlinear Constraints{end_bold}
            \r{pretty_print_df(nonlinear_constr_df)}"""

        return continuous_str.replace("\n", "\n ").replace("\r", "\r ")

    @property
    def constraints_cardinality(self) -> tuple[ContinuousCardinalityConstraint, ...]:
        """Cardinality constraints."""
        return tuple(
            c
            for c in self.constraints_nonlin
            if isinstance(c, ContinuousCardinalityConstraint)
        )

    @property
    def combinatorial_counts_zero_parameters(self) -> int:
        """Return the total number of all possible combinations of zero parameters."""
        # Note that both continuous subspace and continuous cardinality constraint
        # have this property. This property is the counts for the subspace
        # parameters; while the latter one is the counts only for that constraint.
        if self.constraints_cardinality:
            return reduce(
                lambda x, y: x * y,
                [
                    con.combinatorial_counts_zero_parameters
                    for con in self.constraints_cardinality
                ],
            )
        else:
            return 0

    @property
    def combinatorial_zero_parameters(self) -> Iterable[tuple[str, ...]]:
        """Return a combinatorial list of all possible zero parameters on subspace."""
        # The comments on the difference in `combinatorial_counts_zero_parameters`
        # applies here as well.
        if self.constraints_cardinality:
            return product(
                *[
                    con.combinatorial_zero_parameters
                    for con in self.constraints_cardinality
                ]
            )

    @constraints_nonlin.validator
    def _validate_constraints_nonlin(self, _, __) -> None:
        """Validate nonlinear constraints."""
        # Note: The passed constraints are accessed indirectly through the property
        validate_cardinality_constraints_are_nonoverlapping(
            self.constraints_cardinality
        )

        for con in self.constraints_cardinality:
            validate_parameters_bounds_in_cardinality_constraint(self.parameters, con)

    def to_searchspace(self) -> SearchSpace:
        """Turn the subspace into a search space with no discrete part."""
        from baybe.searchspace.core import SearchSpace

        return SearchSpace(continuous=self)

    @classmethod
    def empty(cls) -> SubspaceContinuous:
        """Create an empty continuous subspace."""
        return SubspaceContinuous([])

    @classmethod
    def from_parameter(cls, parameter: ContinuousParameter) -> SubspaceContinuous:
        """Create a subspace from a single parameter.

        Args:
            parameter: The parameter to span the subspace.

        Returns:
            The created subspace.
        """
        return cls.from_product([parameter])

    @classmethod
    def from_product(
        cls,
        parameters: Sequence[ContinuousParameter],
        constraints: Sequence[ContinuousConstraint] | None = None,
    ) -> SubspaceContinuous:
        """See :class:`baybe.searchspace.core.SearchSpace`."""
        constraints = constraints or []
        return SubspaceContinuous(
            parameters=[p for p in parameters if p.is_continuous],  # type:ignore[misc]
            constraints_lin_eq=[  # type:ignore[misc]
                c
                for c in constraints
                if isinstance(c, ContinuousLinearEqualityConstraint)
            ],
            constraints_lin_ineq=[  # type:ignore[misc]
                c
                for c in constraints
                if isinstance(c, ContinuousLinearInequalityConstraint)
            ],
            constraints_nonlin=[
                c for c in constraints if isinstance(c, ContinuousNonlinearConstraint)
            ],
        )

    @classmethod
    def from_bounds(cls, bounds: pd.DataFrame) -> SubspaceContinuous:
        """Create a hyperrectangle-shaped continuous subspace with given bounds.

        Args:
            bounds: The bounds of the parameters.

        Returns:
            The constructed subspace.
        """
        # Assert that the input represents valid bounds
        assert bounds.shape[0] == 2
        assert (np.diff(bounds.values, axis=0) >= 0).all()
        assert bounds.apply(pd.api.types.is_numeric_dtype).all()

        # Create the corresponding parameters and from them the search space
        parameters = [
            NumericalContinuousParameter(cast(str, name), bound)
            for (name, bound) in bounds.items()
        ]
        return SubspaceContinuous(parameters)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Sequence[ContinuousParameter] | None = None,
    ) -> SubspaceContinuous:
        """Create a hyperrectangle-shaped continuous subspace from a dataframe.

        More precisely, create the smallest axis-aligned hyperrectangle-shaped
        continuous subspace that contains the points specified in the given dataframe.

        Args:
            df: The dataframe specifying the points spanning the subspace.
            parameters: Optional parameter objects corresponding to the columns in the
                given dataframe that can be provided to explicitly control parameter
                attributes. If a match between column name and parameter name is found,
                the corresponding parameter object is used. If a column has no match in
                the parameter list, a new
                :class:`baybe.parameters.numerical.NumericalContinuousParameter`
                is created with default optional arguments. For more details, see
                :func:`baybe.parameters.utils.get_parameters_from_dataframe`.

        Raises:
            ValueError: If parameter types other than
                :class:`baybe.parameters.numerical.NumericalContinuousParameter`
                are provided.

        Returns:
            The created continuous subspace.
        """
        # TODO: Add option for convex hull once constraints are in place

        if parameters and not all(
            isinstance(p, NumericalContinuousParameter) for p in parameters
        ):
            raise ValueError(
                "Currently, only parameters of type "
                "'{NumericalContinuousParameter.__name__}' are supported."
            )

        def continuous_parameter_factory(name: str, values: Collection[Any]):
            return NumericalContinuousParameter(name, (min(values), max(values)))

        # Get the full list of both explicitly and implicitly defined parameter
        parameters = get_parameters_from_dataframe(
            df, continuous_parameter_factory, parameters
        )

        return cls(parameters)

    @property
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def param_names(self) -> tuple[str, ...]:
        """Return list of parameter names."""
        return tuple(p.name for p in self.parameters)

    @property
    def param_names_in_cardinality_constraint(self) -> tuple[str, ...]:
        """Return list of parameter names involved in cardinality constraints."""
        params_per_cardinatliy_constraint = [
            c.parameters for c in self.constraints_cardinality
        ]
        return tuple(chain(*params_per_cardinatliy_constraint))

    @property
    def param_bounds_comp(self) -> np.ndarray:
        """Return bounds as numpy array."""
        if not self.parameters:
            return np.empty((2, 0), dtype=DTypeFloatNumpy)
        return np.stack([p.bounds.to_ndarray() for p in self.parameters]).T

    def _drop_parameters(self, parameter_names: Collection[str]) -> SubspaceContinuous:
        """Create a copy of the subspace with certain parameters removed.

        Args:
            parameter_names: The names of the parameter to be removed.

        Returns:
            The reduced subspace.
        """
        return SubspaceContinuous(
            parameters=[p for p in self.parameters if p.name not in parameter_names],
            constraints_lin_eq=[
                c._drop_parameters(parameter_names) for c in self.constraints_lin_eq
            ],
            constraints_lin_ineq=[
                c._drop_parameters(parameter_names) for c in self.constraints_lin_ineq
            ],
        )

    def transform(
        self,
        df: pd.DataFrame | None = None,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool | None = None,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """See :func:`baybe.searchspace.core.SearchSpace.transform`."""
        # >>>>>>>>>> Deprecation
        if not ((df is None) ^ (data is None)):
            raise ValueError(
                "Provide the dataframe to be transformed as argument to `df`."
            )

        if data is not None:
            df = data
            warnings.warn(
                "Providing the dataframe via the `data` argument is deprecated and "
                "will be removed in a future version. Please pass your dataframe "
                "as positional argument instead.",
                DeprecationWarning,
            )

        # Mypy does not infer from the above that `df` must be a dataframe here
        assert isinstance(df, pd.DataFrame)

        if allow_extra is None:
            allow_extra = True
            if set(df) - {p.name for p in self.parameters}:
                warnings.warn(
                    "For backward compatibility, the new `allow_extra` flag is set "
                    "to `True` when left unspecified. However, this behavior will be "
                    "changed in a future version. If you want to invoke the old "
                    "behavior, please explicitly set `allow_extra=True`.",
                    DeprecationWarning,
                )
        # <<<<<<<<<< Deprecation

        # Extract the parameters to be transformed
        parameters = get_transform_parameters(
            self.parameters, df, allow_missing, allow_extra
        )

        # Transform the parameters
        return df[[p.name for p in parameters]]

    def samples_random(self, n_points: int = 1) -> pd.DataFrame:
        """Deprecated!"""  # noqa: D401
        warnings.warn(
            f"The method '{SubspaceContinuous.samples_random.__name__}' "
            f"has been deprecated and will be removed in a future version. "
            f"Use '{SubspaceContinuous.sample_uniform.__name__}' instead.",
            DeprecationWarning,
        )
        return self.sample_uniform(n_points)

    def sample_uniform(self, batch_size: int = 1) -> pd.DataFrame:
        """Draw uniform random parameter configurations from the continuous space.

        Args:
            batch_size: The number of parameter configurations to be sampled.

        Returns:
            A dataframe containing the parameter configurations as rows with columns
            corresponding to the parameter names.

        Raises:
            ValueError: If the subspace contains unsupported nonlinear constraints.
        """
        if not all(
            isinstance(c, ContinuousCardinalityConstraint)
            for c in self.constraints_nonlin
        ):
            raise ValueError(
                f"Currently, only nonlinear constraints of type "
                f"'{ContinuousCardinalityConstraint.__name__}' are supported."
            )

        if not self.parameters:
            return pd.DataFrame(index=pd.RangeIndex(0, batch_size))

        if (
            len(self.constraints_lin_eq) == 0
            and len(self.constraints_lin_ineq) == 0
            and len(self.constraints_cardinality) == 0
        ):
            return self._sample_from_bounds(batch_size, self.param_bounds_comp)

        if len(self.constraints_cardinality) == 0:
            return self._sample_from_polytope(batch_size, self.param_bounds_comp)

        return self._sample_from_polytope_with_cardinality_constraints(batch_size)

    def _sample_from_bounds(self, batch_size: int, bounds: np.ndarray) -> pd.DataFrame:
        """Draw uniform random samples over a hyperrectangle-shaped space."""
        points = np.random.uniform(
            low=bounds[0, :], high=bounds[1, :], size=(batch_size, len(self.parameters))
        )

        return pd.DataFrame(points, columns=self.param_names)

    def _sample_from_polytope(
        self, batch_size: int, bounds: np.ndarray
    ) -> pd.DataFrame:
        """Draw uniform random samples from a polytope."""
        import torch
        from botorch.utils.sampling import get_polytope_samples

        points = get_polytope_samples(
            n=batch_size,
            bounds=torch.from_numpy(bounds),
            equality_constraints=[
                c.to_botorch(self.parameters) for c in self.constraints_lin_eq
            ],
            inequality_constraints=[
                c.to_botorch(self.parameters) for c in self.constraints_lin_ineq
            ],
        )
        return pd.DataFrame(points, columns=self.param_names)

    def _sample_from_polytope_with_cardinality_constraints(
        self, batch_size: int
    ) -> pd.DataFrame:
        """Draw random samples from a polytope with cardinality constraints."""
        if not self.constraints_cardinality:
            raise RuntimeError(
                f"This method should not be called without any constraints of type "
                f"'{ContinuousCardinalityConstraint.__name__}' in place. "
                f"Use '{SubspaceContinuous._sample_from_bounds.__name__}' "
                f"or '{SubspaceContinuous._sample_from_polytope.__name__}' instead."
            )

        # List to store the created samples
        samples: list[pd.DataFrame] = []

        # Counter for failed sampling attempts
        n_fails = 0

        while len(samples) < batch_size:
            # Randomly set some parameters inactive
            inactive_params_sample = self._sample_inactive_parameters(1)[0]

            # Remove the inactive parameters from the search space
            subspace_without_cardinality_constraint = self._drop_parameters(
                inactive_params_sample
            )

            # Sample from the reduced space
            try:
                sample = subspace_without_cardinality_constraint.sample_uniform(1)
                samples.append(sample)
            except ValueError:
                n_fails += 1

            # Avoid infinite loop
            if n_fails >= _MAX_CARDINALITY_SAMPLING_ATTEMPTS:
                raise RuntimeError(
                    f"The number of failed sampling attempts has exceeded the limit "
                    f"of {_MAX_CARDINALITY_SAMPLING_ATTEMPTS}. "
                    f"It appears that the feasible region of the search space is very "
                    f"small. Please review the search space constraints."
                )

        # Combine the samples and fill in inactive parameters
        parameter_names = [p.name for p in self.parameters]
        return (
            pd.concat(samples, ignore_index=True)
            .reindex(columns=parameter_names)
            .fillna(0.0)
        )

    def _sample_inactive_parameters(self, batch_size: int = 1) -> list[set[str]]:
        """Sample inactive parameters according to the given cardinality constraints."""
        inactives_per_constraint = [
            con.sample_inactive_parameters(batch_size)
            for con in self.constraints_cardinality
        ]
        return [set(chain(*x)) for x in zip(*inactives_per_constraint)]

    def _ensure_nonzero_parameters(
        self,
        inactive_parameters: Collection[str],
        zero_threshold: float = ZERO_THRESHOLD,
    ) -> SubspaceContinuous:
        """Create a new subspace with following several actions.

        * Ensure active parameter != 0.0.
        * Remove cardinality constraint.

        Args:
            inactive_parameters: A list of inactive parameters.
            zero_threshold: Threshold for checking whether a value is zero.

        Returns:
            A new subspace object.
        """
        # Active parameters: parameters involved in cardinality constraints
        active_params_sample = set(
            self.param_names_in_cardinality_constraint
        ).difference(set(inactive_parameters))

        constraints_lin_ineq = list(self.constraints_lin_ineq)
        for active_param in active_params_sample:
            index = self.param_names.index(active_param)

            # TODO: Ensure x != 0 when x in [..., 0, ...] is not done. Do we need it?
            # TODO: To ensure the minimum cardinality constraints, shall we keep the x
            #  != 0 operations or shall we instead skip the invalid results at the end
            # Ensure x != 0 when bounds = [..., 0]. This is needed, otherwise
            # the minimum cardinality constraint is easily violated
            if self.parameters[index].bounds.upper == 0:
                constraints_lin_ineq.append(
                    ContinuousLinearInequalityConstraint(
                        parameters=[active_param],
                        coefficients=[-1.0],
                        rhs=min(zero_threshold, -self.parameters[index].bounds.lower),
                    )
                )
            # Ensure x != 0 when bounds = [0, ...]
            elif self.parameters[index].bounds.lower == 0:
                constraints_lin_ineq.append(
                    ContinuousLinearInequalityConstraint(
                        parameters=[active_param],
                        coefficients=[1.0],
                        rhs=min(zero_threshold, self.parameters[index].bounds.upper),
                    ),
                )

        return SubspaceContinuous(
            parameters=tuple(self.parameters),
            constraints_lin_eq=self.constraints_lin_eq,
            constraints_lin_ineq=tuple(constraints_lin_ineq),
        )

    def samples_full_factorial(self, n_points: int = 1) -> pd.DataFrame:
        """Deprecated!"""  # noqa: D401
        warnings.warn(
            f"The method '{SubspaceContinuous.samples_full_factorial.__name__}' "
            f"has been deprecated and will be removed in a future version. "
            f"Use '{SubspaceContinuous.sample_from_full_factorial.__name__}' instead.",
            DeprecationWarning,
        )
        return self.sample_from_full_factorial(n_points)

    def sample_from_full_factorial(self, batch_size: int = 1) -> pd.DataFrame:
        """Draw parameter configurations from the full factorial of the space.

        Args:
            batch_size: The number of parameter configurations to be sampled.

        Returns:
            A dataframe containing the parameter configurations as rows with columns
            corresponding to the parameter names.

        Raises:
            ValueError: If there are not enough points to sample from.
        """
        if len(full_factorial := self.full_factorial) < batch_size:
            raise ValueError(
                f"You are trying to sample {batch_size} points from the full factorial "
                f"of the continuous space bounds, but it has only "
                f"{len(full_factorial)} points."
            )

        return full_factorial.sample(n=batch_size).reset_index(drop=True)

    @property
    def full_factorial(self) -> pd.DataFrame:
        """Get the full factorial of the continuous space."""
        index = pd.MultiIndex.from_product(
            self.param_bounds_comp.T.tolist(), names=self.param_names
        )

        return pd.DataFrame(index=index).reset_index()

    def get_parameters_by_name(
        self, names: Sequence[str]
    ) -> tuple[NumericalContinuousParameter, ...]:
        """Return parameters with the specified names.

        Args:
            names: Sequence of parameter names.

        Returns:
            The named parameters.
        """
        return tuple(p for p in self.parameters if p.name in names)


# Register deserialization hook
converter.register_structure_hook(SubspaceContinuous, select_constructor_hook)
