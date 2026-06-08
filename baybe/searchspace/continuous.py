"""Continuous subspaces."""

from __future__ import annotations

import gc
import math
import random
import warnings
from collections.abc import Collection, Iterator, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

import cattrs.gen
import numpy as np
import pandas as pd
from attrs import define, evolve, field
from typing_extensions import override

from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.constraints.base import ContinuousConstraint, ContinuousNonlinearConstraint
from baybe.constraints.validation import (
    validate_cardinality_constraint_parameter_bounds,
    validate_cardinality_constraints_are_nonoverlapping,
    validate_constraints,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.parameters.base import ContinuousParameter
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.parameters.utils import (
    activate_parameter,
    get_parameters_from_dataframe,
    sort_parameters,
)
from baybe.searchspace.utils import select_via_flat_index
from baybe.searchspace.validation import (
    validate_constraint_parameter_names,
    validate_parameter_names,
)
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.settings import active_settings
from baybe.utils.basic import flatten, is_all_instance, to_tuple
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import get_transform_objects, pretty_print_df

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.searchspace.core import SearchSpace

_MAX_CARDINALITY_SAMPLING_ATTEMPTS = 10_000


@define(init=False)
class SubspaceContinuous(SerialMixin):
    """Class for managing continuous subspaces.

    Builds the subspace from parameter definitions and optional constraints,
    and provides access to candidate sets and different parameter views.
    """

    parameters: tuple[NumericalContinuousParameter, ...] = field(
        converter=sort_parameters,
        validator=lambda _, __, x: validate_parameter_names(x),
    )
    """The parameters spanning the subspace."""

    constraints: tuple[ContinuousConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """Optional constraints filtering the subspace."""

    def __init__(
        self,
        parameters: Sequence[ContinuousParameter] | None = None,
        constraints: Sequence[ContinuousConstraint] | None = None,
        constraints_lin_eq: Sequence[ContinuousLinearConstraint] | None = None,
        constraints_lin_ineq: Sequence[ContinuousLinearConstraint] | None = None,
        constraints_nonlin: Sequence[ContinuousNonlinearConstraint] | None = None,
    ):
        parameters = list(parameters) if parameters is not None else []
        constraints = list(constraints) if constraints is not None else []

        n_constraints = len(constraints)
        if constraints_lin_eq is not None:
            constraints.extend(constraints_lin_eq)
        if constraints_lin_ineq is not None:
            constraints.extend(constraints_lin_ineq)
        if constraints_nonlin is not None:
            constraints.extend(constraints_nonlin)

        if len(constraints) != n_constraints:
            warnings.warn(
                "You are using the deprecated `constraints_lin_eq`, "
                "`constraints_lin_ineq` and/or `constraints_nonlin` arguments to "
                "specify constraints. For backward compatibility, we have "
                "automatically merged their content into the `constraints` attribute. "
                "However, please update your code to directly use the `constraints` "
                "argument instead since the deprecated arguments will be removed in "
                "a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.__attrs_init__(parameters, constraints)

    @override
    def __str__(self) -> str:
        if self.is_empty:
            return ""

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
        lin_eq_df = pd.DataFrame(eq_constraints_list)
        lin_ineq_df = pd.DataFrame(ineq_constraints_list)
        nonlinear_df = pd.DataFrame(nonlin_constraints_list)

        fields = [
            to_string(
                "Continuous Parameters", pretty_print_df(param_df, max_colwidth=None)
            ),
            to_string("Linear Equality Constraints", pretty_print_df(lin_eq_df)),
            to_string("Linear Inequality Constraints", pretty_print_df(lin_ineq_df)),
            to_string("Non-linear Constraints", pretty_print_df(nonlinear_df)),
        ]

        return to_string(self.__class__.__name__, *fields)

    @property
    def constraints_lin_eq(self) -> tuple[ContinuousLinearConstraint, ...]:
        """Linear equality constraints."""
        return tuple(
            c
            for c in self.constraints
            if isinstance(c, ContinuousLinearConstraint) and c.is_eq
        )

    @property
    def constraints_lin_ineq(self) -> tuple[ContinuousLinearConstraint, ...]:
        """Linear inequality constraints."""
        return tuple(
            c
            for c in self.constraints
            if isinstance(c, ContinuousLinearConstraint) and not c.is_eq
        )

    @property
    def constraints_nonlin(self) -> tuple[ContinuousNonlinearConstraint, ...]:
        """Nonlinear constraints."""
        return tuple(
            c for c in self.constraints if isinstance(c, ContinuousNonlinearConstraint)
        )

    @property
    def constraints_cardinality(self) -> tuple[ContinuousCardinalityConstraint, ...]:
        """The cardinality constraints of the subspace."""
        return tuple(
            c
            for c in self.constraints
            if isinstance(c, ContinuousCardinalityConstraint)
        )

    @property
    def n_subsets(self) -> int:
        """The number of possible subset configurations.

        Returns 0 if no cardinality constraints exist, indicating that
        no decomposition is needed.
        """
        if not self.constraints_cardinality:
            return 0
        return math.prod(
            c.n_inactive_parameter_combinations for c in self.constraints_cardinality
        )

    def inactive_parameter_combinations(
        self,
        *,
        mode: Literal["sequential", "shuffled", "replace"] = "shuffled",
    ) -> Iterator[frozenset[str]]:
        """Get an iterator over all possible inactive parameter combinations.

        Args:
            mode: The iteration strategy.

                * ``"sequential"`` iterates all combinations in deterministic order.
                * ``"shuffled"`` iterates all combinations exactly once in random order.
                * ``"replace"`` samples with replacement, producing an infinite iterator
                  where each draw is independent.

        Raises:
            ValueError: If an invalid mode is provided.

        Yields:
            A frozenset of inactive parameter names for the subspace.
        """
        if mode not in (allowed := {"sequential", "shuffled", "replace"}):
            raise ValueError(f"Invalid {mode=}. Must be one of {allowed}.")

        per_constraint = [
            list(c.inactive_parameter_combinations())
            for c in self.constraints_cardinality
        ]

        total = math.prod(len(v) for v in per_constraint)

        if mode == "replace":
            while True:
                yield frozenset(
                    chain(
                        *select_via_flat_index(
                            random.randint(0, total - 1), per_constraint
                        )
                    )
                )
        else:
            order = list(range(total))
            if mode == "shuffled":
                random.shuffle(order)
            for flat_idx in order:
                yield frozenset(chain(*select_via_flat_index(flat_idx, per_constraint)))

    @constraints.validator
    def _validate_constraints(self, _, __) -> None:
        """Validate constraints."""
        validate_constraint_parameter_names(self.constraints, self.parameters)

        validate_cardinality_constraints_are_nonoverlapping(
            self.constraints_cardinality
        )

        for constraint in self.constraints_cardinality:
            validate_cardinality_constraint_parameter_bounds(
                constraint, self.parameters
            )

    def to_searchspace(self) -> SearchSpace:
        """Turn the subspace into a search space with no discrete part."""
        from baybe.searchspace.core import SearchSpace

        return SearchSpace(continuous=self)

    @classmethod
    def empty(cls) -> SubspaceContinuous:
        """Create an empty continuous subspace."""
        return SubspaceContinuous(())

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
        if constraints:
            validate_constraints(constraints, parameters)
        return SubspaceContinuous(parameters, constraints)

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
        """Boolean indicating if the subspace is empty."""
        return len(self.parameters) == 0

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The names of the parameters spanning the subspace."""
        return tuple(p.name for p in self.parameters)

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        return tuple(chain.from_iterable(p.comp_rep_columns for p in self.parameters))

    @property
    def parameter_names_in_cardinality_constraints(self) -> frozenset[str]:
        """The names of all parameters affected by cardinality constraints."""
        names_per_constraint = (c.parameters for c in self.constraints_cardinality)
        return frozenset(chain(*names_per_constraint))

    @property
    def comp_rep_bounds(self) -> pd.DataFrame:
        """The minimum and maximum values of the computational representation."""
        return pd.DataFrame(
            {p.name: p.bounds.to_tuple() for p in self.parameters},
            index=["min", "max"],
            dtype=active_settings.DTypeFloatNumpy,
        )

    @property
    def scaling_bounds(self) -> pd.DataFrame:
        """The bounds used for scaling the surrogate model input."""
        return self.comp_rep_bounds

    def _drop_parameters(self, parameter_names: Collection[str]) -> SubspaceContinuous:
        """Create a copy of the subspace with certain parameters removed.

        Args:
            parameter_names: The names of the parameter to be removed.

        Raises:
            NotImplementedError: If the subspace contains constraints that are not
                linear intrapoint constraints.

        Returns:
            The reduced subspace.
        """
        if not is_all_instance(self.constraints, ContinuousLinearConstraint):
            raise NotImplementedError(
                "Dropping parameters is only supported for subspaces without "
                "constraints or with linear intrapoint constraints."
            )
        return SubspaceContinuous(
            parameters=[p for p in self.parameters if p.name not in parameter_names],
            constraints=[
                c._drop_parameters(parameter_names)
                for c in self.constraints
                if (set(c.parameters) - set(parameter_names))
            ],
        )

    @property
    def is_constrained(self) -> bool:
        """Boolean indicating if the subspace is constrained in any way."""
        return bool(self.constraints)

    @property
    def has_interpoint_constraints(self) -> bool:
        """Boolean indicating if the subspace has any interpoint constraints."""
        return any(
            c.is_interpoint for c in self.constraints_lin_eq + self.constraints_lin_ineq
        )

    def _enforce_cardinality_constraints(
        self,
        inactive_parameter_names: Collection[str],
    ) -> SubspaceContinuous:
        """Create a copy of the subspace with fixed inactive parameters.

        The returned subspace requires no cardinality constraints since – for the
        given separation of parameter into active an inactive sets – the
        cardinality constraints are implemented by fixing the inactive parameters to
        zero and bounding the active parameters away from zero.

        Args:
            inactive_parameter_names: The names of the parameter to be inactivated.

        Returns:
            A new subspace with fixed inactive parameters and no cardinality
            constraints.
        """
        # Extract active parameters involved in cardinality constraints
        active_parameter_names = (
            self.parameter_names_in_cardinality_constraints.difference(
                inactive_parameter_names
            )
        )

        # Adjust parameters depending on their in-/activity assignment
        adjusted_parameters: list[ContinuousParameter] = []
        p_adjusted: ContinuousParameter
        for p in self.parameters:
            if p.name in inactive_parameter_names:
                p_adjusted = _FixedNumericalContinuousParameter(name=p.name, value=0.0)

            elif p.name in active_parameter_names:
                constraints = [
                    c for c in self.constraints_cardinality if p.name in c.parameters
                ]

                # Constraint validation should have ensured that each parameter can
                # be part of at most one cardinality constraint
                assert len(constraints) == 1
                constraint = constraints[0]

                # If the corresponding constraint enforces a minimum cardinality,
                # force-activate the parameter
                if constraint.min_cardinality > 0:
                    p_adjusted = activate_parameter(
                        p, constraint.get_absolute_thresholds(p.bounds)
                    )
                else:
                    p_adjusted = p
            else:
                p_adjusted = p

            adjusted_parameters.append(p_adjusted)

        return evolve(
            self,
            parameters=adjusted_parameters,
            constraints=[
                c
                for c in self.constraints
                if not isinstance(c, ContinuousCardinalityConstraint)
            ],
        )

    def transform(
        self,
        df: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        """See :func:`baybe.searchspace.core.SearchSpace.transform`."""
        # Extract the parameters to be transformed
        parameters = get_transform_objects(
            df, self.parameters, allow_missing=allow_missing, allow_extra=allow_extra
        )

        # Transform the parameters
        return df[[p.name for p in parameters]]

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

        if not self.is_constrained:
            return self._sample_from_bounds(batch_size, self.comp_rep_bounds.to_numpy())

        if len(self.constraints_cardinality) == 0:
            return self._sample_from_polytope(
                batch_size, self.comp_rep_bounds.to_numpy()
            )

        return self._sample_from_polytope_with_cardinality_constraints(batch_size)

    def _sample_from_bounds(self, batch_size: int, bounds: np.ndarray) -> pd.DataFrame:
        """Draw uniform random samples over a hyperrectangle-shaped space."""
        points = np.random.uniform(
            low=bounds[0, :], high=bounds[1, :], size=(batch_size, len(self.parameters))
        )

        return pd.DataFrame(points, columns=self.parameter_names)

    def _sample_from_polytope(
        self, batch_size: int, bounds: np.ndarray
    ) -> pd.DataFrame:
        """Draw uniform random samples from a polytope."""
        import torch
        from botorch.utils.sampling import get_polytope_samples

        # pandas 3 with Copy-on-Write may pass a read-only array; copy only if needed
        if not bounds.flags.writeable:
            bounds = bounds.copy()
        bounds_tensor = torch.from_numpy(bounds)
        if not self.has_interpoint_constraints:
            points = get_polytope_samples(
                n=batch_size,
                bounds=bounds_tensor,
                equality_constraints=flatten(
                    c.to_botorch(self.parameters) for c in self.constraints_lin_eq
                ),
                inequality_constraints=flatten(
                    c.to_botorch(self.parameters) for c in self.constraints_lin_ineq
                ),
            )
        else:
            points = self._sample_from_polytope_with_interpoint_constraints(
                batch_size, bounds_tensor
            )

        return pd.DataFrame(points, columns=self.parameter_names)

    def _sample_from_polytope_with_interpoint_constraints(
        self, batch_size: int, bounds: Tensor
    ) -> Tensor:
        """Draw samples from a polytope with interpoint constraints.

        If the space has interpoint constraints, we need to sample from a larger
        searchspace that models the batch size via additional dimension. This is
        necessary since `get_polytope_samples` cannot handle interpoint
        constraints, see https://github.com/pytorch/botorch/issues/2468

        Args:
            batch_size: The number of samples to draw.
            bounds: The bounds of the parameters as a 2D tensor where the first row
                contains the lower bounds and the second row the upper bounds.

        Returns:
            A tensor of shape (batch_size, n_params) containing the samples.
        """
        from botorch.utils.sampling import get_polytope_samples

        eq_constraints = flatten(
            c.to_botorch(self.parameters, batch_size=batch_size, flatten=True)
            for c in self.constraints_lin_eq
        )
        ineq_constraints = flatten(
            c.to_botorch(self.parameters, batch_size=batch_size, flatten=True)
            for c in self.constraints_lin_ineq
        )

        flattened_bounds = bounds.repeat(1, batch_size)

        points = get_polytope_samples(
            n=1,
            bounds=flattened_bounds,
            equality_constraints=eq_constraints,
            inequality_constraints=ineq_constraints,
        )

        # Reshape to separate batch dimension from parameter dimension
        points_per_batch, remainder = divmod(points.shape[-1], batch_size)
        assert remainder == 0, "Dimensions mismatch."
        return points.reshape(batch_size, points_per_batch)

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
        from botorch.exceptions.errors import InfeasibilityError

        # List to store the created samples
        samples: list[pd.DataFrame] = []

        # Counter for failed sampling attempts
        n_fails = 0

        while len(samples) < batch_size:
            # Randomly set some parameters inactive
            inactive_params_sample = self._sample_inactive_parameters(1)[0]

            # Remove the inactive parameters from the search space. In the first
            # step, the active parameters get activated and inactive parameters are
            # fixed to zero. The first step helps ensure active parameters stay
            # non-zero, especially when one boundary is zero. The second step is
            # optional and it helps reduce the parameter space with certain
            # computational cost.
            subspace_without_cardinality_constraint = (
                self._enforce_cardinality_constraints(
                    inactive_params_sample
                )._drop_parameters(inactive_params_sample)
            )

            # Sample from the reduced space
            try:
                sample = subspace_without_cardinality_constraint.sample_uniform(1)
                samples.append(sample)
            except InfeasibilityError:
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

    def _sample_inactive_parameters(self, batch_size: int = 1) -> list[frozenset[str]]:
        """Sample inactive parameter configurations from the cardinality constraints."""
        inactives_per_constraint = [
            con.sample_inactive_parameters(batch_size)
            for con in self.constraints_cardinality
        ]
        return [frozenset(chain(*x)) for x in zip(*inactives_per_constraint)]

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
            self.comp_rep_bounds.values.T.tolist(), names=self.parameter_names
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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Uncomment when removing the deprecation:
# converter.register_structure_hook(SubspaceContinuous, select_constructor_hook)

# >>>>> Deprecation
_hook = cattrs.gen.make_dict_structure_fn(SubspaceContinuous, converter)


def _structure_hook(specs: dict, cls: type) -> SubspaceContinuous:
    """Structure hook that supports both constructor dispatch and legacy fields."""
    if "constructor" in specs:
        return select_constructor_hook(specs, cls)

    specs = specs.copy()
    specs.pop("type", None)

    # Check if any deprecated constraint fields are present
    deprecated_keys = {
        "constraints_lin_eq",
        "constraints_lin_ineq",
        "constraints_nonlin",
    }
    if deprecated_keys & specs.keys():
        from baybe.constraints.base import (
            ContinuousConstraint,
            ContinuousNonlinearConstraint,
        )

        kwargs: dict[str, Any] = {}
        if "parameters" in specs:
            kwargs["parameters"] = [
                converter.structure(p, NumericalContinuousParameter)
                for p in specs["parameters"]
            ]
        if "constraints" in specs:
            kwargs["constraints"] = [
                converter.structure(c, ContinuousConstraint)
                for c in specs["constraints"]
            ]
        if "constraints_lin_eq" in specs:
            kwargs["constraints_lin_eq"] = [
                converter.structure(c, ContinuousLinearConstraint)
                for c in specs["constraints_lin_eq"]
            ]
        if "constraints_lin_ineq" in specs:
            kwargs["constraints_lin_ineq"] = [
                converter.structure(c, ContinuousLinearConstraint)
                for c in specs["constraints_lin_ineq"]
            ]
        if "constraints_nonlin" in specs:
            kwargs["constraints_nonlin"] = [
                converter.structure(c, ContinuousNonlinearConstraint)
                for c in specs["constraints_nonlin"]
            ]
        return SubspaceContinuous(**kwargs)

    return _hook(specs, cls)


converter.register_structure_hook(SubspaceContinuous, _structure_hook)
# <<<<< Deprecation
