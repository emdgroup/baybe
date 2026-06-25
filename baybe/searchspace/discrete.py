"""Discrete subspaces."""

from __future__ import annotations

import gc
import random
import warnings
from collections.abc import Callable, Collection, Iterator, Sequence
from itertools import islice
from math import prod
from typing import TYPE_CHECKING, Annotated, Any, Literal

import cattrs
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field, fields
from attrs.validators import deep_iterable, instance_of
from cattrs import IterableValidationError
from typing_extensions import Self, override

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER, validate_constraints
from baybe.constraints.base import DiscreteConstraint
from baybe.constraints.discrete import DiscreteBatchConstraint
from baybe.exceptions import DeprecationError
from baybe.parameters import (
    CategoricalEncoding,
    CategoricalParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.utils import get_parameters_from_dataframe, sort_parameters
from baybe.searchspace.utils import build_constrained_product, select_via_flat_index
from baybe.searchspace.validation import validate_parameters
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.settings import active_settings
from baybe.utils.basic import to_tuple
from baybe.utils.boolean import eq_dataframe
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import (
    get_transform_objects,
    normalize_input_dtypes,
    pretty_print_df,
)
from baybe.utils.memory import bytes_to_human_readable

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace


def _deprecate_argument(error: bool, msg: str | Callable[[], str] | None = None):
    """Helper for deprecating legacy arguments."""  # noqa: D401

    def validator(self, attribute, value):
        if value is not None:
            # Generate message lazily if callable, otherwise use provided string
            warning_msg = (msg() if callable(msg) else msg) or (
                f"Providing '{attribute.alias}' to '{self.__class__.__name__}' is no "
                f"longer supported. To proceed, simply drop the argument."
            )
            if error:
                raise DeprecationError(warning_msg)
            else:
                warnings.warn(warning_msg, DeprecationWarning, stacklevel=3)

    return validator


@define(kw_only=True)
class MemorySize:
    """Estimated memory size of a :class:`SubspaceDiscrete`."""

    exp_rep_bytes: float
    """The memory size of the experimental representation dataframe in bytes."""

    exp_rep_shape: tuple[int, int]
    """The shape of the experimental representation dataframe."""

    comp_rep_bytes: float
    """The memory size of the computational representation dataframe in bytes."""

    comp_rep_shape: tuple[int, int]
    """The shape of the computational representation dataframe."""

    @property
    def exp_rep_human_readable(self) -> tuple[float, str]:
        """Human-readable memory size of the experimental representation dataframe.

        Consists of a tuple containing memory size and unit.
        """
        return bytes_to_human_readable(self.exp_rep_bytes)

    @property
    def comp_rep_human_readable(self) -> tuple[float, str]:
        """Human-readable memory size of the computational representation dataframe.

        Consists of a tuple containing memory size and unit.
        """
        return bytes_to_human_readable(self.comp_rep_bytes)


@define
class SubspaceDiscrete(SerialMixin):
    """Class for managing discrete subspaces.

    Builds the subspace from parameter definitions and optional constraints,
    and provides access to candidate sets and different parameter views.
    """

    parameters: tuple[DiscreteParameter, ...] = field(
        converter=sort_parameters,
        validator=[
            deep_iterable(member_validator=instance_of(DiscreteParameter)),
            lambda _, __, x: validate_parameters(x, allow_empty=True),
        ],
    )
    """The parameters spanning the subspace."""

    _exp_rep: pd.DataFrame = field(
        alias="exp_rep", validator=instance_of(pd.DataFrame), eq=eq_dataframe
    )
    """The experimental representation of the subspace."""

    _empty_encoding: Annotated[bool, cattrs.override(omit=True)] = field(
        alias="empty_encoding", default=None, validator=_deprecate_argument(error=False)
    )
    "Ignore! For backwards compatibility only."

    _constraints: Annotated[
        tuple[DiscreteConstraint, ...], cattrs.override(omit=True)
    ] = field(
        alias="constraints",
        default=None,
        validator=_deprecate_argument(
            error=False,
            msg=lambda: _make_constraints_deprecation_msg(),  # noqa: PLW0108
        ),
    )
    "Ignore! For backwards compatibility only."

    _comp_rep: Annotated[pd.DataFrame, cattrs.override(omit=True)] = field(
        alias="comp_rep", default=None, validator=_deprecate_argument(error=True)
    )
    "Ignore! For backwards compatibility only."

    batch_constraints: tuple[DiscreteBatchConstraint, ...] = field(
        default=(),
        converter=to_tuple,
        validator=deep_iterable(member_validator=instance_of(DiscreteBatchConstraint)),
    )
    """Constraints operating on the recommendation batch level."""

    def __attrs_post_init__(self) -> None:
        """Migrate deprecated ``constraints`` argument to ``batch_constraints``."""
        # >>>>>>>>>> Deprecation
        if self._constraints is not None:
            batch: tuple[DiscreteBatchConstraint, ...] = tuple(
                c for c in self._constraints if isinstance(c, DiscreteBatchConstraint)
            )

            if n_non_batch := len(self._constraints) - len(batch):
                warnings.warn(
                    f"You provided {n_non_batch} filtering constraint(s) via "
                    f"'constraints' but filtering constraints are (and always have "
                    f"been) ignored when entered via '__init__'. The latter assumes "
                    f"that all filtering constraints have already been applied to the "
                    f"given experimental candidate representation. To avoid this "
                    f"warning, either drop the filtering constraints or use one of the "
                    f"alternative constructors.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if batch:
                self.batch_constraints = self.batch_constraints + batch

                # attrs validators have already run at this point, so re-validate.
                validate_constraints(self.batch_constraints, self.parameters)
        # <<<<<<<<<< Deprecation

    @override
    def __str__(self) -> str:
        if self.is_empty:
            return ""

        # Convert the lists to dataFrames to be able to use pretty_printing
        param_list = [param.summary() for param in self.parameters]
        batch_constraints_list = [constr.summary() for constr in self.batch_constraints]
        param_df = pd.DataFrame(param_list)
        batch_constraints_df = pd.DataFrame(batch_constraints_list)

        fields = [
            to_string(
                "Discrete Parameters",
                pretty_print_df(param_df, max_colwidth=None),
            ),
            to_string("Batch Constraints", pretty_print_df(batch_constraints_df)),
        ]
        return to_string(self.__class__.__name__, *fields)

    @_exp_rep.validator
    def _validate_exp_rep(  # noqa: DOC101, DOC103
        self, _: Any, exp_rep: pd.DataFrame
    ) -> None:
        """Validate the experimental representation.

        Raises:
            ValueError: If the provided dataframe columns do not match the parameter
                names of the subspace.
            ValueError: If the index of the provided dataframe contains duplicates.
        """
        if set(exp_rep.columns) != {p.name for p in self.parameters}:
            raise ValueError(
                "The columns of the experimental representation must match the "
                "parameter names of the subspace."
            )
        # TODO: We should ideally also also validate that there are no duplicate rows,
        #    but in the current eager implementation this is a costly operation.
        #    To be revisited once the lazy implementation is in place.
        if exp_rep.index.has_duplicates:
            raise ValueError(
                "The index of this search space contains duplicates. "
                "This is not allowed, as it can lead to hard-to-detect bugs."
            )

    @batch_constraints.validator
    def _validate_batch_constraints(self, _, __) -> None:  # noqa: DOC101, DOC103
        """Validate batch constraints."""
        validate_constraints(self.batch_constraints, self.parameters)

    def to_searchspace(self) -> SearchSpace:
        """Turn the subspace into a search space with no continuous part."""
        from baybe.searchspace.core import SearchSpace

        return SearchSpace(discrete=self)

    @classmethod
    def empty(cls) -> Self:
        """Create an empty discrete subspace."""
        return cls(parameters=[], exp_rep=pd.DataFrame())

    @classmethod
    def from_parameter(cls, parameter: DiscreteParameter) -> Self:
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
        parameters: Sequence[DiscreteParameter],
        constraints: Sequence[DiscreteConstraint] | None = None,
        empty_encoding: bool | None = None,
    ) -> Self:
        """See :class:`baybe.searchspace.core.SearchSpace`."""
        validate_parameters(parameters, allow_empty=True)

        if constraints is None:
            constraints = []
        else:
            constraints = sorted(
                constraints,
                key=lambda x: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(x.__class__),
            )
            validate_constraints(constraints, parameters)

        filtering_constraints = [c for c in constraints if c.eval_during_creation]
        batch_constraints = [c for c in constraints if c.eval_during_modeling]
        assert len(filtering_constraints) + len(batch_constraints) == len(
            constraints
        ), (
            "The constraints could not be fully partitioned into filtering and batch "
            "constraints. The current logic assumes that each constraint belongs "
            "exactly to one type."
        )

        df = build_constrained_product(parameters, filtering_constraints)

        return cls(
            parameters=parameters,
            batch_constraints=batch_constraints,
            exp_rep=df,
            empty_encoding=empty_encoding,  # type: ignore[arg-type]
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Sequence[DiscreteParameter] | None = None,
        batch_constraints: Collection[DiscreteBatchConstraint] = (),
        empty_encoding: bool | None = None,
    ) -> Self:
        """Create a discrete subspace with a specified set of configurations.

        Args:
            df: The experimental representation of the search space to be created.
            parameters: Optional parameter objects corresponding to the columns in the
                given dataframe that can be provided to explicitly control parameter
                attributes. If a match between column name and parameter name is found,
                the corresponding parameter object is used. If a column has no match in
                the parameter list, a
                :class:`baybe.parameters.numerical.NumericalDiscreteParameter` is
                created if possible, or a
                :class:`baybe.parameters.categorical.CategoricalParameter` is used as
                fallback. For both types, default values are used for their optional
                arguments. For more details, see
                :func:`baybe.parameters.utils.get_parameters_from_dataframe`.
            batch_constraints: Optional batch constraints to be applied at
                recommendation time.
            empty_encoding: Ignore! For backwards compatibility only.

        Returns:
            The created discrete subspace.
        """

        def discrete_parameter_factory(
            name: str, values: Collection[Any]
        ) -> DiscreteParameter:
            """Try to create a numerical parameter or use a categorical fallback."""
            try:
                if pd.api.types.is_bool_dtype(np.asarray(values)):
                    # Due to the difference between bool and np.bool and pandas'
                    # auto-casting into the latter, the usage of is_bool_dtype and map
                    # is required here.
                    return CategoricalParameter(
                        name=name,
                        values=map(bool, values),
                        encoding=CategoricalEncoding.INT,
                    )
                return NumericalDiscreteParameter(name=name, values=values)
            except IterableValidationError:
                return CategoricalParameter(name=name, values=values)

        # Catch edge case
        if df.shape[1] == 0:
            return cls.empty()

        # Get the full list of both explicitly and implicitly defined parameter
        parameters = get_parameters_from_dataframe(
            df, discrete_parameter_factory, parameters
        )

        # Ensure dtype consistency
        df = normalize_input_dtypes(df, parameters)

        return cls(
            parameters=parameters,
            exp_rep=df,
            batch_constraints=batch_constraints,
            empty_encoding=empty_encoding,  # type: ignore[arg-type]
        )

    @classmethod
    def from_simplex(
        cls,
        max_sum: float,
        simplex_parameters: Sequence[NumericalDiscreteParameter],
        product_parameters: Sequence[DiscreteParameter] | None = None,
        constraints: Sequence[DiscreteConstraint] | None = None,
        min_nonzero: int = 0,
        max_nonzero: int | None = None,
        boundary_only: bool = False,
        tolerance: float = 1e-6,
    ) -> Self:
        """Efficiently create discrete simplex subspaces.

        The same result can be achieved using
        :meth:`baybe.searchspace.discrete.SubspaceDiscrete.from_product` in combination
        with appropriate constraints. However, such an approach is inefficient
        because the Cartesian product involved creates an exponentially large set of
        candidates, most of which do not satisfy the simplex constraints and must be
        subsequently be filtered out by the method.

        By contrast, this method uses a shortcut that removes invalid candidates
        already during the creation of parameter combinations, resulting in a
        significantly faster construction.

        Args:
            max_sum: The maximum sum of the parameter values defining the simplex size.
            simplex_parameters: The parameters to be used for the simplex construction.
            product_parameters: Optional parameters that enter in form of a Cartesian
                product.
            constraints: See :class:`baybe.searchspace.core.SearchSpace`.
            min_nonzero: Optional restriction on the minimum number of nonzero
                parameter values in the simplex construction.
            max_nonzero: Optional restriction on the maximum number of nonzero
                parameter values in the simplex construction.
            boundary_only: Flag determining whether to keep only parameter
                configurations on the simplex boundary.
            tolerance: Numerical tolerance used to validate the simplex constraint.

        Raises:
            ValueError: If the passed simplex parameters are not suitable for a simplex
                construction.
            ValueError: If the passed product parameters are not discrete.
            ValueError: If the passed simplex parameters and product parameters are
                not disjoint.

        Returns:
            The created simplex subspace.

        Note:
            The achieved efficiency gains can vary depending on the particular order in
            which the parameters are passed to this method, as the configuration space
            is built up incrementally from the parameter sequence.
        """
        # Resolve defaults
        if product_parameters is None:
            product_parameters = []
        if constraints is None:
            constraints = []
        if max_nonzero is None:
            max_nonzero = len(simplex_parameters)

        # Validate parameter types
        if not (
            all(isinstance(p, NumericalDiscreteParameter) for p in simplex_parameters)
        ):
            raise ValueError(
                f"All parameters passed via 'simplex_parameters' "
                f"must be of type '{NumericalDiscreteParameter.__name__}'."
            )
        if not all(p.is_discrete for p in product_parameters):
            raise ValueError(
                f"All parameters passed via 'product_parameters' "
                f"must be of subclasses of '{DiscreteParameter.__name__}'."
            )

        # Validate no overlap between simplex parameters and product parameters
        simplex_parameters_names = {p.name for p in simplex_parameters}
        product_parameters_names = {p.name for p in product_parameters}
        if overlap := simplex_parameters_names.intersection(product_parameters_names):
            raise ValueError(
                f"Parameter sets passed via 'simplex_parameters' and "
                f"'product_parameters' must be disjoint but share the following "
                f"parameters: {overlap}."
            )

        # Validate constraints
        if constraints:
            validate_constraints(
                constraints, [*simplex_parameters, *product_parameters]
            )

        # Handle degenerate simplex cases
        if len(simplex_parameters) < 2:
            warnings.warn(
                f"'{cls.from_simplex.__name__}' was called with less than 2 "
                f"simplex parameters, so smart simplex construction has no effect."
                f"Consider using '{cls.from_product.__name__}' instead.",
                UserWarning,
            )
            if len(simplex_parameters) < 1:
                return cls.from_product(product_parameters, constraints)

        # Validate non-negativity
        min_values = [min(p.values) for p in simplex_parameters]
        max_values = [max(p.values) for p in simplex_parameters]
        if not (min(min_values) >= 0.0):
            raise ValueError(
                f"All simplex_parameters passed to '{cls.from_simplex.__name__}' "
                f"must have non-negative values only."
            )

        def drop_invalid(
            df: pd.DataFrame,
            max_sum: float,
            boundary_only: bool,
            min_nonzero: int | None = None,
            max_nonzero: int | None = None,
        ) -> None:
            """Drop rows that violate the specified simplex constraint.

            Args:
                df: The dataframe whose rows should satisfy the simplex constraint.
                max_sum: The maximum row sum defining the simplex size.
                boundary_only: Flag to control if the points represented by the rows
                    may lie inside the simplex or on its boundary only.
                min_nonzero: Minimum number of nonzero parameters required per row.
                max_nonzero: Maximum number of nonzero parameters allowed per row.
            """
            # Apply sum constraints
            row_sums = df.sum(axis=1)
            mask_violated = row_sums > max_sum + tolerance
            if boundary_only:
                mask_violated |= row_sums < max_sum - tolerance

            # Apply optional nonzero constraints
            if (min_nonzero is not None) or (max_nonzero is not None):
                n_nonzero = (df != 0.0).sum(axis=1)
                if min_nonzero is not None:
                    mask_violated |= n_nonzero < min_nonzero
                if max_nonzero is not None:
                    mask_violated |= n_nonzero > max_nonzero

            # Remove violating rows
            idxs_to_drop = df[mask_violated].index
            df.drop(index=idxs_to_drop, inplace=True)

        # Get the minimum sum contributions to come in the upcoming joins (the
        # first item is the minimum possible sum of all parameters starting from the
        # second parameter, the second item is the minimum possible sum starting from
        # the third parameter, and so on ...)
        min_sum_upcoming = np.cumsum(min_values[:0:-1])[::-1]

        # Get the min/max number of nonzero values to come in the upcoming joins (the
        # first item is the min/max number of nonzero parameters starting from the
        # second parameter, the second item is the min/max number starting from
        # the third parameter, and so on ...)
        min_nonzero_upcoming = np.cumsum((np.asarray(min_values) > 0.0)[:0:-1])[::-1]
        max_nonzero_upcoming = np.cumsum((np.asarray(max_values) > 0.0)[:0:-1])[::-1]

        # Incrementally build up the space, dropping invalid configuration along the
        # way. More specifically:
        # * After having cross-joined a new parameter, there must
        #   be enough "room" left for the remaining parameters to fit. That is,
        #   configurations of the current parameter subset that exceed the desired
        #   total value minus the minimum contribution to come from the yet-to-be-added
        #   parameters can be already discarded, because it is already clear that
        #   the total sum will be exceeded once all joins are completed.
        # * Analogously, there must be enough "nonzero slots" left for the yet to be
        #   joined parameters, i.e. parameter subset configurations can be discarded
        #   where the number of nonzero parameters already exceeds the maximum number
        #   of nonzeros minus the number of nonzeros to come, because it is already
        #   clear that the maximum will be exceeded once all joins are completed.
        # * Similarly, it can be verified for each parameter that there are still
        #   enough nonzero parameters to come to even reach the minimum
        #   desired number of nonzero after all joins.
        for i, (
            param,
            min_sum_to_go,
            min_nonzero_to_go,
            max_nonzero_to_go,
        ) in enumerate(
            zip(
                simplex_parameters,
                np.append(min_sum_upcoming, 0),
                np.append(min_nonzero_upcoming, 0),
                np.append(max_nonzero_upcoming, 0),
            )
        ):
            if i == 0:
                exp_rep = pd.DataFrame({param.name: param.values})
            else:
                exp_rep = pd.merge(
                    exp_rep, pd.DataFrame({param.name: param.values}), how="cross"
                )
            drop_invalid(
                exp_rep,
                max_sum=max_sum - min_sum_to_go,
                # the maximum possible number of nonzeros to come dictates if we
                # can achieve our minimum constraint in the end:
                min_nonzero=min_nonzero - max_nonzero_to_go,
                # the minimum possible number of nonzeros to come dictates if we
                # can stay below the targeted maximum in the end:
                max_nonzero=max_nonzero - min_nonzero_to_go,
                boundary_only=False,
            )

        # If requested, keep only the boundary values
        if boundary_only:
            drop_invalid(exp_rep, max_sum, boundary_only=True)

        filtering_constraints = [c for c in constraints if c.eval_during_creation]
        batch_constraints_list = [c for c in constraints if c.eval_during_modeling]
        assert len(filtering_constraints) + len(batch_constraints_list) == len(
            constraints
        ), (
            "The constraints could not be fully partitioned into filtering and batch "
            "constraints. The current logic assumes that each constraint belongs "
            "exactly to one type."
        )

        # Merge product parameters and apply filtering constraints incrementally
        exp_rep = build_constrained_product(
            product_parameters, filtering_constraints, initial_df=exp_rep
        )

        return cls(
            parameters=[*simplex_parameters, *product_parameters],
            exp_rep=exp_rep,
            batch_constraints=batch_constraints_list,
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Deprecated!"""
        from baybe.campaign import Campaign

        raise DeprecationError(
            f"Search spaces no longer carry any metadata to avoid stateful behavior. "
            f"Metadata is now exclusively tracked by the `{Campaign.__name__}` class. "
            f"To dynamically exclude discrete candidates from the search space, "
            f"use its `{Campaign.toggle_discrete_candidates.__name__}` method."
        )

    @property
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return tuple of parameter names."""
        return tuple(p.name for p in self.parameters)

    # >>>>>>>>>> Deprecation
    @property
    def exp_rep(self) -> pd.DataFrame:
        """Deprecated! Use :meth:`get_candidates` instead."""
        get_candidates = type(self).get_candidates.__name__
        warnings.warn(
            f"Accessing 'exp_rep' is deprecated and will be removed in a future "
            f"version. Use '{get_candidates}()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._exp_rep

    @property
    def comp_rep(self) -> pd.DataFrame:
        """Deprecated! Use :meth:`transform` with :meth:`get_candidates` instead."""
        cls = type(self)
        transform = cls.transform.__name__
        get_candidates = cls.get_candidates.__name__
        warnings.warn(
            f"Accessing 'comp_rep' is deprecated and will be removed in a future "
            f"version. Use '{transform}({get_candidates}())' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.transform(self._exp_rep)

    # <<<<<<<<<< Deprecation

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        return tuple(col for p in self.parameters for col in p.comp_rep_columns)

    @property
    def comp_rep_bounds(self) -> pd.DataFrame:
        """The minimum and maximum values of the computational representation."""
        df = pd.concat([p.comp_df for p in self.parameters], axis=1)
        return pd.DataFrame({"min": df.min(), "max": df.max()}).T

    @property
    def scaling_bounds(self) -> pd.DataFrame:
        """The bounds used for scaling the surrogate model input."""
        return (
            pd.concat([p.comp_df.agg(["min", "max"]) for p in self.parameters], axis=1)
            if self.parameters
            else pd.DataFrame(index=["min", "max"])
        )

    @staticmethod
    def estimate_product_space_size(
        parameters: Sequence[DiscreteParameter],
    ) -> MemorySize:
        """Estimate an upper bound for the memory size of a product space.

        Args:
            parameters: The parameters spanning the product space.

        Returns:
            The estimated memory size.
        """
        # Compute the dataframe shapes
        n_cols_exp = len(parameters)
        n_cols_comp = sum(p.comp_df.shape[1] for p in parameters)
        n_rows = prod(len(p.active_values) for p in parameters)

        # Comp rep space is estimated as the size of float times the number of matrix
        # elements in the comp rep. The latter is the total number of parameter
        # configurations (= number of rows) times the total number of columns.
        comp_rep_bytes = (
            np.array([0.0], dtype=active_settings.DTypeFloatNumpy).itemsize
            * n_rows
            * n_cols_comp
        )

        # Exp rep space is estimated as the size of the per-parameter exp rep dataframe
        # times the number of times it will appear in the entire search space. The
        # latter is the total number of parameter configurations (= number of rows)
        # divided by the number of values for the respective parameter. Contributions of
        # all parameters are summed up.
        exp_rep_bytes = sum(
            pd.DataFrame(p.active_values).memory_usage(index=False, deep=True).sum()
            * n_rows
            / len(p.active_values)
            for p in parameters
        )

        return MemorySize(
            exp_rep_bytes=exp_rep_bytes,
            exp_rep_shape=(n_rows, n_cols_exp),
            comp_rep_bytes=comp_rep_bytes,
            comp_rep_shape=(n_rows, n_cols_comp),
        )

    # >>>>>>>>>> Deprecation
    @property
    def constraints_batch(self) -> tuple[DiscreteBatchConstraint, ...]:
        """Deprecated! Use :attr:`batch_constraints` instead."""
        replacement = fields(type(self)).batch_constraints.name
        warnings.warn(
            f"Accessing 'constraints_batch' is deprecated and will be disabled in a "
            f"future version. Use '{replacement}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.batch_constraints

    # <<<<<<<<<< Deprecation

    @property
    def n_subsets(self) -> int:
        """The number of possible subset configurations.

        Returns 0 if no subset-generating constraints exist, indicating that
        no decomposition is needed.
        """
        if not self.batch_constraints:
            return 0
        return prod(
            len(self.get_parameters_by_name([c.parameters[0]])[0].active_values)
            for c in self.batch_constraints
        )

    def subset_masks(
        self,
        min_candidates: int | None = None,
        mode: Literal["sequential", "shuffled", "replace"] = "shuffled",
    ) -> Iterator[npt.NDArray[np.bool_]]:
        """Get an iterator over all possible subset masks.

        Collect masks from each subset-generating constraint, iterates the
        Cartesian product, AND-reduces each combination, and yields feasible
        combined masks.

        Args:
            min_candidates: If provided, combined masks selecting fewer rows
                are silently skipped.
            mode: The iteration strategy.

                * ``"sequential"`` iterates all combinations in deterministic order.
                * ``"shuffled"`` iterates all combinations exactly once in random order.
                * ``"replace"`` samples with replacement, producing an infinite iterator
                  where each draw is independent.

        Raises:
            ValueError: If an invalid mode is provided.

        Yields:
            A Boolean mask selecting the subset's rows.
        """
        if mode not in (allowed := {"sequential", "shuffled", "replace"}):
            raise ValueError(f"Invalid {mode=}. Must be one of {allowed}.")

        per_constraint: list[list[npt.NDArray[np.bool_]]]
        if not self.batch_constraints:
            per_constraint = [[np.ones(len(self.exp_rep), dtype=bool)]]
        else:
            per_constraint = [
                c.subset_masks(self.exp_rep) for c in self.batch_constraints
            ]

        total = prod(len(masks) for masks in per_constraint)

        if mode == "replace":
            candidates = list(range(total))
            while candidates:
                idx_pos = random.randint(0, len(candidates) - 1)
                flat_idx = candidates[idx_pos]
                combined = np.logical_and.reduce(
                    select_via_flat_index(flat_idx, per_constraint)
                )
                if min_candidates is not None and combined.sum() < min_candidates:
                    candidates[idx_pos] = candidates[-1]
                    candidates.pop()
                    continue
                yield combined
        else:
            order = list(range(total))
            if mode == "shuffled":
                random.shuffle(order)
            for flat_idx in order:
                combined = np.logical_and.reduce(
                    select_via_flat_index(flat_idx, per_constraint)
                )
                if min_candidates is not None and combined.sum() < min_candidates:
                    continue
                yield combined

    def sample_subset_masks(
        self,
        n: int,
        min_candidates: int | None = None,
    ) -> list[npt.NDArray[np.bool_]]:
        """Sample subset masks (without replacement).

        Args:
            n: Number of masks to sample.
            min_candidates: If provided, Subsets with fewer matching
                candidates are skipped.

        Returns:
            A list of boolean masks.
        """
        return list(
            islice(
                self.subset_masks(min_candidates),
                n,
            )
        )

    def get_candidates(self) -> pd.DataFrame:
        """Return all candidate parameter configurations."""
        return self._exp_rep

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
        dfs = []
        for param in parameters:
            comp_df = param.transform(df[param.name])
            dfs.append(comp_df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def get_parameters_by_name(
        self, names: Sequence[str]
    ) -> tuple[DiscreteParameter, ...]:
        """Return parameters with the specified names.

        Args:
            names: Sequence of parameter names.

        Returns:
            The named parameters.
        """
        return tuple(p for p in self.parameters if p.name in names)


def validate_simplex_subspace_from_config(specs: dict, _) -> None:
    """Validate the discrete space while skipping costly creation steps."""
    # Validate product inputs without constructing it
    if specs.get("constructor", None) == "from_product":
        parameters = converter.structure(specs["parameters"], list[DiscreteParameter])
        validate_parameters(parameters, allow_empty=True)

        # Support both the current `constraints` key (deprecated) and
        # the new `batch_constraints` key for forward/backward compatibility
        constraints_raw = specs.get("constraints", []) or specs.get(
            "batch_constraints", []
        )
        if constraints_raw:
            constraints = converter.structure(constraints_raw, list[DiscreteConstraint])
            validate_constraints(constraints, parameters)

    # Validate simplex inputs without constructing it
    elif specs.get("constructor", None) == "from_simplex":
        simplex_parameters = converter.structure(
            specs["simplex_parameters"], list[NumericalDiscreteParameter]
        )

        if not all(min(p.values) >= 0.0 for p in simplex_parameters):
            raise ValueError(
                f"All simplex_parameters passed to "
                f"'{SubspaceDiscrete.from_simplex.__name__}' must have non-negative "
                f"values only."
            )

        product_parameters = specs.get("product_parameters", [])
        if product_parameters:
            product_parameters = converter.structure(
                specs["product_parameters"], list[DiscreteParameter]
            )

        validate_parameters(simplex_parameters + product_parameters)

        # Support both the current `constraints` key (deprecated) and
        # the new `batch_constraints` key for forward/backward compatibility
        constraints_raw = specs.get("constraints", []) or specs.get(
            "batch_constraints", []
        )
        if constraints_raw:
            constraints = converter.structure(constraints_raw, list[DiscreteConstraint])
            validate_constraints(constraints, simplex_parameters + product_parameters)

    # For all other types, validate by construction
    else:
        converter.structure(specs, SubspaceDiscrete)


# >>>>>>>>>> Deprecation
def _make_constraints_deprecation_msg() -> str:
    """Generate the constraints deprecation message with programmatic names."""
    # Get field aliases programmatically
    constraints_alias = fields(SubspaceDiscrete)._constraints.alias
    batch_constraints_alias = fields(SubspaceDiscrete).batch_constraints.alias

    return (
        f"Providing '{constraints_alias}' to '{SubspaceDiscrete.__name__}' is no "
        f"longer supported. Please update your code as follows:\n"
        f"  • Use '{batch_constraints_alias}' for '{DiscreteBatchConstraint.__name__}' "
        f"objects. Any batch constraints you have provided have been extracted "
        f"automatically for you. This automatic extraction is temporary and will be "
        f"removed in a future version.\n"
        f"  • Filtering constraints can simply be dropped. Instead, make sure you "
        f"construct the experimental representation to satisfy them."
    )


def _structure_subspace_discrete(specs: dict, cls: type) -> SubspaceDiscrete:
    """Structure hook supporting legacy ``constraints`` key migration."""
    specs = specs.copy()
    if "constraints" in specs and specs["constraints"] is not None:
        warnings.warn(
            _make_constraints_deprecation_msg(),
            DeprecationWarning,
            stacklevel=2,
        )
        legacy_constraints = converter.structure(
            specs.pop("constraints"), list[DiscreteConstraint]
        )
        batch_from_legacy = [
            c for c in legacy_constraints if isinstance(c, DiscreteBatchConstraint)
        ]
        if batch_from_legacy:
            existing = specs.get("batch_constraints", [])
            existing_structured = converter.structure(
                existing, list[DiscreteBatchConstraint]
            )
            specs["batch_constraints"] = [
                c.to_dict() for c in existing_structured + batch_from_legacy
            ]
    else:
        specs.pop("constraints", None)
    return select_constructor_hook(specs, cls)


# Uncomment when removing the deprecation:
# converter.register_structure_hook(SubspaceDiscrete, select_constructor_hook)
converter.register_structure_hook(SubspaceDiscrete, _structure_subspace_discrete)
# <<<<<<<<<< Deprecation

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
