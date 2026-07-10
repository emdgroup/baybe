"""Discrete subspaces."""

from __future__ import annotations

import gc
import random
import warnings
from collections.abc import Collection, Iterator, Sequence
from itertools import islice
from math import prod
from typing import TYPE_CHECKING, Any, Literal

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
from baybe.parameters.utils import get_parameters_from_dataframe
from baybe.searchspace.candidates import (
    CandidatesProtocol,
    EmptyCandidates,
    ProductCandidates,
    TableCandidates,
)
from baybe.searchspace.utils import build_constrained_product, select_via_flat_index
from baybe.searchspace.validation import validate_parameters
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.settings import active_settings
from baybe.utils.basic import UNSPECIFIED, UnspecifiedType, to_tuple
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import (
    get_transform_objects,
    normalize_input_dtypes,
    pretty_print_df,
)
from baybe.utils.memory import bytes_to_human_readable

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace


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

    candidates: CandidatesProtocol = field(validator=instance_of(CandidatesProtocol))
    """The discrete candidates set spanning the subspace."""

    batch_constraints: tuple[DiscreteBatchConstraint, ...] = field(
        default=(),
        converter=to_tuple,
        validator=deep_iterable(member_validator=instance_of(DiscreteBatchConstraint)),
    )
    """Constraints operating on the recommendation batch level."""

    # >>>>>>>>>> Deprecation
    def __init__(
        self,
        candidates: CandidatesProtocol = UNSPECIFIED,  # type: ignore[assignment]
        batch_constraints: Collection[DiscreteBatchConstraint] = (),
        *,
        parameters: Sequence[DiscreteParameter] | UnspecifiedType = UNSPECIFIED,
        exp_rep: pd.DataFrame | UnspecifiedType = UNSPECIFIED,
        empty_encoding: bool | UnspecifiedType = UNSPECIFIED,
        constraints: Sequence[DiscreteConstraint] | UnspecifiedType = UNSPECIFIED,
        comp_rep: Any | UnspecifiedType = UNSPECIFIED,
    ) -> None:
        # Detect legacy positional calls: SubspaceDiscrete([p, ...], df) where
        # the parameters list and exp_rep DataFrame were passed as positional args.
        if (
            candidates is not UNSPECIFIED
            and not isinstance(candidates, CandidatesProtocol)
            and isinstance(candidates, Collection)
            and all(isinstance(p, DiscreteParameter) for p in candidates)
        ):
            parameters = candidates
            candidates = UNSPECIFIED
            if isinstance(batch_constraints, pd.DataFrame):
                exp_rep = batch_constraints
                batch_constraints = ()

        # Legacy input format requires both ``parameters`` and ``exp_rep`` together
        if (parameters is UNSPECIFIED) != (exp_rep is UNSPECIFIED):
            raise ValueError(
                f"When using legacy constructor arguments for "
                f"'{self.__class__.__name__}', provide both 'parameters' and 'exp_rep' "
                f"together. Otherwise, use the '{fields(type(self)).candidates.alias}' "
                f"argument."
            )

        # --- parameters + exp_rep ---
        if parameters is not UNSPECIFIED and exp_rep is not UNSPECIFIED:
            name = fields(self.__class__).candidates.alias
            warnings.warn(
                f"Providing 'parameters' and 'exp_rep' to '{self.__class__.__name__}' "
                f"has been deprecated and support will be dropped in a future version. "
                f"Please use the new '{name}' interface instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            candidates = TableCandidates(
                parameters, normalize_input_dtypes(exp_rep, parameters)
            )

        # --- empty_encoding ---
        if empty_encoding is not UNSPECIFIED:
            _deprecate_argument("empty_encoding", error=False, stacklevel=3)

        # --- comp_rep ---
        if comp_rep is not UNSPECIFIED:
            _deprecate_argument("comp_rep", error=True, stacklevel=3)

        # --- constraints ---
        if constraints is not UNSPECIFIED:
            warnings.warn(
                _make_constraints_deprecation_msg(),
                DeprecationWarning,
                stacklevel=2,
            )
            batch_from_legacy: list[DiscreteBatchConstraint] = [
                c for c in constraints if isinstance(c, DiscreteBatchConstraint)
            ]
            if n_non_batch := len(constraints) - len(batch_from_legacy):
                warnings.warn(
                    f"You provided {n_non_batch} filtering constraint(s) via "
                    f"'constraints' but filtering constraints are (and always have "
                    f"been) ignored when entered via '__init__'. The latter assumes "
                    f"that all filtering constraints have already been applied to the "
                    f"given experimental candidate representation. To avoid this "
                    f"warning, either drop the filtering constraints or use one of "
                    f"the alternative constructors.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if batch_from_legacy:
                batch_constraints = tuple(batch_constraints) + tuple(batch_from_legacy)

        self.__attrs_init__(candidates=candidates, batch_constraints=batch_constraints)  # type: ignore[attr-defined]

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

    @property
    def parameters(self) -> tuple[DiscreteParameter, ...]:
        """The parameters spanning the subspace."""
        return self.candidates.parameters

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
        return cls(candidates=EmptyCandidates())

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

        extra = {"empty_encoding": empty_encoding} if empty_encoding is not None else {}
        return cls(
            candidates=(
                EmptyCandidates()
                if not parameters
                else ProductCandidates(parameters, filtering_constraints)
            ),
            batch_constraints=batch_constraints,
            **extra,
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

        # Get the full list of both explicitly and implicitly defined parameters
        parameters = get_parameters_from_dataframe(
            df, discrete_parameter_factory, parameters
        )

        # Ensure dtype consistency
        df = normalize_input_dtypes(df, parameters)

        extra = {"empty_encoding": empty_encoding} if empty_encoding is not None else {}
        return cls(
            candidates=TableCandidates(parameters, df),
            batch_constraints=batch_constraints,
            **extra,
        )

    @classmethod
    def from_simplex(
        cls,
        max_sum: float,
        simplex_parameters: Sequence[NumericalDiscreteParameter],
        *,
        simplex_coefficients: Sequence[float] | None = None,
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
            max_sum: The maximum (weighted) sum of the parameter values defining the
                simplex size.
            simplex_parameters: The parameters to be used for the simplex construction.
            simplex_coefficients: Optional coefficients for the weighted sum, one per
                entry in ``simplex_parameters``. Defaults to all-ones, i.e. an
                unweighted sum.
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
            ValueError: If the length of ``simplex_coefficients`` does not match the
                number of ``simplex_parameters``.
            ValueError: If ``simplex_coefficients`` contains any zeros.
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
        if simplex_coefficients is None:
            simplex_coefficients = [1.0] * len(simplex_parameters)

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

        # Validate coefficients length
        if len(simplex_coefficients) != len(simplex_parameters):
            raise ValueError(
                f"'simplex_coefficients' must have one entry per 'simplex_parameters' "
                f"entry, but got {len(simplex_coefficients)} coefficient(s) for "
                f"{len(simplex_parameters)} parameter(s)."
            )

        # Validate no zero coefficients
        if any(c == 0.0 for c in simplex_coefficients):
            raise ValueError("All entries in 'simplex_coefficients' must be non-zero.")

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

        # Compute per-parameter minimum weighted contributions.
        # For a positive coefficient c the minimum contribution is c*min_raw; for a
        # negative coefficient the ordering flips and it becomes c*max_raw. Taking
        # min of both products handles any real coefficient correctly.
        min_raw = [min(p.values) for p in simplex_parameters]
        max_raw = [max(p.values) for p in simplex_parameters]
        coeffs = np.asarray(simplex_coefficients, dtype=active_settings.DTypeFloatNumpy)
        if not np.isfinite(coeffs).all():
            raise ValueError(
                f"All simplex_coefficients passed to '{cls.from_simplex.__name__}' "
                f"must be finite numbers."
            )
        min_weighted = np.array(
            [min(c * lo, c * hi) for c, lo, hi in zip(coeffs, min_raw, max_raw)]
        )

        # Get the minimum weighted sum contributions to come in the upcoming joins (the
        # first item is the minimum possible weighted sum of all parameters starting
        # from the second parameter, the second item is the minimum possible weighted
        # sum starting from the third parameter, and so on ...)
        min_sum_upcoming = np.cumsum(min_weighted[:0:-1])[::-1]

        # Get the min/max number of nonzero values to come in the upcoming joins.
        # Nonzero counting is based on raw parameter values, not weighted values,
        # because the cardinality constraint counts zero/nonzero entries regardless
        # of the coefficient signs.
        min_nonzero_upcoming = np.cumsum((np.asarray(min_raw) > 0.0)[:0:-1])[::-1]
        max_nonzero_upcoming = np.cumsum((np.asarray(max_raw) > 0.0)[:0:-1])[::-1]

        # Incrementally build up the space as a numpy array, dropping invalid
        # configurations along the way. Working with raw numpy avoids pandas overhead
        # (index management, BlockManager, merge machinery) in the hot loop.
        #
        # After having cross-joined a new parameter, there must be enough "room" left
        # for the remaining parameters to fit. That is, configurations of the current
        # parameter subset that exceed the desired total value minus the minimum
        # contribution to come from the yet-to-be-added parameters can be already
        # discarded, because it is already clear that the total sum will be exceeded
        # once all joins are completed. Analogously, nonzero cardinality bounds are
        # checked at each step.
        #
        # Instead of materializing the full cross-product before filtering, we use
        # broadcasting to compute the validity mask in 2D (n_old, n_new) and only
        # materialize the surviving combinations. This avoids allocating large
        # intermediate arrays that are mostly discarded.
        arr = np.empty((1, 0), dtype=active_settings.DTypeFloatNumpy)
        partial_sums = np.zeros(1, dtype=active_settings.DTypeFloatNumpy)
        nz_counts = np.zeros(1, dtype=np.intp)

        for coeff, param, min_sum_to_go, min_nonzero_to_go, max_nonzero_to_go in zip(
            coeffs,
            simplex_parameters,
            np.append(min_sum_upcoming, 0.0),
            np.append(min_nonzero_upcoming, 0),
            np.append(max_nonzero_upcoming, 0),
        ):
            values = np.asarray(param.values, dtype=active_settings.DTypeFloatNumpy)
            threshold = (max_sum - min_sum_to_go) + tolerance
            effective_min = min_nonzero - max_nonzero_to_go
            effective_max = max_nonzero - min_nonzero_to_go

            # Compute weighted sums via broadcasting: (n_old, n_new)
            new_contributions = values * coeff
            total_sums = partial_sums[:, None] + new_contributions[None, :]

            # Build 2D validity mask from sum constraint
            mask_2d = total_sums <= threshold

            # Cardinality check via broadcasting
            new_nz = (values != 0.0).astype(np.intp)
            total_nz = nz_counts[:, None] + new_nz[None, :]
            if effective_min > 0:
                mask_2d &= total_nz >= effective_min
            if effective_max < len(simplex_parameters):
                mask_2d &= total_nz <= effective_max

            # Extract surviving indices and materialize only those rows
            old_idx, new_idx = np.where(mask_2d)
            arr = np.column_stack([arr[old_idx], values[new_idx]])
            partial_sums = total_sums[old_idx, new_idx]
            nz_counts = total_nz[old_idx, new_idx]

        # If requested, keep only the boundary values
        if boundary_only:
            mask = np.abs(partial_sums - max_sum) <= tolerance
            arr = arr[mask]

        # Wrap in DataFrame
        exp_rep = pd.DataFrame(arr, columns=[p.name for p in simplex_parameters])

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

        all_parameters = [*simplex_parameters, *product_parameters]
        return cls(
            # TODO: Investigate how off-the-shelf query optimization performs against
            #   our custom TableCandidates construction
            candidates=TableCandidates(all_parameters, exp_rep),
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
        return self.get_candidates()

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
        return self.transform(self.get_candidates())

    # <<<<<<<<<< Deprecation

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        return tuple(col for p in self.parameters for col in p.comp_rep_columns)

    @property
    def comp_rep_bounds(self) -> pd.DataFrame:
        """The minimum and maximum values of the computational representation."""
        if not self.parameters:
            return pd.DataFrame(index=["min", "max"])
        df = pd.concat([p.transform() for p in self.parameters], axis=1)
        return pd.DataFrame({"min": df.min(), "max": df.max()}).T

    @property
    def scaling_bounds(self) -> pd.DataFrame:
        """The bounds used for scaling the surrogate model input."""
        return (
            pd.concat(
                [p.transform().agg(["min", "max"]) for p in self.parameters],
                axis=1,
            )
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
        n_cols_comp = sum(len(p.comp_rep_columns) for p in parameters)
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
            per_constraint = [[np.ones(len(self.get_candidates()), dtype=bool)]]
        else:
            per_constraint = [
                c.subset_masks(self.get_candidates()) for c in self.batch_constraints
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
        return self.candidates.to_lazy().collect().to_pandas()

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
        dfs = [param.transform(df[param.name]) for param in parameters]
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

        simplex_coefficients = specs.get("simplex_coefficients", None)
        if simplex_coefficients is not None:
            try:
                simplex_coefficients = converter.structure(
                    simplex_coefficients, list[float]
                )
            except (IterableValidationError, TypeError, ValueError) as exc:
                raise ValueError(
                    "'simplex_coefficients' must be a list of numeric values."
                ) from exc

            if len(simplex_coefficients) != len(simplex_parameters):
                raise ValueError(
                    f"'simplex_coefficients' must have one entry per "
                    f"'simplex_parameters' entry, but got "
                    f"{len(simplex_coefficients)} coefficient(s) for "
                    f"{len(simplex_parameters)} parameter(s)."
                )

            if any(c == 0.0 for c in simplex_coefficients):
                raise ValueError(
                    "All entries in 'simplex_coefficients' must be non-zero."
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
def _deprecate_argument(arg: str, *, error: bool, stacklevel: int) -> None:
    """Raise a ``DeprecationError`` or emit a ``DeprecationWarning`` for a dropped argument."""  # noqa: E501
    msg = (
        f"Providing '{arg}' to '{SubspaceDiscrete.__name__}' is no longer "
        f"supported. To proceed, simply drop the argument."
    )
    if error:
        raise DeprecationError(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def _make_constraints_deprecation_msg() -> str:
    """Generate the constraints deprecation message."""
    batch_constraints_alias = fields(SubspaceDiscrete).batch_constraints.alias
    return (
        f"Providing 'constraints' to '{SubspaceDiscrete.__name__}' is no longer "
        f"supported. Please update your code as follows:\n"
        f"  • Use '{batch_constraints_alias}' for '{DiscreteBatchConstraint.__name__}' "
        f"objects. Any batch constraints you have provided have been extracted "
        f"automatically for you. This automatic extraction is temporary and will be "
        f"removed in a future version.\n"
        f"  • Filtering constraints can simply be dropped. Instead, make sure you "
        f"construct the experimental representation to satisfy them."
    )


def _structure_subspace_discrete(specs: dict, cls: type) -> SubspaceDiscrete:
    """Structure hook supporting legacy key migration."""
    specs = specs.copy()

    # Migrate legacy ``parameters`` + ``exp_rep`` format to ``candidates``
    if "exp_rep" in specs:
        name = fields(SubspaceDiscrete).candidates.alias
        warnings.warn(
            f"Deserializing a '{SubspaceDiscrete.__name__}' from the legacy "
            f"'parameters' + 'exp_rep' format is deprecated and support will be "
            f"removed in a future version. Please re-serialize your objects to use "
            f"the new '{name}' format.",
            DeprecationWarning,
            stacklevel=2,
        )
        parameters = converter.structure(
            specs.pop("parameters"), list[DiscreteParameter]
        )
        exp_rep_df = converter.structure(specs.pop("exp_rep"), pd.DataFrame)
        specs["candidates"] = converter.unstructure(
            TableCandidates(parameters, exp_rep_df),
            unstructure_as=CandidatesProtocol,
        )

    # Drop legacy ``empty_encoding`` key
    if "empty_encoding" in specs:
        _deprecate_argument("empty_encoding", error=False, stacklevel=2)
        specs.pop("empty_encoding")

    # Reject legacy ``comp_rep`` key
    if "comp_rep" in specs:
        _deprecate_argument("comp_rep", error=True, stacklevel=2)

    # Migrate legacy ``constraints`` key to ``batch_constraints``
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
