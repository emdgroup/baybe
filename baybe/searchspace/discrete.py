"""Discrete subspaces."""

from __future__ import annotations

import gc
import os
import warnings
from collections.abc import Collection, Sequence
from itertools import compress
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attr import define, field
from cattrs import IterableValidationError

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER, validate_constraints
from baybe.constraints.base import DiscreteConstraint
from baybe.exceptions import OptionalImportError
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.utils import get_parameters_from_dataframe, sort_parameters
from baybe.searchspace.validation import (
    get_transform_parameters,
    validate_parameter_names,
    validate_parameters,
)
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.utils.basic import to_tuple
from baybe.utils.boolean import eq_dataframe
from baybe.utils.dataframe import (
    df_drop_single_value_columns,
    fuzzy_row_match,
    pretty_print_df,
)
from baybe.utils.memory import bytes_to_human_readable
from baybe.utils.numerical import DTypeFloatNumpy
from baybe.utils.plotting import to_string

if TYPE_CHECKING:
    import polars as pl

    from baybe.searchspace.core import SearchSpace

_METADATA_COLUMNS = ["was_recommended", "was_measured", "dont_recommend"]


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

    Builds the subspace from parameter definitions and optional constraints, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    parameters: tuple[DiscreteParameter, ...] = field(
        converter=sort_parameters,
        validator=lambda _, __, x: validate_parameter_names(x),
    )
    """The list of parameters of the subspace."""

    exp_rep: pd.DataFrame = field(eq=eq_dataframe)
    """The experimental representation of the subspace."""

    metadata: pd.DataFrame = field(eq=eq_dataframe)
    """The metadata."""

    empty_encoding: bool = field(default=False)
    """Flag encoding whether an empty encoding is used."""

    constraints: tuple[DiscreteConstraint, ...] = field(
        converter=to_tuple, factory=tuple
    )
    """A list of constraints for restricting the space."""

    comp_rep: pd.DataFrame = field(eq=eq_dataframe)
    """The computational representation of the space. Technically not required but added
    as an optional initializer argument to allow ingestion from e.g. serialized objects
    and thereby speed up construction. If not provided, the default hook will derive it
    from ``exp_rep``."""

    def __str__(self) -> str:
        if self.is_empty:
            return ""

        # Convert the lists to dataFrames to be able to use pretty_printing
        param_list = [param.summary() for param in self.parameters]
        constraints_list = [constr.summary() for constr in self.constraints]
        param_df = pd.DataFrame(param_list)
        constraints_df = pd.DataFrame(constraints_list)

        # Get summary information from metadata
        was_recommended_count = len(self.metadata[self.metadata[_METADATA_COLUMNS[0]]])
        was_measured_count = len(self.metadata[self.metadata[_METADATA_COLUMNS[1]]])
        dont_recommend_count = len(self.metadata[self.metadata[_METADATA_COLUMNS[2]]])
        metadata_count = len(self.metadata)

        metadata_fields = [
            to_string(
                f"{_METADATA_COLUMNS[0]}",
                f"{was_recommended_count}/{metadata_count}",
                single_line=True,
            ),
            to_string(
                f"{_METADATA_COLUMNS[1]}",
                f"{was_measured_count}/{metadata_count}",
                single_line=True,
            ),
            to_string(
                f"{_METADATA_COLUMNS[2]}",
                f"{dont_recommend_count}/{metadata_count}",
                single_line=True,
            ),
        ]
        fields = [
            to_string("Discrete Parameters", pretty_print_df(param_df)),
            to_string("Experimental Representation", pretty_print_df(self.exp_rep)),
            to_string("Meta Data", *metadata_fields),
            to_string("Constraints", pretty_print_df(constraints_df)),
            to_string("Computational Representation", pretty_print_df(self.comp_rep)),
        ]
        return to_string(self.__class__.__name__, *fields)

    @exp_rep.validator
    def _validate_exp_rep(  # noqa: DOC101, DOC103
        self, _: Any, exp_rep: pd.DataFrame
    ) -> None:
        """Validate the experimental representation.

        Raises:
            ValueError: If the index of the provided dataframe contains duplicates.
        """
        if exp_rep.index.has_duplicates:
            raise ValueError(
                "The index of this search space contains duplicates. "
                "This is not allowed, as it can lead to hard-to-detect bugs."
            )

    @metadata.default
    def _default_metadata(self) -> pd.DataFrame:
        """Create the default metadata."""
        # If the discrete search space is empty, explicitly return an empty dataframe
        # instead of simply using a zero-length index. Otherwise, the Boolean dtype
        # would be lost during a serialization roundtrip as there would be no
        # data available that allows to determine the type, causing subsequent
        # equality checks to fail.
        # TODO: verify if this is still required
        if self.is_empty:
            return pd.DataFrame(columns=_METADATA_COLUMNS)

        # TODO [16605]: Redesign metadata handling
        # Exclude inactive tasks from search
        df = pd.DataFrame(False, columns=_METADATA_COLUMNS, index=self.exp_rep.index)
        off_task_idxs = ~self._on_task_configurations()
        df.loc[off_task_idxs.values, "dont_recommend"] = True  # type: ignore
        return df

    @metadata.validator
    def _validate_metadata(  # noqa: DOC101, DOC103
        self, _: Any, metadata: pd.DataFrame
    ) -> None:
        """Validate the metadata.

        Raises:
            ValueError: If the provided metadata allows testing parameter configurations
                for inactive tasks.
        """
        # We first check whether there are actually any parameters that need to be
        # checked.
        if self.is_empty:
            return
        off_task_idxs = ~self._on_task_configurations()
        if not metadata.loc[off_task_idxs.values, "dont_recommend"].all():  # type: ignore
            raise ValueError(
                "Inconsistent instructions given: The provided metadata allows "
                "testing parameter configurations for inactive tasks."
            )

    @comp_rep.default
    def _default_comp_rep(self) -> pd.DataFrame:
        """Create the default computational representation."""
        # Create a dataframe containing the computational parameter representation
        comp_rep = self.transform(self.exp_rep)

        # Ignore all columns that do not carry any covariate information
        # TODO[12758]: This logic needs to be refined, i.e. when should we drop columns
        #   and when not (can have undesired/unexpected side-effects). Should this be
        #   configurable at the parameter level? A hotfix was made to exclude task
        #   parameters, but this needs to be revisited as well.
        comp_rep = df_drop_single_value_columns(
            comp_rep, [p.name for p in self.parameters if isinstance(p, TaskParameter)]
        )

        return comp_rep

    def __attrs_post_init__(self) -> None:
        # TODO [16605]: Redesign metadata handling
        if self.is_empty:
            return
        off_task_idxs = ~self._on_task_configurations()
        self.metadata.loc[off_task_idxs.values, "dont_recommend"] = True  # type: ignore

    def to_searchspace(self) -> SearchSpace:
        """Turn the subspace into a search space with no continuous part."""
        from baybe.searchspace.core import SearchSpace

        return SearchSpace(discrete=self)

    def _on_task_configurations(self) -> pd.Series:
        """Retrieve the parameter configurations for the active tasks."""
        # TODO [16932]: This only works for a single parameter
        try:
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
        except StopIteration:
            return pd.Series(True, index=self.exp_rep.index)
        return self.exp_rep[task_param.name].isin(task_param.active_values)

    @classmethod
    def empty(cls) -> SubspaceDiscrete:
        """Create an empty discrete subspace."""
        return SubspaceDiscrete(
            parameters=[],
            exp_rep=pd.DataFrame(),
            metadata=pd.DataFrame(columns=_METADATA_COLUMNS),
        )

    @classmethod
    def from_parameter(cls, parameter: DiscreteParameter) -> SubspaceDiscrete:
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
        empty_encoding: bool = False,
    ) -> SubspaceDiscrete:
        """See :class:`baybe.searchspace.core.SearchSpace`."""
        # Set defaults and order constraints
        constraints = constraints or []
        constraints = sorted(
            constraints,
            key=lambda x: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(x.__class__),
        )

        try:
            # Check for manual deactivation of polars
            if os.environ.get("BAYBE_DEACTIVATE_POLARS", None) is not None:
                raise OptionalImportError(
                    "Polars was deactivated manually via environment variable."
                )

            # Apply polars product and filtering
            lazy_df = parameter_cartesian_prod_polars(parameters)
            lazy_df, mask_missing = _apply_constraint_filter_polars(
                lazy_df, constraints
            )
            df_records = lazy_df.collect(streaming=True).to_dicts()
            df = pd.DataFrame.from_records(df_records)
        except OptionalImportError:
            # Apply pandas product
            df = parameter_cartesian_prod_pandas(parameters)
            mask_missing = [True] * len(constraints)

        # Gather and use constraints not yet applied
        _apply_constraint_filter_pandas(df, list(compress(constraints, mask_missing)))

        return SubspaceDiscrete(
            parameters=parameters,
            constraints=constraints,
            exp_rep=df,
            empty_encoding=empty_encoding,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Sequence[DiscreteParameter] | None = None,
        empty_encoding: bool = False,
    ) -> SubspaceDiscrete:
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
            empty_encoding: See :func:`baybe.searchspace.core.SearchSpace.from_product`.

        Returns:
            The created discrete subspace.
        """

        def discrete_parameter_factory(
            name: str, values: Collection[Any]
        ) -> DiscreteParameter:
            """Try to create a numerical parameter or use a categorical fallback."""
            try:
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

        return cls(parameters=parameters, exp_rep=df, empty_encoding=empty_encoding)

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
    ) -> SubspaceDiscrete:
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

        # Validate constraints
        validate_constraints(constraints, [*simplex_parameters, *product_parameters])

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

        # Construct the product part of the space
        product_space = parameter_cartesian_prod_pandas(product_parameters)
        if not simplex_parameters:
            return cls(parameters=product_parameters, exp_rep=product_space)

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

        # Augment the Cartesian product created from all other parameter types
        if product_parameters:
            exp_rep = pd.merge(exp_rep, product_space, how="cross")

        # Remove entries that violate parameter constraints:
        _apply_constraint_filter_pandas(exp_rep, constraints)

        return cls(
            parameters=[*simplex_parameters, *product_parameters],
            exp_rep=exp_rep,
            constraints=constraints,
        )

    @property
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return tuple of parameter names."""
        return tuple(p.name for p in self.parameters)

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        # We go via `comp_rep` here instead of using the columns of the individual
        # parameters because the search space potentially uses only a subset of the
        # columns due to decorrelation
        return tuple(self.comp_rep.columns)

    @property
    def comp_rep_bounds(self) -> pd.DataFrame:
        """The minimum and maximum values of the computational representation."""
        return pd.DataFrame({"min": self.comp_rep.min(), "max": self.comp_rep.max()}).T

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
        n_rows = prod(p.comp_df.shape[0] for p in parameters)

        # Comp rep space is estimated as the size of float times the number of matrix
        # elements in the comp rep. The latter is the total number of parameter
        # configurations (= number of rows) times the total number of columns.
        comp_rep_bytes = (
            np.array([0.0], dtype=DTypeFloatNumpy).itemsize * n_rows * n_cols_comp
        )

        # Exp rep space is estimated as the size of the per-parameter exp rep dataframe
        # times the number of times it will appear in the entire search space. The
        # latter is the total number of parameter configurations (= number of rows)
        # divided by the number of values for the respective parameter. Contributions of
        # all parameters are summed up.
        exp_rep_bytes = sum(
            pd.DataFrame(p.values).memory_usage(index=False, deep=True).sum()
            * n_rows
            / p.comp_df.shape[0]
            for p in parameters
        )

        return MemorySize(
            exp_rep_bytes=exp_rep_bytes,
            exp_rep_shape=(n_rows, n_cols_exp),
            comp_rep_bytes=comp_rep_bytes,
            comp_rep_shape=(n_rows, n_cols_comp),
        )

    def mark_as_measured(
        self,
        measurements: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool,
    ) -> None:
        """Mark the given elements of the space as measured.

        Args:
            measurements: A dataframe containing parameter settings that should be
                marked as measured.
            numerical_measurements_must_be_within_tolerance: See
                :func:`baybe.utils.dataframe.fuzzy_row_match`.
        """
        idxs_matched = fuzzy_row_match(
            self.exp_rep,
            measurements,
            self.parameters,
            numerical_measurements_must_be_within_tolerance,
        )
        self.metadata.loc[idxs_matched, "was_measured"] = True

    def get_candidates(
        self,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = False,
        exclude: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the set of candidate parameter settings that can be tested.

        Args:
            allow_repeated_recommendations: If ``True``, parameter settings that have
                already been recommended in an earlier iteration are still considered
                valid candidates. This is relevant, for instance, when an earlier
                recommended parameter setting has not been measured by the user (for any
                reason) after the corresponding recommendation was made.
            allow_recommending_already_measured: If ``True``, parameters settings for
                which there are already target values available are still considered as
                valid candidates.
            exclude: Points in experimental representation that should be excluded as
                candidates.

        Returns:
            The candidate parameter settings both in experimental and computational
            representation.
        """
        # Filter the search space down to the candidates
        mask_todrop = self.metadata["dont_recommend"].copy()
        if not allow_repeated_recommendations:
            mask_todrop |= self.metadata["was_recommended"]
        if not allow_recommending_already_measured:
            mask_todrop |= self.metadata["was_measured"]

        # Remove additional excludes
        if exclude is not None:
            mask_todrop |= pd.merge(self.exp_rep, exclude, indicator=True, how="left")[
                "_merge"
            ].eq("both")

        return self.exp_rep.loc[~mask_todrop], self.comp_rep.loc[~mask_todrop]

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
            if set(df.columns) - {p.name for p in self.parameters}:
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

        # If the transformed values are not required, return an empty dataframe
        if self.empty_encoding or len(df) < 1:
            return pd.DataFrame(index=df.index)

        # Transform the parameters
        dfs = []
        for param in parameters:
            comp_df = param.transform(df[param.name])
            dfs.append(comp_df)
        comp_rep = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

        # If the computational representation has already been built (with potentially
        # removing some columns, e.g. due to decorrelation or dropping constant ones),
        # any subsequent transformation should yield the same columns.
        try:
            return comp_rep[self.comp_rep.columns]
        except AttributeError:
            return comp_rep

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


def _apply_constraint_filter_pandas(
    df: pd.DataFrame, constraints: Collection[DiscreteConstraint]
) -> pd.DataFrame:
    """Remove discrete search space entries based on constraints.

    The filtering is done inplace, but the modified object is still returned.

    Args:
        df: The data in experimental representation to be modified inplace.
        constraints: List of discrete constraints.

    Returns:
        The filtered dataframe.
    """
    # Remove entries that violate parameter constraints:
    for constraint in (c for c in constraints if c.eval_during_creation):
        idxs = constraint.get_invalid(df)
        df.drop(index=idxs, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def _apply_constraint_filter_polars(
    ldf: pl.LazyFrame,
    constraints: Sequence[DiscreteConstraint],
) -> tuple[pl.LazyFrame, list[bool]]:
    """Remove discrete search space entries based on constraints.

    Note:
        This will silently skip constraints that have no Polars implementation.

    Args:
        ldf: The data in experimental representation to be filtered.
        constraints: Collection of discrete constraints.

    Returns:
        A tuple containing
            * The Polars lazyframe with undesired rows removed
            * A Boolean mask indicating which constraints have **not** been applied
    """
    mask_missing = []

    for c in constraints:
        try:
            to_keep = c.get_invalid_polars().not_()
            ldf = ldf.filter(to_keep)
            mask_missing.append(False)
        except NotImplementedError:
            mask_missing.append(True)

    return ldf, mask_missing


def parameter_cartesian_prod_polars(parameters: Sequence[Parameter]) -> pl.LazyFrame:
    """Create the Cartesian product of all parameter values using Polars.

    Ignores continuous parameters.

    Args:
        parameters: List of parameter objects.

    Returns:
        A lazy dataframe containing all possible discrete parameter value combinations.
    """
    from baybe._optional.polars import polars as pl

    discrete_parameters = [p for p in parameters if p.is_discrete]
    if not discrete_parameters:
        return pl.LazyFrame()

    # Convert each parameter to a lazy dataframe for cross-join operation
    param_frames = [pl.LazyFrame({p.name: p.values}) for p in discrete_parameters]  # type:ignore[attr-defined]

    # Handling edge cases
    if len(param_frames) == 1:
        return param_frames[0]

    # Cross-join parameters
    res = param_frames[0]
    for frame in param_frames[1:]:
        res = res.join(frame, how="cross", force_parallel=True)

    return res


def parameter_cartesian_prod_pandas(
    parameters: Sequence[Parameter],
) -> pd.DataFrame:
    """Create the Cartesian product of all parameter values using Pandas.

    Ignores continuous parameters.

    Args:
        parameters: List of parameter objects.

    Returns:
        A dataframe containing all possible discrete parameter value combinations.
    """
    discrete_parameters = [p for p in parameters if p.is_discrete]
    if not discrete_parameters:
        return pd.DataFrame()

    index = pd.MultiIndex.from_product(
        [p.values for p in discrete_parameters],  # type:ignore[attr-defined]
        names=[p.name for p in discrete_parameters],
    )
    ret = pd.DataFrame(index=index).reset_index()

    return ret


def validate_simplex_subspace_from_config(specs: dict, _) -> None:
    """Validate the discrete space while skipping costly creation steps."""
    # Validate product inputs without constructing it
    if specs.get("constructor", None) == "from_product":
        parameters = converter.structure(specs["parameters"], list[DiscreteParameter])
        validate_parameters(parameters)

        constraints = specs.get("constraints", None)
        if constraints:
            constraints = converter.structure(
                specs["constraints"], list[DiscreteConstraint]
            )
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

        product_parameters = specs.get("product_parameters", None)
        if product_parameters:
            product_parameters = converter.structure(
                specs["product_parameters"], list[DiscreteParameter]
            )

        validate_parameters(simplex_parameters + product_parameters)

        constraints = specs.get("constraints", None)
        if constraints:
            constraints = converter.structure(
                specs["constraints"], list[DiscreteConstraint]
            )
            validate_constraints(constraints, simplex_parameters + product_parameters)

    # For all other types, validate by construction
    else:
        converter.structure(specs, SubspaceDiscrete)


# Register deserialization hook
converter.register_structure_hook(SubspaceDiscrete, select_constructor_hook)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
