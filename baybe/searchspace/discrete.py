"""Discrete subspaces."""

from __future__ import annotations

from itertools import zip_longest
from typing import Any, Collection, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from attr import define, field
from cattrs import IterableValidationError

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER
from baybe.constraints.base import DiscreteConstraint
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.utils import get_parameters_from_dataframe
from baybe.searchspace.validation import validate_parameter_names
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.utils.boolean import eq_dataframe
from baybe.utils.dataframe import (
    df_drop_single_value_columns,
    fuzzy_row_match,
    pretty_print_df,
)

_METADATA_COLUMNS = ["was_recommended", "was_measured", "dont_recommend"]


@define
class SubspaceDiscrete(SerialMixin):
    """Class for managing discrete subspaces.

    Builds the subspace from parameter definitions and optional constraints, keeps
    track of search metadata, and provides access to candidate sets and different
    parameter views.
    """

    parameters: List[DiscreteParameter] = field(
        validator=lambda _1, _2, x: validate_parameter_names(x)
    )
    """The list of parameters of the subspace."""

    exp_rep: pd.DataFrame = field(eq=eq_dataframe)
    """The experimental representation of the subspace."""

    metadata: pd.DataFrame = field(eq=eq_dataframe)
    """The metadata."""

    empty_encoding: bool = field(default=False)
    """Flag encoding whether an empty encoding is used."""

    constraints: List[DiscreteConstraint] = field(factory=list)
    """A list of constraints for restricting the space."""

    comp_rep: pd.DataFrame = field(eq=eq_dataframe)
    """The computational representation of the space. Technically not required but added
    as an optional initializer argument to allow ingestion from e.g. serialized objects
    and thereby speed up construction. If not provided, the default hook will derive it
    from ``exp_rep``."""

    def __str__(self) -> str:
        if self.is_empty:
            return ""

        start_bold = "\033[1m"
        end_bold = "\033[0m"

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

        # Put all attributes of the discrete class in one string.
        discrete_str = f"""{start_bold}|--> Discrete search space
            \nDiscrete Parameters{end_bold}\n{pretty_print_df(param_df)}
            \n{start_bold}Experimental Representation{end_bold}
            \n{pretty_print_df(self.exp_rep)}\n{start_bold}\nMetadata:{end_bold}
            \r{_METADATA_COLUMNS[0]}: {was_recommended_count}/{metadata_count}
            \r{_METADATA_COLUMNS[1]}: {was_measured_count}/{metadata_count}
            \r{_METADATA_COLUMNS[2]}: {dont_recommend_count}/{metadata_count}
            \n{start_bold}Constraints{end_bold}\n{pretty_print_df(constraints_df)}
            \n{start_bold}Computational representation of the space{end_bold}
            \n{pretty_print_df(self.comp_rep)}\n\n"""

        return discrete_str

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
        # instead of simply using a zero-length index. Otherwise, the boolean dtype
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
        df.loc[off_task_idxs.values, "dont_recommend"] = True
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
        off_task_idxs = ~self._on_task_configurations()
        if not metadata.loc[off_task_idxs.values, "dont_recommend"].all():
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
        off_task_idxs = ~self._on_task_configurations()
        self.metadata.loc[off_task_idxs.values, "dont_recommend"] = True

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
    def from_product(
        cls,
        parameters: List[DiscreteParameter],
        constraints: Optional[List[DiscreteConstraint]] = None,
        empty_encoding: bool = False,
    ) -> SubspaceDiscrete:
        """See :class:`baybe.searchspace.core.SearchSpace`."""
        # Store the input
        if constraints is None:
            constraints = []
        else:
            # Reorder the constraints according to their execution order
            constraints = sorted(
                constraints,
                key=lambda x: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(x.__class__),
            )

        # Create a dataframe representing the experimental search space
        exp_rep = parameter_cartesian_prod_to_df(parameters)

        # Remove entries that violate parameter constraints:
        for constraint in (c for c in constraints if c.eval_during_creation):
            inds = constraint.get_invalid(exp_rep)
            exp_rep.drop(index=inds, inplace=True)
        exp_rep.reset_index(inplace=True, drop=True)

        return SubspaceDiscrete(
            parameters=parameters,
            constraints=constraints,
            exp_rep=exp_rep,
            empty_encoding=empty_encoding,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Optional[List[DiscreteParameter]] = None,
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

        # Get the full list of both explicitly and implicitly defined parameter
        parameters = get_parameters_from_dataframe(
            df, discrete_parameter_factory, parameters
        )

        return cls(parameters=parameters, exp_rep=df, empty_encoding=empty_encoding)

    @classmethod
    def from_simplex(
        cls,
        max_sum: float,
        simplex_parameters: List[NumericalDiscreteParameter],
        product_parameters: Optional[List[DiscreteParameter]] = None,
        min_nonzero: int = 0,
        max_nonzero: Optional[int] = None,
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
        if not all(isinstance(p, DiscreteParameter) for p in product_parameters):
            raise ValueError(
                f"All parameters passed via 'product_parameters' "
                f"must be of subclasses of '{DiscreteParameter.__name__}'."
            )

        # Construct the product part of the space
        product_space = parameter_cartesian_prod_to_df(product_parameters)
        if not simplex_parameters:
            return cls(parameters=product_parameters, exp_rep=product_space)

        # Validate non-negativity
        min_values = [min(p.values) for p in simplex_parameters]
        max_values = [max(p.values) for p in simplex_parameters]
        if not (min(min_values) >= 0.0):
            raise ValueError(
                f"All parameters passed to '{cls.from_simplex.__name__}' "
                f"must have non-negative values only."
            )

        def drop_invalid(
            df: pd.DataFrame,
            max_sum: float,
            boundary_only: bool,
            min_nonzero: Optional[int] = None,
            max_nonzero: Optional[int] = None,
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
            zip_longest(
                simplex_parameters,
                min_sum_upcoming,
                min_nonzero_upcoming,
                max_nonzero_upcoming,
                fillvalue=0,
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

        # Reset the index
        exp_rep.reset_index(drop=True, inplace=True)

        return cls(parameters=simplex_parameters + product_parameters, exp_rep=exp_rep)

    @property
    def is_empty(self) -> bool:
        """Return whether this subspace is empty."""
        return len(self.parameters) == 0

    @property
    def param_bounds_comp(self) -> torch.Tensor:
        """Return bounds as tensor.

        Take bounds from the parameter definitions, but discards bounds belonging to
        columns that were filtered out during the creation of the space.
        """
        if not self.parameters:
            return torch.empty(2, 0)
        bounds = np.hstack(
            [
                np.vstack([p.comp_df[col].min(), p.comp_df[col].max()])
                for p in self.parameters
                for col in p.comp_df
                if col in self.comp_rep.columns
            ]
        )
        return torch.from_numpy(bounds)

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
        inds_matched = fuzzy_row_match(
            self.exp_rep,
            measurements,
            self.parameters,
            numerical_measurements_must_be_within_tolerance,
        )
        self.metadata.loc[inds_matched, "was_measured"] = True

    def get_candidates(
        self,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        return self.exp_rep.loc[~mask_todrop], self.comp_rep.loc[~mask_todrop]

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform parameters from experimental to computational representation.

        Continuous parameters and additional columns are ignored.

        Args:
            data: The data to be transformed. Must contain all specified parameters, can
                contain more columns.

        Returns:
            A dataframe with the parameters in computational representation.
        """
        # If the transformed values are not required, return an empty dataframe
        if self.empty_encoding or len(data) < 1:
            comp_rep = pd.DataFrame(index=data.index)
            return comp_rep

        # Transform the parameters
        dfs = []
        for param in self.parameters:
            comp_df = param.transform_rep_exp2comp(data[param.name])
            dfs.append(comp_df)
        comp_rep = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

        # If the computational representation has already been built (with potentially
        # removing some columns, e.g. due to decorrelation or dropping constant ones),
        # any subsequent transformation should yield the same columns.
        try:
            comp_rep = comp_rep[self.comp_rep.columns]
        except AttributeError:
            pass

        return comp_rep


def parameter_cartesian_prod_to_df(
    parameters: Iterable[Parameter],
) -> pd.DataFrame:
    """Create the Cartesian product of all parameter values.

    Ignores continuous parameters.

    Args:
        parameters: List of parameter objects.

    Returns:
        A dataframe containing all possible discrete parameter value combinations.
    """
    discrete_parameters = [p for p in parameters if isinstance(p, DiscreteParameter)]
    if not discrete_parameters:
        return pd.DataFrame()

    index = pd.MultiIndex.from_product(
        [p.values for p in discrete_parameters],
        names=[p.name for p in discrete_parameters],
    )
    ret = pd.DataFrame(index=index).reset_index()

    return ret


# Register deserialization hook
converter.register_structure_hook(SubspaceDiscrete, select_constructor_hook)
