"""Functionality for managing search spaces."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import cast

import pandas as pd
from attr import define, field

from baybe.constraints import (
    validate_constraints,
)
from baybe.constraints.base import Constraint
from baybe.parameters import SubstanceEncoding, TaskParameter
from baybe.parameters.base import Parameter
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.discrete import (
    MemorySize,
    SubspaceDiscrete,
    validate_simplex_subspace_from_config,
)
from baybe.searchspace.validation import validate_parameters
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.telemetry import TELEM_LABELS, telemetry_record_value
from baybe.utils.plotting import to_string


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

    def __str__(self) -> str:
        fields = [
            to_string("Search Space Type", self.type.name, single_line=True),
        ]
        if not self.discrete.is_empty:
            fields.append(str(self.discrete))
        if not self.continuous.is_empty:
            fields.append(str(self.continuous))
        return to_string(self.__class__.__name__, *fields)

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
    def from_parameter(cls, parameter: Parameter) -> SearchSpace:
        """Create a search space from a single parameter.

        Args:
            parameter: The parameter to span the search space.

        Returns:
            The created search space.
        """
        return cls.from_product([parameter])

    @classmethod
    def from_product(
        cls,
        parameters: Sequence[Parameter],
        constraints: Sequence[Constraint] | None = None,
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

        discrete = SubspaceDiscrete.from_product(
            parameters=[p for p in parameters if p.is_discrete],  # type:ignore[misc]
            constraints=[c for c in constraints if c.is_discrete],  # type:ignore[misc]
            empty_encoding=empty_encoding,
        )
        continuous = SubspaceContinuous.from_product(
            parameters=[p for p in parameters if p.is_continuous],  # type:ignore[misc]
            constraints=[c for c in constraints if c.is_continuous],  # type:ignore[misc]
        )

        return SearchSpace(discrete=discrete, continuous=continuous)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Sequence[Parameter],
    ) -> SearchSpace:
        """Create a search space from a specified set of parameter configurations.

        The way in which the contents of the columns are interpreted depends on the
        types of the corresponding parameter objects provided. For details, see
        :meth:`baybe.searchspace.discrete.SubspaceDiscrete.from_dataframe` and
        :meth:`baybe.searchspace.continuous.SubspaceContinuous.from_dataframe`.

        Args:
            df: A dataframe whose parameter configurations are used as
                search space specification.
            parameters: The corresponding parameter objects, one for each column
                in the provided dataframe.

        Returns:
            The created search space.

        Raises:
            ValueError: If the dataframe columns do not match with the parameters.
        """
        if {p.name for p in parameters} != set(df.columns.values):
            raise ValueError(
                "The provided dataframe columns must match exactly with the specified "
                "parameter names."
            )

        disc_params = [p for p in parameters if p.is_discrete]
        cont_params = [p for p in parameters if p.is_continuous]

        return SearchSpace(
            discrete=SubspaceDiscrete.from_dataframe(
                df[[p.name for p in disc_params]],
                disc_params,  # type:ignore[arg-type]
            ),
            continuous=SubspaceContinuous.from_dataframe(
                df[[p.name for p in cont_params]],
                cont_params,  # type:ignore[arg-type]
            ),
        )

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Return the list of parameters of the search space."""
        return (*self.discrete.parameters, *self.continuous.parameters)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return the constraints of the search space."""
        return (
            *self.discrete.constraints,
            *self.continuous.constraints_lin_eq,
            *self.continuous.constraints_lin_ineq,
            *self.continuous.constraints_nonlin,
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
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        return self.discrete.comp_rep_columns + self.continuous.comp_rep_columns

    @property
    def comp_rep_bounds(self) -> pd.DataFrame:
        """The minimum and maximum values of the computational representation."""
        return pd.concat(
            [self.discrete.comp_rep_bounds, self.continuous.comp_rep_bounds], axis=1
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return tuple of parameter names."""
        return self.discrete.parameter_names + self.continuous.parameter_names

    @property
    def task_idx(self) -> int | None:
        """The column index of the task parameter in computational representation."""
        try:
            # TODO [16932]: Redesign metadata handling
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
        except StopIteration:
            return None
        # TODO[11611]: The current approach has three limitations:
        #   1.  It matches by column name and thus assumes that the parameter name
        #       is used as the column name.
        #   2.  It relies on the current implementation detail that discrete parameters
        #       appear first in the computational dataframe.
        #   3.  It assumes there exists exactly one task parameter
        #   --> Fix this when refactoring the data
        return cast(int, self.discrete.comp_rep.columns.get_loc(task_param.name))

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

    def get_comp_rep_parameter_indices(self, name: str, /) -> tuple[int, ...]:
        """Find a parameter's column indices in the computational representation.

        Args:
            name: The name of the parameter whose columns indices are to be retrieved.

        Raises:
            ValueError: If no parameter with the provided name exists.
            ValueError: If more than one parameter with the provided name exists.

        Returns:
            A tuple containing the integer indices of the columns in the computational
            representation associated with the parameter. When the parameter is not part
            of the computational representation, an empty tuple is returned.
        """
        params = self.get_parameters_by_name([name])
        if len(params) < 1:
            raise ValueError(
                f"There exists no parameter named '{name}' in the search space."
            )
        if len(params) > 1:
            raise ValueError(
                f"There exist multiple parameter matches for '{name}' in the search "
                f"space."
            )
        p = params[0]

        return tuple(
            i
            for i, col in enumerate(self.comp_rep_columns)
            if col in p.comp_rep_columns
        )

    @staticmethod
    def estimate_product_space_size(parameters: Iterable[Parameter]) -> MemorySize:
        """Estimate an upper bound for the memory size of a product space.

        Continuous parameters are ignored because creating a continuous subspace has
        no considerable memory footprint.

        Args:
            parameters: The parameters spanning the product space.

        Returns:
            The estimated memory size.
        """
        discrete_parameters = [p for p in parameters if p.is_discrete]
        return SubspaceDiscrete.estimate_product_space_size(discrete_parameters)  # type: ignore[arg-type]

    def transform(
        self,
        df: pd.DataFrame | None = None,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool | None = None,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Transform parameters from experimental to computational representation.

        Args:
            df: The dataframe to be transformed. The allowed columns of the dataframe
                are dictated by the ``allow_missing`` and ``allow_extra`` flags.
                The ``None`` default value is for temporary backward compatibility only
                and will be removed in a future version.
            allow_missing: If ``False``, each parameter of the space must have
                (exactly) one corresponding column in the given dataframe. If ``True``,
                the dataframe may contain only a subset of parameter columns.
            allow_extra: If ``False``, every column present in the dataframe must
                correspond to (exactly) one parameter of the space. If ``True``, the
                dataframe may contain additional non-parameter-related columns, which
                will be ignored.
                The ``None`` default value is for temporary backward compatibility only
                and will be removed in a future version.
            data: Ignore! For backward compatibility only.

        Raises:
            ValueError: If dataframes are passed to both ``df`` and ``data``.

        Returns:
            A corresponding dataframe with parameters in computational representation.
        """
        # TODO: Remove deprecation-related explanation of `None` default values
        #   from docstring once deprecation expires
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

        # Transform subspaces separately
        df_discrete = self.discrete.transform(
            df, allow_missing=allow_missing, allow_extra=allow_extra
        )
        df_continuous = self.continuous.transform(
            df, allow_missing=allow_missing, allow_extra=allow_extra
        )

        # Combine Subspaces
        comp_rep = pd.concat([df_discrete, df_continuous], axis=1)

        return comp_rep

    @property
    def constraints_augmentable(self) -> tuple[Constraint, ...]:
        """The searchspace constraints that can be considered during augmentation."""
        return tuple(c for c in self.constraints if c.eval_during_augmentation)

    def get_parameters_by_name(self, names: Sequence[str]) -> tuple[Parameter, ...]:
        """Return parameters with the specified names.

        Args:
            names: Sequence of parameter names.

        Returns:
            The named parameters.
        """
        return self.discrete.get_parameters_by_name(
            names
        ) + self.continuous.get_parameters_by_name(names)


def to_searchspace(
    x: Parameter | SubspaceDiscrete | SubspaceContinuous | SearchSpace, /
) -> SearchSpace:
    """Convert a parameter/subspace into a search space (with search space passthrough)."""  # noqa: E501
    return x if isinstance(x, SearchSpace) else x.to_searchspace()


def validate_searchspace_from_config(specs: dict, _) -> None:
    """Validate the search space specifications while skipping costly creation steps."""
    # Validate product inputs without constructing it
    if specs.get("constructor", None) == "from_product":
        parameters = converter.structure(specs["parameters"], list[Parameter])
        validate_parameters(parameters)

        constraints = specs.get("constraints", None)
        if constraints:
            constraints = converter.structure(specs["constraints"], list[Constraint])
            validate_constraints(constraints, parameters)

    else:
        discrete_subspace_specs = specs.get("discrete", {})
        if discrete_subspace_specs.get("constructor", None) == "from_simplex":
            # Validate discrete simplex subspace
            _validation_converter = converter.copy()
            _validation_converter.register_structure_hook(
                SubspaceDiscrete, validate_simplex_subspace_from_config
            )
            _validation_converter.structure(discrete_subspace_specs, SubspaceDiscrete)
        else:
            # For all other types, validate by construction
            converter.structure(specs, SearchSpace)


# Register deserialization hook
converter.register_structure_hook(SearchSpace, select_constructor_hook)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
