"""Product-free input layouts built directly from parameter definitions."""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from attrs import define, field

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.categorical import CategoricalParameter, TaskParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.settings import active_settings
from baybe.utils.dataframe import get_transform_objects

if TYPE_CHECKING:
    from baybe.parameters.selectors import ParameterSelectorProtocol


def _sort_parameters(parameters: Sequence[Parameter]) -> tuple[Parameter, ...]:
    """Sort parameters like :class:`~baybe.searchspace.core.SearchSpace`.

    Discrete parameters come first, followed by continuous parameters, and each
    group is sorted alphabetically by parameter name. Unsupported concrete
    parameter types are rejected before the discrete/continuous split so that
    they cannot silently disappear from the layout.

    Raises:
        IncompatibilityError: If a parameter type or encoding is unsupported.
    """
    parameters = tuple(parameters)
    for parameter in parameters:
        _validate_supported_parameter(parameter)

    discrete = sorted(
        (parameter for parameter in parameters if parameter.is_discrete),
        key=lambda parameter: parameter.name,
    )
    continuous = sorted(
        (parameter for parameter in parameters if parameter.is_continuous),
        key=lambda parameter: parameter.name,
    )
    return (*discrete, *continuous)


def _validate_supported_parameter(parameter: Parameter) -> None:
    """Reject parameter types and encodings outside the product-free contract.

    Raises:
        IncompatibilityError: If the parameter type or encoding is unsupported.
    """
    if isinstance(parameter, TaskParameter):
        raise IncompatibilityError(
            f"Parameter '{parameter.name}' is a '{TaskParameter.__name__}', which is "
            f"not supported by the product-free input layout."
        )
    if isinstance(
        parameter, (NumericalContinuousParameter, NumericalDiscreteParameter)
    ):
        return
    if isinstance(parameter, CategoricalParameter):
        if parameter.encoding not in (
            CategoricalEncoding.OHE,
            CategoricalEncoding.INT,
        ):
            raise IncompatibilityError(
                f"Parameter '{parameter.name}' uses the unsupported categorical "
                f"encoding '{parameter.encoding}'."
            )
        return
    raise IncompatibilityError(
        f"Parameter '{parameter.name}' of type '{type(parameter).__name__}' is not "
        f"supported by the product-free input layout. Supported types are "
        f"'{NumericalContinuousParameter.__name__}', "
        f"'{NumericalDiscreteParameter.__name__}', and "
        f"'{CategoricalParameter.__name__}' with OHE or INT encoding."
    )


def _validate_active_levels(parameter: Parameter) -> None:
    """Check that every declared active level has a complete encoding row.

    Raises:
        IncompatibilityError: If the parameter declares no active levels or an
            active level is missing from the parameter's computational
            representation.
    """
    if not isinstance(parameter, DiscreteParameter):
        return

    active_values = parameter.active_values
    if not active_values:
        raise IncompatibilityError(
            f"Discrete parameter '{parameter.name}' must declare at least one active "
            f"level."
        )

    encoded_levels = set(parameter.comp_df.index)
    unknown = [value for value in active_values if value not in encoded_levels]
    if unknown:
        raise IncompatibilityError(
            f"Parameter '{parameter.name}' declares active levels without a "
            f"computational-representation row: {unknown}."
        )


def _validate_parameter_definitions(parameters: Sequence[Parameter]) -> None:
    """Validate names, levels, encodings, and generated columns parameter-wise.

    Raises:
        IncompatibilityError: If names or generated comp-rep columns are not
            unique, or a parameter is unsupported.
    """
    duplicate_names = sorted(
        name for name, count in Counter(p.name for p in parameters).items() if count > 1
    )
    if duplicate_names:
        raise IncompatibilityError(
            f"Parameter names must be unique in the product-free input layout, but "
            f"found duplicates: {duplicate_names}."
        )

    for parameter in parameters:
        _validate_supported_parameter(parameter)
        _validate_active_levels(parameter)

    columns = [
        column for parameter in parameters for column in parameter.comp_rep_columns
    ]
    duplicate_columns = sorted(
        column for column, count in Counter(columns).items() if count > 1
    )
    if duplicate_columns:
        raise IncompatibilityError(
            f"Computational-representation columns of the product-free input layout "
            f"collide: {duplicate_columns}."
        )


@define(frozen=True, slots=False)
class _FactorizedInputLayout:
    """Immutable parameter layout without any candidate or membership table.

    The layout is the product-free counterpart of a real
    :class:`~baybe.searchspace.core.SearchSpace` for the supported basic parameter
    types. It uses the sorted parameter definitions and their complete
    computational-representation data frames as the single source for names,
    ordering, columns, scaling bounds, and transformations.
    """

    parameters: tuple[Parameter, ...] = field(converter=_sort_parameters)
    """The parameters in BayBE comp-rep order (discrete first, then continuous)."""

    @parameters.validator
    def _validate_parameters(self, _attribute, value) -> None:  # noqa: DOC101, DOC103
        """Validate the parameter definitions without building a product."""
        _validate_parameter_definitions(value)

    @classmethod
    def from_parameters(cls, parameters: Sequence[Parameter]) -> _FactorizedInputLayout:
        """Create a layout from parameter definitions.

        Args:
            parameters: The parameters spanning the layout.

        Returns:
            The immutable product-free input layout.
        """
        return cls(tuple(parameters))

    @property
    def _continuous_parameters(self) -> tuple[NumericalContinuousParameter, ...]:
        """The continuous parameters in comp-rep order."""
        return tuple(
            cast(NumericalContinuousParameter, parameter)
            for parameter in self.parameters
            if parameter.is_continuous
        )

    @property
    def _discrete_parameters(self) -> tuple[DiscreteParameter, ...]:
        """The discrete parameters in comp-rep order."""
        return tuple(
            cast(DiscreteParameter, parameter)
            for parameter in self.parameters
            if parameter.is_discrete
        )

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        return tuple(
            column
            for parameter in self.parameters
            for column in parameter.comp_rep_columns
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameter names in comp-rep order."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def scaling_bounds(self) -> pd.DataFrame:
        """The bounds used for scaling the surrogate model input."""
        discrete = (
            pd.concat(
                [
                    parameter.comp_df.agg(["min", "max"])
                    for parameter in self._discrete_parameters
                ],
                axis=1,
            )
            if self._discrete_parameters
            else pd.DataFrame(index=["min", "max"])
        )
        continuous = (
            pd.DataFrame(
                {
                    parameter.name: parameter.bounds.to_tuple()
                    for parameter in self._continuous_parameters
                },
                index=["min", "max"],
                dtype=active_settings.DTypeFloatNumpy,
            )
            if self._continuous_parameters
            else pd.DataFrame(index=["min", "max"])
        )
        return pd.concat([discrete, continuous], axis=1)

    @property
    def _task_parameter(self) -> TaskParameter | None:
        """The task parameter of the layout, which is always ``None``."""
        return None

    @property
    def task_idx(self) -> int | None:
        """The comp-rep index of the task parameter, which is always ``None``."""
        return None

    @property
    def n_tasks(self) -> int:
        """The number of tasks encoded by the layout, which is always one."""
        return 1

    def active_levels(self, name: str) -> tuple[Any, ...]:
        """Return the active levels of a discrete parameter.

        Args:
            name: The parameter name.

        Returns:
            The active levels in the parameter's stable level order.

        Raises:
            IncompatibilityError: If the parameter is unknown or not discrete.
        """
        try:
            parameter = next(
                parameter for parameter in self.parameters if parameter.name == name
            )
        except StopIteration as ex:
            raise IncompatibilityError(f"Unknown parameter '{name}'.") from ex

        if not isinstance(parameter, DiscreteParameter):
            raise IncompatibilityError(
                f"Parameter '{name}' is not a discrete parameter and has no levels."
            )
        return tuple(parameter.active_values)

    def transform(
        self,
        frame: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        """Transform parameters from experimental to computational representation.

        Args:
            frame: The dataframe to be transformed.
            allow_missing: If ``False``, each parameter of the layout must have
                exactly one corresponding column in the given dataframe. If ``True``,
                the dataframe may contain only a subset of parameter columns.
            allow_extra: If ``False``, each column present in the dataframe must
                correspond to exactly one parameter of the layout. If ``True``, the
                dataframe may contain additional non-parameter-related columns,
                which will be ignored.

        Returns:
            A corresponding dataframe with parameters in computational
            representation.
        """
        discrete_names = [parameter.name for parameter in self._discrete_parameters]
        continuous_names = [parameter.name for parameter in self._continuous_parameters]

        discrete_input = frame.drop(columns=continuous_names, errors="ignore")
        continuous_input = frame.drop(columns=discrete_names, errors="ignore")

        discrete = self._transform_discrete(
            discrete_input, allow_missing=allow_missing, allow_extra=allow_extra
        )
        continuous = self._transform_continuous(
            continuous_input, allow_missing=allow_missing, allow_extra=allow_extra
        )
        return pd.concat([discrete, continuous], axis=1)

    def _transform_discrete(
        self,
        frame: pd.DataFrame,
        /,
        *,
        allow_missing: bool,
        allow_extra: bool,
    ) -> pd.DataFrame:
        """Transform the discrete part like :class:`SubspaceDiscrete`."""
        parameters = get_transform_objects(
            frame,
            self._discrete_parameters,
            allow_missing=allow_missing,
            allow_extra=allow_extra,
        )
        if len(frame) < 1:
            return pd.DataFrame(index=frame.index)

        frames = [
            parameter.transform(frame[parameter.name]) for parameter in parameters
        ]
        comp_rep = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        columns = [
            column
            for parameter in self._discrete_parameters
            for column in parameter.comp_rep_columns
        ]
        return comp_rep[columns]

    def _transform_continuous(
        self,
        frame: pd.DataFrame,
        /,
        *,
        allow_missing: bool,
        allow_extra: bool,
    ) -> pd.DataFrame:
        """Transform the continuous part like :class:`SubspaceContinuous`."""
        parameters = get_transform_objects(
            frame,
            self._continuous_parameters,
            allow_missing=allow_missing,
            allow_extra=allow_extra,
        )
        return frame[[parameter.name for parameter in parameters]]

    def get_comp_rep_parameter_indices(
        self,
        name_or_selector: str | ParameterSelectorProtocol,
        /,
    ) -> tuple[int, ...]:
        """Find comp-rep column indices for a parameter selection.

        Unknown parameter names and selectors without matches yield an empty tuple,
        matching :class:`~baybe.searchspace.core.SearchSpace`.

        Args:
            name_or_selector: Either the name of a single parameter or a selector
                that filters parameters to be included.

        Returns:
            The integer indices of the selected comp-rep columns.
        """
        if isinstance(name_or_selector, str):
            parameters = [
                parameter
                for parameter in self.parameters
                if parameter.name == name_or_selector
            ]
        else:
            parameters = [
                parameter
                for parameter in self.parameters
                if name_or_selector(parameter)
            ]

        columns = self.comp_rep_columns
        return tuple(
            index
            for parameter in parameters
            for index, column in enumerate(columns)
            if column in parameter.comp_rep_columns
        )

    def _get_n_comp_rep_columns(
        self,
        name_or_selector: str | ParameterSelectorProtocol,
        /,
    ) -> int:
        """Get the number of comp-rep columns for a parameter selection.

        Args:
            name_or_selector: Either the name of a single parameter or a selector
                that filters parameters to be included.

        Returns:
            The number of selected comp-rep columns.
        """
        return len(self.get_comp_rep_parameter_indices(name_or_selector))

    def get_parameters_by_name(self, names: Sequence[str]) -> tuple[Parameter, ...]:
        """Return parameters with the specified names.

        Unknown names are ignored, matching
        :class:`~baybe.searchspace.core.SearchSpace`.

        Args:
            names: Sequence of parameter names.

        Returns:
            The named parameters in comp-rep order.
        """
        return tuple(
            parameter for parameter in self.parameters if parameter.name in names
        )

    def _drop_parameters(self, names: Collection[str], /) -> _FactorizedInputLayout:
        """Return a layout without the named parameters.

        Args:
            names: The names of the parameters to remove.

        Raises:
            ValueError: If any name does not match a parameter in the layout.

        Returns:
            A new layout containing only the remaining parameter definitions.
        """
        current_names = set(self.parameter_names)
        names_set = set(names)
        if unknown := names_set - current_names:
            raise ValueError(
                f"Parameter name(s) {sorted(unknown)} not found in the input layout. "
                f"Available: {sorted(current_names)}."
            )
        return _FactorizedInputLayout(
            tuple(
                parameter
                for parameter in self.parameters
                if parameter.name not in names_set
            )
        )


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
