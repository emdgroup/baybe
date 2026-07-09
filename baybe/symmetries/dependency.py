"""Dependency symmetry."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from attrs import Converter, define, field, fields
from attrs.validators import deep_iterable, ge, instance_of, min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.constraints.conditions import Condition
from baybe.symmetries.base import Symmetry
from baybe.utils.augmentation import df_apply_dependency_augmentation
from baybe.utils.conversion import nonstring_to_tuple, sort_tuple
from baybe.utils.validation import validate_unique_values

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace


@define(frozen=True)
class DependencySymmetry(Symmetry):
    """Class for representing dependency symmetries.

    A dependency symmetry expresses that certain parameters are dependent on another
    parameter having a specific value. For instance, the situation "The value of
    parameter y only matters if parameter x has the value 'on'.". In this scenario x
    is the causing parameter and y depends on x.
    """

    _parameter_name: str = field(validator=instance_of(str), alias="parameter_name")
    """The names of the causing parameter others are depending on."""

    # object variables
    condition: Condition = field(validator=instance_of(Condition))
    """The condition specifying the active range of the causing parameter."""

    affected_parameter_names: tuple[str, ...] = field(
        converter=[  # type: ignore[misc]
            Converter(nonstring_to_tuple, takes_self=True, takes_field=True),  # type: ignore[call-overload]
            sort_tuple,
        ],
        validator=(
            validate_unique_values,
            deep_iterable(
                member_validator=instance_of(str), iterable_validator=min_len(1)
            ),
        ),
    )
    """The parameters affected by the dependency."""

    n_discretization_points: int | None = field(
        default=None, validator=optional_v((instance_of(int), ge(2))), kw_only=True
    )
    """Number of evenly spaced points used to sample from continuous parameter ranges
    during augmentation. Must be set when any affected parameter is continuous."""

    @override
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (self._parameter_name, *self.affected_parameter_names)

    @override
    def augment_measurements(
        self,
        measurements: pd.DataFrame,
        parameters: Sequence[Parameter] | None = None,
    ) -> pd.DataFrame:
        # See base class.
        if not self.use_data_augmentation:
            return measurements

        if parameters is None:
            raise ValueError(
                f"A '{self.__class__.__name__}' requires parameter objects "
                f"for data augmentation."
            )

        from baybe.parameters.base import DiscreteParameter

        # The 'causing' entry describes the parameters and the value
        # for which one or more affected parameters become degenerate.
        # 'cond' specifies for which values the affected parameter
        # values are active, i.e. not degenerate. Hence, here we get the
        # values that are not active, as rows containing them should be
        # augmented.
        param = next(
            cast(DiscreteParameter, p)
            for p in parameters
            if p.name == self._parameter_name
        )

        causing_values = [
            x
            for x, flag in zip(
                param.values,
                ~self.condition.evaluate(pd.Series(param.values)),
                strict=True,
            )
            if flag
        ]
        causing = (param.name, causing_values)

        # The 'affected' entry describes the affected parameters and the
        # values they are allowed to take, which are all degenerate if
        # the corresponding condition for the causing parameter is met.
        affected: list[tuple[str, tuple[float, ...]]] = []
        for pn in self.affected_parameter_names:
            p = next(p for p in parameters if p.name == pn)
            if p.is_discrete:
                # Use all values for augmentation
                vals = cast(DiscreteParameter, p).values
            else:
                # Use linear subsample of parameter bounds interval for augmentation.
                # Note: The original value will not necessarily be part of this.
                assert self.n_discretization_points is not None
                vals = tuple(
                    np.linspace(
                        p.bounds.lower,  # type: ignore[attr-defined]
                        p.bounds.upper,  # type: ignore[attr-defined]
                        self.n_discretization_points,
                    )
                )
            affected.append((p.name, vals))

        measurements = df_apply_dependency_augmentation(measurements, causing, affected)

        return measurements

    @override
    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """See base class.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            TypeError: If the causing parameter is not discrete.
            ValueError: If any affected parameter is continuous and
                ``n_discretization_points`` is not set.
        """
        super().validate_searchspace_context(searchspace)

        # Causing parameter must be discrete
        param = searchspace.get_parameters_by_name(self._parameter_name)[0]
        if not param.is_discrete:
            raise TypeError(
                f"In a '{self.__class__.__name__}', the causing parameter must "
                f"be discrete. However, the parameter '{param.name}' is of "
                f"type '{param.__class__.__name__}' and is not discrete."
            )

        # n_discretization_points is required when affected parameters are continuous
        if self.n_discretization_points is None:
            affected = searchspace.get_parameters_by_name(self.affected_parameter_names)
            if any(not p.is_discrete for p in affected):
                raise ValueError(
                    f"'{fields(DependencySymmetry).n_discretization_points.alias}' "
                    f"must be set explicitly when affected parameters are continuous. "
                    f"It specifies the number of evenly spaced points used to sample "
                    f"from the parameter's range for data augmentation."
                )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
