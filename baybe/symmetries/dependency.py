"""Dependency symmetry."""

from __future__ import annotations

import gc
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from attrs import Converter, define, field
from attrs.validators import deep_iterable, ge, instance_of, min_len
from typing_extensions import override

from baybe.constraints.conditions import Condition
from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.symmetries.base import Symmetry
from baybe.utils.augmentation import df_apply_dependency_augmentation
from baybe.utils.conversion import normalize_convertible2str_sequence
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
        converter=Converter(  # type: ignore[misc,call-overload]  # mypy: Converter
            normalize_convertible2str_sequence, takes_self=True, takes_field=True
        ),
        validator=(
            validate_unique_values,
            deep_iterable(
                member_validator=instance_of(str), iterable_validator=min_len(1)
            ),
        ),
    )
    """The parameters affected by the dependency."""

    n_discretization_points: int = field(
        default=3, validator=(instance_of(int), ge(2)), kw_only=True
    )
    """Number of points used when subsampling continuous parameter ranges."""

    @override
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (self._parameter_name,)

    @override
    def augment_measurements(
        self,
        measurements: pd.DataFrame,
        parameters: Iterable[Parameter] | None = None,
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
            IncompatibleSearchSpaceError: If any of the affected parameters is
                not present in the searchspace.
            TypeError: If the causing parameter is not discrete.
        """
        super().validate_searchspace_context(searchspace)

        # Affected parameters must be in the searchspace
        parameters_missing = set(self.affected_parameter_names).difference(
            searchspace.parameter_names
        )
        if parameters_missing:
            raise IncompatibleSearchSpaceError(
                f"The symmetry of type '{self.__class__.__name__}' was set up "
                f"with at least one parameter which is not present in the "
                f"search space: {parameters_missing}."
            )

        # Causing parameter must be discrete
        param = searchspace.get_parameters_by_name(self._parameter_name)[0]
        if not param.is_discrete:
            raise TypeError(
                f"In a '{self.__class__.__name__}', the causing parameter must "
                f"be discrete. However, the parameter '{param.name}' is of "
                f"type '{param.__class__.__name__}' and is not discrete."
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
