"""Functionality for expressing symmetries of the modeling process."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attrs import Converter, define, field
from attrs.validators import deep_iterable, ge, instance_of, min_len
from typing_extensions import override

from baybe.constraints.conditions import Condition
from baybe.serialization import SerialMixin
from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_mirror_augmentation,
    df_apply_permutation_augmentation,
)
from baybe.utils.conversion import normalize_str_sequence
from baybe.utils.validation import validate_is_finite, validate_unique_values

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace


@define(frozen=True)
class Symmetry(SerialMixin, ABC):
    """Abstract base class for symmetries.

    Symmetry is a concept that can be used to configure the modelling process in the
    presence of invariances.
    """

    use_data_augmentation: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """Flag indicating whether data augmentation would be used with surrogates that
    support this."""

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """The names of the parameters affected by the symmetry."""

    def summary(self) -> dict:
        """Return a custom summarization of the symmetry."""
        symmetry_dict = dict(
            Type=self.__class__.__name__, Affected_Parameters=self.parameter_names
        )
        return symmetry_dict

    @abstractmethod
    def augment_measurements(
        self, df: pd.DataFrame, parameters: Iterable[Parameter]
    ) -> pd.DataFrame:
        """Augment the given measurements according to the symmetry.

        Args:
            df: The dataframe containing the measurements to be augmented.
            parameters: Parameter objects carrying additional information (might not be
                needed by all augmentation implementations).

        Returns:
            The augmented dataframe including the original measurements.
        """

    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """Validate that the symmetry is compatible with the given searchspace.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            ValueError: If the symmetry affects parameters not present in the
                searchspace.
        """
        parameters_missing = set(self.parameter_names).difference(
            searchspace.parameter_names
        )
        if parameters_missing:
            raise ValueError(
                f"The symmetry of type {self.__class__.__name__} was set up with at "
                f"least one parameter which is not present in the searchspace: "
                f"{parameters_missing}."
            )


@define(frozen=True)
class PermutationSymmetry(Symmetry):
    """Class for representing permutation symmetries.

    A permutation symmetry expresses that certain parameters can be permuted without
    affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(y,x)$.
    """

    _parameter_names: tuple[str, ...] = field(
        alias="parameter_names",
        converter=Converter(normalize_str_sequence, takes_self=True, takes_field=True),  # type: ignore
        validator=(  # type: ignore
            validate_unique_values,
            deep_iterable(
                member_validator=instance_of(str), iterable_validator=min_len(2)
            ),
        ),
    )
    """The names of the parameters affected by the symmetry."""

    @override
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    # Object variables
    # TODO: Needs inner converter to tuple
    copermuted_groups: tuple[tuple[str, ...], ...] = field(
        factory=tuple, converter=tuple
    )
    """Groups of parameter names that are co-permuted like the other parameters."""

    @copermuted_groups.validator
    def _validate_copermuted_groups(  # noqa: DOC101, DOC103
        self, _: Any, groups: tuple[tuple[str, ...], ...]
    ) -> None:
        """Validate the copermuted groups.

        Raises:
            ValueError: If any of the copermuted groups don't have the same length as
                the primary group.
            ValueError: If any of the copermuted groups contain duplicate parameters.
            ValueError: If any parameter name appears in multiple permutation groups.
        """
        for k, group in enumerate(groups):
            # Ensure all groups have the same length as the primary group
            if len(group) != len(self.parameter_names):
                raise ValueError(
                    f"In the {self.__class__.__name__}, all copermuted groups must "
                    f"have the same length as the primary parameter group "
                    f"({len(self.parameter_names)} in this case). But group {k + 1} "
                    f"has {len(group)} entries: {group}."
                )

            # Ensure parameter names in a group are unique
            if len(set(group)) != len(group):
                raise ValueError(
                    f"In the {self.__class__.__name__}, all parameter names being "
                    f"permuted with each other must be unique. However, the "
                    f"following group contains duplicates: {group}."
                )

        # Ensure there is no overlap between any permutation group
        for a, b in combinations((self.parameter_names, *groups), 2):
            if overlap := set(a) & set(b):
                raise ValueError(
                    f"In the {self.__class__.__name__}, parameter names cannot appear "
                    f"in multiple permutation groups. However, the following parameter "
                    f"names appear in several groups {overlap}."
                )

    @override
    def augment_measurements(
        self, df: pd.DataFrame, _: Iterable[Parameter]
    ) -> pd.DataFrame:
        # See base class.

        if not self.use_data_augmentation:
            return df

        # The input could look like:
        #  - params = ["p_1", "p_2", ...]
        #  - copermuted_groups = [["a_1", "a_2", ...], ["b_1", "b_2", ...]]
        # indicating that the groups "a_k" and "b_k" should be permuted in the same way
        # as the group "p_k".
        # We create `groups` to look like (("p1", "a1", "b1"), ("p2", "a2", "b2"), ...).
        # It results in just (("p1",), ("p2",), ...) if there are no copermuted groups.
        groups = tuple(zip(self.parameter_names, *self.copermuted_groups, strict=True))
        df = df_apply_permutation_augmentation(df, groups)

        return df

    @override
    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """See base class.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            ValueError: If any of the copermuted groups contain parameters not present
                in the searchspace.
            TypeError: If parameters withing a permutation group do not have the same
                type.
            ValueError: If parameters withing a permutation group do not have a
                compatible set of values.
        """
        super().validate_searchspace_context(searchspace)

        # Ensure all copermuted parameters are in the searchspace
        for group in self.copermuted_groups:
            parameters_missing = set(group).difference(searchspace.parameter_names)
            if parameters_missing:
                raise ValueError(
                    f"The symmetry of type {self.__class__.__name__} was set up with "
                    f"at least one parameter which is not present in the searchspace: "
                    f"{parameters_missing}."
                )

        # Ensure permuted parameters all have the same specification.
        # Without this, it could be attempted to read in data that is not allowed for
        # parameters that only allow a subset or different values compared to
        # parameters they are being permuted with.
        for group in (self.parameter_names, *self.copermuted_groups):
            params = searchspace.get_parameters_by_name(group)

            # All parameters in a group must be of the same type
            if len(types := {type(p).__name__ for p in params}) != 1:
                raise TypeError(
                    f"In the {self.__class__.__name__}, all parameters being "
                    f"permuted with each other must have the same type. However, the "
                    f"following multiple types were found in the permutation group "
                    f"{group}: {types}."
                )

            # ALl parameters in a group must have the same values. Numerical parameters
            # are not considered here since technically for them this restriction is not
            # required as al numbers can be added if the tolerance is configured
            # accordingly.
            if all(p.is_discrete and not p.is_numerical for p in params):
                ref_vals = set(params[0].values)
                if any(set(p.values) != ref_vals for p in params):
                    raise ValueError(
                        f"The parameter group '{group}' contains parameters which have "
                        f"different values. All parameters in a group must have the "
                        f"same specification."
                    )


@define(frozen=True)
class MirrorSymmetry(Symmetry):
    """Class for representing mirror symmetries.

    A mirror symmetry expresses that certain parameters can be inflected at a mirror
    point without affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(-x,y)$ (mirror point is 0).
    """

    _parameter_name: str = field(validator=instance_of(str), alias="parameter_name")
    """The name of the single parameter affected by the symmetry."""

    # object variables
    mirror_point: float = field(
        default=0.0, converter=float, validator=validate_is_finite, kw_only=True
    )
    """The mirror point."""

    @override
    @property
    def parameter_names(self) -> tuple[str]:
        return (self._parameter_name,)

    @override
    def augment_measurements(
        self, df: pd.DataFrame, _: Iterable[Parameter]
    ) -> pd.DataFrame:
        # See base class.

        if not self.use_data_augmentation:
            return df

        df = df_apply_mirror_augmentation(
            df, self._parameter_name, mirror_point=self.mirror_point
        )

        return df

    @override
    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """See base class.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            TypeError: If the affected parameter is not numerical.
        """
        super().validate_searchspace_context(searchspace)

        param = searchspace.get_parameters_by_name(self.parameter_names)[0]
        if not param.is_numerical:
            raise TypeError(
                f"In the {self.__class__.__name__}, the affected parameter must be "
                f"numerical. However, the parameter '{param.name}' is of type "
                f"{param.__class__.__name__} and is not numerical."
            )


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
        converter=Converter(normalize_str_sequence, takes_self=True, takes_field=True),  # type: ignore
        validator=(  # type: ignore
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
        self, df: pd.DataFrame, parameters: Iterable[Parameter]
    ) -> pd.DataFrame:
        # See base class.
        if not self.use_data_augmentation:
            return df

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
                vals = p.values
            else:
                # Use linear subsample of parameter bounds interval for augmentation.
                # Note: The original value will not necessarily be part of this.
                vals = tuple(
                    np.linspace(
                        p.bounds.lower, p.bounds.upper, self.n_discretization_points
                    )
                )
            affected.append((p.name, vals))

        df = df_apply_dependency_augmentation(df, causing, affected)

        return df

    @override
    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """See base class.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            ValueError: If any of the affected parameters is not present in the
                searchspace.
            TypeError: If the causing parameter is not discrete.
        """
        super().validate_searchspace_context(searchspace)

        # Affected parameters must be in the searchspace
        parameters_missing = set(self.affected_parameter_names).difference(
            searchspace.parameter_names
        )
        if parameters_missing:
            raise ValueError(
                f"The symmetry of type {self.__class__.__name__} was set up with at "
                f"least one parameter which is not present in the searchspace: "
                f"{parameters_missing}."
            )

        # Causing parameter must be discrete
        param = searchspace.get_parameters_by_name(self._parameter_name)[0]
        if not param.is_discrete:
            raise TypeError(
                f"In the {self.__class__.__name__}, the causing parameter must be "
                f"discrete. However, the parameter '{param.name}' is of type "
                f"'{param.__class__.__name__}' and is not discrete."
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
