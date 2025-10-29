"""Functionality for expressing symmetries of the modeling process."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.constraints.conditions import Condition
from baybe.parameters.validation import validate_unique_values
from baybe.serialization import SerialMixin
from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_mirror_augmentation,
    df_apply_permutation_augmentation,
)
from baybe.utils.conversion import normalize_str_sequence

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter


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
    def augment_measurements(self, df: pd.DataFrame, parameters: Iterable[Parameter]):
        """Augment the given measurements according to the symmetry.

        Args:
            df: The dataframe containing the measurements to be augmented.
            parameters: Parameter objects carrying additional information (might not be
                needed by all augmentation implementations).
        """


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

    # TODO: Validation
    #  - Each entry in copermuted_groups must have the same length as parameter_names
    #  - parameters in each group must have identical specification as otherwise
    #    disallowed parameter values could be produced

    # Object variables
    copermuted_groups: tuple[tuple[str, ...], ...] = field(factory=tuple)
    """Groups of parameter names that are co-permuted like the other parameters."""

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


@define(frozen=True)
class MirrorSymmetry(Symmetry):
    """Class for representing mirror symmetries.

    A mirror symmetry expresses that certain parameters can be inflected at a mirror
    point without affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(-x,y)$ (mirror point is 0).
    """

    _parameter_name: str = field(validator=instance_of(str), alias="parameter_name")
    """The anme of the single parameter affected by the symmetry."""

    # object variables
    mirror_point: float = field(default=0.0, converter=float, kw_only=True)
    """The mirror point."""

    @override
    @property
    def parameter_names(self) -> tuple[str]:
        return (self._parameter_name,)

    # TODO: Validation
    #  - Only numerical parameters are supported

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


@define(frozen=True)
class DependencySymmetry(Symmetry):
    """Class for representing dependency symmetries.

    A dependency symmetry expresses that certain parameters are dependent on another
    parameter having a specific value. For instance, the situation "The value of
    parameter y only matters if parameter x has the value 'on'.".
    """

    _parameter_names: tuple[str, ...] = field(
        alias="parameter_names",
        converter=Converter(normalize_str_sequence, takes_self=True, takes_field=True),  # type: ignore
        validator=(  # type: ignore
            deep_iterable(
                member_validator=instance_of(str), iterable_validator=min_len(1)
            ),
        ),
    )
    """The names of the parameters affected by the symmetry."""

    @override
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    # TODO: Validation
    #  - Only discrete parameters are supported as "causing" due to the conditions
    #  - Length and content of conditions and affected_parameters

    # object variables
    conditions: list[Condition] = field()
    """The list of individual conditions."""

    affected_parameters: list[list[str]] = field()
    """The parameters affected by the individual conditions."""

    @affected_parameters.validator
    def _validate_affected_parameters(  # noqa: DOC101, DOC103
        self, _: Any, value: list[list[str]]
    ) -> None:
        """Validate the affected parameters.

        Raises:
            ValueError: If one set of affected parameters does not have exactly one
                condition.
        """
        if len(self.conditions) != len(value):
            raise ValueError(
                f"For the {self.__class__.__name__}, for each item in the "
                f"affected_parameters list you must provide exactly one condition in "
                f"the conditions list."
            )

    @override
    def augment_measurements(
        self, df: pd.DataFrame, parameters: Iterable[Parameter]
    ) -> pd.DataFrame:
        # See base class.
        if not self.use_data_augmentation:
            return df

        from baybe.parameters.base import DiscreteParameter

        for param_name, cond, affected_param_names in zip(
            self.parameter_names, self.conditions, self.affected_parameters
        ):
            # The 'causing' entry describes the parameters and the value
            # for which one or more affected parameters become degenerate.
            # 'cond' specifies for which values the affected parameter
            # values are active, i.e. not degenerate. Hence, here we get the
            # values that are not active, as rows containing them should be
            # augmented.
            param = next(
                cast(DiscreteParameter, p) for p in parameters if p.name == param_name
            )

            causing_values = [
                x
                for x, flag in zip(
                    param.values,
                    ~cond.evaluate(pd.Series(param.values)),
                    strict=True,
                )
                if flag
            ]
            causing = (param.name, causing_values)

            # The 'affected' entry describes the affected parameters and the
            # values they are allowed to take, which are all degenerate if
            # the corresponding condition for the causing parameter is met.
            affected = [
                (
                    (ap := next(p for p in parameters if p.name == pn)).name,
                    ap.values,  # type: ignore[attr-defined]
                )
                for pn in affected_param_names
            ]

            df = df_apply_dependency_augmentation(df, causing, affected)

        return df


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
