"""Functionality for expressing symmetries of the modeling process."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.constraints.conditions import Condition
from baybe.serialization import SerialMixin
from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_permutation_augmentation,
)

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter


@define(frozen=True)
class Symmetry(SerialMixin, ABC):
    """Abstract base class for symmetries.

    Symmetry is a concept that can be used to configure the modelling process in the
    presence of invariances.
    """

    # Object variables
    parameters: list[str] = field(validator=min_len(1))
    """The list of parameters used for the constraint."""

    use_data_augmentation: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """Flag indicating whether data augmentation would be used with surrogates that
    support this."""

    @parameters.validator
    def _validate_params(  # noqa: DOC101, DOC103
        self, _: Any, params: list[str]
    ) -> None:
        """Validate the parameter list.

        Raises:
            ValueError: If ``params`` contains duplicate values.
        """
        if len(params) != len(set(params)):
            raise ValueError(
                f"The given 'parameters' list must have unique values "
                f"but was: {params}."
            )

    def summary(self) -> dict:
        """Return a custom summarization of the symmetry."""
        symmetry_dict = dict(
            Type=self.__class__.__name__, Affected_Parameters=self.parameters
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

    # TODO: Validation
    #  - Each entry in copermuted_groups must have the same length as parameters

    # Object variables
    copermuted_groups: tuple[tuple[str, ...], ...] = field(factory=tuple)
    """The list of parameters used for the constraint."""

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
        groups = tuple(zip(self.parameters, *self.copermuted_groups, strict=True))

        df = df_apply_permutation_augmentation(df, groups)

        return df


@define(frozen=True)
class MirrorSymmetry(Symmetry):
    """Class for representing mirror symmetries.

    A mirror symmetry expresses that certain parameters can be negated without
    affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(-x,y)$.
    """

    # TODO: Validation
    #  - Only numerical parameters are supported
    @override
    def augment_measurements(
        self, df: pd.DataFrame, _: Iterable[Parameter]
    ) -> pd.DataFrame:
        # See base class.

        raise NotImplementedError(
            "Augmentation for mirror symmetry is not yet implemented."
        )


@define(frozen=True)
class DependencySymmetry(Symmetry):
    """Class for representing dependency symmetries.

    A dependency symmetry expresses that certain parameters are dependent on another
    parameter having a specific value. For instance, the situation "The value of
    parameter y only matters if parameter x has the value 'on'.".
    """

    # TODO: Validation
    #  - Only discrete parameters are supported here due to the conditions
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
            self.parameters, self.conditions, self.affected_parameters
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
