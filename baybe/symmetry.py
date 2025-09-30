"""Functionality for expressing symmetries of the modeling process."""

from __future__ import annotations

import gc
from abc import ABC
from typing import Any

from attrs import define, field
from attrs.validators import instance_of, min_len

from baybe.constraints.conditions import Condition
from baybe.serialization import SerialMixin


@define(frozen=True)
class Symmetry(SerialMixin, ABC):
    """Abstract base class for symmetries.

    Symmetry is a concept that can be used to configure the modelling process in the
    presence of invariances.
    """

    # Object variables
    parameters: list[str] = field(validator=min_len(1))
    """The list of parameters used for the constraint."""

    consider_data_augmentation: bool = field(
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


@define(frozen=True)
class PermutationSymmetry(Symmetry):
    """Class for representing permutation symmetries.

    A permutation symmetry expresses that certain parameters can be permuted without
    affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(y,x)$.
    """


@define(frozen=True)
class MirrorSymmetry(Symmetry):
    """Class for representing mirror symmetries.

    A mirror symmetry expresses that certain parameters can be negated without
    affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(-x,y)$.
    """


@define(frozen=True)
class DependencySymmetry(Symmetry):
    """Class for representing dependency symmetries.

    A dependency symmetry expresses that certain parameters are dependent on another
    parameter having a specific value. For instance, the situation "The value of
    parameter y only matters if parameter x has the value 'on'.".
    """

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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
