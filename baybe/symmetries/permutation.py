"""Permutation symmetry."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.symmetries.base import Symmetry
from baybe.utils.augmentation import df_apply_permutation_augmentation

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace


def _convert_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Convert nested sequences to a tuple of tuples, blocking bare strings."""
    if isinstance(groups, str):
        raise ValueError(
            "The 'permutation_groups' argument must be a sequence of sequences, "
            "not a string."
        )
    converted = []
    for g in groups:
        if isinstance(g, str):
            raise ValueError(
                "Each element in 'permutation_groups' must be a sequence of "
                "parameter names, not a string."
            )
        converted.append(tuple(g))
    return tuple(converted)


@define(frozen=True)
class PermutationSymmetry(Symmetry):
    """Class for representing permutation symmetries.

    A permutation symmetry expresses that certain parameters can be permuted without
    affecting the outcome of the model. For instance, this is the case if
    $f(x,y) = f(y,x)$.
    """

    permutation_groups: tuple[tuple[str, ...], ...] = field(
        converter=_convert_groups,
        validator=(
            min_len(1),
            deep_iterable(
                member_validator=deep_iterable(
                    member_validator=instance_of(str),
                    iterable_validator=min_len(2),
                ),
            ),
        ),
    )
    """The permutation groups. Each group contains parameter names that are permuted
    in lockstep. All groups must have the same length."""

    @permutation_groups.validator
    def _validate_permutation_groups(  # noqa: DOC101, DOC103
        self, _: Any, groups: tuple[tuple[str, ...], ...]
    ) -> None:
        """Validate the permutation groups.

        Raises:
            ValueError: If the groups have different lengths.
            ValueError: If any group contains duplicate parameters.
            ValueError: If any parameter name appears in multiple groups.
        """
        # Ensure all groups have the same length
        if len({len(g) for g in groups}) != 1:
            lengths = {k + 1: len(g) for k, g in enumerate(groups)}
            raise ValueError(
                f"In a '{self.__class__.__name__}', all permutation groups "
                f"must have the same length. Got group lengths: {lengths}."
            )

        # Ensure parameter names in each group are unique
        for group in groups:
            if len(set(group)) != len(group):
                raise ValueError(
                    f"In a '{self.__class__.__name__}', all parameters being "
                    f"permuted with each other must be unique. However, the "
                    f"following group contains duplicates: {group}."
                )

        # Ensure there is no overlap between any permutation group
        for a, b in combinations(groups, 2):
            if overlap := set(a) & set(b):
                raise ValueError(
                    f"In a '{self.__class__.__name__}', parameter names cannot "
                    f"appear in multiple permutation groups. However, the "
                    f"following parameter names appear in several groups: "
                    f"{overlap}."
                )

    @override
    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for group in self.permutation_groups for name in group)

    @override
    def augment_measurements(
        self,
        measurements: pd.DataFrame,
        parameters: Iterable[Parameter] | None = None,
    ) -> pd.DataFrame:
        # See base class.

        if not self.use_data_augmentation:
            return measurements

        measurements = df_apply_permutation_augmentation(
            measurements, self.permutation_groups
        )

        return measurements

    @override
    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """See base class.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            TypeError: If parameters within a permutation group do not have the
                same type.
            ValueError: If parameters within a permutation group do not have a
                compatible set of values.
        """
        super().validate_searchspace_context(searchspace)

        # Ensure permuted parameters all have the same specification.
        # Without this, it could be attempted to read in data that is not allowed
        # for parameters that only allow a subset or different values compared to
        # parameters they are being permuted with.
        for group in self.permutation_groups:
            params = searchspace.get_parameters_by_name(group)

            # All parameters in a group must be of the same type
            if len(types := {type(p).__name__ for p in params}) != 1:
                raise TypeError(
                    f"In a '{self.__class__.__name__}', all parameters being "
                    f"permuted with each other must have the same type. "
                    f"However, the following multiple types were found in the "
                    f"permutation group {group}: {types}."
                )

            # All parameters in a group must have the same values. Numerical
            # parameters are not considered here since technically for them
            # this restriction is not required as all numbers can be added if
            # the tolerance is configured accordingly.
            if all(p.is_discrete and not p.is_numerical for p in params):
                from baybe.parameters.base import DiscreteParameter

                ref_vals = set(cast(DiscreteParameter, params[0]).values)
                if any(
                    set(cast(DiscreteParameter, p).values) != ref_vals
                    for p in params[1:]
                ):
                    raise ValueError(
                        f"The parameter group {group} contains "
                        f"parameters with different values. All "
                        f"parameters in a group must have the same "
                        f"specification."
                    )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
