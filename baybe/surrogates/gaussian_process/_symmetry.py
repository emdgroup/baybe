"""Symmetry-invariant kernel construction for Gaussian process surrogates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection

from baybe.symmetries.base import Symmetry
from baybe.symmetries.dependency import DependencySymmetry
from baybe.symmetries.permutation import PermutationSymmetry

MAX_PERMUTATION_GROUP_SIZE = 5
"""The maximum number of positions in a permutation group (cost grows factorially)."""


def _get_controlled_parameter_names(symmetry: Symmetry, /) -> tuple[str, ...]:
    """Get the names of the parameters whose modeling is changed by a symmetry.

    For dependency symmetries, these are only the affected parameters, since the
    causing parameter keeps its regular role and is merely read.

    Args:
        symmetry: The symmetry.

    Returns:
        The names of the controlled parameters.
    """
    if isinstance(symmetry, DependencySymmetry):
        return symmetry.affected_parameter_names
    return symmetry.parameter_names


def validate_symmetries(symmetries: Collection[Symmetry], /) -> None:
    """Validate that symmetries can be jointly enforced via kernel construction.

    Args:
        symmetries: The symmetries to validate.

    Raises:
        ValueError: If a parameter is controlled by more than one symmetry.
        ValueError: If the causing parameter of a dependency is controlled by a
            symmetry.
        ValueError: If a permutation group exceeds the maximum supported size.
    """
    counts = Counter(n for s in symmetries for n in _get_controlled_parameter_names(s))
    if duplicates := sorted(n for n, c in counts.items() if c > 1):
        raise ValueError(
            f"Each parameter can be controlled by at most one symmetry when "
            f"symmetries are enforced via kernel construction. However, the following "
            f"parameters are controlled by several symmetries: {duplicates}."
        )

    for s in symmetries:
        if (
            isinstance(s, DependencySymmetry)
            # The causing parameter comes first
            and (name := s.parameter_names[0]) in counts
        ):
            raise ValueError(
                f"The causing parameter '{name}' of a '{s.__class__.__name__}' cannot "
                f"be controlled by another symmetry when symmetries are enforced via "
                f"kernel construction."
            )
        if (
            isinstance(s, PermutationSymmetry)
            and (n := len(s.permutation_groups[0])) > MAX_PERMUTATION_GROUP_SIZE
        ):
            raise ValueError(
                f"Enforcing a '{s.__class__.__name__}' via kernel construction "
                f"requires summing over all permutations of a group, which is "
                f"supported for at most {MAX_PERMUTATION_GROUP_SIZE} positions, but a "
                f"group with {n} positions was given. Consider using data augmentation "
                f"instead, which is a different modeling approach whose number of "
                f"training points grows by the same factorial factor."
            )
