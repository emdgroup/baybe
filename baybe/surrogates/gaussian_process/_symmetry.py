"""Symmetry-invariant kernel construction for Gaussian process surrogates."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Collection, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import _ParameterKind
from baybe.surrogates.gaussian_process._override import iter_gpytorch_kernel_tree
from baybe.symmetries.base import Symmetry
from baybe.symmetries.dependency import DependencySymmetry
from baybe.symmetries.mirror import MirrorSymmetry
from baybe.symmetries.permutation import PermutationSymmetry

if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor
    from torch.nn import Module

    from baybe.searchspace import SearchSpace

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


def validate_searchspace_context(
    symmetries: Collection[Symmetry], searchspace: SearchSpace, /
) -> None:
    """Validate that symmetries can be enforced via kernel construction in a space.

    Args:
        symmetries: The symmetries to validate.
        searchspace: The search space the kernel operates on.

    Raises:
        IncompatibleSearchSpaceError: If a symmetry involves a parameter that is not a
            regular parameter, e.g. a task parameter.
    """
    for s in symmetries:
        s.validate_searchspace_context(searchspace)
        if irregular := [
            p.name
            for p in searchspace.get_parameters_by_name(s.parameter_names)
            if p._kind is not _ParameterKind.REGULAR
        ]:
            raise IncompatibleSearchSpaceError(
                f"Symmetries enforced via kernel construction can only involve regular "
                f"parameters, but the '{s.__class__.__name__}' involves the special "
                f"parameters {irregular}, whose kernels have a dedicated purpose."
            )


def validate_mean(mean: GPyTorchMean, /) -> None:
    """Validate that a mean function is compatible with symmetric kernels.

    The posterior of a Gaussian process is only invariant if its prior mean is
    invariant, which is guaranteed for constant means and task-wise constant means.

    Args:
        mean: The mean function to validate.

    Raises:
        ValueError: If the mean function is not (task-wise) constant.
    """
    from gpytorch.means import ConstantMean, ZeroMean

    from baybe.surrogates.gaussian_process.components._gpytorch import (
        HadamardConstantMean,
    )

    means = (
        mean.multitask_mean.base_means if type(mean) is HadamardConstantMean else [mean]
    )
    if not all(type(m) in (ConstantMean, ZeroMean) for m in means):
        raise ValueError(
            f"Symmetries enforced via kernel construction require a constant mean "
            f"function, since otherwise the model predictions are not invariant. "
            f"However, a mean function of type '{type(mean).__name__}' was given."
        )


_PER_DIMENSION_PARAMETERS = ("raw_lengthscale", "raw_period_length")
"""Names of GPyTorch kernel parameters holding one entry per input dimension."""


def make_symmetric_kernel(
    symmetries: Collection[Symmetry],
    searchspace: SearchSpace,
    input_transform: InputTransform,
    kernel: GPyTorchKernel,
) -> GPyTorchKernel:
    """Make a kernel invariant under the given symmetries.

    The kernel is wrapped in three stages:

    1. Dependencies: Inputs are only correlated if they share the same pattern of
       active and inactive dependencies, in which case the affected parameters of the
       inactive dependencies are fixed to a constant.
    2. Permutations: The hyperparameters of permuted positions are shared and the
       kernel is summed over all permutations of its second input.
    3. Mirrors: The kernel is summed over all reflections of both inputs.

    Args:
        symmetries: The symmetries to enforce, validated against the search space.
        searchspace: The search space the kernel operates on.
        input_transform: The transform normalizing the kernel inputs.
        kernel: The kernel to be made invariant. Its hyperparameters of permuted
            positions are shared in place.

    Returns:
        The symmetric kernel.
    """
    from baybe.surrogates.gaussian_process.components._gpytorch import (
        DependencyGatedKernel,
        MirrorInvariantKernel,
        PermutationInvariantKernel,
    )

    dependencies = [s for s in symmetries if isinstance(s, DependencySymmetry)]
    permutations = [s for s in symmetries if isinstance(s, PermutationSymmetry)]
    mirrors = [s for s in symmetries if isinstance(s, MirrorSymmetry)]
    n_columns = len(searchspace.comp_rep_columns)
    slots = [_get_permutation_slots(s, searchspace) for s in permutations]
    index_maps = [_make_permutation_index_maps(s, n_columns) for s in slots]

    _tie_permuted_hyperparameters(kernel, slots, n_columns)
    for symmetry, maps in zip(permutations, index_maps):
        _validate_permutation_invariance(kernel, maps, n_columns, symmetry)

    if dependencies:
        gates = [
            _make_dependency_gate(s, searchspace, input_transform) for s in dependencies
        ]
        kernel = DependencyGatedKernel(kernel, *zip(*gates))
    for maps in index_maps:
        kernel = PermutationInvariantKernel(kernel, maps)
    for mirror in mirrors:
        column, center = _get_mirror_column_and_center(
            mirror, searchspace, input_transform
        )
        kernel = MirrorInvariantKernel(kernel, column, center)
    return kernel


def _get_permutation_slots(
    symmetry: PermutationSymmetry, searchspace: SearchSpace, /
) -> list[list[tuple[int, ...]]]:
    """Get the computational columns of each slot of each group of a permutation.

    Args:
        symmetry: The permutation symmetry.
        searchspace: The search space containing the permuted parameters.

    Returns:
        The column indices, indexed by group and slot.

    Example:
        For the groups ``[["s1", "s2"], ["f1", "f2"]]``, where ``s1`` and ``s2`` are
        one-hot encoded with three columns each, the result is
        ``[[(0, 1, 2), (3, 4, 5)], [(6,), (7,)]]``.
    """
    slots = [
        [tuple(searchspace.get_comp_rep_parameter_indices(n)) for n in group]
        for group in symmetry.permutation_groups
    ]
    # Equivalent parameters have identical representations and normalization bounds
    bounds = searchspace.scaling_bounds.to_numpy()
    assert all(
        len(columns) == len(group[0])
        and (bounds[:, list(columns)] == bounds[:, list(group[0])]).all()
        for group in slots
        for columns in group
    )
    return slots


def _make_permutation_index_maps(
    slots: Sequence[Sequence[tuple[int, ...]]], n_columns: int, /
) -> list[list[int]]:
    """Make one column index map per permutation of the slots.

    Args:
        slots: The column indices, indexed by group and slot.
        n_columns: The total number of computational columns.

    Returns:
        For each permutation, the input column used for each column.

    Example:
        For the slots ``[[(0,), (1,)]]`` and three columns, the maps are ``[0, 1, 2]``
        (identity) and ``[1, 0, 2]`` (swap), i.e. column 0 reads column 1 and vice
        versa, while column 2 stays in place.
    """
    maps = []
    for permutation in itertools.permutations(range(len(slots[0]))):
        index_map = list(range(n_columns))
        for group in slots:
            for slot, source in enumerate(permutation):
                for target_column, source_column in zip(group[slot], group[source]):
                    index_map[target_column] = source_column
        maps.append(index_map)
    return maps


def _share_parameters(source: Module, target: Module, /) -> None:
    """Let a module use the parameters of a structurally identical module.

    Afterward, both modules hold the very same parameter objects, so that their values
    are identical at all times, including during model fitting. Modules that differ
    in structure (submodule types or parameter shapes) are left unchanged.

    Example:
        If two slots are each modeled by an ``RBFKernel``, the kernel of the second
        slot afterward uses the ``raw_lengthscale`` tensor of the kernel of the first
        slot, so that fitting updates both together.

    Args:
        source: The module whose parameters are shared.
        target: The module adopting the parameters of the source module.
    """
    if [(n, type(m)) for n, m in source.named_modules()] != [
        (n, type(m)) for n, m in target.named_modules()
    ] or [(n, p.shape) for n, p in source.named_parameters()] != [
        (n, p.shape) for n, p in target.named_parameters()
    ]:
        return
    for name, parameter in list(source.named_parameters()):
        path, _, attribute = name.rpartition(".")
        setattr(target.get_submodule(path), attribute, parameter)


def _tie_permuted_hyperparameters(
    kernel: GPyTorchKernel,
    slots: Sequence[Sequence[Sequence[tuple[int, ...]]]],
    n_columns: int,
) -> None:
    """Share the hyperparameters of permuted positions within a kernel (in place).

    Kernels acting on a single permuted parameter use the parameters of the
    structurally identical kernel acting on the corresponding first-slot parameter.
    Kernels acting on all positions of permuted parameters jointly (e.g. an ARD kernel)
    tie their per-dimension parameters entry-wise. Kernels for which neither applies
    are left unchanged.

    Example:
        For the default ARD Matérn kernel over ``x1``, ``x2`` and ``z`` with ``x1`` and
        ``x2`` permuted, the lengthscales become ``[l, l, l_z]`` with a single learned
        ``l``. For equivalent kernel overrides of ``x1`` and ``x2``, the kernel of
        ``x2`` uses the parameters of the kernel of ``x1``.

    Args:
        kernel: The kernel to modify.
        slots: For each permutation symmetry, the column indices by group and slot.
        n_columns: The total number of computational columns.
    """
    from torch.nn.utils import parametrize

    from baybe.surrogates.gaussian_process.components._gpytorch import TiedEntries

    kernels = list(iter_gpytorch_kernel_tree(kernel, n_columns))
    groups = [group for symmetry_slots in slots for group in symmetry_slots]

    for group in groups:
        per_slot = [[k for k, c in kernels if c == columns] for columns in group]
        for targets in per_slot[1:]:
            if len(targets) == len(per_slot[0]):
                for source, target in zip(per_slot[0], targets):
                    _share_parameters(source, target)

    # Each column is mapped to the set of columns it is exchanged with
    exchanged = {
        column: frozenset(position)
        for group in groups
        for position in zip(*group)
        for column in position
    }
    for module, columns in kernels:
        keys = [exchanged.get(c, frozenset({c})) for c in columns]
        if len(set(keys)) == len(keys) or not all(k <= set(columns) for k in keys):
            continue
        index = [list(dict.fromkeys(keys)).index(k) for k in keys]
        for name in _PER_DIMENSION_PARAMETERS:
            parameter = module._parameters.get(name)
            if parameter is None or parameter.shape[-1] != len(columns):
                continue
            constraint = module._constraints.get(f"{name}_constraint")
            parametrize.register_parametrization(module, name, TiedEntries(index))
            # GPyTorch looks up constraints by parameter name, which the
            # parametrization changes to "parametrizations.<name>.original"
            getattr(module.parametrizations, name)._constraints = (
                {} if constraint is None else {"original_constraint": constraint}
            )


def _validate_permutation_invariance(
    kernel: GPyTorchKernel,
    index_maps: Sequence[Sequence[int]],
    n_columns: int,
    symmetry: PermutationSymmetry,
) -> None:
    """Validate that a kernel is invariant under jointly permuting both inputs.

    The check is performed numerically on a copy of the kernel with randomized
    hyperparameters, so that it holds regardless of the values found during fitting.

    Args:
        kernel: The kernel to validate.
        index_maps: The column index maps of the permutations.
        n_columns: The total number of computational columns.
        symmetry: The permutation symmetry (for error messages).

    Raises:
        ValueError: If the kernel is not invariant.
    """
    import torch

    probe = deepcopy(kernel)
    generator = torch.Generator().manual_seed(0)
    parameters = list(probe.parameters())
    dtype = parameters[0].dtype if parameters else torch.float64
    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(
                0.5 + torch.rand(parameter.shape, generator=generator, dtype=dtype)
            )
        x1 = torch.rand(6, n_columns, generator=generator, dtype=dtype)
        x2 = torch.rand(5, n_columns, generator=generator, dtype=dtype)
        reference = probe(x1, x2).to_dense()
        for index_map in index_maps:
            if not torch.allclose(
                probe(x1[:, index_map], x2[:, index_map]).to_dense(),
                reference,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    f"The kernel cannot be made invariant under {symmetry}, since the "
                    f"hyperparameters of the permuted parameters cannot be shared. "
                    f"This is the case if the permuted parameters are modeled by "
                    f"kernels of different structure or by a kernel that does not "
                    f"treat them alike. Assign equivalent 'override_kernel's to the "
                    f"permuted parameters or use a kernel (factory) treating them "
                    f"alike."
                )


def _normalize(
    values: Sequence[float] | np.ndarray,
    columns: Sequence[int],
    n_columns: int,
    input_transform: InputTransform,
) -> Tensor:
    """Normalize values of the given computational columns like the model inputs.

    The values are embedded into an otherwise empty full-width input, so that the
    input transform of the model can be reused, and the given columns are extracted
    afterward.

    Example:
        With scaling bounds ``[0, 10]`` for column 2, the raw value ``3`` of that column
        becomes ``0.3``.

    Args:
        values: The values, with one entry per given column (per row).
        columns: The computational columns the values belong to.
        n_columns: The total number of computational columns.
        input_transform: The transform normalizing the model inputs.

    Returns:
        The normalized values, with one column per given column.
    """
    import torch

    values = np.asarray(values, dtype=np.float64).reshape(-1, len(columns))
    full = torch.zeros(len(values), n_columns, dtype=torch.float64)
    full[:, list(columns)] = torch.tensor(values)
    return input_transform.transform(full)[:, list(columns)]


def _get_mirror_column_and_center(
    symmetry: MirrorSymmetry, searchspace: SearchSpace, input_transform: InputTransform
) -> tuple[int, float]:
    """Get the normalized computational column and mirror point of a mirror symmetry.

    Args:
        symmetry: The mirror symmetry.
        searchspace: The search space containing the mirrored parameter.
        input_transform: The transform normalizing the kernel inputs.

    Returns:
        The column index and the normalized mirror point.

    Example:
        For ``MirrorSymmetry("m", mirror_point=3)`` with ``m`` bounded to ``(0, 10)``,
        the result is the column of ``m`` and the normalized mirror point ``0.3``.
    """
    (column,) = searchspace.get_comp_rep_parameter_indices(symmetry.parameter_names[0])
    n_columns = len(searchspace.comp_rep_columns)
    center = _normalize([symmetry.mirror_point], [column], n_columns, input_transform)
    return column, float(center.item())


def _make_dependency_gate(
    symmetry: DependencySymmetry,
    searchspace: SearchSpace,
    input_transform: InputTransform,
) -> tuple[list[int], list[int], Tensor, Tensor]:
    """Make the lookup determining the activity of a dependency from model inputs.

    Args:
        symmetry: The dependency symmetry.
        searchspace: The search space containing the causing parameter.
        input_transform: The transform normalizing the kernel inputs.

    Raises:
        IncompatibleSearchSpaceError: If an active and an inactive value of the causing
            parameter share the same computational representation.

    Returns:
        The columns of the affected parameters, the columns of the causing parameter,
        the normalized representations of its values, and whether each value makes
        the dependency active.

    Example:
        For a one-hot encoded causing parameter ``switch`` with values
        ``("off1", "off2", "on")`` and the condition ``switch == "on"``, the result
        contains the columns of the affected parameters, the columns of ``switch``,
        its three normalized encodings, and the activities ``[False, False, True]``.
    """
    import torch

    from baybe.surrogates.gaussian_process.components._gpytorch import (
        DependencyGatedKernel,
    )

    name = symmetry.parameter_names[0]  # The causing parameter comes first
    (parameter,) = searchspace.get_parameters_by_name((name,))
    assert isinstance(parameter, DiscreteParameter)  # ensured by symmetry validation
    columns = list(searchspace.get_comp_rep_parameter_indices(name))
    representation = parameter.comp_df.loc[
        list(parameter.values), [searchspace.comp_rep_columns[i] for i in columns]
    ]
    encodings = _normalize(
        representation.to_numpy(),
        columns,
        len(searchspace.comp_rep_columns),
        input_transform,
    )
    activities = torch.tensor(
        symmetry.condition.evaluate(pd.Series(parameter.values)).to_numpy(dtype=bool)
    )

    identical = torch.isclose(
        encodings[:, None], encodings[None], rtol=0.0, atol=DependencyGatedKernel.atol
    ).all(-1)
    if (identical & (activities[:, None] != activities[None])).any():
        raise IncompatibleSearchSpaceError(
            f"The causing parameter '{name}' of a '{symmetry.__class__.__name__}' has "
            f"active and inactive values with identical computational representations, "
            f"so that the activity of the dependency cannot be determined by the model."
        )
    affected_columns = [
        i
        for n in symmetry.affected_parameter_names
        for i in searchspace.get_comp_rep_parameter_indices(n)
    ]
    return affected_columns, columns, encodings, activities
