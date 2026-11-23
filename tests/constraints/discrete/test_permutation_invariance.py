"""Tests for the discrete permutation invariance constraint."""

import narwhals.stable.v2 as nw
import pytest
from pytest import param

from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.discrete import (
    DiscreteDependenciesConstraint,
    DiscretePermutationInvarianceConstraint,
)
from baybe.symmetries.permutation import PermutationSymmetry
from tests.constraints.discrete.utils import assert_filter, assert_partial_rejected
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

##### API #####


def test_inverted_dependencies_rejected():
    """Attached dependency constraints cannot be inverted."""
    dependencies = DiscreteDependenciesConstraint(
        ["amount"], [ThresholdCondition(0, ">")], [["a"]], exclude=True
    )
    with pytest.raises(ValueError, match="cannot be inverted"):
        DiscretePermutationInvarianceConstraint(["a", "b"], dependencies)


def test_invalid_dependencies_rejected():
    """Attached dependencies must be a dependency constraint."""
    with pytest.raises(TypeError, match="dependencies"):
        DiscretePermutationInvarianceConstraint(["a", "b"], dependencies="invalid")


@pytest.mark.parametrize("with_dependencies", [False, True], ids=["plain", "linked"])
def test_permutation_to_symmetry(with_dependencies):
    """Conversion preserves the primary and linked permutation groups."""
    dependencies = (
        DiscreteDependenciesConstraint(
            ["amount_a", "amount_b"],
            [ThresholdCondition(0, ">"), ThresholdCondition(0, ">")],
            [["a"], ["b"]],
        )
        if with_dependencies
        else None
    )
    constraint = DiscretePermutationInvarianceConstraint(["a", "b"], dependencies)
    groups = (
        [["a", "b"], ["amount_a", "amount_b"]] if with_dependencies else [["a", "b"]]
    )
    assert constraint.to_symmetry() == PermutationSymmetry(groups)


##### Behavior #####


@pytest.mark.parametrize("exclude", [False, True], ids=["keep", "exclude"])
def test_permutation_invariance(exclude, backend):
    """Keep the first equivalent row, respecting unrelated columns and inversion."""
    data = [
        {"a": "x", "b": "y", "group": "one"},
        {"a": "y", "b": "x", "group": "one"},
        {"a": "y", "b": "x", "group": "two"},
        {"a": "x", "b": "x", "group": "one"},
        {"a": "y", "b": "y", "group": "one"},
        {"a": "x", "b": "x", "group": "one"},
    ]
    constraint = DiscretePermutationInvarianceConstraint(["a", "b"], exclude=exclude)
    expected = [data[1], data[5]] if exclude else [data[i] for i in (0, 2, 3, 4)]
    assert_filter(constraint, data, expected, backend)


@pytest.mark.parametrize(
    "data",
    [
        param(
            [
                {"a": float("nan"), "b": "x"},
                {"a": float("nan"), "b": "x"},
            ],
            id="nan-in-permutation-parameter",
        ),
        param(
            [
                {"a": "x", "b": "x", "other": None},
                {"a": "x", "b": "x", "other": None},
            ],
            id="null-in-unrelated-column",
        ),
    ],
)
def test_permutation_invariance_rejects_missing_values(data, backend):
    """Reject NaN or null in permutation and unrelated columns."""
    constraint = DiscretePermutationInvarianceConstraint(["a", "b"])
    frame = nw.from_dicts(data, backend=backend)
    with pytest.raises(ValueError, match="Null and NaN values are not supported"):
        constraint(frame.to_native())


@pytest.mark.parametrize(
    "expected_indices",
    [
        param((0,), id="legacy-set-semantics"),
        param(
            (0, 1),
            id="permutation-semantics",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="Legacy permutation filtering ignores value multiplicities.",
                strict=True,
            ),
        ),
    ],
)
def test_permutation_legacy_repetition_semantics(expected_indices, backend):
    """Capture the legacy and intended behavior for repeated values."""
    data = [
        {"a": "x", "b": "x", "c": "y"},
        {"a": "x", "b": "y", "c": "y"},
    ]
    assert_filter(
        DiscretePermutationInvarianceConstraint(["a", "b", "c"]),
        data,
        [data[i] for i in expected_indices],
        backend,
    )


def test_permutation_invariance_mixed_types(backend):
    """Different numerical and textual values must not be conflated."""
    data = [{"a": 1, "b": "2"}, {"a": 2, "b": "1"}, {"a": 1, "b": "2"}]
    assert_filter(
        DiscretePermutationInvarianceConstraint(["a", "b"]),
        data,
        data[:2],
        backend,
    )


@pytest.mark.parametrize("exclude", [False, True], ids=["keep", "exclude"])
def test_permutation_with_dependencies(exclude, backend):
    """Apply dependency deduplication after permutation, without mutation."""
    data = [
        {"a": "x", "b": "y", "amount_a": 0.0, "amount_b": 1.0},
        {"a": "z", "b": "y", "amount_a": 0.0, "amount_b": 1.0},
        {"a": "x", "b": "y", "amount_a": 1.0, "amount_b": 0.0},
        {"a": "y", "b": "x", "amount_a": 1.0, "amount_b": 0.0},
    ]
    dependencies = DiscreteDependenciesConstraint(
        ["amount_a", "amount_b"],
        [ThresholdCondition(0, ">"), ThresholdCondition(0, ">")],
        [["a"], ["b"]],
    )
    before = dependencies.to_dict()
    constraint = DiscretePermutationInvarianceConstraint(
        ["a", "b"], dependencies, exclude=exclude
    )
    expected = [data[1], data[3]] if exclude else [data[0], data[2]]
    assert_filter(constraint, data, expected, backend)
    assert dependencies.to_dict() == before


@pytest.mark.parametrize("exclude", [False, True], ids=["keep", "exclude"])
@pytest.mark.parametrize(
    "kept_indices",
    [
        param((0,), id="legacy-label-permutation"),
        param(
            (0, 1),
            id="lockstep-permutation",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="Legacy permutation filtering does not permute in lockstep.",
                strict=True,
            ),
        ),
    ],
)
def test_permutation_with_dependencies_lockstep(kept_indices, exclude, backend):
    """Capture legacy and intended permutation behavior for dependent values."""
    data = [
        {"a": "x", "b": "y", "amount_a": 1.0, "amount_b": 0.0},
        {"a": "y", "b": "x", "amount_a": 1.0, "amount_b": 0.0},
        {"a": "y", "b": "x", "amount_a": 0.0, "amount_b": 1.0},
    ]
    dependencies = DiscreteDependenciesConstraint(
        ["amount_a", "amount_b"],
        [ThresholdCondition(0, ">"), ThresholdCondition(0, ">")],
        [["a"], ["b"]],
    )
    constraint = DiscretePermutationInvarianceConstraint(
        ["a", "b"], dependencies, exclude=exclude
    )
    # TODO: Permute dependent parameters in lockstep with their labels.
    expected = [row for i, row in enumerate(data) if (i in kept_indices) != exclude]
    assert_filter(constraint, data, expected, backend)


def test_permutation_partial_accepted(backend):
    """Non-inverted permutations without dependencies allow safe partial filtering."""
    constraint = DiscretePermutationInvarianceConstraint(["a", "b", "c"])
    data = [{"a": "x", "b": "y"}, {"a": "y", "b": "x"}]
    assert_filter(constraint, data, data[:1], backend)


@pytest.mark.parametrize("exclude", [False, True], ids=["keep", "exclude"])
def test_permutation_with_dependencies_partial_rejected(exclude, backend):
    """Partial filtering is unsafe when connected dependencies are incomplete."""
    dependencies = DiscreteDependenciesConstraint(
        ["amount"], [ThresholdCondition(0, ">")], [["a"]]
    )
    constraint = DiscretePermutationInvarianceConstraint(
        ["a", "b"], dependencies, exclude=exclude
    )
    assert_partial_rejected(constraint, [{"a": "x", "b": "y"}], backend)


@pytest.mark.parametrize(
    ("exclude", "data"),
    [
        param(False, [{"a": "x"}], id="too-few"),
        param(True, [{"a": "x", "b": "y"}], id="exclude-partial"),
    ],
)
def test_permutation_partial_rejected(exclude, data, backend):
    """Incomplete data is rejected when no sound partial filtering is available."""
    constraint = DiscretePermutationInvarianceConstraint(
        ["a", "b", "c"], exclude=exclude
    )
    assert_partial_rejected(constraint, data, backend)
