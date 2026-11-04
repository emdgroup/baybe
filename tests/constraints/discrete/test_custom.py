"""Tests for the discrete custom constraint."""

import narwhals.stable.v2 as nw
import pytest
from pytest import param

from baybe.constraints.discrete import DiscreteCustomConstraint
from tests.constraints.discrete.utils import (
    assert_filter,
    assert_partial_rejected,
    requires_fulltest,
)

pytestmark = requires_fulltest

##### API #####


@pytest.mark.parametrize(
    "validator",
    [
        param(1, id="int"),
        param(None, id="none"),
        param("callable", id="str"),
        param([1], id="list"),
    ],
)
def test_custom_constraint_invalid_validator(validator):
    """Custom constraints require a callable validator."""
    with pytest.raises(TypeError, match="must be callable"):
        DiscreteCustomConstraint(parameters=["temperature"], validator=validator)


##### Behavior #####


_CUSTOM_DATA = [
    {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
    {"temperature": 2.0, "pressure": 1.0, "solvent": "ethanol"},
    {"temperature": 1.0, "pressure": 2.0, "solvent": "methanol"},
    {"temperature": 2.0, "pressure": 2.0, "solvent": "acetone"},
]


@pytest.mark.parametrize(
    ("exclude", "expected_rows"),
    [
        param(
            False,
            [
                {"temperature": 2.0, "pressure": 1.0, "solvent": "ethanol"},
                {"temperature": 1.0, "pressure": 2.0, "solvent": "methanol"},
                {"temperature": 2.0, "pressure": 2.0, "solvent": "acetone"},
            ],
            id="keep",
        ),
        param(
            True,
            [{"temperature": 1.0, "pressure": 1.0, "solvent": "water"}],
            id="exclude",
        ),
    ],
)
def test_custom_constraint(exclude, expected_rows, backend):
    """Custom constraints apply a backend-native validator with optional inversion."""

    def validator(df):
        nw_df = nw.from_native(df, eager_only=True)
        assert nw_df.implementation == backend
        assert set(nw_df.columns) == {"temperature", "pressure"}
        return (nw_df["temperature"] + nw_df["pressure"] >= 3.0).to_native()

    constraint = DiscreteCustomConstraint(
        parameters=["temperature", "pressure"],
        validator=validator,
        exclude=exclude,
    )
    assert_filter(constraint, _CUSTOM_DATA, expected_rows, backend)


def test_custom_constraint_partial_rejected(backend):
    """Custom constraints defer until all parameters are available."""
    constraint = DiscreteCustomConstraint(
        parameters=["temperature", "pressure"],
        validator=lambda df: df,
    )
    assert_partial_rejected(constraint, [{"temperature": 1.0}], backend)
