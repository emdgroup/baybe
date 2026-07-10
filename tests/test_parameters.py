"""Tests for parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.settings import active_settings

if CHEM_INSTALLED:
    from baybe.parameters.substance import SubstanceParameter


def test_transform_numerical():
    """NumericalDiscreteParameter encoding is the identity mapping of its values."""
    p = NumericalDiscreteParameter(name="x", values=[1.0, 2.0, 3.0])
    expected = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.Index([1.0, 2.0, 3.0]),
        dtype=active_settings.DTypeFloatNumpy,
    )
    assert_frame_equal(p.transform(), expected)


def test_transform_categorical_ohe():
    """CategoricalParameter OHE encoding is an identity matrix, one column per value."""
    p = CategoricalParameter(
        name="color", values=["red", "green", "blue"], active_values=["red"]
    )
    # Values get sorted: blue, green, red
    expected = pd.DataFrame(
        np.eye(3, dtype=active_settings.DTypeFloatNumpy),
        columns=["color_blue", "color_green", "color_red"],
        index=pd.Index(["blue", "green", "red"]),
    )
    assert_frame_equal(p.transform(), expected)


def test_transform_categorical_int():
    """CategoricalParameter INT encoding assigns consecutive integers to sorted values."""  # noqa: E501
    p = CategoricalParameter(
        name="size", values=["S", "M", "L"], encoding="INT", active_values=["S"]
    )
    # Values get sorted: L, M, S
    expected = pd.DataFrame(
        {"size": [0.0, 1.0, 2.0]},
        index=pd.Index(["L", "M", "S"]),
        dtype=active_settings.DTypeFloatNumpy,
    )
    assert_frame_equal(p.transform(), expected)


def test_transform_categorical_ohe_bool():
    """CategoricalParameter OHE with Boolean values uses 'b' prefix in column names."""
    p = CategoricalParameter(name="flag", values=[True, False], active_values=[True])
    # Values get sorted by (type_str, value): False before True
    expected = pd.DataFrame(
        np.eye(2, dtype=active_settings.DTypeFloatNumpy),
        columns=["flag_bFalse", "flag_bTrue"],
        index=pd.Index([False, True]),
    )
    assert_frame_equal(p.transform(), expected)


def test_transform_custom():
    """CustomDiscreteParameter encoding prefixes columns and preserves values."""
    p = CustomDiscreteParameter(
        name="mol",
        data=pd.DataFrame(
            {"d1": [1.0, 2.0, 3.0], "d2": [4.0, 5.0, 6.0]},
            index=["A", "B", "C"],
        ),
        decorrelate=False,
        active_values=["A"],
    )
    expected = pd.DataFrame(
        {"mol_d1": [1.0, 2.0, 3.0], "mol_d2": [4.0, 5.0, 6.0]},
        index=pd.Index(["A", "B", "C"]),
        dtype=active_settings.DTypeFloatNumpy,
    )
    assert_frame_equal(p.transform(), expected)


def test_decorrelation_custom():
    """CustomDiscreteParameter encoding drops correlated columns when decorrelate=True."""  # noqa: E501
    data = pd.DataFrame(
        # d2 = 2*d1 (perfectly correlated with d1), d3 independent
        {"d1": [1.0, 2.0, 3.0], "d2": [2.0, 4.0, 6.0], "d3": [1.0, 4.0, 2.0]},
        index=["A", "B", "C"],
    )
    p = CustomDiscreteParameter(name="mol", data=data, decorrelate=True)
    assert tuple(p.transform().columns) == ("mol_d1", "mol_d3")


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
def test_transform_substance():
    """SubstanceParameter encoding has one row per substance, indexed by label."""
    data = {"water": "O", "ethanol": "CCO", "methanol": "CO"}
    p = SubstanceParameter(
        name="solvent", data=data, decorrelate=False, active_values=["water"]
    )

    comp = p.transform()
    assert list(comp.index) == list(p.values)
    assert all(col.startswith("solvent_") for col in comp.columns)


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
def test_decorrelation_substance():
    """SubstanceParameter drops correlated columns when decorrelate=True."""
    data = {"water": "O", "ethanol": "CCO", "methanol": "CO"}
    p_raw = SubstanceParameter(name="solvent", data=data, decorrelate=False)
    p_decorr = SubstanceParameter(name="solvent", data=data, decorrelate=True)
    assert len(p_decorr.transform().columns) < len(p_raw.transform().columns)
