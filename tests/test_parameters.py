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


@pytest.mark.parametrize(
    "index_select",
    [
        pytest.param(None, id="no_input"),
        pytest.param(slice(None), id="full_input"),
        pytest.param([-1, 0], id="partial_input"),
    ],
)
@pytest.mark.parametrize(
    ("param", "expected"),
    [
        pytest.param(
            NumericalDiscreteParameter(name="x", values=[1.0, 2.0, 3.0]),
            pd.DataFrame(
                {"x": [1.0, 2.0, 3.0]},
                index=pd.Index([1.0, 2.0, 3.0]),
                dtype=active_settings.DTypeFloatNumpy,
            ),
            id="numerical",
        ),
        pytest.param(
            # Values get sorted: blue, green, red
            CategoricalParameter(
                name="color", values=["red", "green", "blue"], active_values=["red"]
            ),
            pd.DataFrame(
                np.eye(3, dtype=active_settings.DTypeFloatNumpy),
                columns=["color_blue", "color_green", "color_red"],
                index=pd.Index(["blue", "green", "red"]),
            ),
            id="categorical_ohe",
        ),
        pytest.param(
            # Values get sorted: L, M, S
            CategoricalParameter(
                name="size", values=["S", "M", "L"], encoding="INT", active_values=["S"]
            ),
            pd.DataFrame(
                {"size": [0.0, 1.0, 2.0]},
                index=pd.Index(["L", "M", "S"]),
                dtype=active_settings.DTypeFloatNumpy,
            ),
            id="categorical_int",
        ),
        pytest.param(
            # Values get sorted by (type_str, value): False before True
            CategoricalParameter(
                name="flag", values=[True, False], active_values=[True]
            ),
            pd.DataFrame(
                np.eye(2, dtype=active_settings.DTypeFloatNumpy),
                columns=["flag_bFalse", "flag_bTrue"],
                index=pd.Index([False, True]),
            ),
            id="categorical_ohe_bool",
        ),
        pytest.param(
            CustomDiscreteParameter(
                name="mol",
                data=pd.DataFrame(
                    {"d1": [1.0, 2.0, 3.0], "d2": [4.0, 5.0, 6.0]},
                    index=["A", "B", "C"],
                ),
                decorrelate=False,
                active_values=["A"],
            ),
            pd.DataFrame(
                {"mol_d1": [1.0, 2.0, 3.0], "mol_d2": [4.0, 5.0, 6.0]},
                index=pd.Index(["A", "B", "C"]),
                dtype=active_settings.DTypeFloatNumpy,
            ),
            id="custom",
        ),
    ],
)
def test_transform(param, expected, index_select):
    """Parameter encodings return the correct rows with the correct index."""
    if index_select is None:
        assert_frame_equal(param.transform(), expected)
    else:
        labels = list(expected.index[index_select])
        series = pd.Series(labels, index=range(10, 10 + len(labels)), name=param.name)
        assert_frame_equal(
            param.transform(series),
            expected.reindex(labels).set_index(series.index),
        )


@pytest.mark.skipif(
    not CHEM_INSTALLED, reason="Optional chem dependency not installed."
)
@pytest.mark.parametrize(
    "subset",
    [
        pytest.param(None, id="no_input"),
        pytest.param(["ethanol", "water", "methanol"], id="full_input"),
        pytest.param(["methanol", "water"], id="partial_input"),
    ],
)
def test_transform_substance(subset):
    """SubstanceParameter encoding returns the correct rows with the correct index."""
    data = {"water": "O", "ethanol": "CCO", "methanol": "CO"}
    p = SubstanceParameter(
        name="solvent", data=data, decorrelate=False, active_values=["water"]
    )
    full = p.transform()
    if subset is None:
        assert list(full.index) == list(p.values)
        assert all(col.startswith("solvent_") for col in full.columns)
    else:
        series = pd.Series(subset, index=range(10, 10 + len(subset)), name=p.name)
        assert_frame_equal(
            p.transform(series),
            full.reindex(subset).set_index(series.index),
        )


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
def test_decorrelation_substance():
    """SubstanceParameter drops correlated columns when decorrelate=True."""
    data = {"water": "O", "ethanol": "CCO", "methanol": "CO"}
    p_raw = SubstanceParameter(name="solvent", data=data, decorrelate=False)
    p_decorr = SubstanceParameter(name="solvent", data=data, decorrelate=True)
    assert len(p_decorr.transform().columns) < len(p_raw.transform().columns)
