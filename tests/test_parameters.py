"""Tests for parameters."""

from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals.stable.v2.dependencies import is_into_series
from narwhals.stable.v2.typing import IntoFrame

from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.parameters.sequence import SequenceEncoderProtocol, SequenceParameter
from baybe.settings import active_settings


class _SequenceEncoder(SequenceEncoderProtocol):
    def encode(
        self,
        values: nw.Series,
        _alphabet: frozenset[str],
        *,
        key: str,
        name: str,
    ) -> nw.DataFrame:
        """Passthrough encoder: returns a two-column frame with key and name columns."""
        return values.rename(key).to_frame().with_columns(values.rename(name))


if CHEM_INSTALLED:
    from baybe.parameters.substance import SubstanceParameter


def _to_pd(frame: IntoFrame) -> pd.DataFrame:
    """Normalize any native frame to pandas."""
    return nw.from_native(frame).lazy().collect(backend=pd).to_native()


def assert_frame_equal(left: IntoFrame, right: IntoFrame, **kwargs) -> None:
    """Cross-backend frame equality assertion via pandas normalization."""
    pd.testing.assert_frame_equal(
        _to_pd(left).reset_index(drop=True),
        _to_pd(right).reset_index(drop=True),
        **kwargs,
    )


def _pd_series(name, values):
    return pd.Series(values, name=name)


def _pl_series(name, values):
    return pl.Series(name=name, values=values)


def _nw_series(name, values):
    return nw.new_series(name=name, values=values, backend=pl)


def _list(name, values):
    return list(values)


@pytest.mark.parametrize(
    ("index_select", "series_factory"),
    [
        pytest.param(None, None, id="no_input"),
        pytest.param([-1, 0], _pd_series, id="partial_input-pd"),
        pytest.param([-1, 0], _pl_series, id="partial_input-pl"),
        pytest.param([-1, 0], _nw_series, id="partial_input-nw"),
        pytest.param([-1, 0], _list, id="partial_input-list"),
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
        pytest.param(
            SequenceParameter(
                name="seq",
                alphabet={"A", "C"},
                encoder=_SequenceEncoder(),
                min_length=1,
                max_length=1,
            ),
            pd.DataFrame({"seq": ["A", "C"]}, index=pd.Index(["A", "C"])),
            id="sequence",
        ),
        pytest.param(
            SubstanceParameter(
                name="solvent",
                data={"water": "O", "ethanol": "CCO", "methanol": "CO"},
                decorrelate=False,
                active_values=["water"],
            )
            if CHEM_INSTALLED
            else None,
            None,  # expected computed at runtime from transform()
            id="substance",
            marks=pytest.mark.skipif(
                not CHEM_INSTALLED, reason="Optional chem dependency not installed."
            ),
        ),
    ],
)
def test_transform(param, expected, index_select, series_factory):
    """Parameter encodings return the correct rows, index and backend."""
    if expected is None:
        assert isinstance(param, SubstanceParameter)
        expected = nw.from_native(param.transform(), eager_only=True).to_pandas()
        expected.index = pd.Index(param.values)
        assert all(col.startswith(f"{param.name}_") for col in expected.columns)

    if index_select is None:
        result = param.transform()
        assert_frame_equal(result, expected)
    else:
        labels = list(expected.index[index_select])
        positions = expected.index.get_indexer(labels).tolist()

        series = series_factory(param.name, labels)
        result = param.transform(series)

        if is_into_series(series):
            assert nw.get_native_namespace(result) is nw.get_native_namespace(series)
            with pytest.raises(ValueError, match="does not match parameter name"):
                param.transform(series_factory("wrong name", labels))

        assert_frame_equal(result, nw.from_native(expected)[positions])


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
