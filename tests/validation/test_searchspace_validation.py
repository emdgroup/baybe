"""Validation tests for search spaces."""

import pandas as pd
import pytest
from pytest import param

from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.utils.dataframe import get_transform_objects

parameters = [NumericalDiscreteParameter("d1", [0, 1])]


@pytest.mark.parametrize(
    ("df", "match"),
    [
        param(
            pd.DataFrame(columns=[]),
            r"object\(s\) named \{'d1'\} cannot be matched",
            id="missing",
        ),
        param(
            pd.DataFrame(columns=["d1", "d2"]),
            r"column\(s\) \{'d2'\} cannot be matched",
            id="extra",
        ),
    ],
)
def test_invalid_transforms(df, match):
    """Transforming dataframes with incorrect columns raises an error."""
    with pytest.raises(ValueError, match=match):
        get_transform_objects(df, parameters, allow_missing=False, allow_extra=False)


@pytest.mark.parametrize(
    ("df", "missing", "extra"),
    [
        param(pd.DataFrame(columns=["d1"]), False, False, id="equal"),
        param(pd.DataFrame(columns=[]), True, False, id="missing"),
        param(pd.DataFrame(columns=["d1", "d2"]), False, True, id="extra"),
    ],
)
def test_valid_transforms(df, missing, extra):
    """When providing the appropriate flags, the columns of the dataframe to be transformed can be flexibly chosen."""  # noqa
    get_transform_objects(df, parameters, allow_missing=missing, allow_extra=extra)
