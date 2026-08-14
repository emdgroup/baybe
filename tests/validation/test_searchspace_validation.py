"""Validation tests for search spaces."""

import pandas as pd
import pytest
from pytest import param

from baybe.parameters.categorical import TaskParameter
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.utils.dataframe import get_transform_objects

parameters = [NumericalDiscreteParameter("d1", [0, 1])]

_task = TaskParameter("task", values=["a", "b"])
_cat_fid_1 = CategoricalFidelityParameter("f1", ["lo", "hi"], costs=[1, 10], zeta=5)
_cat_fid_2 = CategoricalFidelityParameter("f2", ["a", "b"], costs=[1, 5], zeta=5)
_num_fid_1 = NumericalDiscreteFidelityParameter("f1", values=[0.5, 1.0], costs=[1, 10])
_num_fid_2 = NumericalDiscreteFidelityParameter("f2", values=[0.2, 1.0], costs=[1, 5])


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


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        param([_cat_fid_1, _cat_fid_2], "at most one fidelity", id="two_categorical"),
        param([_cat_fid_1, _num_fid_2], "at most one fidelity", id="mixed_types"),
        param([_num_fid_1, _num_fid_2], "at most one fidelity", id="two_numerical"),
        param(
            [_task, _cat_fid_1],
            "task parameters with fidelity parameters",
            id="task_plus_fidelity",
        ),
    ],
)
def test_invalid_fidelity_parameter_combinations(parameters, match):
    """Search spaces with invalid fidelity parameter combinations are rejected."""
    with pytest.raises(NotImplementedError, match=match):
        SearchSpace.from_product(parameters)
