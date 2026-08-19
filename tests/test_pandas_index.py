"""Tests verifying that transformation methods preserve the pandas index."""

import functools

import pandas as pd
import pytest

from baybe._optional.info import CHEM_INSTALLED
from baybe.objectives import (
    DesirabilityObjective,
    ParetoObjective,
    SingleTargetObjective,
)
from baybe.objectives.base import Objective
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.categorical import CategoricalParameter, TaskParameter
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets._deprecated import LegacyTarget
from baybe.targets.base import Target
from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import get_subclasses

_INDEX = [10, 20, 30]
_DATA = pd.DataFrame(
    {
        "num": [1.0, 2.0, 3.0],
        "cat": ["a", "b", "c"],
        "task": ["A", "B", "C"],
        "mol": ["Water", "THF", "DMF"],
        "mol_custom": ["mol1", "mol2", "mol3"],
        "t1": [1.0, 2.0, 3.0],
        "t2": [4.0, 5.0, 6.0],
        "binary": ["yes", "no", "yes"],
        "conti": [0.1, 0.5, 0.9],
    },
    index=_INDEX,
)

_custom_parameter_data = pd.DataFrame(
    {"D1": [1.1, 1.4, 1.7], "D2": [11, 23, 55]},
    index=["mol1", "mol2", "mol3"],
)
_substance_data = {"Water": "O", "THF": "C1CCOC1", "DMF": "CN(C)C=O"}
_desirability_targets = [
    NumericalTarget.normalized_ramp("t1", cutoffs=(0, 10)),
    NumericalTarget.normalized_ramp("t2", cutoffs=(0, 10)),
]


_targets: list[Target] = [
    NumericalTarget("num"),
    BinaryTarget("binary", success_value="yes", failure_value="no"),
]

_objectives: list[Objective] = [
    SingleTargetObjective(NumericalTarget("t1")),
    DesirabilityObjective(_desirability_targets),
    ParetoObjective((NumericalTarget("t1"), NumericalTarget("t2", minimize=True))),
]

_parameters: list[DiscreteParameter] = [
    NumericalDiscreteParameter("num", values=[1.0, 2.0, 3.0]),
    CategoricalParameter("cat", values=["a", "b", "c"]),
    TaskParameter("task", values=["A", "B", "C"]),
    CustomDiscreteParameter("mol_custom", data=_custom_parameter_data),
]
if CHEM_INSTALLED:
    from baybe.parameters.substance import SubstanceParameter

    _parameters.append(SubstanceParameter("mol", data=_substance_data))

_searchspaces: list[tuple[str, SearchSpace]] = [
    (
        "SubspaceDiscrete-numerical",
        SearchSpace.from_product(
            [NumericalDiscreteParameter("num", values=[1.0, 2.0, 3.0])]
        ),
    ),
    (
        "SubspaceDiscrete-categorical",
        SearchSpace.from_product([CategoricalParameter("cat", values=["a", "b", "c"])]),
    ),
    (
        "SubspaceContinuous",
        SearchSpace.from_product(
            [NumericalContinuousParameter("conti", bounds=(0, 1))]
        ),
    ),
    (
        "SearchSpace-hybrid",
        SearchSpace.from_product(
            [
                NumericalDiscreteParameter("num", values=[1.0, 2.0, 3.0]),
                NumericalContinuousParameter("conti", bounds=(0, 1)),
            ]
        ),
    ),
]


assert set(get_subclasses(Target)) - {LegacyTarget} == {t.__class__ for t in _targets}
assert set(get_subclasses(Objective)) == {o.__class__ for o in _objectives}
assert set(get_subclasses(DiscreteParameter)) == {p.__class__ for p in _parameters}


@pytest.mark.parametrize(
    "callable,data",
    [
        *(
            pytest.param(t.transform, _DATA[t.name], id=t.__class__.__name__)
            for t in _targets
        ),
        *(
            pytest.param(
                functools.partial(o.transform, allow_extra=True),
                _DATA,
                id=o.__class__.__name__,
            )
            for o in _objectives
        ),
        *(
            pytest.param(p.transform, _DATA[p.name], id=p.__class__.__name__)
            for p in _parameters
        ),
        *(
            pytest.param(
                functools.partial(ss.transform, allow_extra=True), _DATA, id=id_
            )
            for id_, ss in _searchspaces
        ),
    ],
)
def test_transform_preserves_pandas_index(callable, data):
    """Transformation callables preserve a non-default pandas index."""
    result = callable(data)
    assert list(pd.DataFrame(result).index) == _INDEX
