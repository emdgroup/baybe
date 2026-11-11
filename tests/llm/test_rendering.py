"""Tests for LLM prompt rendering helpers."""

import pandas as pd
import pytest
from pytest import param

from baybe._llm.rendering import _drop_empty, to_prompt_dict
from baybe._optional.info import CHEM_INSTALLED
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)


@pytest.mark.parametrize(
    ("parameter", "expected_values"),
    [
        param(
            CategoricalParameter("Cat", ("A", "B", "C")),
            ["A", "B", "C"],
            id="categorical",
        ),
        param(
            CategoricalParameter("Cat", ("A", "B", "C"), active_values=("A", "B")),
            ["A", "B"],
            id="categorical_active_values",
        ),
        param(
            CategoricalParameter("Cat", ("A", "B", "C"), active_values=("A",)),
            ["A"],
            id="categorical_single_active_value",
        ),
        param(
            TaskParameter("Task", ("T1", "T2", "T3"), active_values=("T1",)),
            ["T1"],
            id="task_active_values",
        ),
        param(
            CustomDiscreteParameter(
                "Custom",
                pd.DataFrame({"d1": [1.0, 2.0, 3.0]}, index=["L1", "L2", "L3"]),
                active_values=("L1", "L2"),
            ),
            ["L1", "L2"],
            id="custom_active_values",
        ),
        param(
            NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0)),
            [1.0, 2.0, 3.0],
            id="numerical_discrete",
        ),
    ],
)
def test_to_prompt_dict(parameter, expected_values):
    """Only the active values of a parameter are shown, without encoding data."""
    serialized = to_prompt_dict(parameter)
    assert serialized["name"] == parameter.name
    assert serialized["values"] == expected_values
    assert "active_values" not in serialized
    assert "data" not in serialized


@pytest.mark.skipif(not CHEM_INSTALLED, reason="Chem dependencies not installed.")
def test_to_prompt_dict_substance():
    """Substance parameters show the SMILES of their active values only."""
    from baybe.parameters import SubstanceParameter

    parameter = SubstanceParameter(
        "Solvent",
        data={"Water": "O", "Ethanol": "CCO", "Benzene": "c1ccccc1"},
        active_values=("Water", "Ethanol"),
    )
    serialized = to_prompt_dict(parameter)
    assert serialized["values"] == ["Water", "Ethanol"]
    assert serialized["data"] == {"Water": "O", "Ethanol": "CCO"}
    assert "active_values" not in serialized


def test_drop_empty():
    """None values and (resulting) empty dictionaries are dropped recursively."""
    serialized = {
        "none": None,
        "empty": {},
        "emptied": {"none": None},
        "nested": [{"none": None, "kept": 1}],
        "falsy": 0,
        "empty_list": [],
    }
    assert _drop_empty(serialized) == {
        "nested": [{"kept": 1}],
        "falsy": 0,
        "empty_list": [],
    }
