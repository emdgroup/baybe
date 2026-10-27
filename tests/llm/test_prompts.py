"""Tests for LLM prompt construction."""

import pandas as pd
import pytest
from pytest import param

from baybe._optional.info import CHEM_INSTALLED, JINJA2_INSTALLED, LLM_INSTALLED
from baybe.constraints import (
    ContinuousLinearConstraint,
    DiscreteSelectionConstraint,
    SubSelectionCondition,
)
from baybe.exceptions import LLMResponseError
from baybe.objectives import (
    DesirabilityObjective,
    ParetoObjective,
    SingleTargetObjective,
)
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    TaskParameter,
)
from baybe.recommenders.pure.llm._prompts import (
    _drop_empty,
    _sample_example_configurations,
    _to_prompt_dict,
    make_prompt,
)
from baybe.searchspace import SearchSpace
from baybe.targets import BinaryTarget, NumericalTarget

if not JINJA2_INSTALLED:
    pytest.skip("Jinja2 not installed.", allow_module_level=True)


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
    serialized = _to_prompt_dict(parameter)
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
    serialized = _to_prompt_dict(parameter)
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


_SEARCHSPACE = SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))])
_DESCRIPTION = "Test experiment."
_T1 = NumericalTarget("yield")
_T2 = NumericalTarget("cost", minimize=True)


@pytest.mark.skipif(not LLM_INSTALLED, reason="LLM dependencies not installed.")
@pytest.mark.parametrize(
    ("objective", "expected_content"),
    [
        param(SingleTargetObjective(_T2), '"minimize": true', id="direction"),
        param(
            SingleTargetObjective(
                NumericalTarget.match_bell("pH", match_value=7.0, sigma=0.5)
            ),
            '"match_value": 7.0',
            id="match_target",
        ),
        param(
            DesirabilityObjective(
                [
                    NumericalTarget.normalized_ramp("yield", (0, 100)),
                    NumericalTarget.normalized_ramp("cost", (0, 10), descending=True),
                ],
                weights=[2, 1],
            ),
            '"weights": [2.0, 1.0]',
            id="desirability",
        ),
        param(ParetoObjective([_T1, _T2]), '"type": "ParetoObjective"', id="pareto"),
        param(
            SingleTargetObjective(
                BinaryTarget("success", success_value="yes", failure_value="no")
            ),
            '"success_value": "yes"',
            id="binary",
        ),
        param(
            SingleTargetObjective(_T1, metadata={"description": "Maximize yield"}),
            '"description": "Maximize yield"',
            id="metadata",
        ),
    ],
)
def test_make_prompt_objective(objective, expected_content):
    """The objective is included with all information relevant for optimizing."""
    prompt = make_prompt(
        batch_size=1,
        searchspace=_SEARCHSPACE,
        objective=objective,
        experiment_description=_DESCRIPTION,
    )
    print(prompt)
    assert expected_content in prompt


_ALL_OPTIONAL_SECTIONS = {
    "OPTIMIZATION OBJECTIVE",
    "CONSTRAINTS",
    "PREVIOUS MEASUREMENTS",
    "PENDING EXPERIMENTS",
    "WHAT WENT WRONG",
    "ORIGINAL RESPONSE",
}


@pytest.mark.skipif(not LLM_INSTALLED, reason="LLM dependencies not installed.")
@pytest.mark.parametrize(
    ("kwargs", "present_sections"),
    [
        param(
            {},
            set(),
            id="no_optionals",
        ),
        param(
            {
                "objective": SingleTargetObjective(
                    target=_T1,
                    metadata={"description": "Maximize yield"},
                ),
            },
            {"OPTIMIZATION OBJECTIVE"},
            id="with_objective",
        ),
        param(
            {
                "searchspace": SearchSpace.from_product(
                    [CategoricalParameter("Cat", ("A", "B", "C"))],
                    [
                        DiscreteSelectionConstraint(
                            ["Cat"], [SubSelectionCondition(["C"])], exclude=True
                        )
                    ],
                ),
            },
            {"CONSTRAINTS"},
            id="with_discrete_constraint",
        ),
        param(
            {
                "searchspace": SearchSpace.from_product(
                    [NumericalContinuousParameter("x", (0.0, 1.0))],
                    [ContinuousLinearConstraint(["x"], "<=", [1.0], 0.5)],
                ),
            },
            {"CONSTRAINTS"},
            id="with_continuous_constraint",
        ),
        param(
            {
                "measurements": pd.DataFrame({"Cat": ["A"], "yield": [42.0]}),
            },
            {"PREVIOUS MEASUREMENTS"},
            id="with_measurements",
        ),
        param(
            {
                "pending_experiments": pd.DataFrame({"Cat": ["B"]}),
            },
            {"PENDING EXPERIMENTS"},
            id="with_pending",
        ),
        param(
            {"recovery": (LLMResponseError("test error"), "bad response")},
            {"WHAT WENT WRONG", "ORIGINAL RESPONSE"},
            id="with_recovery",
        ),
        param(
            {"recovery": (LLMResponseError("test error"), None)},
            {"WHAT WENT WRONG"},
            id="with_recovery_without_response",
        ),
        param(
            {
                "measurements": pd.DataFrame(columns=["Cat", "yield"]),
            },
            set(),
            id="empty_measurements_omitted",
        ),
        param(
            {
                "pending_experiments": pd.DataFrame(columns=["Cat"]),
            },
            set(),
            id="empty_pending_omitted",
        ),
    ],
)
def test_make_prompt_optional_sections(kwargs, present_sections):
    """Optional prompt sections appear only when their arguments are provided."""
    prompt = make_prompt(
        **{
            "batch_size": 1,
            "searchspace": _SEARCHSPACE,
            "experiment_description": _DESCRIPTION,
            **kwargs,
        }
    )
    for section in present_sections:
        assert section in prompt
    for section in _ALL_OPTIONAL_SECTIONS - present_sections:
        assert section not in prompt


@pytest.mark.skipif(not LLM_INSTALLED, reason="LLM dependencies not installed.")
def test_make_prompt_excludes_metadata_columns():
    """Only parameter and target columns of the dataframes appear in the prompt."""
    prompt = make_prompt(
        batch_size=1,
        searchspace=_SEARCHSPACE,
        objective=SingleTargetObjective(_T1),
        measurements=pd.DataFrame(
            {"Cat": ["A"], "yield": [42.0], "BatchNr": [1], "FitNr": [1.0]}
        ),
        pending_experiments=pd.DataFrame({"Cat": ["B"], "BatchNr": [2]}),
        experiment_description=_DESCRIPTION,
    )
    assert "42.0" in prompt
    assert "BatchNr" not in prompt
    assert "FitNr" not in prompt


_DISCRETE_SS = SearchSpace.from_product(
    [
        CategoricalParameter("Cat", ("A", "B", "C")),
        NumericalDiscreteParameter("Num", (1.0, 2.0)),
    ]
)


_MIXTURE_NAMES = ["a", "b", "c"]


@pytest.mark.parametrize(
    ("searchspace", "n_expected"),
    [
        param(_DISCRETE_SS, 3, id="discrete"),
        param(
            SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))]),
            2,
            id="fewer_candidates",
        ),
        param(
            SearchSpace.from_product(
                [NumericalContinuousParameter(n, (0, 100)) for n in _MIXTURE_NAMES],
                [ContinuousLinearConstraint(_MIXTURE_NAMES, "=", (1, 1, 1), 100)],
            ),
            3,
            id="continuous",
        ),
        param(
            SearchSpace.from_product(
                [
                    CategoricalParameter("Cat", ("A", "B", "C")),
                    NumericalContinuousParameter("x", (0, 1)),
                ]
            ),
            3,
            id="hybrid",
        ),
    ],
)
def test_sample_example_configurations(searchspace, n_expected):
    """The examples are configurations of the search space."""
    examples = _sample_example_configurations(searchspace)
    assert len(examples) == n_expected
    searchspace._validate_configurations(examples)


@pytest.mark.skipif(not LLM_INSTALLED, reason="LLM dependencies not installed.")
@pytest.mark.parametrize(
    ("recovery_kwargs", "expected_present", "expected_absent"),
    [
        param(
            {},
            ["Recommend 3 distinct new experimental conditions"],
            ["WHAT WENT WRONG", "ORIGINAL RESPONSE"],
            id="without_recovery",
        ),
        param(
            {"recovery": (LLMResponseError("test error"), "bad response")},
            [
                "WHAT WENT WRONG",
                "ORIGINAL RESPONSE",
                "bad response",
                "corrected set of 3",
            ],
            ["Recommend"],
            id="with_recovery",
        ),
    ],
)
def test_full_prompt(recovery_kwargs, expected_present, expected_absent):
    """A fully populated prompt contains all sections and no template residue."""
    prompt = make_prompt(
        batch_size=3,
        searchspace=_DISCRETE_SS,
        objective=SingleTargetObjective(
            target=NumericalTarget("yield"),
            metadata={"description": "Maximize yield"},
        ),
        measurements=pd.DataFrame({"Cat": ["A"], "Num": [1.0], "yield": [42.0]}),
        pending_experiments=pd.DataFrame({"Cat": ["B"], "Num": [2.0]}),
        experiment_description="A test experiment.",
        **recovery_kwargs,
    )

    common_sections = [
        "EXPERIMENT DESCRIPTION",
        "A test experiment.",
        "OPTIMIZATION OBJECTIVE",
        "Maximize yield",
        "yield",
        "PARAMETERS",
        "EXAMPLE CONFIGURATIONS",
        "Cat",
        "Num",
        "PREVIOUS MEASUREMENTS",
        "PENDING EXPERIMENTS",
        "explanation",
        "parameters",
    ]
    for section in common_sections + expected_present:
        assert section in prompt
    for section in expected_absent:
        assert section not in prompt
    # Jinja expressions are written with inner spaces, unlike nested JSON objects
    for residue in ("{{ ", " }}", "{%", "%}"):
        assert residue not in prompt
