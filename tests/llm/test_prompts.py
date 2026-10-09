"""Tests for LLM prompt construction."""

import pandas as pd
import pytest
from attrs import evolve
from pytest import param

from baybe._optional.info import CHEM_INSTALLED, JINJA2_INSTALLED
from baybe.exceptions import LLMResponseError
from baybe.objectives import (
    DesirabilityObjective,
    ParetoObjective,
    SingleTargetObjective,
)
from baybe.objectives.enum import Scalarizer
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.pure.llm._prompts import (
    _forbidden_configurations,
    _objective_prompt_info,
    _parameter_prompt_info,
    _target_prompt_info,
    make_prompt,
)
from baybe.searchspace import SearchSpace
from baybe.searchspace._filtered import FilteredSubspaceDiscrete
from baybe.targets import BinaryTarget, NumericalTarget
from baybe.transformations import AffineTransformation

if not JINJA2_INSTALLED:
    pytest.skip("Jinja2 not installed.", allow_module_level=True)


@pytest.mark.parametrize(
    ("parameter", "expected_kind", "domain_contains", "domain_excludes"),
    [
        param(
            CategoricalParameter("Cat", ("A", "B", "C")),
            "categorical",
            ["A", "B", "C"],
            [],
            id="categorical",
        ),
        param(
            CategoricalParameter("Cat", ("A", "B", "C"), active_values=("A", "B")),
            "categorical",
            ["A", "B"],
            ["C"],
            id="categorical_active_values",
        ),
        param(
            NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0)),
            "discrete_numeric",
            ["1.0", "2.0", "3.0"],
            ["numpy", "np."],
            id="numerical_discrete",
        ),
        param(
            NumericalContinuousParameter("Cont", bounds=(0.0, 10.0)),
            "continuous",
            ["0.0", "10.0"],
            [],
            id="continuous",
        ),
    ],
)
def test_parameter_prompt_info(
    parameter, expected_kind, domain_contains, domain_excludes
):
    """The prompt view has the correct kind and domain for each parameter type."""
    info = _parameter_prompt_info(parameter)
    assert info["name"] == parameter.name
    assert info["kind"] == expected_kind
    for substr in domain_contains:
        assert substr in info["domain"]
    for substr in domain_excludes:
        assert substr not in info["domain"]


@pytest.mark.skipif(not CHEM_INSTALLED, reason="Chem dependencies not installed.")
def test_parameter_prompt_info_substance():
    """Substance parameters expose SMILES in the domain string."""
    from baybe.parameters import SubstanceParameter

    p = SubstanceParameter(
        "Solvent",
        data={"Water": "O", "Ethanol": "CCO"},
        encoding="MORDRED",
    )
    info = _parameter_prompt_info(p)
    assert info["kind"] == "substance"
    assert "Water" in info["domain"]
    assert "O" in info["domain"]
    assert "Ethanol" in info["domain"]
    assert "CCO" in info["domain"]


@pytest.mark.parametrize(
    ("parameter", "expected_description", "expected_unit", "expected_misc"),
    [
        param(
            CategoricalParameter(
                "Cat",
                ("A", "B"),
                metadata={
                    "description": "A category",
                    "unit": "mg",
                    "purity": "high",
                },
            ),
            "A category",
            "mg",
            (("purity", "high"),),
            id="with_metadata",
        ),
        param(
            CategoricalParameter("Cat", ("A", "B")),
            None,
            None,
            (),
            id="no_metadata",
        ),
    ],
)
def test_parameter_prompt_info_metadata(
    parameter, expected_description, expected_unit, expected_misc
):
    """Metadata fields are forwarded when present and None/empty otherwise."""
    info = _parameter_prompt_info(parameter)
    assert info["description"] == expected_description
    assert info["unit"] == expected_unit
    assert info["misc"] == expected_misc


@pytest.mark.parametrize(
    ("target", "goal_contains", "exp_transformation", "exp_desc", "exp_unit"),
    [
        param(
            NumericalTarget("yield"),
            ["maximize"],
            None,
            None,
            None,
            id="numerical_maximize",
        ),
        param(
            NumericalTarget("yield", minimize=True),
            ["minimize"],
            None,
            None,
            None,
            id="numerical_minimize",
        ),
        param(
            NumericalTarget("yield", transformation=AffineTransformation(factor=2.0)),
            ["maximize"],
            "AffineTransformation(factor=2.0, shift=0.0)",
            None,
            None,
            id="numerical_with_transformation",
        ),
        param(
            BinaryTarget("success", success_value="yes", failure_value="no"),
            ["yes", "no"],
            None,
            None,
            None,
            id="binary",
        ),
        param(
            NumericalTarget(
                "yield",
                metadata={"description": "Reaction yield", "unit": "%"},
            ),
            ["maximize"],
            None,
            "Reaction yield",
            "%",
            id="numerical_with_metadata",
        ),
    ],
)
def test_target_prompt_info(
    target, goal_contains, exp_transformation, exp_desc, exp_unit
):
    """The prompt view has the correct fields for each target type."""
    info = _target_prompt_info(target)
    assert info["name"] == target.name
    assert info["description"] == exp_desc
    assert info["unit"] == exp_unit
    assert info["transformation"] == exp_transformation
    for substr in goal_contains:
        assert substr in info["goal"]


_T1 = NumericalTarget("yield")
_T2 = NumericalTarget("cost", minimize=True)


@pytest.mark.parametrize(
    ("objective", "combination_contains", "expected_description"),
    [
        param(
            SingleTargetObjective(target=_T1),
            None,
            None,
            id="single_target",
        ),
        param(
            DesirabilityObjective(
                targets=[_T1, _T2],
                scalarizer=Scalarizer.MEAN,
                require_normalization=False,
            ),
            "weighted arithmetic mean",
            None,
            id="desirability_mean",
        ),
        param(
            DesirabilityObjective(
                targets=[
                    NumericalTarget.normalized_sigmoid("yield", [[0, 0.1], [100, 0.9]]),
                    NumericalTarget.normalized_sigmoid("cost", [[0, 0.9], [100, 0.1]]),
                ],
                scalarizer=Scalarizer.GEOM_MEAN,
            ),
            "weighted geometric mean",
            None,
            id="desirability_geom_mean",
        ),
        param(
            ParetoObjective(targets=[_T1, _T2]),
            "Pareto",
            None,
            id="pareto",
        ),
        param(
            SingleTargetObjective(
                target=_T1,
                metadata={"description": "Maximize yield"},
            ),
            None,
            "Maximize yield",
            id="single_target_with_metadata",
        ),
    ],
)
def test_objective_prompt_info(objective, combination_contains, expected_description):
    """The prompt view has the correct fields per objective type."""
    info = _objective_prompt_info(objective)
    assert len(info["targets"]) == len(objective.targets)
    assert info["description"] == expected_description
    if combination_contains is None:
        assert info["combination"] is None
    else:
        assert combination_contains in info["combination"]


_SEARCHSPACE = SearchSpace.from_product([CategoricalParameter("Cat", ("A", "B"))])
_DESCRIPTION = "Test experiment."
_ALL_OPTIONAL_SECTIONS = {
    "OPTIMIZATION OBJECTIVE",
    "TARGETS",
    "PREVIOUS MEASUREMENTS",
    "PENDING EXPERIMENTS",
    "WHAT WENT WRONG",
    "ORIGINAL RESPONSE",
}


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
            {"OPTIMIZATION OBJECTIVE", "TARGETS"},
            id="with_objective",
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
            {
                "error": LLMResponseError("test error"),
                "original_response": "bad response",
            },
            {"WHAT WENT WRONG", "ORIGINAL RESPONSE"},
            id="with_recovery",
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
        batch_size=1,
        searchspace=_SEARCHSPACE,
        experiment_description=_DESCRIPTION,
        **kwargs,
    )
    for section in present_sections:
        assert section in prompt
    for section in _ALL_OPTIONAL_SECTIONS - present_sections:
        assert section not in prompt


@pytest.mark.parametrize(
    ("error", "original_response"),
    [
        param(LLMResponseError("err"), None, id="error_only"),
        param(None, "response text", id="response_only"),
    ],
)
def test_make_prompt_recovery_requires_both(error, original_response):
    """Providing only one of ``error``/``original_response`` raises."""
    with pytest.raises(ValueError, match="must be provided together"):
        make_prompt(
            batch_size=1,
            searchspace=_SEARCHSPACE,
            experiment_description=_DESCRIPTION,
            error=error,
            original_response=original_response,
        )


def _filter_searchspace(searchspace, mask):
    """Create a filtered copy of a search space."""
    return evolve(
        searchspace,
        discrete=FilteredSubspaceDiscrete.from_subspace(
            searchspace.discrete, mask.to_numpy()
        ),
    )


_DISCRETE_SS = SearchSpace.from_product(
    [
        CategoricalParameter("Cat", ("A", "B", "C")),
        NumericalDiscreteParameter("Num", (1.0, 2.0)),
    ]
)
_HYBRID_SS = SearchSpace.from_product(
    [
        CategoricalParameter("Cat", ("A", "B", "C")),
        NumericalContinuousParameter("Cont", bounds=(0.0, 10.0)),
    ]
)
_CONTINUOUS_SS = SearchSpace.from_product(
    [NumericalContinuousParameter("x", bounds=(0.0, 10.0))]
)


@pytest.mark.parametrize(
    "searchspace",
    [
        param(_DISCRETE_SS, id="unfiltered_discrete"),
        param(_CONTINUOUS_SS, id="continuous_only"),
    ],
)
def test_forbidden_configurations_returns_none(searchspace):
    """Without filtering or without a discrete part, the helper returns None."""
    assert _forbidden_configurations(searchspace) is None


@pytest.mark.parametrize(
    ("searchspace", "expected"),
    [
        param(
            _filter_searchspace(
                _DISCRETE_SS,
                _DISCRETE_SS.discrete.exp_rep["Cat"] != "B",
            ),
            "Cat  Num\n  B  1.0\n  B  2.0",
            id="single_value_forbidden",
        ),
        param(
            _filter_searchspace(
                _DISCRETE_SS,
                _DISCRETE_SS.discrete.exp_rep["Cat"].isin(["A"]),
            ),
            "Cat  Num\n  B  1.0\n  B  2.0\n  C  1.0\n  C  2.0",
            id="multiple_values_forbidden",
        ),
        param(
            _filter_searchspace(
                _DISCRETE_SS,
                ~(
                    (_DISCRETE_SS.discrete.exp_rep["Cat"] == "A")
                    & (_DISCRETE_SS.discrete.exp_rep["Num"] == 1.0)
                ),
            ),
            "Cat  Num\n  A  1.0",
            id="single_row_forbidden",
        ),
        param(
            _filter_searchspace(
                _HYBRID_SS,
                _HYBRID_SS.discrete.exp_rep["Cat"] != "B",
            ),
            "Cat\n  B",
            id="hybrid_forbidden",
        ),
    ],
)
def test_forbidden_configurations(searchspace, expected):
    """The helper returns the correct forbidden rows."""
    assert _forbidden_configurations(searchspace) == expected


@pytest.mark.parametrize(
    ("recovery_kwargs", "expected_present", "expected_absent"),
    [
        param(
            {},
            ["Please suggest 3 new experimental conditions"],
            ["WHAT WENT WRONG", "ORIGINAL RESPONSE"],
            id="without_recovery",
        ),
        param(
            {
                "error": LLMResponseError("test error"),
                "original_response": "bad response",
            },
            [
                "WHAT WENT WRONG",
                "ORIGINAL RESPONSE",
                "bad response",
                "corrected set of 3",
            ],
            ["Please suggest"],
            id="with_recovery",
        ),
    ],
)
def test_full_prompt(recovery_kwargs, expected_present, expected_absent):
    """A fully populated prompt contains all sections and no template residue."""
    searchspace = _filter_searchspace(
        _DISCRETE_SS,
        _DISCRETE_SS.discrete.exp_rep["Cat"] != "B",
    )
    prompt = make_prompt(
        batch_size=3,
        searchspace=searchspace,
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
        "TARGETS",
        "yield",
        "PARAMETERS",
        "Cat",
        "Num",
        "PREVIOUS MEASUREMENTS",
        "PENDING EXPERIMENTS",
        "FORBIDDEN CONFIGURATIONS",
        "explanation",
        "parameters",
    ]
    for section in common_sections + expected_present:
        assert section in prompt
    for section in expected_absent:
        assert section not in prompt
    for residue in ("{{", "}}", "{%", "%}"):
        assert residue not in prompt
