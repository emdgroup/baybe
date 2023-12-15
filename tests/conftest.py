"""PyTest configuration."""

from __future__ import annotations

import os
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch

from baybe.campaign import Campaign
from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
    DiscreteCustomConstraint,
    DiscreteDependenciesConstraint,
    DiscreteExcludeConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
    SubSelectionCondition,
    ThresholdCondition,
)
from baybe.objective import Objective
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    SubstanceEncoding,
    TaskParameter,
)
from baybe.recommenders.base import Recommender
from baybe.recommenders.bayesian import SequentialGreedyRecommender
from baybe.recommenders.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy
from baybe.strategies.composite import (
    SequentialStrategy,
    StreamingSequentialStrategy,
    TwoPhaseStrategy,
)
from baybe.surrogates import _ONNX_INSTALLED, GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.telemetry import (
    VARNAME_TELEMETRY_ENABLED,
    VARNAME_TELEMETRY_HOSTNAME,
    VARNAME_TELEMETRY_USERNAME,
)
from baybe.utils import add_fake_results, add_parameter_noise, hilberts_factory
from baybe.utils.chemistry import _MORDRED_INSTALLED, _RDKIT_INSTALLED

_CHEM_INSTALLED = _MORDRED_INSTALLED and _RDKIT_INSTALLED
if _CHEM_INSTALLED:
    from baybe.parameters.substance import SubstanceParameter

if _ONNX_INSTALLED:
    from baybe.surrogates.custom import CustomONNXSurrogate

# All fixture functions have prefix 'fixture_' and explicitly declared name so they
# can be reused by other fixtures, see
# https://docs.pytest.org/en/stable/reference/reference.html#pytest-fixture


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry():
    """Disables telemetry during pytesting via fixture."""
    # Remember the original value of the environment variables
    telemetry_enabled_before = os.environ.get(VARNAME_TELEMETRY_ENABLED)
    telemetry_userhash_before = os.environ.get(VARNAME_TELEMETRY_USERNAME)
    telemetry_hosthash_before = os.environ.get(VARNAME_TELEMETRY_HOSTNAME)

    # Set the environment variable to a certain value for the duration of the tests
    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
    os.environ[VARNAME_TELEMETRY_USERNAME] = "PYTEST"
    os.environ[VARNAME_TELEMETRY_HOSTNAME] = "PYTEST"

    # Yield control to the tests
    yield

    # Restore the original value of the environment variables
    if telemetry_enabled_before is not None:
        os.environ[VARNAME_TELEMETRY_ENABLED] = telemetry_enabled_before
    else:
        os.environ.pop(VARNAME_TELEMETRY_ENABLED)

    if telemetry_userhash_before is not None:
        os.environ[VARNAME_TELEMETRY_USERNAME] = telemetry_userhash_before
    else:
        os.environ.pop(VARNAME_TELEMETRY_USERNAME)

    if telemetry_hosthash_before is not None:
        os.environ[VARNAME_TELEMETRY_HOSTNAME] = telemetry_hosthash_before
    else:
        os.environ.pop(VARNAME_TELEMETRY_HOSTNAME)


# Add option to only run fast tests
def pytest_addoption(parser):
    """Changes pytest parser."""
    parser.addoption("--fast", action="store_true", help="fast: Runs reduced tests")


def pytest_configure(config):
    """Changes pytest marker configuration."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Marks slow tests as skip if flag is set."""
    if not config.getoption("--fast"):
        return

    skip_slow = pytest.mark.skip(reason="skip with --fast")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(params=[2], name="n_iterations", ids=["i2"])
def fixture_n_iterations(request):
    """Number of iterations ran in tests."""
    return request.param


@pytest.fixture(
    params=[pytest.param(1, marks=pytest.mark.slow), 3],
    name="batch_quantity",
    ids=["b1", "b3"],
)
def fixture_batch_quantity(request):
    """Number of recommendations requested per iteration.

    Testing 1 as edge case and 3 as a case for >1.
    """
    return request.param


@pytest.fixture(
    params=[5, pytest.param(8, marks=pytest.mark.slow)],
    name="n_grid_points",
    ids=["grid5", "grid8"],
)
def fixture_n_grid_points(request):
    """Number of grid points used in e.g. the mixture tests.

    Test an even number (5 grid points will cause 4 sections) and a number that causes
    division into numbers that have no perfect floating point representation (8 grid
    points will cause 7 sections).
    """
    return request.param


@pytest.fixture(name="good_reference_values")
def fixture_good_reference_values():
    """Define some good reference values.

    These are used by the utility function to
    generate fake good results. These only make sense for discrete parameters.
    """
    return {"Categorical_1": ["B"], "Categorical_2": ["OK"]}


@pytest.fixture(name="mock_substances")
def fixture_mock_substances():
    """A set of test substances."""
    substances = {
        "Water": "O",
        "THF": "C1CCOC1",
        "DMF": "CN(C)C=O",
        "Hexane": "CCCCCC",
    }

    return substances


@pytest.fixture(name="mock_categories")
def fixture_mock_categories():
    """A set of mock categories for categorical parameters."""
    return ["Type1", "Type2", "Type3"]


@pytest.fixture(name="parameters")
def fixture_parameters(
    parameter_names: List[str], mock_substances, mock_categories, n_grid_points
):
    """Provides example parameters via specified names."""
    # FIXME: n_grid_points causes duplicate test cases if the argument is not used

    # Required for the selection to work as intended (if the input was a single string,
    # the list comprehension would match substrings instead)
    assert isinstance(parameter_names, list)

    valid_parameters = [
        CategoricalParameter(
            name="Categorical_1",
            values=["A", "B", "C"],
            encoding="OHE",
        ),
        CategoricalParameter(
            name="Categorical_2",
            values=["bad", "OK", "good"],
            encoding="OHE",
        ),
        CategoricalParameter(
            name="Switch_1",
            values=["on", "off"],
            encoding="OHE",
        ),
        CategoricalParameter(
            name="Switch_2",
            values=["left", "right"],
            encoding="OHE",
        ),
        CategoricalParameter(
            name="Frame_A",
            values=mock_categories,
        ),
        CategoricalParameter(
            name="Frame_B",
            values=mock_categories,
        ),
        CategoricalParameter(
            name="SomeSetting",
            values=["slow", "normal", "fast"],
            encoding="INT",
        ),
        NumericalDiscreteParameter(
            name="Num_disc_1",
            values=[1, 2, 7],
            tolerance=0.3,
        ),
        NumericalDiscreteParameter(
            name="Fraction_1",
            values=list(np.linspace(0, 100, n_grid_points)),
            tolerance=0.2,
        ),
        NumericalDiscreteParameter(
            name="Fraction_2",
            values=list(np.linspace(0, 100, n_grid_points)),
            tolerance=0.5,
        ),
        NumericalDiscreteParameter(
            name="Fraction_3",
            values=list(np.linspace(0, 100, n_grid_points)),
            tolerance=0.5,
        ),
        NumericalDiscreteParameter(
            name="Temperature",
            values=list(np.linspace(100, 200, n_grid_points)),
        ),
        NumericalDiscreteParameter(
            name="Pressure",
            values=list(np.linspace(0, 6, n_grid_points)),
        ),
        NumericalContinuousParameter(
            name="Conti_finite1",
            bounds=(0, 1),
        ),
        NumericalContinuousParameter(
            name="Conti_finite2",
            bounds=(-1, 0),
        ),
        NumericalContinuousParameter(
            name="Conti_finite3",
            bounds=(-1, 1),
        ),
        CustomDiscreteParameter(
            name="Custom_1",
            data=pd.DataFrame(
                {
                    "D1": [1.1, 1.4, 1.7],
                    "D2": [11, 23, 55],
                    "D3": [-4, -13, 4],
                },
                index=["mol1", "mol2", "mol3"],
            ),
        ),
        CustomDiscreteParameter(
            name="Custom_2",
            data=pd.DataFrame(
                {
                    "desc1": [1.1, 1.4, 1.7],
                    "desc2": [55, 23, 3],
                    "desc3": [4, 5, 6],
                },
                index=["A", "B", "C"],
            ),
        ),
        TaskParameter(
            name="Task",
            values=["A", "B", "C"],
            active_values=["A", "B"],
        ),
    ]

    if _CHEM_INSTALLED:
        valid_parameters += [
            *[
                SubstanceParameter(
                    name=f"Solvent_{k+1}",
                    data=mock_substances,
                )
                for k in range(3)
            ],
            *[
                SubstanceParameter(
                    name=f"Substance_1_{encoding}",
                    data=mock_substances,
                    encoding=encoding,
                )
                for encoding in SubstanceEncoding
            ],
        ]
    else:
        valid_parameters += [
            *[
                CategoricalParameter(
                    name=f"Solvent_{k+1}",
                    values=list(mock_substances.keys()),
                )
                for k in range(3)
            ],
        ]

    return [p for p in valid_parameters if p.name in parameter_names]


@pytest.fixture(name="targets")
def fixture_targets(target_names: List[str]):
    """Provides example targets via specified names."""
    # Required for the selection to work as intended (if the input was a single string,
    # the list comprehension would match substrings instead)
    assert isinstance(target_names, list)

    valid_targets = [
        NumericalTarget(
            name="Target_max",
            mode="MAX",
        ),
        NumericalTarget(
            name="Target_min",
            mode="MIN",
        ),
        NumericalTarget(
            name="Target_max_bounded",
            mode="MAX",
            bounds=(0, 100),
            transformation="LINEAR",
        ),
        NumericalTarget(
            name="Target_min_bounded",
            mode="MIN",
            bounds=(0, 100),
            transformation="LINEAR",
        ),
        NumericalTarget(
            name="Target_match_bell",
            mode="MATCH",
            bounds=(0, 100),
            transformation="BELL",
        ),
        NumericalTarget(
            name="Target_match_triangular",
            mode="MATCH",
            bounds=(0, 100),
            transformation="TRIANGULAR",
        ),
    ]
    return [t for t in valid_targets if t.name in target_names]


@pytest.fixture(name="constraints")
def fixture_constraints(constraint_names: List[str], mock_substances, n_grid_points):
    """Provides example constraints via specified names."""
    # Required for the selection to work as intended (if the input was a single string,
    # the list comprehension would match substrings instead)
    assert isinstance(constraint_names, list)

    def custom_function(series: pd.Series) -> bool:
        if series.Solvent_1 == "water":
            if series.Temperature > 120 and series.Pressure > 5:
                return False
            if series.Temperature > 180 and series.Pressure > 3:
                return False
        if series.Solvent_1 == "C3":
            if series.Temperature < 150 and series.Pressure > 3:
                return False
        return True

    valid_constraints = {
        "Constraint_1": DiscreteDependenciesConstraint(
            parameters=["Switch_1", "Switch_2"],
            conditions=[
                SubSelectionCondition(selection=["on"]),
                SubSelectionCondition(selection=["right"]),
            ],
            affected_parameters=[
                ["Solvent_1", "Fraction_1"],
                ["Frame_A", "Frame_B"],
            ],
        ),
        "Constraint_2": DiscreteDependenciesConstraint(
            parameters=["Switch_1"],
            conditions=[SubSelectionCondition(selection=["on"])],
            affected_parameters=[["Solvent_1", "Fraction_1"]],
        ),
        "Constraint_3": DiscreteDependenciesConstraint(
            parameters=["Switch_2"],
            conditions=[SubSelectionCondition(selection=["right"])],
            affected_parameters=[["Frame_A", "Frame_B"]],
        ),
        "Constraint_4": DiscreteExcludeConstraint(
            parameters=["Temperature", "Solvent_1"],
            combiner="AND",
            conditions=[
                ThresholdCondition(threshold=151, operator=">"),
                SubSelectionCondition(selection=list(mock_substances)[:2]),
            ],
        ),
        "Constraint_5": DiscreteExcludeConstraint(
            parameters=["Pressure", "Solvent_1"],
            combiner="AND",
            conditions=[
                ThresholdCondition(threshold=5, operator=">"),
                SubSelectionCondition(selection=list(mock_substances)[-2:]),
            ],
        ),
        "Constraint_6": DiscreteExcludeConstraint(
            parameters=["Pressure", "Temperature"],
            combiner="AND",
            conditions=[
                ThresholdCondition(threshold=3, operator="<"),
                ThresholdCondition(threshold=120, operator=">"),
            ],
        ),
        "Constraint_7": DiscreteNoLabelDuplicatesConstraint(
            parameters=["Solvent_1", "Solvent_2", "Solvent_3"],
        ),
        "Constraint_8": DiscreteSumConstraint(
            parameters=["Fraction_1", "Fraction_2"],
            condition=ThresholdCondition(threshold=150, operator="<="),
        ),
        "Constraint_9": DiscreteProductConstraint(
            parameters=["Fraction_1", "Fraction_2"],
            condition=ThresholdCondition(threshold=30, operator=">="),
        ),
        "Constraint_10": DiscreteSumConstraint(
            parameters=["Fraction_1", "Fraction_2"],
            condition=ThresholdCondition(threshold=100, operator="="),
        ),
        "Constraint_11": DiscretePermutationInvarianceConstraint(
            parameters=["Solvent_1", "Solvent_2", "Solvent_3"],
            dependencies=DiscreteDependenciesConstraint(
                parameters=["Fraction_1", "Fraction_2", "Fraction_3"],
                conditions=[
                    ThresholdCondition(threshold=0.0, operator=">"),
                    ThresholdCondition(threshold=0.0, operator=">"),
                    SubSelectionCondition(
                        selection=list(np.linspace(0, 100, n_grid_points)[1:])
                    ),
                ],
                affected_parameters=[["Solvent_1"], ["Solvent_2"], ["Solvent_3"]],
            ),
        ),
        "Constraint_12": DiscreteSumConstraint(
            parameters=["Fraction_1", "Fraction_2", "Fraction_3"],
            condition=ThresholdCondition(threshold=100, operator="=", tolerance=0.01),
        ),
        "Constraint_13": DiscreteCustomConstraint(
            parameters=["Pressure", "Solvent_1", "Temperature"],
            validator=custom_function,
        ),
        "ContiConstraint_1": ContinuousLinearEqualityConstraint(
            parameters=["Conti_finite1", "Conti_finite2"],
            coefficients=[1.0, 1.0],
            rhs=0.3,
        ),
        "ContiConstraint_2": ContinuousLinearEqualityConstraint(
            parameters=["Conti_finite1", "Conti_finite2"],
            coefficients=[1.0, 3.0],
            rhs=0.3,
        ),
        "ContiConstraint_3": ContinuousLinearInequalityConstraint(
            parameters=["Conti_finite1", "Conti_finite2"],
            coefficients=[1.0, 1.0],
            rhs=0.3,
        ),
        "ContiConstraint_4": ContinuousLinearInequalityConstraint(
            parameters=["Conti_finite1", "Conti_finite2"],
            coefficients=[1.0, 3.0],
            rhs=0.3,
        ),
    }
    return [
        c_item
        for c_name, c_item in valid_constraints.items()
        if c_name in constraint_names
    ]


@pytest.fixture(name="target_names")
def fixture_default_target_selection():
    """The default targets to be used if not specified differently."""
    return ["Target_max"]


@pytest.fixture(name="parameter_names")
def fixture_default_parameter_selection():
    """Default parameters used if not specified differently."""
    return ["Categorical_1", "Categorical_2", "Num_disc_1"]


@pytest.fixture(name="constraint_names")
def fixture_default_constraint_selection():
    """Default constraints used if not specified differently."""
    return []


@pytest.fixture(name="campaign")
def fixture_campaign(parameters, constraints, strategy, objective):
    """Returns a campaign."""
    return Campaign(
        searchspace=SearchSpace.from_product(
            parameters=parameters, constraints=constraints
        ),
        strategy=strategy,
        objective=objective,
    )


@pytest.fixture(name="twophase_strategy")
def fixture_default_twophase_strategy(recommender, initial_recommender):
    """The default ```TwoPhaseStrategy``` to be used if not specified differently."""
    return TwoPhaseStrategy(
        recommender=recommender,
        initial_recommender=initial_recommender,
        allow_repeated_recommendations=False,
        allow_recommending_already_measured=False,
    )


@pytest.fixture(name="sequential_strategy")
def fixture_default_sequential_strategy():
    """The default ```SequentialStrategy``` to be used if not specified differently."""
    return SequentialStrategy(
        recommenders=[RandomRecommender(), SequentialGreedyRecommender()],
        mode="reuse_last",
        allow_repeated_recommendations=False,
        allow_recommending_already_measured=False,
    )


@pytest.fixture(name="streaming_sequential_strategy")
def fixture_default_streaming_sequential_strategy():
    """The default ```StreamingSequentialStrategy``` to be used."""
    return StreamingSequentialStrategy(
        recommenders=chain(
            (RandomRecommender(),), hilberts_factory(SequentialGreedyRecommender)
        ),
        allow_repeated_recommendations=False,
        allow_recommending_already_measured=False,
    )


@pytest.fixture(name="strategy")
def fixture_select_strategy(
    request, twophase_strategy, sequential_strategy, streaming_sequential_strategy
):
    """Returns the requested strategy."""
    if not hasattr(request, "param") or (request.param == TwoPhaseStrategy):
        return twophase_strategy
    if request.param == SequentialStrategy:
        return sequential_strategy
    if request.param == StreamingSequentialStrategy:
        return streaming_sequential_strategy
    raise NotImplementedError("unknown strategy type")


@pytest.fixture(name="acquisition_function_cls")
def fixture_default_acquisition_function():
    """The default acquisition function to be used if not specified differently."""
    return "qEI"


@pytest.fixture(name="surrogate_model")
def fixture_default_surrogate_model(request, onnx_surrogate):
    """The default surrogate model to be used if not specified differently."""
    if hasattr(request, "param") and request.param == "onnx":
        return onnx_surrogate
    return GaussianProcessSurrogate()


@pytest.fixture(name="recommender")
def fixture_recommender(surrogate_model, acquisition_function_cls):
    """The default recommender to be used if not specified differently."""
    return SequentialGreedyRecommender(
        surrogate_model=surrogate_model,
        acquisition_function_cls=acquisition_function_cls,
    )


@pytest.fixture(name="initial_recommender")
def fixture_initial_recommender():
    """The default initial recommender to be used if not specified differently."""
    return RandomRecommender()


@pytest.fixture(name="objective")
def fixture_default_objective(targets):
    """The default objective to be used if not specified differently."""
    mode = "SINGLE" if len(targets) == 1 else "DESIRABILITY"
    return Objective(mode=mode, targets=targets)


@pytest.fixture(name="config")
def fixture_default_config():
    """The default config to be used if not specified differently."""
    # TODO: Once `to_config` is implemented, generate the default config from the
    #   default campaign object instead of hardcoding it here. This avoids redundant
    #   code and automatically keeps them synced.
    cfg = """{
        "parameters": [
            {
                "type": "NumericalDiscreteParameter",
                "name": "Temp_C",
                "values": [10, 20, 30, 40]
            },
            {
                "type": "NumericalDiscreteParameter",
                "name": "Concentration",
                "values": [0.2, 0.3, 1.4]
            },
            __fillin__
            {
                "type": "CategoricalParameter",
                "name": "Base",
                "values": ["base1", "base2", "base3", "base4", "base5"]
            }
        ],
        "constraints": [],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {"name": "Yield", "mode": "MAX"}
            ]
        },
        "strategy": {
            "type": "TwoPhaseStrategy",
            "initial_recommender": {
                "type": "RandomRecommender"
            },
            "recommender": {
                "type": "SequentialGreedyRecommender",
                "acquisition_function_cls": "qEI"
            },
            "switch_after": 1,
            "allow_repeated_recommendations": false,
            "allow_recommending_already_measured": false
        }
    }
    """.replace(
        "__fillin__",
        """
                {
                "type": "SubstanceParameter",
                "name": "Solvent",
                "data": {"sol1":"C", "sol2":"CC", "sol3":"CCC"},
                "decorrelate": true,
                "encoding": "MORDRED"
            },"""
        if _CHEM_INSTALLED
        else """
                {
                "type": "CategoricalParameter",
                "name": "Solvent",
                "values": ["sol1", "sol2", "sol3"],
                "encoding": "OHE"
            },""",
    )
    return cfg


@pytest.fixture(name="onnx_str")
def fixture_default_onnx_str() -> Union[bytes, None]:
    """The default ONNX model string to be used if not specified differently."""
    # TODO [19298]: There should be a cleaner way than returning None.
    if not _ONNX_INSTALLED:
        return None

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import BayesianRidge

    # Train sklearn model
    train_x = torch.arange(10).view(-1, 1)
    train_y = torch.arange(10).view(-1, 1)
    model = BayesianRidge()
    model.fit(train_x, train_y)

    # Convert to ONNX string
    input_dim = train_x.size(dim=1)
    onnx_input_name = "input"
    initial_types = [(onnx_input_name, FloatTensorType([None, input_dim]))]
    options = {type(model): {"return_std": True}}
    binary = convert_sklearn(
        model, initial_types=initial_types, options=options
    ).SerializeToString()

    return binary


@pytest.fixture(name="onnx_surrogate")
def fixture_default_onnx_surrogate(onnx_str) -> Union["CustomONNXSurrogate", None]:
    """The default ONNX model to be used if not specified differently."""
    # TODO [19298]: There should be a cleaner way than returning None.
    if not _ONNX_INSTALLED:
        return None
    return CustomONNXSurrogate(onnx_input_name="input", onnx_str=onnx_str)


# Reusables


# TODO consider turning this into a fixture returning a campaign after running some
#  fake iterations
def run_iterations(
    campaign: Campaign, n_iterations: int, batch_quantity: int, add_noise: bool = True
) -> None:
    """Run a campaign for some fake iterations.

    Args:
        campaign: The campaign encapsulating the experiments.
        n_iterations: Number of iterations run.
        batch_quantity: Number of recommended points per iteration.
        add_noise: Flag whether measurement noise should be added every 2nd iteration.
    """
    for k in range(n_iterations):
        rec = campaign.recommend(batch_quantity=batch_quantity)
        # dont use parameter noise for these tests

        add_fake_results(rec, campaign)
        if add_noise and (k % 2):
            add_parameter_noise(rec, campaign.parameters, noise_level=0.1)

        campaign.add_measurements(rec)


def get_dummy_training_data(length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create column-less input and target dataframes of specified length."""
    df = pd.DataFrame(np.empty((length, 0)))
    return df, df


def get_dummy_searchspace() -> SearchSpace:
    """Create a dummy searchspace whose actual content is irrelevant."""
    parameters = [NumericalDiscreteParameter(name="test", values=[0, 1])]
    return SearchSpace.from_product(parameters)


def select_recommender(strategy: Strategy, training_size: int) -> Recommender:
    """Select a recommender for given training dataset size."""
    searchspace = get_dummy_searchspace()
    df_x, df_y = get_dummy_training_data(training_size)
    return strategy.select_recommender(searchspace, train_x=df_x, train_y=df_y)
