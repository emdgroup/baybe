"""Tests for executing fixed configurations with a caller-provided evaluator."""

from __future__ import annotations

import pandas as pd
import pytest

from baybe import multistep
from baybe.exceptions import IncompatibilityError
from baybe.multistep import MultiStepCampaign, MultiStepDAG, MultiStepNode
from baybe.multistep.node import FeasibilityCondition
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.targets.numerical import NumericalTarget


@pytest.fixture(name="campaign")
def fixture_campaign():
    """Create a two-node chain with fixed inputs and a root failure condition."""
    root = MultiStepNode(
        name="root",
        output="root_output",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
        feasibility_conditions=(FeasibilityCondition("root_output", ">=", 0.0),),
    )
    child = MultiStepNode(
        name="child",
        output="utility",
        inputs=(NumericalContinuousParameter("z", (0.0, 1.0)),),
        parents=(root,),
    )
    dag = MultiStepDAG(
        nodes=(root, child), objective=NumericalTarget("utility").to_objective()
    )
    return MultiStepCampaign.from_dag(dag)


def _make_result(action: pd.DataFrame, **outputs: object) -> pd.DataFrame:
    """Return an evaluator response that preserves its action identity."""
    return action.assign(**{name: [value] for name, value in outputs.items()})


def _evaluate_configuration(campaign, experiment_id, evaluator):
    """Call the public execution helper through its intended package export."""
    return multistep.evaluate_configuration(campaign, experiment_id, evaluator)


def test_full_execution_never_recommends_or_fits(campaign, monkeypatch):
    """Run each fixed node once and store a complete real process measurement."""
    registered = campaign.register_configuration({"x": 0.25, "z": 0.75})
    experiment_id = registered.loc[0, "experiment_id"]
    called_nodes = []

    def forbidden(*args, **kwargs):
        raise AssertionError("execution must not recommend or fit")

    monkeypatch.setattr(campaign, "recommend", forbidden)
    monkeypatch.setattr(campaign, "recommend_configuration", forbidden)
    monkeypatch.setattr(campaign, "_fit_node_models", forbidden)

    def evaluator(action):
        node = action.loc[0, "node"]
        called_nodes.append(node)
        if node == "root":
            assert action.loc[0, "x"] == 0.25
            response = action.copy(deep=True)
            action.loc[0, "x"] = 0.9
            return _make_result(response, root_output=2.0)
        assert node == "child"
        assert action.loc[0, "z"] == 0.75
        assert action.loc[0, "root_output"] == 2.0
        return _make_result(action, utility=4.0)

    result = _evaluate_configuration(campaign, experiment_id, evaluator)

    assert result is None
    assert called_nodes == ["root", "child"]
    assert [measurement.node for measurement in campaign.data.measurements] == [
        "root",
        "child",
    ]
    complete = campaign.data.complete_measurement_frame(campaign.dag)
    assert complete.loc[0, ["x", "z", "root_output", "utility"]].tolist() == [
        0.25,
        0.75,
        2.0,
        4.0,
    ]


def test_failure_stops_after_real_failure_measurement(campaign):
    """Preserve the true failing output and do not evaluate later nodes."""
    experiment_id = campaign.register_configuration({"x": 0.25, "z": 0.75}).loc[
        0, "experiment_id"
    ]
    called_nodes = []

    def evaluator(action):
        called_nodes.append(action.loc[0, "node"])
        return _make_result(action, root_output=-1.0)

    assert _evaluate_configuration(campaign, experiment_id, evaluator) is None

    assert called_nodes == ["root"]
    measurements = campaign.data.measurements_of_experiment(experiment_id)
    assert len(measurements) == 1
    assert measurements[0].output == -1.0
    assert campaign.data.complete_measurement_frame(campaign.dag).empty


def test_technical_error_preserves_history_and_pending_child(campaign):
    """Resume after a technical error without re-evaluating the successful root."""
    experiment_id = campaign.register_configuration({"x": 0.25, "z": 0.75}).loc[
        0, "experiment_id"
    ]
    calls = []

    def failing_evaluator(action):
        node = action.loc[0, "node"]
        calls.append(node)
        if node == "root":
            return _make_result(action, root_output=2.0)
        raise RuntimeError("external system unavailable")

    with pytest.raises(RuntimeError, match="external system unavailable"):
        _evaluate_configuration(campaign, experiment_id, failing_evaluator)

    context = campaign.data._contexts[experiment_id]
    assert calls == ["root", "child"]
    assert [measurement.node for measurement in campaign.data.measurements] == ["root"]
    assert set(context.pending) == {"child"}
    assert not context.closed

    def resumed_evaluator(action):
        calls.append(action.loc[0, "node"])
        return _make_result(action, utility=4.0)

    assert _evaluate_configuration(campaign, experiment_id, resumed_evaluator) is None
    assert calls == ["root", "child", "child"]
    assert [measurement.node for measurement in campaign.data.measurements] == [
        "root",
        "child",
    ]


@pytest.mark.parametrize(
    "invalid_response",
    [
        pytest.param(lambda action: action.iloc[:0], id="empty"),
        pytest.param(
            lambda action: pd.concat(
                [_make_result(action, root_output=1.0)] * 2, ignore_index=True
            ),
            id="multiple-rows",
        ),
        pytest.param(
            lambda action: _make_result(
                action.assign(experiment_id="exp-other"), root_output=1.0
            ),
            id="wrong-id",
        ),
        pytest.param(
            lambda action: _make_result(action.assign(node="other"), root_output=1.0),
            id="wrong-node",
        ),
        pytest.param(
            lambda action: _make_result(action.assign(x=0.9), root_output=1.0),
            id="wrong-parameter",
        ),
        pytest.param(lambda action: {"root_output": 1.0}, id="not-dataframe"),
    ],
)
def test_invalid_evaluator_response_is_atomic(campaign, invalid_response):
    """Keep a reserved action pending when the evaluator violates its contract."""
    experiment_id = campaign.register_configuration({"x": 0.25, "z": 0.75}).loc[
        0, "experiment_id"
    ]

    def evaluator(action):
        return invalid_response(action)

    with pytest.raises(IncompatibilityError, match="evaluator response"):
        _evaluate_configuration(campaign, experiment_id, evaluator)

    context = campaign.data._contexts[experiment_id]
    assert not campaign.data.measurements
    assert set(context.pending) == {"root"}
    assert not context.bindings


def test_unknown_feasibility_blocks_execution(campaign):
    """Do not treat missing feasibility as either failure or successful completion."""
    root = MultiStepNode(
        name="only",
        output="utility",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
        feasibility_conditions=(FeasibilityCondition("valid", "is_true"),),
    )
    dag = MultiStepDAG(
        nodes=(root,), objective=NumericalTarget("utility").to_objective()
    )
    campaign = MultiStepCampaign.from_dag(dag)
    experiment_id = campaign.register_configuration({"x": 0.25}).loc[0, "experiment_id"]
    calls = []

    def evaluator(action):
        calls.append(action.loc[0, "node"])
        return _make_result(action, utility=1.0)

    with pytest.raises(IncompatibilityError, match="Feasibility.*unknown.*blocked"):
        _evaluate_configuration(campaign, experiment_id, evaluator)

    assert calls == ["only"]
    assert len(campaign.data.measurements_of_experiment(experiment_id)) == 1
    assert not campaign.data._contexts[experiment_id].closed


def test_repeated_execution_after_completion_does_not_call_evaluator(campaign):
    """Treat a second helper call on a complete attempt as a no-op."""
    experiment_id = campaign.register_configuration({"x": 0.25, "z": 0.75}).loc[
        0, "experiment_id"
    ]
    calls = []

    def evaluator(action):
        node = action.loc[0, "node"]
        calls.append(node)
        if node == "root":
            return _make_result(action, root_output=2.0)
        return _make_result(action, utility=4.0)

    _evaluate_configuration(campaign, experiment_id, evaluator)
    _evaluate_configuration(campaign, experiment_id, evaluator)

    assert calls == ["root", "child"]
