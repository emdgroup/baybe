"""Tests for reserved fixed-configuration process state."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from baybe.exceptions import IncompatibilityError
from baybe.multistep import MultiStepCampaign, MultiStepDAG, MultiStepNode
from baybe.multistep.data import MultiStepData
from baybe.multistep.node import FeasibilityCondition
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.pure.experimental_feasibility import InfeasibleMeasurementMode
from baybe.targets.numerical import NumericalTarget


@pytest.fixture(name="campaign")
def fixture_campaign():
    """Create a two-node process with a measured parent output as history."""
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


def _root_measurement(experiment_id: str, value: float) -> pd.DataFrame:
    """Create one real root-node result."""
    return pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "node": "root",
                "x": 0.25,
                "root_output": value,
            }
        ]
    )


def test_fixed_configuration_uses_real_history(campaign):
    """Reuse the reserved child parameter and only its real parent output."""
    registered = campaign.register_configuration({"x": 0.25, "z": 0.75})
    experiment_id = registered.iloc[0]["experiment_id"]
    root = campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(_root_measurement(experiment_id, 3.0))

    child = campaign.get_next_configuration_action(experiment_id)

    assert child.iloc[0]["experiment_id"] == experiment_id
    assert child.iloc[0]["node"] == "child"
    assert child.iloc[0]["z"] == 0.75
    assert child.iloc[0]["root_output"] == 3.0
    assert root.iloc[0]["x"] == 0.25


def test_configuration_survives_roundtrip_before_and_between_nodes(campaign):
    """Persist configurations before execution, while pending, and after results."""
    registered = campaign.register_configuration({"x": 0.25, "z": 0.75})
    experiment_id = registered.iloc[0]["experiment_id"]

    before = MultiStepData.from_json(campaign.data.to_json())
    assert before._contexts[experiment_id].configuration == {"x": 0.25, "z": 0.75}

    campaign.get_next_configuration_action(experiment_id)
    pending = MultiStepData.from_json(campaign.data.to_json())
    assert pending._contexts[experiment_id].configuration == {
        "x": 0.25,
        "z": 0.75,
    }
    assert "root" in pending._contexts[experiment_id].pending

    campaign.add_measurements(_root_measurement(experiment_id, 3.0))
    between = MultiStepData.from_json(campaign.data.to_json())
    assert between._contexts[experiment_id].configuration == {
        "x": 0.25,
        "z": 0.75,
    }
    assert "root" in between._contexts[experiment_id].bindings


def test_configuration_failure_preserves_outputs_and_stops(campaign):
    """Keep actual failure outputs but never produce a complete EI reference."""
    registered = campaign.register_configuration({"x": 0.25, "z": 0.75})
    experiment_id = registered.iloc[0]["experiment_id"]
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(_root_measurement(experiment_id, -2.0))

    next_action = campaign.get_next_configuration_action(experiment_id)
    stored = campaign.data.measurements_of_experiment(experiment_id)
    complete = campaign.data.complete_measurement_frame(campaign.dag)

    assert next_action.empty
    assert len(stored) == 1
    assert stored[0].output == -2.0
    assert complete.empty


def test_new_configuration_can_follow_a_failed_attempt(campaign):
    """Ignore closed failures when checking for another open process context."""
    first = campaign.register_configuration({"x": 0.25, "z": 0.75})
    failed_id = first.iloc[0]["experiment_id"]
    campaign.get_next_configuration_action(failed_id)
    campaign.add_measurements(_root_measurement(failed_id, -2.0))
    campaign.get_next_configuration_action(failed_id)

    second = campaign.register_configuration({"x": 0.5, "z": 0.25})

    assert second.iloc[0]["experiment_id"] != failed_id
    assert second.iloc[0]["x"] == 0.5


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param({"x": 0.25}, id="missing-key"),
        pytest.param({"x": 0.25, "z": 0.75, "other": 1.0}, id="extra-key"),
        pytest.param({"x": True, "z": 0.75}, id="boolean-is-not-numeric"),
        pytest.param({"x": float("nan"), "z": 0.75}, id="nan"),
        pytest.param({"x": float("inf"), "z": 0.75}, id="infinity"),
        pytest.param({"x": 2.0, "z": 0.75}, id="out-of-bounds"),
    ],
)
def test_invalid_configuration_does_not_reserve_state(campaign, configuration):
    """Validate all configuration fields before allocating an experiment ID."""
    before = campaign.data

    with pytest.raises((IncompatibilityError, TypeError, ValueError)):
        campaign.register_configuration(configuration)

    assert campaign.data is before
    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts


def test_non_mapping_configuration_raises_type_error_without_reserving(campaign):
    """Reject non-mappings directly rather than coercing them to dictionaries."""
    with pytest.raises(TypeError, match="configuration must be a mapping"):
        campaign.register_configuration([("x", 0.25), ("z", 0.75)])

    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts


def test_unknown_configuration_identifier_is_rejected(campaign):
    """Do not interpret an unknown experiment ID as a finished rollout."""
    with pytest.raises(IncompatibilityError, match="Unknown configuration"):
        campaign.get_next_configuration_action("exp-7")


def test_local_recommendation_is_blocked_before_fit_for_open_configuration(
    campaign, monkeypatch
):
    """Do not fit local models while a complete configuration is active."""
    campaign.register_configuration({"x": 0.25, "z": 0.75})

    def fail_if_fitted():
        raise AssertionError("local models must not be fit")

    monkeypatch.setattr(campaign, "_fit_node_models", fail_if_fitted)
    with pytest.raises(
        IncompatibilityError, match="full configuration attempt is open"
    ):
        campaign.recommend()


def test_configuration_prerequisites_reject_functional_and_no_repeats(campaign):
    """Enforce process identity and actual-repeat storage prerequisites."""
    functional = MultiStepCampaign.from_dag(campaign.dag, execution_mode="functional")
    no_repeats = MultiStepCampaign.from_dag(campaign.dag, allow_repeats=False)

    with pytest.raises(IncompatibilityError, match="process execution mode"):
        functional.register_configuration({"x": 0.25, "z": 0.75})
    with pytest.raises(IncompatibilityError, match="allow_repeats=True"):
        no_repeats.register_configuration({"x": 0.25, "z": 0.75})

    dummy_node = MultiStepNode(
        name="only",
        output="utility",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
        feasibility_conditions=(FeasibilityCondition("utility", ">=", 0.0),),
        infeasible_measurement_mode=InfeasibleMeasurementMode.DUMMY,
        infeasible_dummy_value=0.0,
    )
    dummy_dag = MultiStepDAG(
        nodes=(dummy_node,),
        objective=NumericalTarget("utility").to_objective(),
    )
    dummy_campaign = MultiStepCampaign.from_dag(dummy_dag)
    with pytest.raises(IncompatibilityError, match="OBSERVED and EXCLUDE"):
        dummy_campaign.register_configuration({"x": 0.25})


def test_fork_join_waits_for_real_parents_and_preserves_lineage():
    """Join actions carry only both real parent results under one attempt ID."""
    root_a = MultiStepNode(
        name="a",
        output="a_output",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
    )
    root_b = MultiStepNode(
        name="b",
        output="b_output",
        inputs=(NumericalContinuousParameter("y", (0.0, 1.0)),),
    )
    join = MultiStepNode(
        name="join",
        output="utility",
        inputs=(NumericalContinuousParameter("z", (0.0, 1.0)),),
        parents=(root_a, root_b),
    )
    dag = MultiStepDAG(
        nodes=(root_a, root_b, join),
        objective=NumericalTarget("utility").to_objective(),
    )
    campaign = MultiStepCampaign.from_dag(dag)
    experiment_id = campaign.register_configuration(
        {"x": 0.1, "y": 0.2, "z": 0.3}
    ).iloc[0]["experiment_id"]

    first = campaign.get_next_configuration_action(experiment_id)
    assert first.iloc[0]["node"] == "a"
    campaign.add_measurements(
        pd.DataFrame(
            [{"experiment_id": experiment_id, "node": "a", "x": 0.1, "a_output": 2.0}]
        )
    )
    second = campaign.get_next_configuration_action(experiment_id)
    assert second.iloc[0]["node"] == "b"
    campaign.add_measurements(
        pd.DataFrame(
            [{"experiment_id": experiment_id, "node": "b", "y": 0.2, "b_output": 4.0}]
        )
    )
    third = campaign.get_next_configuration_action(experiment_id)

    assert third.iloc[0]["node"] == "join"
    assert third.iloc[0]["experiment_id"] == experiment_id
    assert third.iloc[0]["z"] == 0.3
    assert third.iloc[0]["a_output"] == 2.0
    assert third.iloc[0]["b_output"] == 4.0


def test_root_failure_stops_unexecuted_independent_root():
    """A known process failure stops every remaining branch of the attempt."""
    from baybe.objectives.desirability import DesirabilityObjective

    root_a = MultiStepNode(
        name="a",
        output="a_output",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
        feasibility_conditions=(FeasibilityCondition("a_output", ">=", 0.0),),
    )
    root_b = MultiStepNode(
        name="b",
        output="utility",
        inputs=(NumericalContinuousParameter("y", (0.0, 1.0)),),
    )
    dag = MultiStepDAG(
        nodes=(root_a, root_b),
        objective=DesirabilityObjective(
            [NumericalTarget("a_output"), NumericalTarget("utility")],
            [1, 1],
            scalarizer="MEAN",
            require_normalization=False,
        ),
    )
    campaign = MultiStepCampaign.from_dag(dag)
    experiment_id = campaign.register_configuration({"x": 0.1, "y": 0.2}).iloc[0][
        "experiment_id"
    ]
    action = campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(
        pd.DataFrame(
            [{"experiment_id": experiment_id, "node": "a", "x": 0.1, "a_output": -1.0}]
        )
    )

    assert action.iloc[0]["node"] == "a"
    assert campaign.get_next_configuration_action(experiment_id).empty
    assert [measurement.node for measurement in campaign.data.measurements] == ["a"]


def test_new_configuration_is_blocked_by_open_local_context(campaign):
    """Do not mix a local node process with a reserved full configuration."""
    action = campaign.recommend()
    assert len(action) == 1

    with pytest.raises(IncompatibilityError, match="another process context is open"):
        campaign.register_configuration({"x": 0.25, "z": 0.75})


def test_sample_configuration_is_seeded_and_reserves_state(campaign):
    """Reserve the factorized sampler result under one new identifier."""
    first = campaign.sample_configuration(sampling_seed=41)
    second_campaign = MultiStepCampaign.from_dag(campaign.dag)
    second = second_campaign.sample_configuration(sampling_seed=41)

    assert first.iloc[0]["experiment_id"] == "exp-1"
    assert first.loc[:, ["x", "z"]].equals(second.loc[:, ["x", "z"]])
    assert set(first.columns) == {"experiment_id", "x", "z"}
    assert campaign.data._contexts["exp-1"].configuration == {
        "x": first.iloc[0]["x"],
        "z": first.iloc[0]["z"],
    }


def test_parameterless_configuration_has_a_row_and_executable_action():
    """Execute a parameterless child with the reserved parent output as history."""
    root = MultiStepNode(
        name="root",
        output="root_output",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
    )
    node = MultiStepNode(name="only", output="utility", parents=(root,))
    dag = MultiStepDAG(
        nodes=(root, node), objective=NumericalTarget("utility").to_objective()
    )
    campaign = MultiStepCampaign.from_dag(dag)

    configuration = campaign.register_configuration({"x": 0.25})
    experiment_id = configuration.iloc[0]["experiment_id"]
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(_root_measurement(experiment_id, 3.0))
    action = campaign.get_next_configuration_action(experiment_id)

    assert configuration.shape == (1, 2)
    assert action.loc[0, ["experiment_id", "node"]].tolist() == [experiment_id, "only"]
    assert action.iloc[0]["root_output"] == 3.0


def test_discrete_configuration_rejects_nonmember_without_reserving():
    """Reject a numerically bounded but impermissible discrete value."""
    node = MultiStepNode(
        name="only",
        output="utility",
        inputs=(NumericalDiscreteParameter("choice", (0.0, 1.0)),),
    )
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    campaign = MultiStepCampaign.from_dag(dag)

    with pytest.raises(IncompatibilityError, match="outside the allowed values"):
        campaign.register_configuration({"choice": 0.5})

    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts


def test_unknown_feasibility_blocks_configuration_rollout():
    """Preserve an unknown terminal label as an explicit execution blocker."""
    root = MultiStepNode(
        name="root",
        output="utility",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
        feasibility_conditions=(FeasibilityCondition("valid", "is_true"),),
    )
    dag = MultiStepDAG(
        nodes=(root,), objective=NumericalTarget("utility").to_objective()
    )
    campaign = MultiStepCampaign.from_dag(dag)
    experiment_id = campaign.register_configuration({"x": 0.25}).iloc[0][
        "experiment_id"
    ]
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "node": "root",
                    "x": 0.25,
                    "utility": 2.0,
                }
            ]
        )
    )

    with pytest.raises(IncompatibilityError, match="Feasibility.*unknown"):
        campaign.get_next_configuration_action(experiment_id)


def test_repeated_real_configuration_measurements_are_stored(campaign):
    """Keep an actual repeated execution under its distinct experiment ID."""
    identifiers = []
    for _ in range(2):
        registered = campaign.register_configuration({"x": 0.25, "z": 0.75})
        experiment_id = registered.iloc[0]["experiment_id"]
        identifiers.append(experiment_id)
        campaign.get_next_configuration_action(experiment_id)
        campaign.add_measurements(_root_measurement(experiment_id, 3.0))
        campaign.get_next_configuration_action(experiment_id)
        campaign.add_measurements(
            pd.DataFrame(
                [
                    {
                        "experiment_id": experiment_id,
                        "node": "child",
                        "z": 0.75,
                        "root_output": 3.0,
                        "utility": 5.0,
                    }
                ]
            )
        )

    assert identifiers[0] != identifiers[1]
    assert (
        sum(measurement.node == "root" for measurement in campaign.data.measurements)
        == 2
    )


def test_configuration_returns_defensive_copy_and_pending_action(campaign):
    """Protect stored parameters and return the same existing pending action."""
    configuration = {"x": 0.25, "z": 0.75}
    registered = campaign.register_configuration(configuration)
    configuration["x"] = 0.9
    experiment_id = registered.iloc[0]["experiment_id"]
    registered.loc[0, "z"] = 0.1

    first = campaign.get_next_configuration_action(experiment_id)
    second = campaign.get_next_configuration_action(experiment_id)

    assert first.iloc[0]["x"] == 0.25
    assert_frame_equal(first, second)
    assert len(campaign.data._contexts[experiment_id].pending) == 1
    campaign.add_measurements(_root_measurement(experiment_id, 3.0))
    child = campaign.get_next_configuration_action(experiment_id)
    assert child.iloc[0]["z"] == 0.75


def test_old_local_snapshot_without_configuration_remains_loadable(campaign):
    """Interpret a serialized context without the new optional field as local."""
    registered = campaign.register_configuration({"x": 0.25, "z": 0.75})
    experiment_id = registered.iloc[0]["experiment_id"]
    state = json.loads(campaign.data.to_json())
    context = state["contexts"][experiment_id]
    context.pop("configuration", None)

    restored = MultiStepData.from_json(json.dumps(state))

    assert restored._contexts[experiment_id].configuration is None
