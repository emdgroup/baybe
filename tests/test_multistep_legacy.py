"""Regression tests characterizing the unchanged multi-step legacy path.

The tests use small local chain and fork/join DAGs. They record the existing
history modes, failure-treatment rules, real parent references, remaining-cost
accounting, pending-action handling, successes-only PREDICTED_MEAN imputation, and
the finite p-KGFN reduction arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from baybe.exceptions import IncompatibilityError
from baybe.multistep import (
    ContextMode,
    FeasibilityCondition,
    MultiStepContext,
    MultiStepDAG,
    MultiStepData,
    MultiStepNode,
    NodeMeasurement,
    PartialKnowledgeGradientFN,
)
from baybe.multistep.data import (
    _EvaluationRef,
    _Experiment,
    _history_inputs,
    _PendingAction,
    make_executable_plans,
    register_action,
)
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.experimental_feasibility import (
    InfeasibleMeasurementMode,
)
from baybe.targets.numerical import NumericalTarget


def _ref(index: int, node: str) -> _EvaluationRef:
    """Create a concrete evaluation reference."""
    return _EvaluationRef(index=index, node=node)


def _make_chain(mode: ContextMode = ContextMode.PARENT_OUTPUTS) -> MultiStepDAG:
    """Build a three-node chain with distinct cumulative node costs."""
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    x1 = NumericalContinuousParameter("x1", bounds=(0.0, 1.0))
    x2 = NumericalContinuousParameter("x2", bounds=(0.0, 1.0))
    root = MultiStepNode(name="root", output="root_out", inputs=(x0,), cost=1.0)
    mid = MultiStepNode(
        name="mid", output="mid_out", inputs=(x1,), parents=(root,), cost=2.0
    )
    leaf = MultiStepNode(
        name="leaf", output="leaf_out", inputs=(x2,), parents=(mid,), cost=4.0
    )
    return MultiStepDAG(
        nodes=(root, mid, leaf),
        objective=SingleTargetObjective(NumericalTarget("leaf_out")),
        context=MultiStepContext(mode),
    )


def _make_chain_measurements() -> tuple[
    NodeMeasurement, NodeMeasurement, NodeMeasurement
]:
    """Return one complete successful chain evaluation in experiment ``exp-1``."""
    root = NodeMeasurement(
        node="root",
        inputs={"x0": 0.1},
        output=1.0,
        experiment_id="exp-1",
    )
    mid = NodeMeasurement(
        node="mid",
        inputs={"x1": 0.2, "root_out": 1.0},
        output=2.0,
        experiment_id="exp-1",
        parent_refs={"root": _ref(0, "root")},
    )
    leaf = NodeMeasurement(
        node="leaf",
        inputs={"x2": 0.3, "mid_out": 2.0},
        output=3.0,
        experiment_id="exp-1",
        parent_refs={"mid": _ref(1, "mid")},
    )
    return root, mid, leaf


def _make_chain_data(include_leaf: bool = True) -> MultiStepData:
    """Build a store with the prefix or the complete chain evaluation."""
    root, mid, leaf = _make_chain_measurements()
    measurements = (root, mid, leaf) if include_leaf else (root, mid)
    return MultiStepData.empty().add(*measurements)


def _make_diamond(mode: ContextMode = ContextMode.PARENT_OUTPUTS) -> MultiStepDAG:
    """Build a fork/join DAG whose shared root is reachable via two paths."""
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    x1 = NumericalContinuousParameter("x1", bounds=(0.0, 1.0))
    x2 = NumericalContinuousParameter("x2", bounds=(0.0, 1.0))
    x3 = NumericalContinuousParameter("x3", bounds=(0.0, 1.0))
    root = MultiStepNode(name="root", output="root_out", inputs=(x0,))
    left = MultiStepNode(name="left", output="left_out", inputs=(x1,), parents=(root,))
    right = MultiStepNode(
        name="right", output="right_out", inputs=(x2,), parents=(root,)
    )
    join = MultiStepNode(
        name="join",
        output="join_out",
        inputs=(x3,),
        parents=(left, right),
    )
    return MultiStepDAG(
        nodes=(root, left, right, join),
        objective=SingleTargetObjective(NumericalTarget("join_out")),
        context=MultiStepContext(mode),
    )


def _make_condition_node(mode: InfeasibleMeasurementMode) -> MultiStepNode:
    """Build a single node whose auxiliary Boolean column defines success."""
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    options = {}
    if mode is InfeasibleMeasurementMode.DUMMY:
        options["infeasible_dummy_value"] = -999.0
    return MultiStepNode(
        name="n",
        output="y",
        inputs=(x0,),
        feasibility_conditions=(FeasibilityCondition(column="ok", operator="is_true"),),
        infeasible_measurement_mode=mode,
        **options,
    )


def _make_condition_data() -> MultiStepData:
    """Return two successes, one unknown, and two failures in mixed order."""
    failure_high = NodeMeasurement(
        node="n", inputs={"x0": 0.75}, output=4.0, condition_values={"ok": False}
    )
    success_low = NodeMeasurement(
        node="n", inputs={"x0": 0.0}, output=1.0, condition_values={"ok": True}
    )
    unknown = NodeMeasurement(node="n", inputs={"x0": 0.5}, output=3.0)
    success_mid = NodeMeasurement(
        node="n", inputs={"x0": 0.125}, output=1.5, condition_values={"ok": True}
    )
    failure_low = NodeMeasurement(
        node="n", inputs={"x0": 0.25}, output=2.0, condition_values={"ok": False}
    )
    return MultiStepData.empty().add(
        failure_high, success_low, unknown, success_mid, failure_low
    )


@pytest.mark.parametrize(
    ("mode", "expected_x0", "expected_y"),
    [
        pytest.param(
            InfeasibleMeasurementMode.OBSERVED,
            [0.0, 0.125, 0.5, 0.75, 0.25],
            [1.0, 1.5, 3.0, 4.0, 2.0],
            id="observed",
        ),
        pytest.param(
            InfeasibleMeasurementMode.EXCLUDE,
            [0.0, 0.125, 0.5],
            [1.0, 1.5, 3.0],
            id="exclude",
        ),
        pytest.param(
            InfeasibleMeasurementMode.DUMMY,
            [0.0, 0.125, 0.5, 0.75, 0.25],
            [1.0, 1.5, 3.0, -999.0, -999.0],
            id="dummy",
        ),
    ],
)
def test_node_training_data_modes_group_rows_by_success(
    mode: InfeasibleMeasurementMode,
    expected_x0: list[float],
    expected_y: list[float],
) -> None:
    """Mixed input rows are grouped into successes, unknowns, then failures."""
    node = _make_condition_node(mode)
    dag = MultiStepDAG(
        nodes=(node,), objective=SingleTargetObjective(NumericalTarget("y"))
    )
    frame = _make_condition_data().node_training_data(dag, "n")
    assert list(frame.columns) == ["x0", "y"]
    assert frame["x0"].tolist() == expected_x0
    assert frame["y"].tolist() == expected_y


def test_node_training_data_predicted_mean_fits_successes_only() -> None:
    """The real PREDICTED_MEAN path fits successes and groups mixed rows.

    The expected values are the observed posterior means of the successful-only
    surrogate in the pinned environment; the tolerance is ``abs=1e-9``.
    """
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    node = MultiStepNode(
        name="n",
        output="y",
        inputs=(x0,),
        feasibility_conditions=(FeasibilityCondition(column="ok", operator="is_true"),),
        infeasible_measurement_mode=InfeasibleMeasurementMode.PREDICTED_MEAN,
    )
    dag = MultiStepDAG(
        nodes=(node,),
        objective=SingleTargetObjective(NumericalTarget("y")),
    )
    successes = [
        NodeMeasurement(
            node="n",
            inputs={"x0": value},
            output=2.0 * value,
            condition_values={"ok": True},
        )
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    unknown = NodeMeasurement(node="n", inputs={"x0": 0.875}, output=9.0)
    failures = [
        NodeMeasurement(
            node="n",
            inputs={"x0": value},
            output=99.0,
            condition_values={"ok": False},
        )
        for value in (0.125, 0.625)
    ]
    data = MultiStepData.empty().add(
        successes[0],
        failures[0],
        successes[1],
        unknown,
        successes[2],
        failures[1],
        successes[3],
        successes[4],
    )

    frame = data.node_training_data(dag, "n")

    assert frame["x0"].tolist() == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        0.875,
        0.125,
        0.625,
    ]
    assert frame["y"].iloc[:6].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 9.0]
    assert frame["y"].iloc[6] == pytest.approx(0.21159159183542264, abs=1e-9)
    assert frame["y"].iloc[7] == pytest.approx(1.2360392343754556, abs=1e-9)


def test_default_infeasible_measurement_mode_is_observed() -> None:
    """Nodes keep the existing observed-output default without a dummy value."""
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    node = MultiStepNode(name="n", output="y", inputs=(x0,))
    assert node.infeasible_measurement_mode is InfeasibleMeasurementMode.OBSERVED
    assert node.infeasible_dummy_value is None


@pytest.mark.parametrize(
    ("mode", "expected_join"),
    [
        pytest.param(
            ContextMode.PARENT_OUTPUTS,
            ("x3", "left_out", "right_out"),
            id="parent-outputs",
        ),
        pytest.param(
            ContextMode.ANCESTOR_OUTPUTS,
            (
                "x3",
                "left_out",
                "__history__4_left__4_root__root_out",
                "right_out",
                "__history__5_right__4_root__root_out",
            ),
            id="ancestor-outputs",
        ),
    ],
)
def test_context_mode_qualifies_shared_ancestor_paths(
    mode: ContextMode, expected_join: tuple[str, ...]
) -> None:
    """Repeated fork/join ancestors are represented once per real path."""
    dag = _make_diamond(mode)
    assert dag.context.input_names(dag.node_by_name["join"]) == expected_join


def test_real_parent_references_and_complete_rows() -> None:
    """Stored chain observations resolve to real references and one global row."""
    dag = _make_chain()
    data = _make_chain_data()
    data.validate(dag)
    assert data.evaluation_refs(dag, "mid") == (_ref(1, "mid"),)
    assert data.experiment_bindings("exp-1") == {
        "root": _ref(0, "root"),
        "mid": _ref(1, "mid"),
        "leaf": _ref(2, "leaf"),
    }
    assert data.is_experiment_complete(dag, "exp-1")

    expected = {
        "x0": 0.1,
        "x1": 0.2,
        "x2": 0.3,
        "root_out": 1.0,
        "mid_out": 2.0,
        "leaf_out": 3.0,
    }
    assert data.assemble_complete_row(dag, "exp-1").iloc[0].to_dict() == expected

    process = data.complete_measurement_frame(dag, execution_mode="process")
    assert len(process) == 1
    assert process.loc[0, "experiment_id"] == "exp-1"
    assert process.loc[0, list(expected)].to_dict() == expected

    functional = data.complete_measurement_frame(dag, execution_mode="functional")
    assert len(functional) == 1
    assert functional.loc[0, "experiment_id"] == "functional:root=0,mid=1,leaf=2"
    assert functional.loc[0, list(expected)].to_dict() == expected


def test_invalid_parent_reference_is_rejected() -> None:
    """A reference to the wrong node cannot stand in for a real parent."""
    dag = _make_chain()
    root = NodeMeasurement(
        node="root", inputs={"x0": 0.1}, output=1.0, experiment_id="exp-1"
    )
    mid = NodeMeasurement(
        node="mid",
        inputs={"x1": 0.2, "root_out": 1.0},
        output=2.0,
        experiment_id="exp-1",
        parent_refs={"root": _ref(0, "mid")},
    )
    data = MultiStepData.empty().add(root, mid)
    with pytest.raises(IncompatibilityError, match="parent reference.*invalid"):
        data.validate(dag)


def test_register_action_records_real_parent_references() -> None:
    """A child action is reserved with its concrete successful parent."""
    dag = _make_chain()
    root = _make_chain_measurements()[0]
    data = MultiStepData.empty().add(root)
    contexts = {
        "exp-1": _Experiment(experiment_id="exp-1", bindings={"root": _ref(0, "root")})
    }
    contexts, counter = register_action(
        contexts,
        0,
        experiment_id="exp-1",
        node="mid",
        inputs={"x1": 0.2, "root_out": 1.0},
        parent_refs={"root": _ref(0, "root")},
        data=data,
        dag=dag,
    )
    assert counter == 1
    action = contexts["exp-1"].pending["mid"]
    assert dict(action.parent_refs) == {"root": _ref(0, "root")}
    assert dict(action.inputs) == {"x1": 0.2, "root_out": 1.0}


def test_register_action_rejects_inconsistent_parent_output() -> None:
    """A reserved child input must agree with the referenced parent observation."""
    dag = _make_chain()
    root = _make_chain_measurements()[0]
    data = MultiStepData.empty().add(root)
    with pytest.raises(IncompatibilityError, match="uses parent outputs"):
        register_action(
            {},
            0,
            experiment_id="exp-1",
            node="mid",
            inputs={"x1": 0.2, "root_out": 9.0},
            parent_refs={"root": _ref(0, "root")},
            data=data,
            dag=dag,
        )


def test_register_action_missing_parent_reference_is_rejected() -> None:
    """A plan cannot omit a concrete parent evaluation."""
    dag = _make_chain()
    root = _make_chain_measurements()[0]
    data = MultiStepData.empty().add(root)
    with pytest.raises(IncompatibilityError, match="must reference exactly its parent"):
        register_action(
            {},
            0,
            experiment_id="exp-1",
            node="mid",
            inputs={"x1": 0.2, "root_out": 1.0},
            parent_refs={},
            data=data,
            dag=dag,
        )


def test_remaining_cost_process_and_functional() -> None:
    """Fresh and continuing plans charge exactly the legacy remaining costs."""
    dag = _make_chain()
    data = _make_chain_data(include_leaf=False)
    contexts = {
        "exp-1": _Experiment(
            experiment_id="exp-1",
            bindings={"root": _ref(0, "root"), "mid": _ref(1, "mid")},
        )
    }

    process = make_executable_plans(dag, data, contexts, mode="process", first_new_id=2)
    by_node = {plan.node: plan for plan in process}
    assert by_node["leaf"].is_new is False
    assert by_node["leaf"].experiment_id == "exp-1"
    assert by_node["leaf"].remaining_cost == 4.0
    assert by_node["leaf"].parent_refs == {"mid": _ref(1, "mid")}
    assert by_node["leaf"].inputs == {"mid_out": 2.0}
    assert by_node["root"].is_new is True
    assert by_node["root"].experiment_id == "exp-2"
    assert by_node["root"].remaining_cost == 7.0

    functional = make_executable_plans(
        dag, data, contexts, mode="functional", first_new_id=2
    )
    by_node = {plan.node: plan for plan in functional}
    assert by_node["root"].remaining_cost == 7.0
    assert by_node["mid"].remaining_cost == 6.0
    assert by_node["mid"].parent_refs == {"root": _ref(0, "root")}
    assert by_node["mid"].inputs == {"root_out": 1.0}
    assert by_node["leaf"].remaining_cost == 4.0
    assert by_node["leaf"].parent_refs == {"mid": _ref(1, "mid")}
    assert by_node["leaf"].inputs == {"mid_out": 2.0}


def test_pending_leaf_is_not_reoffered() -> None:
    """A pending execution is neither executable nor charged twice."""
    dag = _make_chain()
    data = _make_chain_data(include_leaf=False)
    pending = _PendingAction(
        node="leaf",
        inputs={"x2": 0.3, "mid_out": 2.0},
        experiment_id="exp-1",
        parent_refs={"mid": _ref(1, "mid")},
    )
    contexts = {
        "exp-1": _Experiment(
            experiment_id="exp-1",
            bindings={"root": _ref(0, "root"), "mid": _ref(1, "mid")},
            pending={"leaf": pending},
        )
    }
    plans = make_executable_plans(dag, data, contexts, mode="process", first_new_id=2)
    assert all(plan.node != "leaf" for plan in plans)


def test_pending_duplicates_are_rejected() -> None:
    """The same node and local input cannot be reserved twice."""
    dag = _make_chain()
    data = MultiStepData.empty()
    contexts, counter = register_action(
        {},
        0,
        experiment_id="exp-1",
        node="root",
        inputs={"x0": 0.1},
        parent_refs={},
        data=data,
        dag=dag,
    )
    with pytest.raises(IncompatibilityError, match="already pending"):
        register_action(
            contexts,
            counter,
            experiment_id="exp-2",
            node="root",
            inputs={"x0": 0.1},
            parent_refs={},
            data=data,
            dag=dag,
        )
    with pytest.raises(IncompatibilityError, match="already has a pending action"):
        register_action(
            contexts,
            counter,
            experiment_id="exp-1",
            node="root",
            inputs={"x0": 0.1},
            parent_refs={},
            data=data,
            dag=dag,
        )
    with pytest.raises(IncompatibilityError, match="already has a pending action"):
        contexts["exp-1"].with_pending(contexts["exp-1"].pending["root"])


def test_legacy_gain_clamps_after_fantasy_mean() -> None:
    """The documented two-fantasy example clamps only after averaging."""
    means = torch.tensor([[12.0, 8.0], [7.0, 6.0]], dtype=torch.double)
    gain = PartialKnowledgeGradientFN._compute_pkgfn_gain(means, torch.tensor(10.0))
    assert gain.item() == 0.0


def test_legacy_gain_averages_fantasies_before_clamping() -> None:
    """Opposite-signed fantasy gains are averaged before the final clamp."""
    means = torch.tensor([[[12.0], [8.0]], [[9.0], [9.0]]], dtype=torch.double)
    gain = PartialKnowledgeGradientFN._compute_pkgfn_gain(means, torch.tensor([10.0]))
    assert torch.equal(gain, torch.tensor([0.5], dtype=torch.double))


def test_legacy_gain_preserves_candidate_order() -> None:
    """The reduction keeps candidate order and clamps each batch element."""
    means = torch.tensor(
        [
            [[12.0, 7.0, 20.0], [8.0, 6.0, 15.0]],
            [[11.0, 8.0, 15.0], [9.0, 6.0, 25.0]],
        ],
        dtype=torch.double,
    )
    gain = PartialKnowledgeGradientFN._compute_pkgfn_gain(
        means, torch.tensor([10.0, 10.0, 10.0], dtype=torch.double)
    )
    assert torch.equal(gain, torch.tensor([1.5, 0.0, 12.5], dtype=torch.double))


def test_legacy_gain_rejects_nonfinite_means() -> None:
    """The legacy estimator refuses non-finite inputs."""
    means = torch.tensor([[float("inf"), 1.0]], dtype=torch.double)
    with pytest.raises(IncompatibilityError, match="finite means"):
        PartialKnowledgeGradientFN._compute_pkgfn_gain(means, torch.tensor(0.0))


def test_legacy_gain_rejects_nonbroadcastable_baseline() -> None:
    """A baseline must broadcast to the best-terminal fantasy means."""
    means = torch.zeros((2, 2, 3), dtype=torch.double)
    with pytest.raises(IncompatibilityError, match="does not broadcast"):
        PartialKnowledgeGradientFN._compute_pkgfn_gain(
            means, torch.zeros(2, dtype=torch.double)
        )


CHAIN_CONTEXT_CASES = [
    pytest.param(
        ContextMode.PARENT_OUTPUTS,
        ("x1", "root_out"),
        ("x2", "mid_out"),
        {"root_out": 1.0},
        {"mid_out": 2.0},
        id="parent-outputs",
    ),
    pytest.param(
        ContextMode.PARENT_INPUTS_OUTPUTS,
        ("x1", "x0", "root_out"),
        ("x2", "x1", "mid_out"),
        {"x0": 0.1, "root_out": 1.0},
        {"x1": 0.2, "mid_out": 2.0},
        id="parent-inputs-outputs",
    ),
    pytest.param(
        ContextMode.ANCESTOR_OUTPUTS,
        ("x1", "root_out"),
        ("x2", "mid_out", "root_out"),
        {"root_out": 1.0},
        {"mid_out": 2.0, "root_out": 1.0},
        id="ancestor-outputs",
    ),
    pytest.param(
        ContextMode.ANCESTOR_INPUTS_OUTPUTS,
        ("x1", "x0", "root_out"),
        ("x2", "x1", "mid_out", "x0", "root_out"),
        {"x0": 0.1, "root_out": 1.0},
        {"x1": 0.2, "mid_out": 2.0, "x0": 0.1, "root_out": 1.0},
        id="ancestor-inputs-outputs",
    ),
]


@pytest.mark.parametrize(
    (
        "mode",
        "expected_mid",
        "expected_leaf",
        "expected_mid_inputs",
        "expected_leaf_inputs",
    ),
    CHAIN_CONTEXT_CASES,
)
def test_context_mode_history_columns_and_values_chain(
    mode: ContextMode,
    expected_mid: tuple[str, ...],
    expected_leaf: tuple[str, ...],
    expected_mid_inputs: dict[str, float],
    expected_leaf_inputs: dict[str, float],
) -> None:
    """The chain history columns and resolved real values are pinned."""
    dag = _make_chain(mode)
    assert dag.context.mode is mode
    assert dag.context.input_names(dag.node_by_name["mid"]) == expected_mid
    assert dag.context.input_names(dag.node_by_name["leaf"]) == expected_leaf

    data = _make_chain_data()
    mid_ref = _ref(0, "root")
    leaf_ref = _ref(1, "mid")
    assert _history_inputs(dag, data, "mid", {"root": mid_ref}) == expected_mid_inputs
    assert _history_inputs(dag, data, "leaf", {"mid": leaf_ref}) == expected_leaf_inputs

    plans = {
        plan.node: plan
        for plan in make_executable_plans(dag, data, {}, mode="functional")
    }
    assert dict(plans["mid"].parent_refs) == {"root": mid_ref}
    assert dict(plans["mid"].inputs) == expected_mid_inputs
    assert dict(plans["leaf"].parent_refs) == {"mid": leaf_ref}
    assert dict(plans["leaf"].inputs) == expected_leaf_inputs
