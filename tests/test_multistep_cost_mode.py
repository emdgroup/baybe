"""Tests for multi-step action-cost selection."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from cattrs import ClassValidationError

from baybe.acquisition import qExpectedImprovement
from baybe.exceptions import IncompatibilityError
from baybe.multistep import (
    MultiStepCampaign,
    MultiStepDAG,
    MultiStepData,
    MultiStepNode,
)
from baybe.multistep import acquisition as multistep_acquisition
from baybe.multistep import campaign as multistep_campaign
from baybe.multistep.acquisition import GraphAwareAcquisitionProtocol
from baybe.multistep.data import (
    _EvaluationRef,
    _ExecutablePlan,
    _PendingAction,
    pending_actions,
    register_action,
)
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.targets.numerical import NumericalTarget


@pytest.fixture(name="dag_and_plan")
def fixture_dag_and_plan() -> tuple[MultiStepDAG, _ExecutablePlan]:
    """Create a two-node DAG and a fresh plan for its first node."""
    node_a = MultiStepNode(
        name="A",
        output="a_output",
        inputs=(NumericalContinuousParameter("a_input", (0.0, 1.0)),),
        cost=2.0,
    )
    node_b = MultiStepNode(
        name="B",
        output="b_output",
        inputs=(NumericalContinuousParameter("b_input", (0.0, 1.0)),),
        parents=(node_a,),
        cost=8.0,
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b), objective=NumericalTarget("b_output").to_objective()
    )
    plan = _ExecutablePlan(
        experiment_id="exp-1",
        node="A",
        remaining_cost=10.0,
        is_new=True,
    )
    return dag, plan


def _discrete_scoring_dag() -> MultiStepDAG:
    """Create the same cost comparison with finite candidates for actual optimizers."""
    node_a = MultiStepNode(
        name="A",
        output="a_output",
        inputs=(NumericalDiscreteParameter("a_input", (0.1, 0.5, 0.9)),),
        cost=2.0,
    )
    node_b = MultiStepNode(
        name="B",
        output="b_output",
        inputs=(NumericalDiscreteParameter("b_input", (0.25, 0.75, 0.95)),),
        parents=(node_a,),
        cost=8.0,
    )
    return MultiStepDAG(
        nodes=(node_a, node_b),
        objective=NumericalTarget("b_output").to_objective(),
    )


class _ConstantGainPolicy:
    """Structural test policy returning one constant raw gain per action."""

    gain_scale = "linear"
    supports_signed_gains = False

    def validate_context(
        self, dag: MultiStepDAG, data: object, batch_size: int
    ) -> None:
        """Accept the test DAG without model fitting prerequisites."""

    def validate_model(self, model: object) -> None:
        """Accept the structural fake model."""

    def make_gain_function(self, model: object, pending: tuple = ()) -> object:
        """Return a placeholder raw-gain function."""
        return lambda context, node, inputs: (
            torch.full(
                inputs.shape[:-2],
                6.0,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            + inputs[..., 0, 0] * 0.0
        )


class _ConstantStockBranch:
    """Return one constant raw score while retaining stock optimization."""

    def __init__(self, model: object) -> None:
        self.model = model

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a shape-compatible constant gain."""
        return (
            torch.full(inputs.shape[:-2], 6.0, dtype=inputs.dtype, device=inputs.device)
            + inputs[..., 0, 0] * 0.0
        )


def _record_campaign_action(
    campaign: MultiStepCampaign,
    *,
    experiment_id: str,
    node: str,
    inputs: dict[str, float],
    parent_refs: dict[str, _EvaluationRef],
    outputs: dict[str, float],
) -> None:
    """Reserve and measure one real action in a campaign context."""
    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id=experiment_id,
        node=node,
        inputs=inputs,
        parent_refs=parent_refs,
        data=campaign.data,
        dag=campaign.dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)
    campaign.add_measurements(
        pd.DataFrame(
            [{"experiment_id": experiment_id, "node": node, **inputs, **outputs}]
        )
    )


def _populated_campaign(
    dag: MultiStepDAG,
    *,
    recommender_path: str | None = None,
    execution_mode: str = "process",
    cost_mode: str | None = None,
    include_pending: bool = True,
) -> MultiStepCampaign:
    """Create observed lineages, one complete process, and optional pending state."""
    kwargs: dict[str, Any] = {"execution_mode": execution_mode}
    if cost_mode is not None:
        kwargs["cost_mode"] = cost_mode
    if recommender_path == "graph-aware":
        kwargs.update(
            acquisition_function=_ConstantGainPolicy(),
            recommender=TwoPhaseMetaRecommender(switch_after=1),
        )
    elif recommender_path == "stock":
        kwargs.update(
            acquisition_function=qExpectedImprovement(),
            recommender=TwoPhaseMetaRecommender(switch_after=1),
        )
    return _populate_with_campaign_options(dag, kwargs, include_pending=include_pending)


def _populate_with_campaign_options(
    dag: MultiStepDAG, kwargs: dict[str, Any], *, include_pending: bool
) -> MultiStepCampaign:
    """Build a campaign and record valid measurements in three process contexts."""
    campaign = MultiStepCampaign.from_dag(dag, **kwargs)
    refs: dict[str, _EvaluationRef] = {}
    for experiment_id, a_input, a_output in (
        ("exp-1", 0.1, 0.25),
        ("exp-2", 0.5, 0.5),
        ("exp-3", 0.9, 0.75),
    ):
        _record_campaign_action(
            campaign,
            experiment_id=experiment_id,
            node="A",
            inputs={"a_input": a_input},
            parent_refs={},
            outputs={"a_output": a_output},
        )
        refs[experiment_id] = campaign.data.experiment_bindings(experiment_id)["A"]

    _record_campaign_action(
        campaign,
        experiment_id="exp-1",
        node="B",
        inputs={"a_output": 0.25, "b_input": 0.75},
        parent_refs={"A": refs["exp-1"]},
        outputs={"b_output": 0.75},
    )
    if include_pending:
        contexts, counter = register_action(
            campaign.data._contexts,
            campaign.data._experiment_counter,
            experiment_id="exp-3",
            node="B",
            inputs={"a_output": 0.75, "b_input": 0.95},
            parent_refs={"A": refs["exp-3"]},
            data=campaign.data,
            dag=dag,
        )
        campaign.data = campaign.data.with_process_state(contexts, counter)
    return campaign


def _install_controlled_recommender_inputs(
    campaign: MultiStepCampaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub costly model predictions and gains without replacing cost scoring."""
    model = campaign.model
    columns = {"A": ("a_input",), "B": ("b_input", "a_output")}
    monkeypatch.setattr(model, "get_node_model_columns", lambda node: columns[node])
    monkeypatch.setattr(
        model,
        "transform_node_inputs",
        lambda node, frame: torch.as_tensor(
            frame[list(columns[node])].to_numpy(copy=True), dtype=torch.float64
        ),
    )
    monkeypatch.setattr(
        model, "parameters", lambda: iter((torch.zeros(1, dtype=torch.float64),))
    )
    monkeypatch.setattr(
        multistep_acquisition,
        "_make_feasibility_scoring_factor",
        lambda model, node, inputs: torch.full(
            inputs.shape[:-2], 0.5, dtype=inputs.dtype, device=inputs.device
        ),
    )
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)
    if isinstance(campaign.acquisition_function, qExpectedImprovement):
        monkeypatch.setattr(
            multistep_acquisition,
            "_make_stock_acquisition_branches",
            lambda model, pending, acquisition_function, measurements: (
                _ConstantStockBranch(model),
            ),
        )


@pytest.mark.parametrize(
    ("cost_mode", "expected_cost"),
    [
        pytest.param("node", 2.0, id="next-node"),
        pytest.param("remaining", 10.0, id="remaining-process"),
    ],
)
def test_action_cost_selects_existing_value(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
    cost_mode: str,
    expected_cost: float,
) -> None:
    """Select the requested existing cost without changing its meaning."""
    dag, plan = dag_and_plan

    assert multistep_acquisition._get_action_cost(dag, plan, cost_mode) == expected_cost


def test_action_cost_rejects_unknown_mode(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """Reject a cost selector value outside the supported modes."""
    dag, plan = dag_and_plan

    with pytest.raises(ValueError, match="Unknown cost mode"):
        multistep_acquisition._get_action_cost(dag, plan, "automatic")


def test_action_cost_does_not_mutate_plan(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """Selecting either cost leaves the plan and DAG costs unchanged."""
    dag, plan = dag_and_plan

    assert multistep_acquisition._get_action_cost(dag, plan, "node") == 2.0
    assert multistep_acquisition._get_action_cost(dag, plan, "remaining") == 10.0
    assert plan.remaining_cost == 10.0
    assert dag.node_by_name["A"].cost == 2.0


@pytest.mark.parametrize(
    ("cost", "expected_score"),
    [
        pytest.param(2.0, 1.5, id="node-cost"),
        pytest.param(10.0, 0.3, id="remaining-cost"),
    ],
)
def test_cost_basis_preserves_linear_and_log_scoring(
    cost: float, expected_score: float
) -> None:
    """Cost selection preserves linear and log score arithmetic for either basis."""
    gains = torch.tensor([6.0], dtype=torch.float64)
    probabilities = torch.tensor([0.5], dtype=torch.float64)

    linear_score = multistep_acquisition._score_actions(
        gains, probabilities, cost, scale="linear"
    )
    log_score = multistep_acquisition._score_actions(
        gains.log(), probabilities, cost, scale="log"
    )

    assert linear_score.item() == expected_score
    torch.testing.assert_close(log_score.exp(), linear_score, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "scale", [pytest.param("linear", id="linear"), pytest.param("log", id="log")]
)
def test_zero_success_remains_unselectable(scale: str) -> None:
    """A zero local success probability continues to produce negative infinity."""
    values = torch.tensor([6.0], dtype=torch.float64)
    if scale == "log":
        values = values.log()
    probabilities = torch.tensor([0.0], dtype=torch.float64)

    score = multistep_acquisition._score_actions(
        values, probabilities, 2.0, scale=scale
    )

    assert torch.isneginf(score).all()


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
    ],
)
def test_invalid_scoring_cost_is_rejected(cost: float) -> None:
    """Invalid selected costs remain rejected by the existing score validator."""
    with pytest.raises(IncompatibilityError, match="Action cost"):
        multistep_acquisition._score_actions(
            torch.tensor([6.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            cost,
            scale="linear",
        )


def test_campaign_cost_mode_defaults_to_remaining(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """Both campaign constructors preserve the existing default cost basis."""
    dag, _ = dag_and_plan

    assert MultiStepCampaign(dag=dag).cost_mode == "remaining"
    assert MultiStepCampaign.from_dag(dag).cost_mode == "remaining"


@pytest.mark.parametrize(
    "cost_mode",
    [pytest.param("node", id="node"), pytest.param("remaining", id="remaining")],
)
def test_campaign_accepts_both_cost_modes(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan], cost_mode: str
) -> None:
    """Both campaign constructors retain either explicitly selected mode."""
    dag, _ = dag_and_plan

    assert MultiStepCampaign(dag=dag, cost_mode=cost_mode).cost_mode == cost_mode
    assert MultiStepCampaign.from_dag(dag, cost_mode=cost_mode).cost_mode == cost_mode


@pytest.mark.parametrize(
    "cost_mode",
    [
        pytest.param("automatic", id="unknown-string"),
        pytest.param(None, id="none"),
        pytest.param(1, id="integer"),
    ],
)
@pytest.mark.parametrize(
    "constructor",
    [pytest.param("direct", id="direct"), pytest.param("from_dag", id="factory")],
)
def test_campaign_rejects_invalid_cost_mode(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
    cost_mode: object,
    constructor: str,
) -> None:
    """Campaign construction validates the cost mode before recommendation."""
    dag, _ = dag_and_plan

    with pytest.raises(ValueError, match="cost_mode"):
        if constructor == "direct":
            MultiStepCampaign(dag=dag, cost_mode=cost_mode)  # type: ignore[arg-type]
        else:
            MultiStepCampaign.from_dag(dag, cost_mode=cost_mode)  # type: ignore[arg-type]


def test_campaign_validates_cost_mode_assignment(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """The mutable campaign validates subsequent cost-mode assignments."""
    dag, _ = dag_and_plan
    campaign = MultiStepCampaign(dag=dag)

    campaign.cost_mode = "node"
    assert campaign.cost_mode == "node"
    with pytest.raises(ValueError, match="cost_mode"):
        campaign.cost_mode = "automatic"  # type: ignore[assignment]


def test_campaign_cost_mode_is_keyword_only(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """The campaign field does not change positional constructor arguments."""
    dag, _ = dag_and_plan

    assert (
        inspect.signature(MultiStepCampaign).parameters["cost_mode"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    campaign = MultiStepCampaign(
        dag, MultiStepData.empty(), None, "process", True, cost_mode="remaining"
    )
    assert campaign.dag is dag
    assert campaign.execution_mode == "process"


def test_recommender_helpers_default_to_remaining() -> None:
    """Every helper exposes the unchanged keyword-only default cost mode."""
    for function in (
        multistep_acquisition.recommend_graph_aware,
        multistep_acquisition._optimize_node_actions,
        multistep_acquisition.recommend_stock,
        multistep_acquisition._optimize_stock_context,
    ):
        parameter = inspect.signature(function).parameters["cost_mode"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == "remaining"


def test_selected_cost_reaches_score_optimizer(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both optimizer adapters receive the chosen cost before optimization."""
    dag, plan = dag_and_plan
    reference = torch.zeros(1, dtype=torch.float64)
    model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter((reference,)),
        get_node_model_columns=lambda node: ("a_input",),
        transform_node_inputs=lambda node, frame: torch.as_tensor(
            frame[["a_input"]].to_numpy(copy=True), dtype=torch.float64
        ),
    )
    captured_costs: list[float] = []

    def make_score_acquisition(*, cost: float, **kwargs: object) -> object:
        captured_costs.append(cost)
        return object()

    def optimize_node(*, bounds: torch.Tensor, **kwargs: object):
        return (
            torch.zeros((1, bounds.shape[-1]), dtype=torch.float64),
            torch.tensor([1.0]),
        )

    def optimize_stock(*, bounds: torch.Tensor, **kwargs: object):
        return (
            torch.zeros((1, 1, bounds.shape[-1]), dtype=torch.float64),
            torch.tensor([1.0]),
        )

    import botorch.optim

    monkeypatch.setattr(
        multistep_acquisition, "_make_tensor_score_acquisition", make_score_acquisition
    )
    monkeypatch.setattr(botorch.optim, "optimize_acqf_mixed", optimize_node)
    monkeypatch.setattr(botorch.optim, "optimize_acqf_discrete", optimize_node)

    for cost_mode, expected_cost in (("node", 2.0), ("remaining", 10.0)):
        multistep_acquisition._optimize_node_actions(
            model=model,
            problems=(plan,),
            gain_function=lambda context, node, inputs: torch.ones(1),
            gain_scale="linear",
            supports_signed_gains=False,
            feasibility_threshold=0.0,
            pending=(),
            cost_mode=cost_mode,
        )
        monkeypatch.setattr(botorch.optim, "optimize_acqf_mixed", optimize_stock)
        multistep_acquisition._optimize_stock_context(
            model=model,
            plan=plan,
            discrete_candidate={},
            fixed_parameters={},
            branches=(SimpleNamespace(model=model),),
            score_scale="linear",
            pending=(),
            feasibility_threshold=0.0,
            cost_mode=cost_mode,
        )
        assert captured_costs[-2:] == [expected_cost, expected_cost]
        monkeypatch.setattr(botorch.optim, "optimize_acqf_mixed", optimize_node)


@pytest.mark.parametrize(
    "recommender_path",
    [pytest.param("graph-aware", id="graph-aware"), pytest.param("stock", id="stock")],
)
def test_recommender_helpers_forward_default_cost_mode(
    recommender_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct helper calls without a mode forward remaining to their real optimizer."""
    campaign = _populated_campaign(
        _discrete_scoring_dag(),
        recommender_path=recommender_path,
        include_pending=False,
    )
    _install_controlled_recommender_inputs(campaign, monkeypatch)
    plans = campaign._executable_plans()
    pending = pending_actions(campaign.data._contexts)
    received_modes: list[str] = []

    if recommender_path == "graph-aware":
        optimizer_name = "_optimize_node_actions"
        recommender = multistep_acquisition.recommend_graph_aware
        args = (campaign.model, plans, pending)
        kwargs = {"acquisition": campaign.acquisition_function, "batch_size": 1}
    else:
        optimizer_name = "_optimize_stock_context"
        recommender = multistep_acquisition.recommend_stock
        args = (campaign.model, campaign.data, campaign.data._contexts, plans, pending)
        kwargs = {
            "acquisition_function": campaign.acquisition_function,
            "score_scale": "linear",
            "batch_size": 1,
        }

    optimizer = getattr(multistep_acquisition, optimizer_name)

    def capture_mode(**kwargs: Any) -> Any:
        received_modes.append(kwargs["cost_mode"])
        return optimizer(**kwargs)

    monkeypatch.setattr(multistep_acquisition, optimizer_name, capture_mode)
    recommender(*args, **kwargs)

    assert received_modes
    assert set(received_modes) == {"remaining"}


@pytest.mark.parametrize(
    "recommender_path",
    [pytest.param("graph-aware", id="graph-aware"), pytest.param("stock", id="stock")],
)
@pytest.mark.parametrize(
    ("cost_mode", "expected_node"),
    [
        pytest.param("node", "A", id="node-cost"),
        pytest.param("remaining", "B", id="remaining-cost"),
    ],
)
def test_cost_mode_changes_action_selection(
    recommender_path: str,
    cost_mode: str,
    expected_node: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real campaign scoring selects observed continuation or a new process."""
    dag = _discrete_scoring_dag()
    campaign = _populated_campaign(
        dag, recommender_path=recommender_path, cost_mode=cost_mode
    )
    _install_controlled_recommender_inputs(campaign, monkeypatch)

    actions = campaign.recommend()

    assert len(actions) == 1
    assert actions.iloc[0]["node"] == expected_node


@pytest.mark.parametrize(
    "recommender_path",
    [pytest.param("graph-aware", id="graph-aware"), pytest.param("stock", id="stock")],
)
def test_default_matches_explicit_remaining(
    recommender_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real recommendations and resulting pending state match the default mode."""
    dag = _discrete_scoring_dag()
    campaigns = (
        _populated_campaign(dag, recommender_path=recommender_path),
        _populated_campaign(
            dag, recommender_path=recommender_path, cost_mode="remaining"
        ),
    )
    for campaign in campaigns:
        _install_controlled_recommender_inputs(campaign, monkeypatch)

    default_actions, explicit_actions = (campaign.recommend() for campaign in campaigns)

    pd.testing.assert_frame_equal(default_actions, explicit_actions)
    pd.testing.assert_frame_equal(
        campaigns[0].pending_experiments, campaigns[1].pending_experiments
    )


@pytest.mark.parametrize(
    "execution_mode",
    [
        pytest.param("process", id="process"),
        pytest.param("functional", id="functional"),
    ],
)
def test_cost_mode_preserves_generated_fork_contexts(
    execution_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated plans retain real fork refs, pending exclusions, and mode costs."""
    node_a = MultiStepNode(
        name="A",
        output="a_output",
        inputs=(NumericalContinuousParameter("a_input", (0.0, 1.0)),),
        cost=2.0,
    )
    node_b = MultiStepNode(
        name="B",
        output="b_output",
        inputs=(NumericalDiscreteParameter("b_input", (0.25, 0.75)),),
        parents=(node_a,),
        cost=8.0,
    )
    node_c = MultiStepNode(
        name="C",
        output="c_output",
        inputs=(NumericalDiscreteParameter("c_input", (0.25, 0.75)),),
        parents=(node_a,),
        cost=4.0,
    )
    node_d = MultiStepNode(
        name="D",
        output="d_output",
        inputs=(NumericalContinuousParameter("d_input", (0.0, 1.0)),),
        parents=(node_b, node_c),
        cost=1.0,
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b, node_c, node_d),
        objective=NumericalTarget("d_output").to_objective(),
    )
    campaign = MultiStepCampaign.from_dag(dag, execution_mode=execution_mode)
    refs = []
    for experiment_id, a_input, a_output in (
        ("exp-1", 0.1, 0.25),
        ("exp-2", 0.9, 0.75),
    ):
        _record_campaign_action(
            campaign,
            experiment_id=experiment_id,
            node="A",
            inputs={"a_input": a_input},
            parent_refs={},
            outputs={"a_output": a_output},
        )
        refs.append(campaign.data.experiment_bindings(experiment_id)["A"])
    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id="exp-1",
        node="B",
        inputs={"a_output": 0.25, "b_input": 0.25},
        parent_refs={"A": refs[0]},
        data=campaign.data,
        dag=dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)

    plans = campaign._executable_plans()
    pending = pending_actions(campaign.data._contexts)
    problems = multistep_acquisition._make_node_actions(dag, plans, pending)
    assert len(pending) == 1
    assert pending[0].parent_refs == {"A": refs[0]}
    assert not any(
        problem.node == "B"
        and problem.parent_refs == pending[0].parent_refs
        and dict(problem.inputs) == dict(pending[0].inputs)
        for problem in problems
    )
    assert any(problem.node == "C" for problem in problems)
    if execution_mode == "functional":
        assert any(
            problem.node == "B"
            and problem.parent_refs == pending[0].parent_refs
            and problem.inputs["b_input"] == 0.75
            for problem in problems
        )

    parented_plans = [plan for plan in plans if plan.parent_refs]
    assert parented_plans
    for plan in parented_plans:
        parent_index = refs.index(plan.parent_refs["A"])
        if execution_mode == "process":
            assert plan.experiment_id == ("exp-1", "exp-2")[parent_index]
            assert not plan.is_new
        else:
            assert plan.is_new
            assert plan.experiment_id not in {"exp-1", "exp-2"}

    for index, experiment_id in enumerate(("exp-1", "exp-2")):
        expected_refs = {"A": refs[index]}
        for node in ("B", "C"):
            matching = [
                plan
                for plan in plans
                if plan.node == node and plan.parent_refs == expected_refs
            ]
            if execution_mode == "process" and index == 0 and node == "B":
                assert not matching
                continue
            assert matching
            assert all(
                plan.is_new is (execution_mode == "functional") for plan in matching
            )
            expected_remaining = (
                13.0 if execution_mode == "functional" or index == 1 else 5.0
            )
            assert {plan.remaining_cost for plan in matching} == {expected_remaining}

    assert {
        plan.remaining_cost for plan in plans if plan.is_new and plan.node == "A"
    } == {15.0}

    optimizer_groups: dict[str, tuple[Any, ...]] = {}
    for mode in ("node", "remaining"):
        groups: list[Any] = []

        def capture_group(**kwargs: Any) -> tuple[_PendingAction, torch.Tensor]:
            group = kwargs["problems"]
            groups.append(
                tuple(
                    (
                        plan.node,
                        plan.experiment_id,
                        tuple(plan.parent_refs.items()),
                        tuple(sorted(plan.inputs.items())),
                        plan.is_new,
                        plan.remaining_cost,
                    )
                    for plan in group
                )
            )
            plan = group[0]
            return (
                _PendingAction(
                    node=plan.node,
                    inputs=plan.inputs,
                    experiment_id=plan.experiment_id,
                    parent_refs=plan.parent_refs,
                ),
                torch.ones(1),
            )

        monkeypatch.setattr(
            multistep_acquisition, "_optimize_node_actions", capture_group
        )
        multistep_acquisition.recommend_graph_aware(
            campaign.model,
            plans,
            pending,
            acquisition=_ConstantGainPolicy(),
            batch_size=1,
            cost_mode=mode,
        )
        optimizer_groups[mode] = tuple(groups)

    assert optimizer_groups["node"] == optimizer_groups["remaining"]
    groups = optimizer_groups["node"]
    group_keys = {
        (group[0][0], group[0][1], group[0][2], group[0][5]) for group in groups
    }
    assert len(group_keys) == len(groups)
    assert all(
        len({(plan[0], plan[1], plan[2], plan[5]) for plan in group}) == 1
        for group in groups
    )
    for index, ref in enumerate(refs):
        nodes_for_parent = {
            group[0][0] for group in groups if group[0][2] == (("A", ref),)
        }
        expected_nodes = (
            {"C"} if execution_mode == "process" and index == 0 else {"B", "C"}
        )
        assert nodes_for_parent >= expected_nodes


@pytest.mark.parametrize(
    "recommender_path",
    [pytest.param("graph-aware", id="graph-aware"), pytest.param("stock", id="stock")],
)
def test_campaign_forwards_cost_mode_to_both_recommenders(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
    recommender_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Campaign dispatch forwards the selected mode to either recommender path."""
    dag, plan = dag_and_plan
    policy = _ConstantGainPolicy()
    campaign = MultiStepCampaign.from_dag(
        dag,
        acquisition_function=(
            policy if recommender_path == "graph-aware" else qExpectedImprovement()
        ),
        cost_mode="node",
    )
    selected = campaign.recommender.recommender
    received_modes: list[str] = []

    monkeypatch.setattr(
        TwoPhaseMetaRecommender,
        "select_recommender",
        lambda self, **kwargs: selected,
    )
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)
    monkeypatch.setattr(campaign, "_executable_plans", lambda: (plan,))
    campaign.model = SimpleNamespace(dag=dag)

    def recommend(*args: object, **kwargs: object) -> list[_PendingAction]:
        received_modes.append(kwargs["cost_mode"])
        return []

    if recommender_path == "graph-aware":
        monkeypatch.setattr(multistep_campaign, "recommend_graph_aware", recommend)
    else:
        monkeypatch.setattr(
            BotorchRecommender,
            "_get_acquisition_function",
            lambda self, objective: SimpleNamespace(
                supports_multi_output=True, is_analytic=False
            ),
        )
        monkeypatch.setattr(
            multistep_campaign, "_resolve_score_scale", lambda _: "linear"
        )
        monkeypatch.setattr(multistep_campaign, "recommend_stock", recommend)

    campaign.recommend()

    assert received_modes == ["node"]


@pytest.mark.parametrize(
    "cost_mode",
    [pytest.param("node", id="node"), pytest.param("remaining", id="remaining")],
)
def test_cost_mode_json_roundtrip(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan], cost_mode: str
) -> None:
    """Serialization retains mode and pending continuation with its real parent ref."""
    dag, _ = dag_and_plan
    campaign = _populated_campaign(
        dag, execution_mode="functional", cost_mode=cost_mode
    )
    parent_ref = campaign.data.experiment_bindings("exp-3")["A"]
    pending = pending_actions(campaign.data._contexts)
    assert pending[0].node == "B"
    assert pending[0].parent_refs == {"A": parent_ref}
    assert campaign.data.measurements[3].parent_refs == {
        "A": campaign.data.experiment_bindings("exp-1")["A"]
    }

    restored = MultiStepCampaign.from_json(campaign.to_json())

    assert restored.cost_mode == cost_mode
    assert restored.execution_mode == "functional"
    assert restored.data._contexts == campaign.data._contexts
    assert restored.data.measurements == campaign.data.measurements


def test_legacy_snapshot_without_cost_mode(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """A valid pre-field snapshot loads with the historical remaining default."""
    dag, _ = dag_and_plan
    snapshot = MultiStepCampaign.from_dag(dag).to_dict()
    del snapshot["cost_mode"]

    restored = MultiStepCampaign.from_dict(snapshot)

    assert restored.cost_mode == "remaining"


def test_invalid_snapshot_cost_mode_is_rejected(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
) -> None:
    """Snapshot loading rejects unknown modes instead of migrating them."""
    dag, _ = dag_and_plan
    snapshot = MultiStepCampaign.from_dag(dag).to_dict()
    snapshot["cost_mode"] = "automatic"

    with pytest.raises(ClassValidationError) as error:
        MultiStepCampaign.from_dict(snapshot)
    assert any(
        "automatic" in str(cause) and "Literal" in str(cause)
        for cause in error.value.exceptions
    )


def test_random_stage_ignores_cost_mode(
    dag_and_plan: tuple[MultiStepDAG, _ExecutablePlan],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal random seeds produce equal initial actions without score weighting."""
    dag, _ = dag_and_plan
    campaigns = [
        MultiStepCampaign.from_dag(
            dag, recommender=TwoPhaseMetaRecommender(switch_after=1), cost_mode=mode
        )
        for mode in ("node", "remaining")
    ]
    score_calls: list[bool] = []
    monkeypatch.setattr(
        multistep_acquisition,
        "_make_tensor_score_acquisition",
        lambda **kwargs: score_calls.append(True),
    )

    np.random.seed(17)
    node_actions = campaigns[0].recommend()
    np.random.seed(17)
    remaining_actions = campaigns[1].recommend()

    pd.testing.assert_frame_equal(node_actions, remaining_actions)
    assert campaigns[0].data._contexts == campaigns[1].data._contexts
    assert score_calls == []


def test_existing_graph_aware_protocol_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structural policy without a mode attribute can make a real recommendation."""
    dag = _discrete_scoring_dag()
    policy = _ConstantGainPolicy()

    assert not hasattr(policy, "cost_mode")
    assert isinstance(policy, GraphAwareAcquisitionProtocol)
    campaign = _populated_campaign(
        dag, recommender_path="graph-aware", include_pending=False
    )
    assert isinstance(campaign.acquisition_function, _ConstantGainPolicy)
    _install_controlled_recommender_inputs(campaign, monkeypatch)

    actions = campaign.recommend()

    assert len(actions) == 1
    assert actions.iloc[0]["node"] == "B"
