"""Focused integration tests for the direct policy's existing cost modes.

The direct path uses one explicit cost basis for the local action search and
the final selection score. The campaign tests drive the real recommendation
and reservation path and only replace the expensive action search with
analytic constants.
"""

from __future__ import annotations

from types import SimpleNamespace

import hypothesis.strategies as st
import pandas as pd
import pytest
import torch
from hypothesis import given, settings

from baybe.multistep import (
    DirectPartialKnowledgeGradientFN,
    MultiStepCampaign,
    MultiStepDAG,
    MultiStepData,
    MultiStepNode,
    NodeMeasurement,
    TerminalSearchProfile,
)
from baybe.multistep.data import _ExecutablePlan, _PendingAction, register_action
from baybe.multistep.kgfn import direct as multistep_direct
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.searchspace import SearchSpace
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.targets.numerical import NumericalTarget
from tests.serialization.utils import assert_roundtrip_consistency

terminal_search_profiles = st.builds(
    TerminalSearchProfile,
    num_network_samples=st.integers(min_value=1, max_value=4),
    sampling_seed=st.integers(min_value=0, max_value=10),
    raw_samples=st.integers(min_value=2, max_value=4),
    num_restarts=st.integers(min_value=1, max_value=2),
    tolerance=st.sampled_from([1e-9, 1e-6, 1e-3]),
)
direct_policies = st.builds(
    DirectPartialKnowledgeGradientFN,
    num_fantasies=st.integers(min_value=1, max_value=3),
    search_profile=terminal_search_profiles,
)


def _make_dag() -> MultiStepDAG:
    """Build the two-node A -> B chain used by all cost comparisons."""
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
    return MultiStepDAG(
        nodes=(node_a, node_b),
        objective=NumericalTarget("b_output").to_objective(),
        product_free=True,
    )


def _make_policy() -> DirectPartialKnowledgeGradientFN:
    """Return a small deterministic direct policy for fake-solver tests."""
    return DirectPartialKnowledgeGradientFN(
        num_fantasies=2,
        search_profile=TerminalSearchProfile(
            num_network_samples=1,
            sampling_seed=0,
            raw_samples=1,
            num_restarts=1,
            tolerance=1e-9,
        ),
    )


def _make_plans() -> tuple[_ExecutablePlan, _ExecutablePlan]:
    """Return one plan per node with the brief's exact cost values."""
    return (
        _ExecutablePlan(
            experiment_id="exp-A",
            node="A",
            remaining_cost=10.0,
            is_new=True,
        ),
        _ExecutablePlan(
            experiment_id="exp-B",
            node="B",
            remaining_cost=8.0,
            is_new=True,
        ),
    )


def _fake_model(dag: MultiStepDAG) -> SimpleNamespace:
    """Return only the model surface used by the direct evaluator."""
    node_model = SimpleNamespace(
        batch_shape=torch.Size(),
        condition_on_observations=lambda *args, **kwargs: None,
    )
    return SimpleNamespace(
        dag=dag,
        parameters=lambda: iter((torch.zeros(1, dtype=torch.double),)),
        get_node_model=lambda name: node_model,
    )


def _plan_action(plan: _ExecutablePlan, dag: MultiStepDAG) -> _PendingAction:
    """Build a complete local action for a fake optimizer result."""
    node = dag.node_by_name[plan.node]
    inputs = dict(plan.inputs)
    for parameter in node.inputs:
        if parameter.is_continuous:
            inputs[parameter.name] = parameter.bounds.lower
        else:
            inputs[parameter.name] = parameter.active_values[0]
    return _PendingAction(
        node=plan.node,
        inputs=inputs,
        experiment_id=plan.experiment_id,
        parent_refs=plan.parent_refs,
    )


def _install_analytic_direct(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gain: float,
    probability: float = 0.5,
) -> list[tuple[str, float]]:
    """Fix gain and local probability while keeping the real cost arithmetic."""
    captured: list[tuple[str, float]] = []

    def fake_optimize_action(*, model, plan, cost, context_index, **_kwargs):
        score = gain * probability / cost
        captured.append((plan.node, score))
        return _plan_action(plan, model.dag), score, {
            "context_index": context_index,
            "node": plan.node,
            "experiment_id": plan.experiment_id,
            "score": score,
            "references": 1,
            "branches": 1,
            "best_value": score,
        }

    monkeypatch.setattr(multistep_direct, "_optimize_action", fake_optimize_action)
    monkeypatch.setattr(
        multistep_direct,
        "_make_reference_points",
        lambda model, profile: torch.zeros((1, 1, 1), dtype=torch.double),
    )
    monkeypatch.setattr(
        multistep_direct,
        "_make_gain_context",
        lambda *args, **kwargs: multistep_direct._GainContext(
            evaluate=lambda node, inputs: torch.ones(
                inputs.shape[:-2], dtype=torch.double
            ),
            num_branches=1,
            num_references=1,
        ),
    )
    return captured


def _make_campaign(*, cost_mode: str | None = None) -> MultiStepCampaign:
    """Build a functional campaign with one observed A so B is executable."""
    kwargs: dict[str, object] = {
        "execution_mode": "functional",
        "acquisition_function": _make_policy(),
        "recommender": TwoPhaseMetaRecommender(switch_after=1),
    }
    if cost_mode is not None:
        kwargs["cost_mode"] = cost_mode
    campaign = MultiStepCampaign.from_dag(_make_dag(), **kwargs)
    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id="exp-1",
        node="A",
        inputs={"a_input": 0.5},
        parent_refs={},
        data=campaign.data,
        dag=campaign.dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": "exp-1",
                    "node": "A",
                    "a_input": 0.5,
                    "a_output": 0.25,
                }
            ]
        )
    )
    return campaign


def _attach_fake_campaign_model(
    campaign: MultiStepCampaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace only expensive model fitting and scoring internals."""
    monkeypatch.setattr(campaign, "model", _fake_model(campaign.dag))
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)


@pytest.mark.parametrize(
    ("cost_arguments", "expected_node", "expected_scores"),
    [
        pytest.param({}, "B", (0.3, 0.375), id="default-remaining"),
        pytest.param(
            {"cost_mode": "remaining"}, "B", (0.3, 0.375), id="explicit-remaining"
        ),
        pytest.param({"cost_mode": "node"}, "A", (1.5, 0.375), id="node"),
    ],
)
def test_direct_cost_mode_selects_existing_cost(
    cost_arguments: dict[str, str],
    expected_node: str,
    expected_scores: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local action search and final selection use the selected cost basis."""
    captured = _install_analytic_direct(monkeypatch, gain=6.0)

    action, diagnostics = multistep_direct.recommend_direct(
        _fake_model(_make_dag()),
        _make_plans(),
        (),
        policy=_make_policy(),
        **cost_arguments,
    )

    assert action.node == expected_node
    assert diagnostics["selected_score"] == pytest.approx(
        expected_scores[0 if expected_node == "A" else 1]
    )
    assert tuple(
        context["score"] for context in diagnostics["contexts"]
    ) == pytest.approx(expected_scores)
    assert tuple(value for _, value in captured) == pytest.approx(expected_scores)


def test_direct_cost_mode_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid direct cost mode remains an explicit failure."""
    _install_analytic_direct(monkeypatch, gain=6.0)

    with pytest.raises(ValueError, match="Unknown cost mode"):
        multistep_direct.recommend_direct(
            _fake_model(_make_dag()),
            _make_plans(),
            (),
            policy=_make_policy(),
            cost_mode="automatic",
        )


@pytest.mark.parametrize(
    ("campaign_cost_mode", "expected_node", "expected_optimizer_values"),
    [
        pytest.param(None, "B", (0.3, 0.375), id="default-remaining"),
        pytest.param(
            "remaining", "B", (0.3, 0.375), id="explicit-remaining"
        ),
        pytest.param("node", "A", (1.5, 0.375), id="node"),
    ],
)
def test_campaign_direct_policy_selects_and_reserves_with_cost_mode(
    campaign_cost_mode: str | None,
    expected_node: str,
    expected_optimizer_values: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real campaign dispatches to the direct policy and reserves its action."""
    captured = _install_analytic_direct(monkeypatch, gain=6.0)
    campaign = _make_campaign(cost_mode=campaign_cost_mode)
    _attach_fake_campaign_model(campaign, monkeypatch)

    actions = campaign.recommend()

    assert len(actions) == 1
    assert actions.iloc[0]["node"] == expected_node
    assert campaign.pending_experiments["node"].tolist() == [expected_node]
    assert campaign.pending_experiments["experiment_id"].tolist() == actions[
        "experiment_id"
    ].tolist()
    assert len(campaign.data.measurements) == 1
    assert len(campaign.data._contexts) == 2
    assert tuple(value for _, value in captured) == pytest.approx(
        expected_optimizer_values
    )


def test_campaign_default_cost_mode_matches_explicit_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted campaign cost mode equals explicit remaining, including reservations."""
    _install_analytic_direct(monkeypatch, gain=6.0)
    outcomes = []
    for cost_mode in (None, "remaining"):
        campaign = _make_campaign(cost_mode=cost_mode)
        _attach_fake_campaign_model(campaign, monkeypatch)
        outcomes.append((campaign.recommend(), campaign.pending_experiments))

    pd.testing.assert_frame_equal(outcomes[0][0], outcomes[1][0])
    pd.testing.assert_frame_equal(outcomes[0][1], outcomes[1][1])


def test_direct_random_stage_is_product_free_and_cost_mode_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial actions avoid products and fitting and reserve distinct pending work."""
    campaigns = []
    for mode in (None, "remaining", "node"):
        kwargs = {} if mode is None else {"cost_mode": mode}
        campaigns.append(
            MultiStepCampaign.from_dag(
                _make_dag(),
                acquisition_function=_make_policy(),
                recommender=TwoPhaseMetaRecommender(switch_after=1),
                **kwargs,
            )
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The random stage built a product or fit a model.")

    monkeypatch.setattr(SearchSpace, "from_product", forbidden)
    monkeypatch.setattr(SubspaceContinuous, "from_product", forbidden)
    monkeypatch.setattr(SubspaceDiscrete, "from_product", forbidden)
    outcomes = []
    for campaign in campaigns:
        monkeypatch.setattr(campaign, "_fit_node_models", forbidden)
        first = campaign.recommend()
        second = campaign.recommend()
        pending = campaign.pending_experiments
        assert first["node"].tolist() == second["node"].tolist() == ["A"]
        assert first.iloc[0]["a_input"] != second.iloc[0]["a_input"]
        assert pending["experiment_id"].tolist() == ["exp-1", "exp-2"]
        assert pending["node"].tolist() == ["A", "A"]
        assert len(campaign.data.measurements) == 0
        outcomes.append((first, second, pending))

    for outcome in outcomes[1:]:
        for default_frame, explicit_frame in zip(outcomes[0], outcome):
            pd.testing.assert_frame_equal(default_frame, explicit_frame)


@pytest.mark.parametrize("cost_mode", ["remaining", "node"])
def test_direct_campaign_uses_real_model_and_solver_without_products(
    cost_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small synthetic campaign fits and recommends through the real pipeline."""
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(NumericalDiscreteParameter("x", (0.0, 1.0)),),
        cost=2.0,
    )
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("y").to_objective(), product_free=True
    )
    data = MultiStepData.empty().add(
        NodeMeasurement(node="leaf", inputs={"x": 0.0}, output=0.0),
        NodeMeasurement(node="leaf", inputs={"x": 1.0}, output=1.0),
    )
    policy = DirectPartialKnowledgeGradientFN(
        num_fantasies=1,
        search_profile=TerminalSearchProfile(
            num_network_samples=2,
            raw_samples=2,
            num_restarts=1,
        ),
    )
    campaign = MultiStepCampaign.from_dag(
        dag, data=data, acquisition_function=policy, cost_mode=cost_mode
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The direct pipeline built a parameter product.")

    monkeypatch.setattr(SearchSpace, "from_product", forbidden)
    monkeypatch.setattr(SubspaceContinuous, "from_product", forbidden)
    monkeypatch.setattr(SubspaceDiscrete, "from_product", forbidden)
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        action = campaign.recommend()

    assert action["node"].tolist() == ["leaf"]
    assert action.iloc[0]["x"] in (0.0, 1.0)
    assert campaign.pending_experiments["experiment_id"].tolist() == ["exp-1"]
    assert len(campaign.data.measurements) == 2


@pytest.mark.parametrize(
    "cost_mode",
    [pytest.param("remaining", id="remaining"), pytest.param("node", id="node")],
)
def test_campaign_direct_policy_serialization_roundtrip(cost_mode: str) -> None:
    """The campaign retains a direct policy together with either cost mode."""
    policy = _make_policy()
    campaign = MultiStepCampaign.from_dag(
        _make_dag(),
        execution_mode="functional",
        acquisition_function=policy,
        cost_mode=cost_mode,
    )

    restored = MultiStepCampaign.from_json(campaign.to_json())

    assert isinstance(restored.acquisition_function, DirectPartialKnowledgeGradientFN)
    assert restored.acquisition_function == policy
    assert restored.cost_mode == cost_mode
    assert restored.execution_mode == "functional"
    assert restored.dag.product_free is True
    assert restored.data.to_dict() == campaign.data.to_dict()


def test_campaign_direct_policy_without_cost_mode_defaults_to_remaining() -> None:
    """A legacy snapshot lacking cost_mode still restores the direct policy."""
    campaign = MultiStepCampaign.from_dag(
        _make_dag(),
        execution_mode="functional",
        acquisition_function=_make_policy(),
    )
    snapshot = campaign.to_dict()
    del snapshot["cost_mode"]

    restored = MultiStepCampaign.from_dict(snapshot)

    assert isinstance(restored.acquisition_function, DirectPartialKnowledgeGradientFN)
    assert restored.acquisition_function == campaign.acquisition_function
    assert restored.cost_mode == "remaining"


@settings(max_examples=10, deadline=None)
@given(terminal_search_profiles)
def test_terminal_search_profile_serialization_roundtrip(
    profile: TerminalSearchProfile,
) -> None:
    """Generated solver profiles survive the standard serialization roundtrip."""
    assert_roundtrip_consistency(profile)


@settings(max_examples=10, deadline=None)
@given(direct_policies)
def test_direct_policy_serialization_roundtrip(
    policy: DirectPartialKnowledgeGradientFN,
) -> None:
    """Generated direct policies survive the standard serialization roundtrip."""
    assert_roundtrip_consistency(policy)


def test_recommend_direct_stable_tie_breaking(monkeypatch):
    def fake_optimize_action(*, model, plan, cost, context_index, **_kwargs):
        return _plan_action(plan, model.dag), 0.5, {
            "context_index": context_index,
            "node": plan.node,
            "experiment_id": plan.experiment_id,
            "score": 0.5,
            "references": 1,
            "branches": 1,
            "best_value": 0.5,
        }

    monkeypatch.setattr(multistep_direct, "_optimize_action", fake_optimize_action)
    monkeypatch.setattr(
        multistep_direct,
        "_make_reference_points",
        lambda model, profile: torch.zeros((1, 1, 1), dtype=torch.double),
    )
    monkeypatch.setattr(
        multistep_direct,
        "_make_gain_context",
        lambda *args, **kwargs: multistep_direct._GainContext(
            evaluate=lambda node, inputs: torch.ones(1, dtype=torch.double),
            num_branches=1,
            num_references=1,
        ),
    )

    action, diagnostics = multistep_direct.recommend_direct(
        _fake_model(_make_dag()),
        _make_plans(),
        (),
        policy=_make_policy(),
    )

    assert action.node == "A"
    assert diagnostics["selected_context_index"] == 0
    assert diagnostics["tie_count"] == 2


def test_direct_random_stage_uses_sampling_seed():
    """The random stage derives its generator from the profile's sampling seed."""

    def first_input(seed: int) -> float:
        policy = DirectPartialKnowledgeGradientFN(
            num_fantasies=2,
            search_profile=TerminalSearchProfile(
                num_network_samples=1,
                sampling_seed=seed,
                raw_samples=1,
                num_restarts=1,
                tolerance=1e-9,
            ),
        )
        campaign = MultiStepCampaign.from_dag(
            _make_dag(),
            acquisition_function=policy,
            recommender=TwoPhaseMetaRecommender(switch_after=1),
        )
        return campaign.recommend().iloc[0]["a_input"]

    assert first_input(3) == first_input(3)
    assert first_input(3) != first_input(4)
