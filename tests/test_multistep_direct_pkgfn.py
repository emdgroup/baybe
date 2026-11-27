"""Focused tests for the direct finite-reference p-KGFN policy.

The tests cover the exact reduction arithmetic, reference construction, local
action encoding, deterministic local search, serialization, and small real
recommendation paths. They deliberately use small explicit profiles instead of
the released search defaults.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baybe.exceptions import IncompatibilityError
from baybe.multistep import (
    DirectPartialKnowledgeGradientFN,
    MultiStepDAG,
    MultiStepData,
    MultiStepNode,
    TerminalSearchProfile,
)
from baybe.multistep.acquisition import PartialKnowledgeGradientFN
from baybe.multistep.data import _ExecutablePlan
from baybe.multistep.kgfn import direct as multistep_direct
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.targets.numerical import NumericalTarget


def _profile(**overrides) -> TerminalSearchProfile:
    """Return a small explicit solver profile for tests."""
    values = {
        "num_network_samples": 4,
        "sampling_seed": 0,
        "raw_samples": 10,
        "num_restarts": 2,
        "tolerance": 1e-9,
    }
    return TerminalSearchProfile(**{**values, **overrides})


def test_terminal_search_profile_defaults_and_immutability():
    """The released profile defaults are exact and the value is immutable."""
    profile = TerminalSearchProfile()
    assert profile.num_network_samples == 64
    assert profile.sampling_seed == 0
    assert profile.raw_samples == 64
    assert profile.num_restarts == 10
    assert profile.tolerance == 1e-9
    with pytest.raises(AttributeError):
        profile.optimizer_seed
    with pytest.raises(AttributeError):
        profile.raw_samples = 3


def test_terminal_search_profile_rejects_fewer_raw_samples_than_restarts():
    """A profile cannot seed more restarts than it draws raw samples."""
    with pytest.raises(ValueError, match="raw_samples"):
        TerminalSearchProfile(raw_samples=1, num_restarts=2)


def test_terminal_search_profile_ignores_removed_fields():
    """Serialized payloads from the removed solver fields still deserialize."""
    payload = TerminalSearchProfile().to_dict()
    payload["optimizer_seed"] = 3
    payload["max_block_sweeps"] = 7
    assert TerminalSearchProfile.from_dict(payload) == TerminalSearchProfile()


def test_direct_policy_rejects_stock_botorch_use():
    """The direct policy must not silently fall back to a stock acquisition."""
    policy = DirectPartialKnowledgeGradientFN()
    with pytest.raises(IncompatibilityError, match="MultiStepCampaign"):
        policy.to_botorch(None, None, None, None)


def _version_dag() -> MultiStepDAG:
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
    )
    return MultiStepDAG(
        nodes=(node,),
        objective=NumericalTarget("y").to_objective(),
        product_free=True,
    )


def test_validate_context_rejects_botorch_before_016(monkeypatch):
    import botorch

    monkeypatch.setattr(botorch, "__version__", "0.15.1")
    with pytest.raises(IncompatibilityError, match="BoTorch >= 0.16"):
        DirectPartialKnowledgeGradientFN().validate_context(
            _version_dag(), MultiStepData.empty(), 1
        )


def test_validate_context_accepts_botorch_016(monkeypatch):
    import botorch

    monkeypatch.setattr(botorch, "__version__", "0.16.0")
    DirectPartialKnowledgeGradientFN().validate_context(
        _version_dag(), MultiStepData.empty(), 1
    )


def _reference_dag() -> MultiStepDAG:
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(
            NumericalContinuousParameter("x", (0.0, 1.0)),
            NumericalDiscreteParameter("d", (0.0, 2.0)),
            CategoricalParameter("c", ("a", "b")),
            CategoricalParameter("fixed", ("u", "v"), active_values=("v",)),
        ),
    )
    return MultiStepDAG(
        nodes=(node,),
        objective=NumericalTarget("y").to_objective(),
        product_free=True,
    )


class _ReferenceModel:
    def __init__(self, dag: MultiStepDAG) -> None:
        self.dag = dag

    def parameters(self):
        return iter((torch.zeros(1, dtype=torch.double),))


def test_reference_points_are_reproducible_product_free_and_finite(monkeypatch):
    dag = _reference_dag()
    profile = TerminalSearchProfile(
        num_network_samples=1,
        sampling_seed=3,
        raw_samples=8,
        num_restarts=2,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("The reference sampling built a parameter product.")

    monkeypatch.setattr(SearchSpace, "from_product", forbidden)
    monkeypatch.setattr(SubspaceContinuous, "from_product", forbidden)
    monkeypatch.setattr(SubspaceDiscrete, "from_product", forbidden)

    first = multistep_direct._make_reference_points(_ReferenceModel(dag), profile)
    second = multistep_direct._make_reference_points(_ReferenceModel(dag), profile)
    other = multistep_direct._make_reference_points(
        _ReferenceModel(dag),
        TerminalSearchProfile(
            num_network_samples=1,
            sampling_seed=4,
            raw_samples=8,
            num_restarts=2,
        ),
    )

    assert first.shape == (8, 1, len(dag.comp_rep_columns))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.isfinite(first).all()
    assert (first[..., 0, 0] >= 0.0).all()
    assert (first[..., 0, 0] <= 1.0).all()


def test_reference_points_encode_active_discrete_levels():
    dag = _reference_dag()
    points = multistep_direct._make_reference_points(
        _ReferenceModel(dag),
        TerminalSearchProfile(
            num_network_samples=1,
            sampling_seed=0,
            raw_samples=16,
            num_restarts=2,
        ),
    )
    layout = dag._factorized_layout
    columns = layout.comp_rep_columns
    d_index = columns.index("d")
    c_indices = [index for index, name in enumerate(columns) if name.startswith("c_")]
    fixed_index = columns.index("fixed_v")

    assert set(points[..., 0, d_index].tolist()) <= {0.0, 2.0}
    assert set(points[..., 0, fixed_index].tolist()) == {1.0}
    for row in points[..., 0, :]:
        assert row[c_indices].sum().item() == pytest.approx(1.0)


def _action_node() -> MultiStepNode:
    return MultiStepNode(
        name="leaf",
        output="y",
        inputs=(
            NumericalContinuousParameter("x", (0.0, 1.0)),
            NumericalDiscreteParameter("d", (0.0, 2.0)),
            CategoricalParameter("c", ("a", "b")),
            CategoricalParameter("fixed", ("u", "v"), active_values=("v",)),
        ),
    )


class _ActionModel:
    def __init__(self, node: MultiStepNode) -> None:
        self.node = node
        self.layout = node._own_parameter_layout

    def transform_node_inputs(self, name, frame):
        comp = self.layout.transform(frame)
        return torch.as_tensor(
            comp.to_numpy(dtype=float, copy=True), dtype=torch.double
        )

    def get_node_model_columns(self, name):
        return self.layout.comp_rep_columns


def test_action_search_space_encodes_only_free_own_parameters():
    node = _action_node()
    layout = node._own_parameter_layout
    space = multistep_direct._ActionSearchSpace.from_layout(layout)

    assert space.dimension == 3
    assert space.continuous_dimension == 1
    assert space.cat_dims == {1: [0.0, 1.0]}
    assert space.discrete_dims == {2: [0.0, 1.0]}
    np.testing.assert_allclose(
        space.bounds,
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    )

    point = torch.tensor([[0.25, 0.0, 1.0]], dtype=torch.double)
    assert space.decode_inputs(point, {}) == [
        {"c": "a", "d": 2.0, "fixed": "v", "x": 0.25}
    ]


def test_action_search_space_lists_all_discrete_levels():
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(NumericalDiscreteParameter("d", (0.0, 1.0, 2.0, 3.0)),),
    )
    space = multistep_direct._ActionSearchSpace.from_layout(
        node._own_parameter_layout
    )

    assert space.discrete_dims == {0: [0.0, 1.0, 2.0, 3.0]}


def test_action_search_space_lists_all_categorical_levels():
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(CategoricalParameter("c", ("a", "b", "c")),),
    )
    space = multistep_direct._ActionSearchSpace.from_layout(
        node._own_parameter_layout
    )

    assert space.cat_dims == {0: [0.0, 1.0, 2.0]}


def test_decode_inputs_clamps_continuous_values_to_bounds():
    node = _action_node()
    space = multistep_direct._ActionSearchSpace.from_layout(
        node._own_parameter_layout
    )
    point = torch.tensor([[-1e-3, 0.0, 1.0], [1.0 + 1e-3, 1.0, 0.0]])

    assert space.decode_inputs(point, {}) == [
        {"c": "a", "d": 2.0, "fixed": "v", "x": 0.0},
        {"c": "b", "d": 0.0, "fixed": "v", "x": 1.0},
    ]


def test_action_comp_rep_is_one_hot_and_preserves_continuous_gradients():
    node = _action_node()
    space = multistep_direct._ActionSearchSpace.from_layout(
        node._own_parameter_layout
    )
    point = torch.tensor([[0.25, 0.0, 1.0]], dtype=torch.double)
    point.requires_grad_()

    model = _ActionModel(node)
    comp = multistep_direct._action_comp_rep_batch(
        model,
        node,
        _ExecutablePlan(experiment_id="exp-1", node="leaf", is_new=True),
        space,
        point,
    )
    columns = list(model.get_node_model_columns("leaf"))
    assert comp.shape == (1, len(columns))
    assert comp[0, columns.index("c_a")].item() == 1.0
    assert comp[0, columns.index("c_b")].item() == 0.0
    assert comp[0, columns.index("d")].item() == 2.0
    assert comp[0, columns.index("fixed_v")].item() == 1.0

    comp[0, columns.index("x")].backward()
    assert point.grad is not None
    assert point.grad[0, 0].item() == pytest.approx(1.0)


def test_action_without_free_parameters_has_zero_dimension():
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(CategoricalParameter("fixed", ("u", "v"), active_values=("v",)),),
    )
    space = multistep_direct._ActionSearchSpace.from_layout(
        node._own_parameter_layout
    )

    assert space.dimension == 0
    assert space.continuous_dimension == 0
    point = torch.empty((2, 0), dtype=torch.double)
    assert space.decode_inputs(point, {}) == [
        {"fixed": "v"},
        {"fixed": "v"},
    ]


def test_action_score_control_example_with_p_scaling():
    means = torch.tensor([[1.0, 4.0], [2.0, 0.0]], dtype=torch.double)
    gain = PartialKnowledgeGradientFN._compute_pkgfn_gain(
        means, torch.tensor(1.0, dtype=torch.double)
    )
    kg, score = multistep_direct._action_score(
        gain, torch.tensor(0.5, dtype=torch.double), 2.0
    )

    assert gain.item() == pytest.approx(2.0, abs=1e-12)
    assert kg.item() == pytest.approx(1.0, abs=1e-12)
    assert score.item() == pytest.approx(0.5, abs=1e-12)


def test_action_score_negative_gain_is_zero():
    means = torch.tensor([[0.0, 0.5], [0.2, 0.1]], dtype=torch.double)
    gain = PartialKnowledgeGradientFN._compute_pkgfn_gain(
        means, torch.tensor(1.0, dtype=torch.double)
    )
    kg, score = multistep_direct._action_score(
        gain, torch.tensor(0.5, dtype=torch.double), 2.0
    )

    assert gain.item() == 0.0
    assert kg.item() == 0.0
    assert score.item() == 0.0


def test_action_score_rejects_invalid_inputs():
    gain = torch.tensor([1.0], dtype=torch.double)
    with pytest.raises(IncompatibilityError, match="probability"):
        multistep_direct._action_score(
            gain, torch.tensor([1.5], dtype=torch.double), 1.0
        )
    with pytest.raises(IncompatibilityError, match="cost"):
        multistep_direct._action_score(
            gain, torch.tensor([0.5], dtype=torch.double), 0.0
        )
    with pytest.raises(IncompatibilityError, match="nonnegative"):
        multistep_direct._action_score(
            -gain, torch.tensor([0.5], dtype=torch.double), 1.0
        )


class _OptimizationModel(_ActionModel):
    def __init__(self, node: MultiStepNode) -> None:
        super().__init__(node)
        self.dag = SimpleNamespace(
            product_free=True,
            node_by_name={"leaf": node},
        )
        self._node_model = SimpleNamespace(
            batch_shape=torch.Size(),
            condition_on_observations=lambda *args, **kwargs: None,
        )

    def parameters(self):
        return iter((torch.zeros(1, dtype=torch.double),))

    def get_node_model(self, name):
        return self._node_model

    def _predict_feasibility_tensor(self, name, inputs):
        return torch.ones(inputs.shape[:-2], dtype=torch.double)


def _optimize_test_action(
    model,
    gain_context,
    *,
    threshold: float = 1e-6,
    cost: float = 1.0,
    profile: TerminalSearchProfile | None = None,
):
    return multistep_direct._optimize_action(
        model=model,
        plan=_ExecutablePlan(experiment_id="exp-1", node="leaf", is_new=True),
        gain_context=gain_context,
        profile=profile
        or TerminalSearchProfile(
            num_network_samples=1,
            sampling_seed=0,
            raw_samples=8,
            num_restarts=2,
            tolerance=1e-6,
        ),
        feasibility_threshold=threshold,
        cost=cost,
        pending=(),
        context_index=0,
    )


def test_optimize_action_finds_mixed_local_optimum():
    node = _action_node()
    model = _OptimizationModel(node)
    layout = node._own_parameter_layout
    columns = list(layout.comp_rep_columns)
    x_index = columns.index("x")
    d_index = columns.index("d")
    c_a_index = columns.index("c_a")

    def analytic_gain(node_arg, inputs):
        comp = inputs[..., 0, :]
        x = comp[..., x_index]
        d = comp[..., d_index] / 2.0
        c_a = comp[..., c_a_index]
        return 1.0 - (x - 0.3) ** 2 + 0.5 * d + 0.25 * c_a

    context = multistep_direct._GainContext(
        evaluate=analytic_gain, num_branches=1, num_references=1
    )
    action, score, diagnostics = _optimize_test_action(model, context)

    assert action.node == "leaf"
    assert action.inputs["x"] == pytest.approx(0.3, abs=1e-2)
    assert action.inputs["d"] == 2.0
    assert action.inputs["c"] == "a"
    assert score > 0.0
    assert diagnostics["references"] == 1
    assert diagnostics["branches"] == 1
    assert np.isfinite(diagnostics["best_value"])


def test_optimize_action_reaches_middle_numerical_discrete_level():
    """A four-level discrete parameter can select an interior level."""
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(
            NumericalContinuousParameter("x", (0.0, 1.0)),
            NumericalDiscreteParameter("d", (0.0, 1.0, 2.0, 3.0)),
        ),
    )
    model = _OptimizationModel(node)
    columns = list(node._own_parameter_layout.comp_rep_columns)
    x_index = columns.index("x")
    d_index = columns.index("d")

    def analytic_gain(node_arg, inputs):
        comp = inputs[..., 0, :]
        raw = (
            1.0
            - (comp[..., d_index] - 1.0) ** 2
            - (comp[..., x_index] - 0.5) ** 2
        )
        return torch.clamp(raw, min=0.0)

    context = multistep_direct._GainContext(
        evaluate=analytic_gain, num_branches=1, num_references=1
    )
    action, score, _ = _optimize_test_action(model, context)

    assert action.inputs["d"] == 1.0
    assert action.inputs["x"] == pytest.approx(0.5, abs=1e-2)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_optimize_action_reaches_middle_categorical_level():
    """A three-level categorical parameter can select an interior label."""
    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(
            NumericalContinuousParameter("x", (0.0, 1.0)),
            CategoricalParameter("c", ("a", "b", "c")),
        ),
    )
    model = _OptimizationModel(node)
    columns = list(node._own_parameter_layout.comp_rep_columns)
    x_index = columns.index("x")
    c_b_index = columns.index("c_b")

    def analytic_gain(node_arg, inputs):
        comp = inputs[..., 0, :]
        raw = comp[..., c_b_index] - (comp[..., x_index] - 0.5) ** 2
        return torch.clamp(raw, min=0.0)

    context = multistep_direct._GainContext(
        evaluate=analytic_gain, num_branches=1, num_references=1
    )
    action, score, _ = _optimize_test_action(model, context)

    assert action.inputs["c"] == "b"
    assert action.inputs["x"] == pytest.approx(0.5, abs=1e-2)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_optimize_action_skips_botorch_without_free_dimensions(monkeypatch):
    import botorch.optim

    node = MultiStepNode(
        name="leaf",
        output="y",
        inputs=(CategoricalParameter("fixed", ("u", "v"), active_values=("v",)),),
    )
    model = _OptimizationModel(node)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("BoTorch was called for a dimension-free action.")

    monkeypatch.setattr(botorch.optim, "optimize_acqf", forbidden)
    monkeypatch.setattr(botorch.optim, "optimize_acqf_mixed_alternating", forbidden)
    context = multistep_direct._GainContext(
        evaluate=lambda node_arg, inputs: torch.ones(
            inputs.shape[:-2], dtype=torch.double
        ),
        num_branches=1,
        num_references=1,
    )
    action, score, _ = _optimize_test_action(
        model,
        context,
        profile=TerminalSearchProfile(
            num_network_samples=1, raw_samples=2, num_restarts=1
        ),
    )

    assert action.inputs == {"fixed": "v"}
    assert score == pytest.approx(1.0)


def test_action_scores_return_sentinel_for_excluded_points():
    node = _action_node()
    model = _OptimizationModel(node)
    space = multistep_direct._ActionSearchSpace.from_layout(
        node._own_parameter_layout
    )
    columns = list(node._own_parameter_layout.comp_rep_columns)
    x_index = columns.index("x")
    context = multistep_direct._GainContext(
        evaluate=lambda node_arg, inputs: torch.ones(
            inputs.shape[:-2], dtype=torch.double
        ),
        num_branches=1,
        num_references=1,
    )
    model._predict_feasibility_tensor = (
        lambda name, inputs: (inputs[..., 0, x_index] > 0.25).to(torch.double)
    )
    scores = multistep_direct._action_scores(
        torch.tensor([[0.1, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=torch.double),
        model=model,
        node=node,
        plan=_ExecutablePlan(experiment_id="exp-1", node="leaf", is_new=True),
        space=space,
        gain_context=context,
        feasibility_threshold=1e-6,
        cost=1.0,
        pending=(),
    )

    assert scores[0].item() == -1.0
    assert scores[1].item() == pytest.approx(1.0)


def test_optimize_action_rejects_excluded_optimizer_result(monkeypatch):
    import botorch.optim

    node = _action_node()
    model = _OptimizationModel(node)
    model._predict_feasibility_tensor = (
        lambda name, inputs: torch.zeros(inputs.shape[:-2], dtype=torch.double)
    )
    context = multistep_direct._GainContext(
        evaluate=lambda node_arg, inputs: torch.ones(
            inputs.shape[:-2], dtype=torch.double
        ),
        num_branches=1,
        num_references=1,
    )
    monkeypatch.setattr(
        botorch.optim,
        "optimize_acqf_mixed_alternating",
        lambda *args, **kwargs: (
            torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.double),
            torch.tensor([-1.0], dtype=torch.double),
        ),
    )
    with pytest.raises(IncompatibilityError, match="not proof"):
        _optimize_test_action(model, context)


def test_optimize_action_fails_closed_on_optimization_warning(monkeypatch):
    import warnings

    import botorch.optim
    from botorch.exceptions.warnings import OptimizationWarning

    node = _action_node()
    model = _OptimizationModel(node)

    def optimizer(*args, **kwargs):
        warnings.warn("bad local step", OptimizationWarning)
        return (
            torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.double),
            torch.tensor([0.5], dtype=torch.double),
        )

    monkeypatch.setattr(botorch.optim, "optimize_acqf_mixed_alternating", optimizer)
    context = multistep_direct._GainContext(
        evaluate=lambda node_arg, inputs: torch.ones(
            inputs.shape[:-2], dtype=torch.double
        ),
        num_branches=1,
        num_references=1,
    )
    with pytest.raises(IncompatibilityError, match="optimization warning"):
        _optimize_test_action(model, context)


def test_optimize_action_ignores_bad_initial_candidates_warning(monkeypatch):
    import warnings

    import botorch.optim
    from botorch.exceptions.warnings import BadInitialCandidatesWarning

    node = _action_node()
    model = _OptimizationModel(node)

    def optimizer(*args, **kwargs):
        warnings.warn("bad initial candidates", BadInitialCandidatesWarning)
        return (
            torch.tensor([[0.5, 0.0, 1.0]], dtype=torch.double),
            torch.tensor([0.5], dtype=torch.double),
        )

    monkeypatch.setattr(botorch.optim, "optimize_acqf_mixed_alternating", optimizer)
    context = multistep_direct._GainContext(
        evaluate=lambda node_arg, inputs: torch.ones(
            inputs.shape[:-2], dtype=torch.double
        ),
        num_branches=1,
        num_references=1,
    )
    action, score, _ = _optimize_test_action(model, context)
    # The candidate is re-evaluated after the search; the fake gain is 1.0.
    assert score == pytest.approx(1.0)


def test_optimize_action_rejects_nonfinite_probability():
    node = _action_node()
    model = _OptimizationModel(node)
    model._predict_feasibility_tensor = (
        lambda name, inputs: torch.full(
            inputs.shape[:-2], float("nan"), dtype=torch.double
        )
    )
    context = multistep_direct._GainContext(
        evaluate=lambda node_arg, inputs: torch.ones(
            inputs.shape[:-2], dtype=torch.double
        ),
        num_branches=1,
        num_references=1,
    )
    with pytest.raises(IncompatibilityError, match="not finite"):
        _optimize_test_action(model, context)
