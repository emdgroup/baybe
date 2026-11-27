"""Focused support and coherence tests for model-reference Thompson paths."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from baybe.exceptions import IncompatibilityError
from baybe.multistep import (
    MREI,
    ModelReferenceExpectedImprovement,
    MultiStepCampaign,
    MultiStepDAG,
    MultiStepModel,
    MultiStepNode,
)
from baybe.multistep._model_reference_ei import _make_network_thompson_draw
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.targets.numerical import NumericalTarget


def _make_path_model(node_model):
    """Make the smallest model registry needed by the support gate."""
    node = SimpleNamespace(name="root", output_names=("utility",))
    return SimpleNamespace(
        dag=SimpleNamespace(topological_order=("root",), node_by_name={"root": node}),
        get_node_model=lambda _name: node_model,
    )


@pytest.fixture(name="single_node_model")
def fixture_single_node_model(single_node_dag):
    """Return a reusable fitted one-node GP model for path tests."""
    import torch
    from botorch.models import SingleTaskGP

    node_model = SingleTaskGP(
        train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
    )
    return MultiStepModel.from_node_models(single_node_dag, {"root": node_model})


@pytest.fixture(name="single_node_dag")
def fixture_single_node_dag():
    """Return a shared-shape, three-choice one-node DAG for path tests."""
    parameter = NumericalDiscreteParameter("x", (0.0, 0.5, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    return MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )


@pytest.fixture(name="make_single_node_campaign")
def fixture_make_single_node_campaign(single_node_dag):
    """Create a switched, one-node campaign with a selectable MREI policy."""

    def make_campaign(acquisition, *, cost_mode="remaining"):
        campaign = MultiStepCampaign.from_dag(
            single_node_dag,
            acquisition_function=acquisition,
            cost_mode=cost_mode,
            recommender=TwoPhaseMetaRecommender(switch_after=1, remain_switched=True),
        )
        campaign.recommender._has_switched = True
        return campaign

    return make_campaign


@pytest.mark.parametrize(
    ("kernel_name", "scaled"),
    [
        pytest.param("RBFKernel", False, id="rbf"),
        pytest.param("MaternKernel", False, id="matern"),
        pytest.param("RBFKernel", True, id="nested-scale-rbf"),
    ],
)
def test_path_support_accepts_scalar_gaussian_exact_gp(kernel_name, scaled):
    import torch
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

    kernel = {"RBFKernel": RBFKernel, "MaternKernel": MaternKernel}[kernel_name]()
    if scaled:
        kernel = ScaleKernel(ScaleKernel(kernel))
    node_model = SingleTaskGP(
        train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        covar_module=kernel,
    )

    MREI(reference="thompson_max").validate_model(_make_path_model(node_model))


def test_path_support_rejects_non_exact_gp_with_node_context():
    import torch

    model = _make_path_model(torch.nn.Linear(1, 1))

    with pytest.raises(IncompatibilityError, match="ExactGP.*root"):
        MREI(reference="thompson_max").validate_model(model)


@pytest.mark.parametrize(
    ("unsupported_model", "message"),
    [
        pytest.param(
            "non_gaussian", "Gaussian likelihood.*root.*utility", id="likelihood"
        ),
        pytest.param(
            "batched",
            "unbatched scalar ExactGP.*root.*utility.*batch shape",
            id="batch",
        ),
        pytest.param("kernel", "support only RBF.*root.*utility", id="kernel"),
    ],
)
def test_path_support_rejects_unsupported_likelihood_batch_and_kernel(
    unsupported_model, message
):
    import torch
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import LinearKernel
    from gpytorch.likelihoods import BernoulliLikelihood

    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    if unsupported_model == "batched":
        node_model = SingleTaskGP(train_x.unsqueeze(0), train_y.unsqueeze(0))
    elif unsupported_model == "kernel":
        node_model = SingleTaskGP(train_x, train_y, covar_module=LinearKernel())
    else:
        node_model = SingleTaskGP(train_x, train_y)
        node_model.likelihood = BernoulliLikelihood()

    with pytest.raises(IncompatibilityError, match=message):
        MREI(reference="thompson_max").validate_model(_make_path_model(node_model))


def test_discrete_support_limit_is_checked_before_full_searchspace_access():
    parameter = NumericalDiscreteParameter(
        "x", tuple(float(value) for value in range(4097))
    )

    class DagWithoutFullSearchspace:
        """Expose only inputs the cheap context check is allowed to inspect."""

        parameter_by_name = {"x": parameter}
        objective = NumericalTarget("utility").to_objective()
        nodes = ()

        @property
        def full_searchspace(self):
            pytest.fail("support limit must be checked before searchspace access")

    with pytest.raises(IncompatibilityError, match="4097 combinations"):
        MREI().validate_context(DagWithoutFullSearchspace(), object(), 1)


def test_context_rejects_batches_and_constrained_node_spaces():
    objective = NumericalTarget("utility").to_objective()
    dag = SimpleNamespace(
        objective=objective,
        parameter_by_name={},
        nodes=(SimpleNamespace(searchspace=SimpleNamespace(is_constrained=True)),),
    )

    with pytest.raises(IncompatibilityError, match="batch_size=1"):
        MREI().validate_context(dag, object(), 2)
    with pytest.raises(IncompatibilityError, match="constrained node search spaces"):
        MREI().validate_context(dag, object(), 1)


def test_thompson_draw_is_stable_across_reordered_and_chunked_inputs(
    single_node_model,
):
    import torch

    draw = _make_network_thompson_draw(
        MREI(n_path_features=64), single_node_model, call_seed=3
    )
    inputs = torch.tensor([[[0.0]], [[0.5]], [[1.0]]], dtype=torch.double)

    whole = draw(inputs)
    reordered = draw(inputs[[2, 0, 1]])
    chunked = torch.cat((draw(inputs[:1]), draw(inputs[1:])), dim=0)

    torch.testing.assert_close(reordered, whole[[2, 0, 1]], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(chunked, whole, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n_features", [1, 3])
def test_thompson_paths_accept_odd_positive_feature_counts(
    n_features, monkeypatch, single_node_model
):
    import torch
    from botorch.sampling.pathwise import posterior_samplers

    paths = []
    draw_matheron_paths = posterior_samplers.draw_matheron_paths

    def capture_path(*args, **kwargs):
        path = draw_matheron_paths(*args, **kwargs)
        paths.append(path)
        return path

    monkeypatch.setattr(posterior_samplers, "draw_matheron_paths", capture_path)
    draw = _make_network_thompson_draw(
        MREI(n_path_features=n_features), single_node_model, call_seed=2
    )

    utilities = draw(torch.tensor([[[0.5]]], dtype=torch.double))

    assert tuple(utilities.shape) == (1, 1)
    assert torch.isfinite(utilities).all()
    assert paths[0].paths["prior_paths"].weight.shape[-1] == n_features


def test_odd_rff_feature_map_adds_the_random_phase_once(monkeypatch):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.sampling.pathwise import prior_samplers

    parameter = NumericalContinuousParameter("x", (0.0, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    node_model = SingleTaskGP(
        train_X=torch.empty((0, 1), dtype=torch.double),
        train_Y=torch.empty((0, 1), dtype=torch.double),
        outcome_transform=None,
    )
    model = MultiStepModel.from_node_models(dag, {"root": node_model})
    feature_maps = []
    draw_kernel_feature_paths = prior_samplers.draw_kernel_feature_paths

    def capture_feature_map(gp, sample_shape, **kwargs):
        map_generator = kwargs["map_generator"]

        def record_map(**map_kwargs):
            feature_map = map_generator(**map_kwargs)
            feature_maps.append(feature_map)
            return feature_map

        return draw_kernel_feature_paths(
            gp, sample_shape, **(kwargs | {"map_generator": record_map})
        )

    monkeypatch.setattr(
        prior_samplers, "draw_kernel_feature_paths", capture_feature_map
    )
    draw = _make_network_thompson_draw(MREI(n_path_features=1), model, call_seed=2)
    inputs = torch.tensor([[[0.0]]], dtype=torch.double)
    draw(inputs)

    (feature_map,) = feature_maps
    (phase_transform,) = tuple(
        module
        for module in feature_map.output_transform.modules()
        if hasattr(module, "phase")
    )
    linear_phase = inputs.squeeze(-2) @ feature_map.weight.transpose(-2, -1)
    expected = phase_transform.scale * torch.cos(linear_phase + phase_transform.phase)
    torch.testing.assert_close(
        feature_map(inputs.squeeze(-2)), expected, rtol=1e-10, atol=1e-12
    )


def test_zero_row_thompson_uses_kernel_prior_without_matheron_update(monkeypatch):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.sampling.pathwise import posterior_samplers

    parameter = NumericalDiscreteParameter("x", (0.0, 0.5, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    node_model = SingleTaskGP(
        train_X=torch.empty((0, 1), dtype=torch.double),
        train_Y=torch.empty((0, 1), dtype=torch.double),
        outcome_transform=None,
    )
    model = MultiStepModel.from_node_models(dag, {"root": node_model})
    monkeypatch.setattr(
        posterior_samplers,
        "draw_matheron_paths",
        lambda *_args, **_kwargs: pytest.fail("zero-row GP must use its pure prior"),
    )

    draw = _make_network_thompson_draw(MREI(n_path_features=64), model, call_seed=2)
    utilities = draw(torch.tensor([[[0.5]]], dtype=torch.double))

    assert tuple(utilities.shape) == (1, 1)
    assert torch.isfinite(utilities).all()


def test_one_row_thompson_path_preserves_input_and_raw_outcome_transforms(
    monkeypatch,
):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.sampling.pathwise import posterior_samplers

    parameter = NumericalDiscreteParameter("x", (0.0, 1.0, 2.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    node_model = SingleTaskGP(
        train_X=torch.tensor([[0.0]], dtype=torch.double),
        train_Y=torch.tensor([[20.0]], dtype=torch.double),
        input_transform=Normalize(
            d=1, bounds=torch.tensor([[0.0], [2.0]], dtype=torch.double)
        ),
        outcome_transform=Standardize(m=1),
    )
    model = MultiStepModel.from_node_models(dag, {"root": node_model})
    matheron_paths = []
    draw_matheron_paths = posterior_samplers.draw_matheron_paths

    def capture_path(*args, **kwargs):
        path = draw_matheron_paths(*args, **kwargs)
        matheron_paths.append(path)
        return path

    monkeypatch.setattr(posterior_samplers, "draw_matheron_paths", capture_path)
    draw = _make_network_thompson_draw(MREI(n_path_features=64), model, call_seed=3)
    inputs = torch.tensor([[[1.0]]], dtype=torch.double)

    utilities = draw(inputs)
    path_values = matheron_paths[0](inputs.squeeze(-2))

    torch.testing.assert_close(utilities, path_values, rtol=1e-10, atol=1e-12)
    assert path_values.item() > 5.0

    untransformed_model = SingleTaskGP(
        train_X=torch.tensor([[0.0]], dtype=torch.double),
        train_Y=torch.tensor([[0.0]], dtype=torch.double),
    )
    untransformed = MultiStepModel.from_node_models(dag, {"root": untransformed_model})
    normalized_input = torch.tensor([[[0.5]]], dtype=torch.double)
    normalized_draw = _make_network_thompson_draw(
        MREI(n_path_features=64), untransformed, call_seed=3
    )(normalized_input)

    torch.testing.assert_close(
        utilities,
        normalized_draw + 20.0,
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        node_model.input_transform(inputs.squeeze(-2)),
        normalized_input.squeeze(-2),
        rtol=1e-10,
        atol=1e-12,
    )
    assert node_model.outcome_transform.means.item() == pytest.approx(20.0)
    assert node_model.outcome_transform.stdvs.item() == pytest.approx(1.0)


def test_thompson_dag_reuses_shared_ancestor_across_fork_and_join(monkeypatch):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.sampling.pathwise import posterior_samplers

    x = NumericalDiscreteParameter("x", (0.0, 1.0))
    y = NumericalDiscreteParameter("y", (0.0, 1.0))
    z = NumericalDiscreteParameter("z", (0.0, 1.0))
    w = NumericalDiscreteParameter("w", (0.0, 1.0))
    root = MultiStepNode(name="root", output="root_value", inputs=(x,))
    left = MultiStepNode(name="left", output="left_value", inputs=(y,), parents=(root,))
    right = MultiStepNode(
        name="right", output="right_value", inputs=(z,), parents=(root,)
    )
    join = MultiStepNode(
        name="join", output="utility", inputs=(w,), parents=(left, right)
    )
    dag = MultiStepDAG(
        nodes=(root, left, right, join),
        objective=NumericalTarget("utility").to_objective(),
    )
    node_models = {
        "root": SingleTaskGP(
            train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        ),
        "left": SingleTaskGP(
            train_X=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        ),
        "right": SingleTaskGP(
            train_X=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        ),
        "join": SingleTaskGP(
            train_X=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double
            ),
            train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        ),
    }
    model = MultiStepModel.from_node_models(dag, node_models)
    sampled_models = []

    def fake_path_sampler(component, **_kwargs):
        sampled_models.append(component)
        if component is node_models["root"]:
            return lambda inputs: inputs[..., 0] + 1.0
        if component is node_models["left"]:
            return lambda inputs: inputs[..., -1] + 1.0
        if component is node_models["right"]:
            return lambda inputs: inputs[..., -1] + 2.0
        return lambda inputs: inputs[..., 0] + inputs[..., -2] + inputs[..., -1]

    monkeypatch.setattr(posterior_samplers, "draw_matheron_paths", fake_path_sampler)
    draw = _make_network_thompson_draw(MREI(n_path_features=64), model, call_seed=0)
    inputs = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0]]], dtype=torch.double
    )

    utilities = draw(inputs)

    torch.testing.assert_close(
        utilities[:, 0],
        torch.tensor([5.0, 8.0], dtype=torch.double),
        rtol=1e-10,
        atol=1e-12,
    )
    assert sampled_models == list(node_models.values())


@pytest.mark.parametrize(
    "parent_output_names",
    [
        pytest.param(("parent_value",), id="scalar-parent"),
        pytest.param(("second", "first"), id="declared-multi-output-order"),
    ],
)
def test_snapshot_thompson_paths_match_explicit_values_and_gradients(
    monkeypatch, parent_output_names
):
    import torch
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.sampling.pathwise import posterior_samplers

    x = NumericalContinuousParameter("x", (0.0, 1.0))
    y = NumericalContinuousParameter("y", (0.0, 1.0))
    parent_output = (
        parent_output_names[0] if len(parent_output_names) == 1 else parent_output_names
    )
    parent = MultiStepNode(name="parent", output=parent_output, inputs=(x,))
    child = MultiStepNode(
        name="child", output="utility", inputs=(y,), parents=(parent,)
    )
    dag = MultiStepDAG(
        nodes=(parent, child), objective=NumericalTarget("utility").to_objective()
    )
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    parent_components = tuple(
        SingleTaskGP(train_x, train_x * (index + 1))
        for index in range(len(parent_output_names))
    )
    parent_model = (
        parent_components[0]
        if len(parent_components) == 1
        else ModelListGP(*parent_components)
    )
    child_train_x = torch.tensor(
        [
            [0.0] * (len(parent_output_names) + 1),
            [1.0] * (len(parent_output_names) + 1),
        ],
        dtype=torch.double,
    )
    child_model = SingleTaskGP(
        child_train_x, torch.tensor([[0.0], [1.0]], dtype=torch.double)
    )
    model = MultiStepModel.from_node_models(
        dag, {"parent": parent_model, "child": child_model}
    )
    original_models = model._node_models.copy()
    path_by_component = {}
    path_factory_calls = []
    draw_matheron_paths = posterior_samplers.draw_matheron_paths

    def capture_path(component, **kwargs):
        path_factory_calls.append(component)
        path = draw_matheron_paths(component, **kwargs)
        path_by_component[id(component)] = path
        return path

    monkeypatch.setattr(posterior_samplers, "draw_matheron_paths", capture_path)
    draw = _make_network_thompson_draw(MREI(n_path_features=32), model, call_seed=6)

    parent_paths = tuple(
        path_by_component[id(component)] for component in parent_components
    )
    child_path = path_by_component[id(child_model)]

    inputs = torch.tensor(
        [[[0.2, 0.4]], [[0.8, 0.7]]], dtype=torch.double, requires_grad=True
    )
    global_indices = {
        column: index
        for index, column in enumerate(dag.full_searchspace.comp_rep_columns)
    }
    parent_inputs = inputs[..., global_indices["x"]].unsqueeze(-1)
    parent_values = torch.cat(
        [path(parent_inputs).unsqueeze(-1) for path in parent_paths], dim=-1
    )
    features = {
        **{
            name: parent_values[..., index : index + 1]
            for index, name in enumerate(parent_output_names)
        },
        "y": inputs[..., global_indices["y"]].unsqueeze(-1),
    }
    child_inputs = torch.cat(
        [features[column] for column in model.get_node_model_columns("child")],
        dim=-1,
    )
    expected = child_path(child_inputs)
    actual = draw(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    expected_gradient = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)[
        0
    ]
    actual_gradient = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12
    )
    assert path_factory_calls == [*parent_components, child_model]
    assert all(
        model._node_models[name] is node_model
        for name, node_model in original_models.items()
    )


def test_normal_torch_seed_reproduces_recommendation_draw_sequences(single_node_model):
    import torch

    from baybe.multistep._model_reference_ei import _draw_model_reference_seed

    inputs = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.double)
    acquisition = MREI(n_path_features=64, sampling_seed=173)

    def draw_sequence():
        """Draw two fresh TS paths using consecutive ordinary Torch seeds."""
        seeds = tuple(_draw_model_reference_seed() for _ in range(2))
        paths = tuple(
            _make_network_thompson_draw(acquisition, single_node_model, call_seed=seed)(
                inputs
            )
            for seed in seeds
        )
        return seeds, paths

    torch.manual_seed(173)
    call_seeds, (first, following) = draw_sequence()
    torch.manual_seed(173)
    repeated_seeds, (repeated_first, repeated_following) = draw_sequence()

    assert call_seeds == repeated_seeds
    torch.testing.assert_close(first, repeated_first, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(following, repeated_following, rtol=1e-10, atol=1e-12)
    assert torch.max(torch.abs(first - following)).item() > 1e-8


def test_actual_discrete_optimizer_maximizes_ts_value_seven_not_mean_three(
    monkeypatch,
):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf_discrete

    from baybe.multistep._model_reference_ei import (
        _compute_model_reference,
        _make_reference_acquisition,
    )

    parameter = NumericalDiscreteParameter("x", (0.0, 0.5, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    node_model = SingleTaskGP(
        train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        train_Y=torch.tensor([[2.0], [4.0]], dtype=torch.double),
    )
    model = MultiStepModel.from_node_models(dag, {"root": node_model})
    maximizing_input = torch.tensor([[0.5]], dtype=torch.double)
    torch.testing.assert_close(
        node_model.posterior(maximizing_input).mean,
        torch.tensor([[3.0]], dtype=torch.double),
        rtol=1e-10,
        atol=1e-12,
    )

    def fixed_thompson_path(_acquisition, _model, *, call_seed):
        assert call_seed == 5

        def evaluate(inputs):
            coordinate = inputs[..., 0]
            return 3.0 + 16.0 * coordinate * (1.0 - coordinate)

        return evaluate

    monkeypatch.setattr(
        "baybe.multistep._model_reference_ei._make_network_thompson_draw",
        fixed_thompson_path,
    )
    acquisition = MREI(reference="thompson_max", n_network_samples=2)
    reference = _compute_model_reference(acquisition, model, call_seed=5)
    choices_frame = dag.full_searchspace.discrete.exp_rep
    choices = torch.as_tensor(
        dag.full_searchspace.discrete.transform(choices_frame).to_numpy().copy(),
        dtype=torch.double,
    )
    reference_acquisition = _make_reference_acquisition(
        model,
        lambda inputs: fixed_thompson_path(acquisition, model, call_seed=5)(
            inputs
        ).squeeze(-1),
    )
    candidate, _ = optimize_acqf_discrete(
        acq_function=reference_acquisition, q=1, choices=choices
    )
    repeated_candidate, _ = optimize_acqf_discrete(
        acq_function=reference_acquisition, q=1, choices=choices
    )

    assert reference == pytest.approx(7.0)
    torch.testing.assert_close(
        candidate, torch.tensor([[0.5]], dtype=torch.double), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(repeated_candidate, candidate, atol=1e-7, rtol=0)


def test_real_mean_and_ei_network_draws_use_separate_fresh_streams(
    monkeypatch, single_node_model
):
    import torch

    from baybe.multistep._model_reference_ei import _derive_acquisition_seed
    from baybe.multistep.model import _derive_seed

    model = single_node_model
    acquisition = MREI(
        reference="mean",
        n_network_samples=4,
        n_reference_points=6,
        sampling_seed=71,
    )
    observed_seeds = []
    draw_network_samples = model.draw_network_samples

    def capture_real_draws(inputs, n_paths, *, sampling_seed=None):
        observed_seeds.append(sampling_seed)
        return draw_network_samples(inputs, n_paths, sampling_seed=sampling_seed)

    monkeypatch.setattr(model, "draw_network_samples", capture_real_draws)
    candidate = torch.tensor([[[0.5]]], dtype=torch.double)
    stream_groups = []
    for call_seed in (17, 29):
        (branch,) = acquisition.make_botorch_branches(model, (), call_seed=call_seed)
        value = branch(candidate)
        assert torch.isfinite(value).all()
        mean_seed = _derive_acquisition_seed(
            acquisition, call_seed, "mean-reference-network-samples"
        )
        ei_seed = _derive_seed(
            acquisition.sampling_seed + call_seed,
            "model-reference-ei",
            "independent-ei-samples-0",
        )
        ts_seed = _derive_acquisition_seed(
            acquisition,
            call_seed,
            "thompson-path-0",
            node_name="root",
        )
        assert observed_seeds[-2:] == [mean_seed, ei_seed]
        assert len({mean_seed, ei_seed, ts_seed}) == 3
        stream_groups.append((mean_seed, ei_seed, ts_seed))

    assert len(set(stream_groups[0]) | set(stream_groups[1])) == 6


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        pytest.param("max_mean", 6.5, id="maximum-of-final-means"),
        pytest.param("mean", 49.0 / 6.0, id="uniform-mean-of-final-means"),
    ],
)
def test_reference_branch_uses_global_final_utility_reference(
    monkeypatch, reference, expected, single_node_model
):
    import torch

    model = single_node_model

    def fake_draws(inputs, n_paths, *, sampling_seed=None):
        """Return controlled utility-reference and independent EI draws."""
        if inputs.shape[-3] == 3:
            utilities = inputs.new_tensor([1.0, 4.0, 2.0]).reshape(1, 3, 1, 1)
            return utilities.expand(n_paths, -1, -1, -1)
        return inputs.new_tensor([9.0, 12.0]).reshape(2, 1, 1, 1)

    monkeypatch.setattr(model, "draw_network_samples", fake_draws)
    acquisition = MREI(
        reference=reference,
        n_network_samples=2,
        n_reference_points=3,
    )
    (branch,) = acquisition.make_botorch_branches(model, (), call_seed=0)

    value = branch(torch.tensor([[[0.0]]], dtype=torch.double))
    assert value.item() == pytest.approx(expected)


def test_uniform_mean_reference_samples_declared_discrete_categories_equally():
    import torch

    from baybe.multistep._model_reference_ei import _uniform_global_configurations

    stage = NumericalDiscreteParameter("stage", (0.0, 1.0, 9.0))
    color = CategoricalParameter("color", ("red", "green", "blue"))
    temperature = NumericalContinuousParameter("temperature", (-2.0, 4.0))
    node = MultiStepNode(
        name="root", output="utility", inputs=(stage, color, temperature)
    )
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    searchspace = dag.full_searchspace
    points = _uniform_global_configurations(
        dag, 18000, torch.zeros(1, dtype=torch.double), seed=71
    ).squeeze(-2)
    discrete_rows = searchspace.discrete.transform(searchspace.discrete.exp_rep)
    discrete_indices = [
        searchspace.comp_rep_columns.index(column) for column in discrete_rows.columns
    ]
    declared_rows = torch.as_tensor(
        discrete_rows.to_numpy(),
        dtype=torch.double,
    )
    row_ids = torch.cdist(points[:, discrete_indices], declared_rows).argmin(dim=1)
    counts = torch.bincount(row_ids, minlength=len(declared_rows))
    temperature_index = searchspace.comp_rep_columns.index("temperature")
    bins = torch.histc(points[:, temperature_index], bins=3, min=-2.0, max=4.0)

    assert len(declared_rows) == 9
    assert int(counts.max() - counts.min()) < 180
    assert int(bins.max() - bins.min()) < 180


def test_pending_ei_branches_keep_their_own_models_and_random_streams(
    monkeypatch, single_node_model
):
    import torch

    from baybe.multistep import _model_reference_ei as model_reference_ei

    model = single_node_model
    dag = model.dag
    branch_models = (SimpleNamespace(dag=dag), SimpleNamespace(dag=dag))
    conditioned = []

    def condition(_pending, *, num_fantasies, sampling_seed):
        conditioned.append((num_fantasies, sampling_seed))
        return branch_models

    monkeypatch.setattr(model, "condition_on_pending", condition)
    reference_calls = []
    monkeypatch.setattr(
        model_reference_ei,
        "_compute_model_reference",
        lambda *_args, **_kwargs: reference_calls.append(True) or 0.0,
    )
    draws = []

    def draw_samples(branch_model, inputs, *, n_samples, sampling_seed):
        draws.append((branch_model, sampling_seed))
        utility = 1.0 if branch_model is branch_models[0] else 3.0
        return inputs.new_full((n_samples, *inputs.shape[:-1]), utility)

    monkeypatch.setattr(model_reference_ei, "_draw_final_utility_samples", draw_samples)
    branches = MREI(n_network_samples=2).make_botorch_branches(
        model, (object(),), call_seed=4
    )
    inputs = torch.tensor([[[0.5]]], dtype=torch.double)

    values = [branch(inputs).item() for branch in branches]

    assert values == pytest.approx([1.0, 3.0])
    assert len(conditioned) == 1
    assert reference_calls == [True]
    assert [model_ for model_, _seed in draws] == list(branch_models)
    assert len({seed for _model, seed in draws}) == 2


def test_mean_reference_does_not_require_thompson_path_support(
    monkeypatch, single_node_model
):
    from baybe.multistep import _model_reference_ei as model_reference_ei

    model = single_node_model
    monkeypatch.setattr(
        model, "condition_on_pending", lambda *_args, **_kwargs: (model,)
    )
    monkeypatch.setattr(
        model_reference_ei, "_compute_model_reference", lambda *_a, **_k: 0.0
    )
    monkeypatch.setattr(
        model_reference_ei,
        "_validate_path_models",
        lambda *_args: pytest.fail("mean reference must not build Thompson paths"),
    )

    branches = MREI(reference="mean").make_botorch_branches(model, (), call_seed=0)

    assert len(branches) == 1


def test_thompson_reference_maximizes_the_sampled_path_value(monkeypatch):
    import torch

    from baybe.multistep import _model_reference_ei as model_reference_ei

    def fake_path(_acquisition, _model, *, call_seed):
        assert call_seed == 5
        return lambda inputs: inputs[..., 0]

    def maximize(_model, score):
        candidates = torch.tensor([[[5.0]], [[7.0]], [[6.0]]], dtype=torch.double)
        assert score(candidates).tolist() == pytest.approx([5.0, 7.0, 6.0])
        return 7.0

    monkeypatch.setattr(model_reference_ei, "_make_network_thompson_draw", fake_path)
    monkeypatch.setattr(model_reference_ei, "_maximize_global_reference", maximize)

    value = model_reference_ei._compute_model_reference(
        MREI(reference="thompson_max"), object(), call_seed=5
    )

    assert value == 7.0


def test_thompson_reference_branch_uses_independent_ei_samples(
    monkeypatch, single_node_model
):
    import torch

    from baybe.multistep import _model_reference_ei as model_reference_ei

    model = single_node_model
    monkeypatch.setattr(
        model, "condition_on_pending", lambda *_args, **_kwargs: (model,)
    )
    monkeypatch.setattr(
        model_reference_ei,
        "_compute_model_reference",
        lambda acquisition, *_args, **_kwargs: (
            10.0 if acquisition.reference == "thompson_max" else pytest.fail()
        ),
    )

    def independent_ei_samples(_model, inputs, *, n_samples, sampling_seed):
        assert n_samples == 2
        return inputs.new_tensor([9.0, 12.0]).reshape(2, *inputs.shape[:-1])

    monkeypatch.setattr(
        model_reference_ei, "_draw_final_utility_samples", independent_ei_samples
    )
    (branch,) = MREI(
        reference="thompson_max", n_network_samples=2
    ).make_botorch_branches(model, (), call_seed=1)

    value = branch(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.item() == pytest.approx(1.0)


@pytest.mark.parametrize("reference", ["max_mean", "mean", "thompson_max"])
@pytest.mark.parametrize("cost_mode", ["remaining", "node"])
def test_campaign_recommends_from_priors_without_complete_measurements(
    reference, cost_mode, make_single_node_campaign
):
    """MREI uses its model reference and does not require real terminal outcomes."""
    campaign = make_single_node_campaign(
        MREI(
            reference=reference,
            n_network_samples=2,
            n_reference_points=4,
            n_path_features=64,
        ),
        cost_mode=cost_mode,
    )

    recommendation = campaign.recommend()

    assert len(recommendation) == 1
    assert recommendation.iloc[0]["node"] == "root"
    pending = campaign.pending_experiments
    assert len(pending) == 1
    assert pending.iloc[0]["node"] == "root"


@pytest.mark.parametrize("reference", ["max_mean", "mean", "thompson_max"])
@pytest.mark.parametrize("cost_mode", ["remaining", "node"])
def test_campaign_recommends_next_node_from_upstream_only_data(reference, cost_mode):
    a_input = NumericalDiscreteParameter("a_input", (0.0, 1.0))
    b_input = NumericalDiscreteParameter("b_input", (0.0, 1.0))
    node_a = MultiStepNode(name="A", output="a_output", inputs=(a_input,), cost=20.0)
    node_b = MultiStepNode(
        name="B", output="utility", inputs=(b_input,), parents=(node_a,), cost=3.0
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b),
        objective=NumericalTarget("utility").to_objective(),
    )
    campaign = MultiStepCampaign.from_dag(
        dag,
        acquisition_function=MREI(
            reference=reference,
            n_network_samples=2,
            n_reference_points=6,
            n_path_features=32,
        ),
        cost_mode=cost_mode,
        recommender=TwoPhaseMetaRecommender(switch_after=1, remain_switched=True),
    )
    campaign.recommender._has_switched = True
    root_action = campaign.recommend().iloc[0]
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": root_action["experiment_id"],
                    "node": "A",
                    "a_input": root_action["a_input"],
                    "a_output": 2.0,
                }
            ]
        )
    )

    assert campaign.data.complete_measurement_frame(campaign.dag).empty
    next_action = campaign.recommend()

    assert len(next_action) == 1
    assert next_action.iloc[0]["node"] == "B"


def test_campaign_deduplicates_repeated_pending_actions(make_single_node_campaign):
    campaign = make_single_node_campaign(
        MREI(reference="mean", n_network_samples=2, n_reference_points=8)
    )

    first = campaign.recommend().iloc[0]
    second = campaign.recommend().iloc[0]

    assert first["node"] == second["node"] == "root"
    assert first["x"] != second["x"]


def test_started_mrei_attempt_consumes_one_global_seed_before_fit_fails(
    monkeypatch, make_single_node_campaign
):
    import torch

    from baybe.multistep._model_reference_ei import _draw_model_reference_seed

    campaign = make_single_node_campaign(MREI())
    monkeypatch.setattr(
        campaign, "_fit_node_models", lambda: (_ for _ in ()).throw(RuntimeError("fit"))
    )
    expected_generator = torch.Generator().manual_seed(413)
    expected_seed = int(
        torch.randint(0, 2**32, (), generator=expected_generator).item()
    )
    expected_next = torch.randint(0, 2**32, (), generator=expected_generator)
    drawn_seeds = []

    def draw_seed():
        seed = _draw_model_reference_seed()
        drawn_seeds.append(seed)
        return seed

    monkeypatch.setattr(
        "baybe.multistep.campaign._draw_model_reference_seed", draw_seed
    )
    torch.manual_seed(413)

    with pytest.raises(RuntimeError, match="fit"):
        campaign.recommend()

    torch.testing.assert_close(torch.randint(0, 2**32, ()), expected_next)
    assert drawn_seeds == [expected_seed]


def test_mrei_failure_before_action_selection_does_not_reserve_partial_state(
    monkeypatch, make_single_node_campaign
):
    campaign = make_single_node_campaign(
        MREI(n_network_samples=2, n_reference_points=4)
    )
    contexts_before = campaign.data._contexts
    drawn_seeds = []

    def draw_seed():
        drawn_seeds.append(419)
        return 419

    monkeypatch.setattr(
        "baybe.multistep.campaign._draw_model_reference_seed", draw_seed
    )

    def fail_to_build_branches(*_args, **_kwargs):
        raise RuntimeError("branch construction failed")

    monkeypatch.setattr(
        ModelReferenceExpectedImprovement,
        "make_botorch_branches",
        fail_to_build_branches,
    )

    with pytest.raises(RuntimeError, match="branch construction failed"):
        campaign.recommend()

    assert campaign.data._contexts == contexts_before
    assert drawn_seeds == [419]


@pytest.mark.parametrize(
    ("cost_mode", "expected_score"),
    [("remaining", 0.4), ("node", 4.0 / 3.0)],
)
def test_mrei_stock_adapter_uses_next_node_probability_and_selected_cost(
    monkeypatch, cost_mode, expected_score
):
    import torch

    from baybe.multistep import acquisition as multistep_acquisition
    from baybe.multistep.data import register_action

    a_input = NumericalDiscreteParameter("a_input", (0.0, 1.0))
    b_input = NumericalDiscreteParameter("b_input", (0.0, 1.0))
    c_input = NumericalDiscreteParameter("c_input", (0.0, 1.0))
    node_a = MultiStepNode(name="A", output="a_output", inputs=(a_input,), cost=2.0)
    node_b = MultiStepNode(
        name="B", output="b_output", inputs=(b_input,), parents=(node_a,), cost=3.0
    )
    node_c = MultiStepNode(
        name="C", output="utility", inputs=(c_input,), parents=(node_b,), cost=7.0
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b, node_c),
        objective=NumericalTarget("utility").to_objective(),
    )
    campaign = MultiStepCampaign.from_dag(dag)
    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id="exp-1",
        node="A",
        inputs={"a_input": 0.0},
        parent_refs={},
        data=campaign.data,
        dag=dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": "exp-1",
                    "node": "A",
                    "a_input": 0.0,
                    "a_output": 1.0,
                }
            ]
        )
    )
    (plan,) = tuple(plan for plan in campaign._executable_plans() if plan.node == "B")
    assert plan.node == "B"
    assert plan.remaining_cost == 10.0

    model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter((torch.zeros(1, dtype=torch.double),)),
    )

    class ConstantEI:
        """Provide the raw sample-EI utility to the unmodified stock adapter."""

        def __init__(self, branch_model):
            self.model = branch_model

        def __call__(self, inputs):
            return inputs.new_full(inputs.shape[:-2], 5.0)

    probability_nodes = []

    def next_node_probability(_model, action_plan, inputs, _pending):
        probability_nodes.append(action_plan.node)
        return inputs.new_full(inputs.shape[:-2], 0.8)

    monkeypatch.setattr(
        multistep_acquisition,
        "_stock_action_probability",
        next_node_probability,
    )
    discrete_candidate = {
        parameter.name: parameter.values[0]
        for parameter in dag.full_searchspace.discrete.parameters
    }

    _candidate, score = multistep_acquisition._optimize_stock_context(
        model=model,
        plan=plan,
        discrete_candidate=discrete_candidate,
        fixed_parameters={},
        branches=(ConstantEI(model),),
        score_scale="linear",
        pending=(),
        feasibility_threshold=0.0,
        cost_mode=cost_mode,
    )

    assert score.item() == pytest.approx(expected_score)
    assert probability_nodes == ["B"]


def test_mrei_risk_weighting_changes_the_projected_discrete_action(monkeypatch):
    import torch

    from baybe.multistep import acquisition as multistep_acquisition
    from baybe.multistep.acquisition import recommend_stock
    from baybe.multistep.data import _ExecutablePlan

    parameter = NumericalDiscreteParameter("x", (0.0, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter((torch.zeros(1, dtype=torch.double),)),
    )

    class ControlledEI:
        """Give the riskier action more raw EI but less weighted value."""

        def __init__(self, model):
            self.model = model

        def __call__(self, inputs):
            return torch.where(inputs[..., 0] == 0.0, 10.0, 6.0).squeeze(-1)

    branch = ControlledEI(model)
    monkeypatch.setattr(
        ModelReferenceExpectedImprovement,
        "make_botorch_branches",
        lambda *_args, **_kwargs: (branch,),
    )
    monkeypatch.setattr(
        multistep_acquisition,
        "_stock_action_probability",
        lambda _model, _plan, inputs, _pending: torch.where(
            inputs[..., 0] == 0.0, 0.1, 1.0
        ).squeeze(-1),
    )
    plan = _ExecutablePlan(experiment_id="exp-1", node="root", is_new=True)

    (selected,) = recommend_stock(
        model,
        SimpleNamespace(),
        {},
        (plan,),
        (),
        acquisition_function=MREI(),
        score_scale="linear",
        batch_size=1,
    )

    assert selected.inputs["x"] == 1.0


@pytest.mark.parametrize("cost_mode", ["remaining", "node"])
def test_stock_recommendation_selects_between_competing_next_nodes(
    monkeypatch, cost_mode
):
    import torch

    from baybe.multistep import acquisition as multistep_acquisition
    from baybe.multistep.data import _ExecutablePlan

    a_input = NumericalDiscreteParameter("a_input", (0.0, 1.0))
    b_input = NumericalDiscreteParameter("b_input", (0.0, 1.0))
    c_input = NumericalDiscreteParameter("c_input", (0.0, 1.0))
    d_input = NumericalDiscreteParameter("d_input", (0.0, 1.0))
    node_a = MultiStepNode(name="A", output="a_output", inputs=(a_input,))
    node_b = MultiStepNode(
        name="B", output="b_output", inputs=(b_input,), parents=(node_a,), cost=3.0
    )
    node_c = MultiStepNode(
        name="C", output="c_output", inputs=(c_input,), parents=(node_a,), cost=5.0
    )
    node_d = MultiStepNode(
        name="D",
        output="utility",
        inputs=(d_input,),
        parents=(node_b, node_c),
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b, node_c, node_d),
        objective=NumericalTarget("utility").to_objective(),
    )
    model = SimpleNamespace(dag=dag)
    data = SimpleNamespace(
        complete_measurement_frame=lambda *_args, **_kwargs: pytest.fail(
            "MREI must not request complete real measurements"
        )
    )
    plans = (
        _ExecutablePlan(
            experiment_id="exp-B",
            node="B",
            inputs={"a_output": 1.0},
            remaining_cost=10.0,
            is_new=True,
        ),
        _ExecutablePlan(
            experiment_id="exp-C",
            node="C",
            inputs={"a_output": 1.0},
            remaining_cost=12.0,
            is_new=True,
        ),
    )
    candidate = {
        "a_input": 0.0,
        "b_input": 0.0,
        "c_input": 0.0,
        "d_input": 0.0,
    }
    weighted_scores = {
        "remaining": {"B": 0.25, "C": 1.0 / 3.0},
        "node": {"B": 5.0 / 6.0, "C": 0.8},
    }
    optimized_nodes = []

    monkeypatch.setattr(
        ModelReferenceExpectedImprovement,
        "make_botorch_branches",
        lambda *_args, **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        multistep_acquisition, "_stock_fixed_parameters", lambda *_args: {}
    )
    monkeypatch.setattr(
        multistep_acquisition,
        "_stock_discrete_candidates",
        lambda *_args: (candidate,),
    )

    def optimize(*, plan, **_kwargs):
        optimized_nodes.append(plan.node)
        return pd.Series(candidate), torch.tensor(weighted_scores[cost_mode][plan.node])

    monkeypatch.setattr(multistep_acquisition, "_optimize_stock_context", optimize)

    (selected,) = multistep_acquisition.recommend_stock(
        model,
        data,
        {},
        plans,
        (),
        acquisition_function=MREI(),
        score_scale="linear",
        batch_size=1,
        cost_mode=cost_mode,
        call_seed=71,
    )

    assert optimized_nodes == ["B", "C"]
    assert selected.node == ("C" if cost_mode == "remaining" else "B")


@pytest.mark.parametrize("cost_mode", ["remaining", "node"])
def test_real_mrei_stock_helper_selects_between_competing_nodes(cost_mode, request):
    import torch

    from baybe.multistep.acquisition import recommend_stock
    from baybe.multistep.data import register_action

    rng_state = torch.random.get_rng_state()
    request.addfinalizer(lambda: torch.random.set_rng_state(rng_state))
    torch.manual_seed(193)

    a_input = NumericalDiscreteParameter("a_input", (0.0, 1.0))
    d_input = NumericalContinuousParameter("d_input", (0.0, 1.0))
    node_a = MultiStepNode(name="A", output="a_output", inputs=(a_input,), cost=2.0)
    node_b = MultiStepNode(name="B", output="b_output", parents=(node_a,), cost=3.0)
    node_c = MultiStepNode(name="C", output="c_output", parents=(node_a,), cost=5.0)
    node_d = MultiStepNode(
        name="D",
        output="utility",
        inputs=(d_input,),
        parents=(node_b, node_c),
        cost=7.0,
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b, node_c, node_d),
        objective=NumericalTarget("utility").to_objective(),
    )
    acquisition = MREI(
        reference="mean",
        n_network_samples=2,
        n_reference_points=8,
        sampling_seed=71,
    )
    campaign = MultiStepCampaign.from_dag(
        dag, acquisition_function=acquisition, cost_mode=cost_mode
    )
    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id="exp-1",
        node="A",
        inputs={"a_input": 0.0},
        parent_refs={},
        data=campaign.data,
        dag=dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": "exp-1",
                    "node": "A",
                    "a_input": 0.0,
                    "a_output": 1.0,
                }
            ]
        )
    )
    campaign._fit_node_models()
    plans = tuple(
        plan for plan in campaign._executable_plans() if plan.node in {"B", "C"}
    )
    assert [(plan.node, plan.remaining_cost) for plan in plans] == [
        ("B", 15.0),
        ("C", 15.0),
    ]

    call_seed = 193
    reference_tensor = next(campaign.model.parameters())
    assert reference_tensor.device.type == "cpu"
    assert reference_tensor.dtype == torch.double
    branches = acquisition.make_botorch_branches(
        campaign.model, (), call_seed=call_seed
    )
    configurations = pd.DataFrame(
        {"a_input": [0.0] * 21, "d_input": torch.linspace(0.0, 1.0, 21).tolist()}
    )
    grid = torch.as_tensor(
        dag.full_searchspace.transform(configurations).to_numpy().copy(),
        dtype=torch.double,
    ).unsqueeze(-2)
    raw_ei = torch.stack([branch(grid) for branch in branches]).mean(dim=0)
    assert torch.isfinite(raw_ei).all()
    assert raw_ei.max().item() > 0.0

    # Independently score a fixed dense grid from real MREI values, real local
    # success predictions, and the plan costs; do not reuse the selector.
    from baybe.multistep.acquisition import _stock_action_probability

    expected_scores = {}
    expected_probabilities = {}
    for plan in plans:
        probabilities = _stock_action_probability(campaign.model, plan, grid, ())
        expected_probabilities[plan.node] = probabilities
        expected_cost = (
            plan.remaining_cost
            if cost_mode == "remaining"
            else dag.node_by_name[plan.node].cost
        )
        expected_scores[plan.node] = (raw_ei * probabilities / expected_cost).max()
    assert all(
        torch.equal(probability, torch.ones_like(probability))
        for probability in expected_probabilities.values()
    )
    if cost_mode == "remaining":
        assert expected_scores["B"].item() == pytest.approx(
            expected_scores["C"].item(), rel=1e-12, abs=1e-14
        )
        expected_node = "B"  # Equal scores preserve executable-plan order.
    else:
        assert expected_scores["B"].item() > expected_scores["C"].item()
        expected_node = "B"

    (selected,) = recommend_stock(
        campaign.model,
        campaign.data,
        campaign.data._contexts,
        plans,
        (),
        acquisition_function=acquisition,
        score_scale="linear",
        batch_size=1,
        cost_mode=cost_mode,
        call_seed=call_seed,
    )

    assert selected.node == expected_node


def test_pending_branch_reuses_existing_fork_join_remaining_cost():
    from baybe.multistep.data import register_action

    a_input = NumericalDiscreteParameter("a_input", (0.0, 1.0))
    b_input = NumericalDiscreteParameter("b_input", (0.0, 1.0))
    c_input = NumericalDiscreteParameter("c_input", (0.0, 1.0))
    d_input = NumericalDiscreteParameter("d_input", (0.0, 1.0))
    node_a = MultiStepNode(name="A", output="a_output", inputs=(a_input,), cost=2.0)
    node_b = MultiStepNode(
        name="B", output="b_output", inputs=(b_input,), parents=(node_a,), cost=3.0
    )
    node_c = MultiStepNode(
        name="C", output="c_output", inputs=(c_input,), parents=(node_a,), cost=5.0
    )
    node_d = MultiStepNode(
        name="D",
        output="utility",
        inputs=(d_input,),
        parents=(node_b, node_c),
        cost=7.0,
    )
    dag = MultiStepDAG(
        nodes=(node_a, node_b, node_c, node_d),
        objective=NumericalTarget("utility").to_objective(),
    )
    campaign = MultiStepCampaign.from_dag(dag, acquisition_function=MREI())
    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id="exp-1",
        node="A",
        inputs={"a_input": 0.0},
        parent_refs={},
        data=campaign.data,
        dag=dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": "exp-1",
                    "node": "A",
                    "a_input": 0.0,
                    "a_output": 1.0,
                }
            ]
        )
    )
    reference = campaign.data.experiment_bindings("exp-1")["A"]

    assert {
        plan.node: plan.remaining_cost
        for plan in campaign._executable_plans()
        if plan.node in {"B", "C"}
    } == {"B": 15.0, "C": 15.0}

    contexts, counter = register_action(
        campaign.data._contexts,
        campaign.data._experiment_counter,
        experiment_id="exp-1",
        node="C",
        inputs={"a_output": 1.0, "c_input": 0.0},
        parent_refs={"A": reference},
        data=campaign.data,
        dag=dag,
    )
    campaign.data = campaign.data.with_process_state(contexts, counter)

    plans = [plan for plan in campaign._executable_plans() if plan.node != "A"]
    assert [(plan.node, plan.remaining_cost) for plan in plans] == [("B", 10.0)]


@pytest.mark.parametrize("cost_mode", ["remaining", "node"])
@pytest.mark.parametrize(
    ("provided_seed", "expected_seed"),
    [
        pytest.param(None, 29, id="stock-helper-draws-seed"),
        pytest.param(13, 13, id="caller-seed-is-forwarded"),
    ],
)
def test_stock_path_builds_one_shared_reference_branch_set(
    monkeypatch, cost_mode, provided_seed, expected_seed, single_node_dag
):
    import pandas as pd
    import torch

    from baybe.multistep import _model_reference_ei as model_reference_ei
    from baybe.multistep import acquisition as multistep_acquisition
    from baybe.multistep.data import _ExecutablePlan

    model = SimpleNamespace(dag=single_node_dag)
    data = SimpleNamespace(
        complete_measurement_frame=lambda *_args, **_kwargs: pytest.fail(
            "MREI must not request complete real measurements"
        )
    )
    plans = (
        _ExecutablePlan(experiment_id="exp-1", node="root", is_new=True),
        _ExecutablePlan(experiment_id="exp-2", node="root", is_new=True),
    )
    built = []
    seed_draws = []
    optimized_branches = []
    optimized_cost_modes = []
    branches = (object(), object())

    def make_branches(_self, _model, pending, *, call_seed):
        built.append((pending, call_seed))
        return branches

    def draw_seed():
        seed_draws.append(29)
        return 29

    def optimize(**kwargs):
        optimized_branches.append(kwargs["branches"])
        optimized_cost_modes.append(kwargs["cost_mode"])
        return pd.Series({"x": 0.0}), torch.tensor(1.0)

    monkeypatch.setattr(
        ModelReferenceExpectedImprovement, "make_botorch_branches", make_branches
    )
    monkeypatch.setattr(model_reference_ei, "_draw_model_reference_seed", draw_seed)
    monkeypatch.setattr(
        multistep_acquisition, "_stock_fixed_parameters", lambda *_args: {}
    )
    monkeypatch.setattr(
        multistep_acquisition, "_stock_discrete_candidates", lambda *_args: ({},)
    )
    monkeypatch.setattr(multistep_acquisition, "_optimize_stock_context", optimize)
    monkeypatch.setattr(
        multistep_acquisition,
        "_project_candidate",
        lambda plan, *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        multistep_acquisition,
        "select_scored_actions",
        lambda scored, **_kwargs: [action for action, _score in scored],
    )

    result = multistep_acquisition.recommend_stock(
        model,
        data,
        {},
        plans,
        (),
        acquisition_function=MREI(),
        score_scale="linear",
        batch_size=1,
        call_seed=provided_seed,
        cost_mode=cost_mode,
    )

    assert result == list(plans)
    assert built == [((), expected_seed)]
    assert seed_draws == ([] if provided_seed is not None else [29])
    assert optimized_branches == [branches, branches]
    assert optimized_cost_modes == [cost_mode, cost_mode]


def test_ordinary_stock_acquisition_keeps_complete_measurement_requirement(
    single_node_dag,
):
    from baybe.acquisition import qExpectedImprovement
    from baybe.multistep.acquisition import recommend_stock
    from baybe.multistep.data import _ExecutablePlan

    model = SimpleNamespace(dag=single_node_dag)
    data = SimpleNamespace(
        complete_measurement_frame=lambda *_args, **_kwargs: pd.DataFrame()
    )
    plan = _ExecutablePlan(experiment_id="exp-1", node="root", is_new=True)

    with pytest.raises(IncompatibilityError, match="complete, reference-consistent"):
        recommend_stock(
            model,
            data,
            {},
            (plan,),
            (),
            acquisition_function=qExpectedImprovement(),
            score_scale="linear",
            batch_size=1,
        )


def test_campaign_roundtrip_adds_no_mrei_rng_state_and_loads_legacy_snapshot(
    make_single_node_campaign,
):
    campaign = make_single_node_campaign(MREI())
    snapshot = campaign.to_dict()

    assert "_model_reference_ei_call_index" not in snapshot
    legacy_snapshot = {**snapshot, "_model_reference_ei_call_index": 7}
    restored = MultiStepCampaign.from_dict(legacy_snapshot)

    assert restored.acquisition_function == campaign.acquisition_function
    assert not hasattr(restored, "_model_reference_ei_call_index")


@pytest.mark.parametrize("stage", ["random", "invalid-batch"])
def test_random_or_invalid_campaign_path_draws_no_mrei_seed(monkeypatch, stage):
    drawn_seeds = []

    def draw_seed():
        drawn_seeds.append(431)
        return 431

    monkeypatch.setattr(
        "baybe.multistep.campaign._draw_model_reference_seed", draw_seed
    )
    parameter = NumericalDiscreteParameter("x", (0.0, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    campaign = MultiStepCampaign.from_dag(dag, acquisition_function=MREI())
    if stage == "invalid-batch":
        campaign.recommender.remain_switched = True
        campaign.recommender._has_switched = True

    if stage == "random":
        recommendation = campaign.recommend()
        assert len(recommendation) == 1
    else:
        with pytest.raises(IncompatibilityError, match="batch_size=1"):
            campaign.recommend(batch_size=2)

    assert drawn_seeds == []
