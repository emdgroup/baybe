"""Tests for ordinary Monte Carlo EI over complete multi-step configurations."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from baybe.exceptions import IncompatibilityError, IncompatibleAcquisitionFunctionError
from baybe.multistep import (
    FeasibilityCondition,
    MultiStepCampaign,
    MultiStepDAG,
    MultiStepNode,
)
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.targets.numerical import NumericalTarget


@pytest.fixture(name="campaign")
def fixture_campaign():
    """Create a one-node numerical campaign with one owned parameter."""
    node = MultiStepNode(
        name="only",
        output="utility",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
    )
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    return MultiStepCampaign.from_dag(dag)


def test_missing_complete_reference_fails_before_fit_or_id_allocation(
    campaign, monkeypatch
):
    """Reject EI without a full success reference before fitting any node model."""

    def fail_if_fitted():
        raise AssertionError("models must not be fit without a complete reference")

    monkeypatch.setattr(campaign, "_fit_node_models", fail_if_fitted)
    with pytest.raises(IncompatibilityError, match="complete successful"):
        campaign.recommend_configuration(sampling_seed=17)

    assert not campaign.data._contexts


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        pytest.param(
            {"sampling_seed": True},
            TypeError,
            "sampling_seed must be an integer",
            id="bool-seed",
        ),
        pytest.param(
            {"sampling_seed": -1},
            ValueError,
            "sampling_seed must be at least 0",
            id="negative-seed",
        ),
        pytest.param(
            {"sampling_seed": 1, "n_mc_samples": 0},
            ValueError,
            "n_mc_samples must be at least 1",
            id="zero-samples",
        ),
        pytest.param(
            {"sampling_seed": 1, "num_restarts": 0},
            ValueError,
            "num_restarts must be at least 1",
            id="zero-restarts",
        ),
        pytest.param(
            {"sampling_seed": 1, "num_restarts": 3, "raw_samples": 2},
            ValueError,
            "raw_samples must be at least num_restarts",
            id="too-few-raw-samples",
        ),
    ],
)
def test_recommendation_arguments_are_validated_before_state_change(
    campaign, kwargs, exception, message
):
    """Require genuine seeds and consistent positive optimization budgets."""
    with pytest.raises(exception, match=message):
        campaign.recommend_configuration(**kwargs)

    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts
    assert campaign.data._experiment_counter == 0


def test_non_numeric_configuration_parameters_fail_before_fit():
    """Reject categorical parameters before reference lookup or model fitting."""
    node = MultiStepNode(
        name="only",
        output="utility",
        inputs=(CategoricalParameter("kind", ("a", "b")),),
    )
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    campaign = MultiStepCampaign.from_dag(dag)

    def fail_if_fitted():
        raise AssertionError("models must not be fit for unsupported parameters")

    campaign._fit_node_models = fail_if_fitted
    with pytest.raises(IncompatibilityError, match="only numerical"):
        campaign.recommend_configuration(sampling_seed=7)

    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts


def test_unsupported_acquisition_fails_before_fit_or_identifier_allocation(
    campaign, monkeypatch
):
    """Reject non-EI methods before model fitting or a reserved identifier."""

    def fail_if_fitted():
        raise AssertionError("unsupported acquisitions must fail before fitting")

    monkeypatch.setattr(campaign, "_fit_node_models", fail_if_fitted)
    with pytest.raises(IncompatibleAcquisitionFunctionError, match="only BayBE qEI"):
        campaign.recommend_configuration(acquisition_function="qUCB", sampling_seed=11)

    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts


def test_incompatible_sampler_api_fails_before_fit(campaign, monkeypatch):
    """Reject an installed sampler API missing required keywords pre-fit."""
    from botorch.sampling.get_sampler import get_sampler

    def unsupported_sampler():
        return get_sampler

    monkeypatch.setattr("botorch.sampling.get_sampler.get_sampler", unsupported_sampler)

    def fail_if_fitted():
        raise AssertionError("incompatible APIs must fail before fitting")

    monkeypatch.setattr(campaign, "_fit_node_models", fail_if_fitted)
    with pytest.raises(IncompatibleAcquisitionFunctionError, match="lacks required"):
        campaign.recommend_configuration(sampling_seed=12)

    assert campaign.data._experiment_counter == 0
    assert not campaign.data._contexts


def test_mixed_optimizer_options_are_verified_before_fit(monkeypatch):
    """Reject a mixed optimizer that cannot verify the required options pre-fit."""
    from botorch.optim import optimize_acqf_mixed_alternating

    from baybe.multistep._configuration_acquisition import (
        validate_configuration_optimization_api,
    )

    def undocumented_optimizer(
        acq_function,
        bounds,
        discrete_dims=None,
        cat_dims=None,
        options=None,
        q=1,
        raw_samples=1024,
        num_restarts=20,
        post_processing_func=None,
        sequential=True,
        fixed_features=None,
        inequality_constraints=None,
        equality_constraints=None,
        return_acq_values=True,
    ):
        return optimize_acqf_mixed_alternating(
            acq_function=acq_function,
            bounds=bounds,
            discrete_dims=discrete_dims,
            cat_dims=cat_dims,
            options=options,
            q=q,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
            post_processing_func=post_processing_func,
            sequential=sequential,
            fixed_features=fixed_features,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            return_acq_values=return_acq_values,
        )

    monkeypatch.setattr(
        "botorch.optim.optimize_acqf_mixed_alternating", undocumented_optimizer
    )

    with pytest.raises(IncompatibleAcquisitionFunctionError, match="options"):
        validate_configuration_optimization_api("qEI", has_discrete_parameters=True)


def test_real_failure_outputs_are_not_complete_ei_references(monkeypatch):
    """Do not use real prefix outputs as a terminal-utility EI reference."""
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
    campaign = MultiStepCampaign.from_dag(dag)
    experiment_id = campaign.register_configuration({"x": 0.25, "z": 0.75}).iloc[0][
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
                    "root_output": -1.0,
                }
            ]
        )
    )
    assert campaign.data.complete_measurement_frame(dag).empty

    def fail_if_fitted():
        raise AssertionError("failed process prefixes must not reach fitting")

    monkeypatch.setattr(campaign, "_fit_node_models", fail_if_fitted)
    with pytest.raises(IncompatibilityError, match="complete successful"):
        campaign.recommend_configuration(sampling_seed=13)

    assert campaign.data._experiment_counter == 1
    assert set(campaign.data._contexts) == {experiment_id}


def test_historical_complete_process_reference_is_eligible(campaign, monkeypatch):
    """Accept a complete real process row without a configuration reservation."""
    import numpy as np

    import baybe.multistep._configuration_acquisition as configuration_acquisition

    historical_action = campaign.recommend()
    historical_id = historical_action.iloc[0]["experiment_id"]
    campaign.add_measurements(historical_action.assign(utility=3.0))
    seen_references = []
    seen_seeds = {}
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)

    def make_acquisition(
        model, references, acquisition_function, *, sampling_seed, n_mc_samples
    ):
        seen_references.append(references.copy())
        seen_seeds["mc"] = sampling_seed
        return object()

    def optimize(model, acquisition, *, sampling_seed, num_restarts, raw_samples):
        seen_seeds["optimizer"] = sampling_seed
        return pd.DataFrame({"x": [0.75]})

    monkeypatch.setattr(
        configuration_acquisition,
        "make_configuration_acquisition",
        make_acquisition,
    )
    monkeypatch.setattr(
        configuration_acquisition,
        "optimize_configuration",
        optimize,
    )

    recommendation = campaign.recommend_configuration(
        sampling_seed=61, n_mc_samples=2, num_restarts=1, raw_samples=1
    )

    assert len(seen_references) == 1
    assert seen_references[0]["experiment_id"].tolist() == [historical_id]
    assert seen_references[0]["utility"].tolist() == [3.0]
    assert recommendation.loc[0, "experiment_id"] != historical_id
    assert recommendation.loc[0, "x"] == 0.75
    expected_seeds = np.random.SeedSequence(61).spawn(2)
    assert seen_seeds == {
        "mc": int(expected_seeds[0].generate_state(1, dtype="uint32")[0]),
        "optimizer": int(expected_seeds[1].generate_state(1, dtype="uint32")[0]),
    }


def test_invalid_optimized_configuration_does_not_reserve_state(campaign, monkeypatch):
    """Validate the candidate again at the campaign boundary before reserving it."""
    import baybe.multistep._configuration_acquisition as configuration_acquisition

    registered = campaign.register_configuration({"x": 0.25})
    experiment_id = registered.iloc[0]["experiment_id"]
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "node": "only",
                    "x": 0.25,
                    "utility": 3.0,
                }
            ]
        )
    )
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)
    monkeypatch.setattr(
        configuration_acquisition,
        "make_configuration_acquisition",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        configuration_acquisition,
        "optimize_configuration",
        lambda *args, **kwargs: pd.DataFrame({"x": [2.0]}),
    )
    prior_contexts = dict(campaign.data._contexts)
    prior_counter = campaign.data._experiment_counter

    with pytest.raises(
        IncompatibilityError, match="outside the allowed values or bounds"
    ):
        campaign.recommend_configuration(sampling_seed=71)

    assert campaign.data._contexts == prior_contexts
    assert campaign.data._experiment_counter == prior_counter


def test_configuration_ei_ignores_local_cost_and_risk_settings(campaign, monkeypatch):
    """Keep complete-configuration selection independent of local scoring policy."""
    import baybe.multistep._configuration_acquisition as configuration_acquisition

    outputs = []
    for cost_mode, feasibility_threshold in (("remaining", 0.0), ("node", 0.9)):
        current = MultiStepCampaign.from_dag(
            campaign.dag,
            cost_mode=cost_mode,
            feasibility_threshold=feasibility_threshold,
        )
        registered = current.register_configuration({"x": 0.25})
        experiment_id = registered.iloc[0]["experiment_id"]
        current.get_next_configuration_action(experiment_id)
        current.add_measurements(
            pd.DataFrame(
                [
                    {
                        "experiment_id": experiment_id,
                        "node": "only",
                        "x": 0.25,
                        "utility": 3.0,
                    }
                ]
            )
        )
        monkeypatch.setattr(current, "_fit_node_models", lambda: None)
        monkeypatch.setattr(
            current,
            "recommend",
            lambda *args, **kwargs: pytest.fail("local recommendation was invoked"),
        )
        monkeypatch.setattr(
            current,
            "_executable_plans",
            lambda: pytest.fail("local cost and feasibility scoring was invoked"),
        )
        monkeypatch.setattr(
            configuration_acquisition,
            "make_configuration_acquisition",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            configuration_acquisition,
            "optimize_configuration",
            lambda *args, **kwargs: pd.DataFrame({"x": [0.75]}),
        )

        outputs.append(
            current.recommend_configuration(
                sampling_seed=67, n_mc_samples=2, num_restarts=1, raw_samples=1
            )[["x"]]
        )

    pd.testing.assert_frame_equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    ("minimize", "expected_best_f", "expected_ei"),
    [(False, 2.0, 0.75), (True, -2.0, 0.25)],
    ids=["maximize", "minimize"],
)
def test_qei_uses_model_mean_reference_and_objective_direction(
    campaign, monkeypatch, minimize, expected_best_f, expected_ei
):
    """Use controlled samples to verify BayBE's model-mean reference and EI formula."""
    import torch
    from botorch.models.deterministic import GenericDeterministicModel
    from botorch.sampling.base import MCSampler

    from baybe.multistep._configuration_acquisition import (
        make_configuration_acquisition,
    )

    dag = campaign.dag
    objective_target = NumericalTarget("utility", minimize=minimize)
    dag = MultiStepDAG(nodes=dag.nodes, objective=objective_target.to_objective())
    campaign.dag = dag
    reference_posterior_mean = 2.0
    adapter = GenericDeterministicModel(
        lambda X: torch.full(
            X.shape[:-1] + (1,),
            reference_posterior_mean,
            dtype=X.dtype,
            device=X.device,
        ),
        num_outputs=1,
    )
    base_model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))]
        ),
    )
    monkeypatch.setattr(
        "baybe.multistep._configuration_acquisition._make_acquisition_model_adapter",
        lambda _model: adapter,
        raising=False,
    )

    class _FixedSampler(MCSampler):
        def __init__(self):
            super().__init__(sample_shape=torch.Size([4]), seed=13)

        def forward(self, posterior):
            """Return four analytically known raw target draws."""
            values = posterior.mean.new_tensor([1.0, 2.0, 3.0, 4.0])
            return values.reshape(4, 1, 1, 1).expand((4, *posterior.mean.shape))

    sampler_calls = []

    def get_fixed_sampler(posterior, sample_shape, *, seed=None):
        """Return controlled draws behind the installed BoTorch sampler API."""
        sampler_calls.append((sample_shape, seed))
        return _FixedSampler()

    monkeypatch.setattr("botorch.sampling.get_sampler.get_sampler", get_fixed_sampler)
    references = pd.DataFrame({"x": [0.25], "utility": [99.0]})

    acquisition = make_configuration_acquisition(
        base_model,
        references,
        "qEI",
        sampling_seed=13,
        n_mc_samples=4,
    )
    candidate = torch.tensor([[[0.5]]], dtype=torch.double)

    assert acquisition.best_f.item() == expected_best_f
    assert acquisition(candidate).item() == pytest.approx(expected_ei)
    assert sampler_calls == [(torch.Size([4]), 13)]


def test_scalarized_qei_uses_target_columns_in_objective_order(monkeypatch):
    """Build scalarized EI from real target columns and objective-ordered outputs."""
    import importlib

    import torch
    from botorch.models.deterministic import GenericDeterministicModel
    from botorch.sampling.base import MCSampler

    from baybe.multistep._configuration_acquisition import (
        make_configuration_acquisition,
    )
    from baybe.objectives.desirability import DesirabilityObjective

    node_a = MultiStepNode(
        name="a-node",
        output="target-a",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
    )
    node_b = MultiStepNode(
        name="b-node",
        output="target-b",
        inputs=(NumericalContinuousParameter("y", (0.0, 1.0)),),
    )
    objective = DesirabilityObjective(
        [
            NumericalTarget("target-b", minimize=True),
            NumericalTarget("target-a"),
        ],
        scalarizer="MEAN",
        require_normalization=False,
    )
    dag = MultiStepDAG(nodes=(node_a, node_b), objective=objective)
    adapter = GenericDeterministicModel(
        lambda X: X.new_tensor([2.0, 4.0]).expand(X.shape[:-1] + (2,)),
        num_outputs=2,
    )
    model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))]
        ),
    )
    monkeypatch.setattr(
        "baybe.multistep._configuration_acquisition._make_acquisition_model_adapter",
        lambda _model: adapter,
        raising=False,
    )

    class _FixedSampler(MCSampler):
        def __init__(self):
            super().__init__(sample_shape=torch.Size([2]), seed=17)

        def forward(self, posterior):
            """Return two known, objective-ordered samples for both targets."""
            values = posterior.mean.new_tensor([[1.0, 5.0], [3.0, 7.0]])
            return values.reshape(2, 1, 1, 2).expand((2, *posterior.mean.shape))

    sampler_module = importlib.import_module("botorch.sampling.get_sampler")
    monkeypatch.setattr(
        sampler_module,
        "get_sampler",
        lambda posterior, sample_shape, *, seed=None: _FixedSampler(),
    )
    references = pd.DataFrame(
        {"x": [0.25], "y": [0.75], "target-a": [99.0], "target-b": [-99.0]}
    )

    acquisition = make_configuration_acquisition(
        model,
        references,
        "qEI",
        sampling_seed=17,
        n_mc_samples=2,
    )

    assert acquisition.best_f.item() == pytest.approx(1.5)
    candidate = torch.tensor([[[0.5, 0.5]]], dtype=torch.double)
    assert acquisition(candidate).item() == pytest.approx(1.0)


def test_qlogei_matches_botorch_with_identical_model_and_samples(campaign, monkeypatch):
    """Preserve BoTorch's smoothed log-EI, not log of a plug-in qEI estimate."""
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.models.deterministic import GenericDeterministicModel
    from botorch.sampling.base import MCSampler

    from baybe.multistep._configuration_acquisition import (
        make_configuration_acquisition,
    )

    adapter = GenericDeterministicModel(
        lambda X: torch.full(X.shape[:-1] + (1,), 2.0, dtype=X.dtype, device=X.device),
        num_outputs=1,
    )
    model = SimpleNamespace(
        dag=campaign.dag,
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))]
        ),
    )
    monkeypatch.setattr(
        "baybe.multistep._configuration_acquisition._make_acquisition_model_adapter",
        lambda _model: adapter,
        raising=False,
    )

    class _FixedSampler(MCSampler):
        def __init__(self):
            super().__init__(sample_shape=torch.Size([4]), seed=23)

        def forward(self, posterior):
            """Return the same fixed samples to both acquisition constructors."""
            values = posterior.mean.new_tensor([1.0, 2.0, 3.0, 4.0])
            return values.reshape(4, 1, 1, 1).expand((4, *posterior.mean.shape))

    sampler_calls = []

    def get_fixed_sampler(posterior, sample_shape, *, seed=None):
        """Return controlled draws behind the installed BoTorch sampler API."""
        sampler_calls.append((sample_shape, seed))
        return _FixedSampler()

    monkeypatch.setattr("botorch.sampling.get_sampler.get_sampler", get_fixed_sampler)
    references = pd.DataFrame({"x": [0.25], "utility": [99.0]})
    actual = make_configuration_acquisition(
        model,
        references,
        "qLogEI",
        sampling_seed=23,
        n_mc_samples=4,
    )
    expected = qLogExpectedImprovement(
        model=adapter,
        best_f=2.0,
        sampler=_FixedSampler(),
        objective=campaign.dag.objective.to_botorch(),
        fat=True,
        tau_relu=1e-6,
        tau_max=1e-2,
    )
    candidate = torch.tensor([[[0.5]]], dtype=torch.double)

    assert actual(candidate).item() == pytest.approx(expected(candidate).item())
    assert sampler_calls == [(torch.Size([4]), 23)]


def test_mixed_optimizer_uses_exact_dimensions_without_global_product(monkeypatch):
    """Route twenty ten-valued inputs without touching the global product."""
    import torch

    from baybe.multistep._configuration_acquisition import optimize_configuration

    nodes = []
    parent = None
    for index in range(20):
        node = MultiStepNode(
            name=f"node-{index:02d}",
            output="utility" if index == 19 else f"output-{index:02d}",
            inputs=(
                NumericalDiscreteParameter(f"x{19 - index:02d}", tuple(range(10))),
            ),
            parents=() if parent is None else (parent,),
        )
        nodes.append(node)
        parent = node
    dag = MultiStepDAG(
        nodes=tuple(nodes), objective=NumericalTarget("utility").to_objective()
    )
    model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))]
        ),
    )

    def fail_if_materialized(self):
        raise AssertionError("the complete configuration product was materialized")

    monkeypatch.setattr(
        MultiStepDAG, "full_searchspace", property(fail_if_materialized)
    )
    calls = {}

    def optimize_mixed(
        acq_function,
        bounds,
        discrete_dims=None,
        cat_dims=None,
        options=None,
        q=1,
        raw_samples=1024,
        num_restarts=20,
        post_processing_func=None,
        sequential=True,
        fixed_features=None,
        inequality_constraints=None,
        equality_constraints=None,
        return_acq_values=True,
    ):
        """Record the mixed optimizer contract and return a valid point."""
        calls.update(
            bounds=bounds,
            discrete_dims=discrete_dims,
            options=options,
            q=q,
            sequential=sequential,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        return torch.ones((1, 20), dtype=bounds.dtype, device=bounds.device), None

    from botorch.optim import optimize_acqf_mixed_alternating as installed_optimizer

    optimize_mixed.__doc__ = installed_optimizer.__doc__
    monkeypatch.setattr("botorch.optim.optimize_acqf_mixed_alternating", optimize_mixed)
    candidate = optimize_configuration(
        model,
        acquisition=object(),
        sampling_seed=31,
        num_restarts=3,
        raw_samples=12,
    )

    assert len(candidate.columns) == 20
    assert (candidate.iloc[0] == 1.0).all()
    assert calls["q"] == 1
    assert calls["sequential"] is True
    assert calls["num_restarts"] == 3
    assert calls["raw_samples"] == 12
    assert calls["discrete_dims"] == {index: tuple(range(10)) for index in range(20)}
    assert calls["options"] == {
        "initialization_strategy": "random",
        "maxiter_alternating": 64,
        "maxiter_discrete": 4,
        "maxiter_continuous": 8,
        "tol": 1e-8,
        "batch_limit": 1,
        "init_batch_limit": 1,
        "num_spray_points": 0,
        "max_discrete_values": 20,
    }


def test_continuous_optimizer_is_reproducible_and_restores_global_rng(
    campaign, monkeypatch
):
    """Scope optimizer randomness and make a seed reproduce its candidate."""
    import torch

    from baybe.multistep._configuration_acquisition import optimize_configuration

    model = SimpleNamespace(
        dag=campaign.dag,
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))]
        ),
    )
    calls = {}

    def optimize_continuous(
        acq_function, bounds, q, num_restarts, raw_samples=None, options=None
    ):
        """Make the optimizer's candidate depend on the scoped Torch RNG."""
        calls.update(
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )
        return torch.rand((1, bounds.shape[-1]), dtype=bounds.dtype), None

    monkeypatch.setattr("botorch.optim.optimize_acqf", optimize_continuous)
    torch.manual_seed(811)
    initial_state = torch.random.get_rng_state()
    expected_next = torch.rand(1)
    torch.random.set_rng_state(initial_state)

    first = optimize_configuration(
        model, object(), sampling_seed=19, num_restarts=2, raw_samples=5
    )
    actual_next = torch.rand(1)
    second = optimize_configuration(
        model, object(), sampling_seed=19, num_restarts=2, raw_samples=5
    )

    assert first.equals(second)
    assert torch.equal(actual_next, expected_next)
    assert calls["bounds"].shape == (2, 1)
    assert calls["q"] == 1
    assert calls["num_restarts"] == 2
    assert calls["raw_samples"] == 5
    assert calls["options"] == {"maxiter": 200, "batch_limit": 1}


def test_optimizer_failure_does_not_publish_a_configuration(campaign, monkeypatch):
    """Keep context state and identifier allocation unchanged on optimizer errors."""
    import baybe.multistep._configuration_acquisition as configuration_acquisition

    registered = campaign.register_configuration({"x": 0.25})
    experiment_id = registered.iloc[0]["experiment_id"]
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "node": "only",
                    "x": 0.25,
                    "utility": 3.0,
                }
            ]
        )
    )
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)
    monkeypatch.setattr(
        configuration_acquisition,
        "make_configuration_acquisition",
        lambda *args, **kwargs: object(),
    )

    def fail_optimization(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(
        configuration_acquisition, "optimize_configuration", fail_optimization
    )
    prior_contexts = dict(campaign.data._contexts)
    prior_counter = campaign.data._experiment_counter

    with pytest.raises(RuntimeError, match="optimizer failed"):
        campaign.recommend_configuration(sampling_seed=29)

    assert campaign.data._contexts == prior_contexts
    assert campaign.data._experiment_counter == prior_counter


def test_parameterless_optimizer_returns_one_empty_row_without_api(monkeypatch):
    """Represent the sole empty configuration without importing an optimizer."""
    import baybe.multistep._configuration_acquisition as configuration_acquisition
    from baybe.multistep._configuration_space import _ConfigurationSpace

    model = SimpleNamespace(
        dag=SimpleNamespace(_configuration_space=_ConfigurationSpace(()))
    )

    def fail_if_checked(*, has_discrete_parameters):
        raise AssertionError("no optimizer API is needed for an empty configuration")

    monkeypatch.setattr(
        configuration_acquisition,
        "_validate_optimizer_api",
        fail_if_checked,
    )

    result = configuration_acquisition.optimize_configuration(
        model,
        object(),
        sampling_seed=3,
        num_restarts=1,
        raw_samples=1,
    )

    assert result.shape == (1, 0)


def test_invalid_discrete_optimizer_result_is_rejected(monkeypatch):
    """Reject an optimizer's in-bounds value that is not an allowed choice."""
    import torch

    from baybe.multistep._configuration_acquisition import optimize_configuration

    node = MultiStepNode(
        name="only",
        output="utility",
        inputs=(NumericalDiscreteParameter("choice", (0.0, 1.0)),),
    )
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    model = SimpleNamespace(
        dag=dag,
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))]
        ),
    )

    def return_nonmember(
        acq_function,
        bounds,
        discrete_dims=None,
        cat_dims=None,
        options=None,
        q=1,
        raw_samples=1024,
        num_restarts=20,
        post_processing_func=None,
        sequential=True,
        fixed_features=None,
        inequality_constraints=None,
        equality_constraints=None,
        return_acq_values=True,
    ):
        return torch.tensor([[0.5]], dtype=torch.double), None

    from botorch.optim import optimize_acqf_mixed_alternating as installed_optimizer

    return_nonmember.__doc__ = installed_optimizer.__doc__
    monkeypatch.setattr(
        "botorch.optim.optimize_acqf_mixed_alternating", return_nonmember
    )
    with pytest.raises(IncompatibilityError, match="outside the allowed values"):
        optimize_configuration(
            model,
            object(),
            sampling_seed=5,
            num_restarts=1,
            raw_samples=1,
        )


@pytest.mark.parametrize("acquisition_name", ["qEI", "qLogEI"])
def test_two_node_gp_acquisition_has_finite_value_gradient_and_fixed_samples(
    acquisition_name,
):
    """Exercise ordinary EI and a native sampler on a real two-node network GP."""
    import torch
    from botorch.models import SingleTaskGP

    from baybe.multistep import MultiStepModel
    from baybe.multistep._configuration_acquisition import (
        make_configuration_acquisition,
    )

    root = MultiStepNode(
        name="root",
        output="root_output",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
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
    node_models = {
        "root": SingleTaskGP(
            train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        ),
        "child": SingleTaskGP(
            train_X=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        ),
    }
    model = MultiStepModel.from_node_models(dag, node_models)
    references = pd.DataFrame({"x": [0.25], "z": [0.75], "utility": [100.0]})
    acquisition = make_configuration_acquisition(
        model,
        references,
        acquisition_name,
        sampling_seed=43,
        n_mc_samples=16,
    )
    candidate = torch.tensor([[0.4, 0.6]], dtype=torch.double).unsqueeze(-2)
    candidate.requires_grad_(True)

    first = acquisition(candidate)
    first_gradient = torch.autograd.grad(first.sum(), candidate)[0]
    sampler = acquisition.sampler
    first_base_samples = sampler.base_samples.detach().clone()
    second = acquisition(candidate)

    assert first.numel() == 1
    assert torch.isfinite(first).all()
    assert torch.isfinite(first_gradient).all()
    assert torch.isfinite(second).all()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(sampler.base_samples, first_base_samples)


def test_campaign_recommendation_reserves_every_parameter_from_real_gp(
    monkeypatch,
):
    """Optimize a real two-node GP and reserve its full unprojected configuration."""
    import torch
    from botorch.models import SingleTaskGP

    from baybe.multistep import MultiStepModel

    root = MultiStepNode(
        name="root",
        output="root_output",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
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
    campaign = MultiStepCampaign.from_dag(dag)
    first = campaign.register_configuration({"x": 0.25, "z": 0.75})
    experiment_id = first.iloc[0]["experiment_id"]
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "node": "root",
                    "x": 0.25,
                    "root_output": 0.5,
                }
            ]
        )
    )
    campaign.get_next_configuration_action(experiment_id)
    campaign.add_measurements(
        pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "node": "child",
                    "z": 0.75,
                    "root_output": 0.5,
                    "utility": 0.8,
                }
            ]
        )
    )
    campaign.model = MultiStepModel.from_node_models(
        dag,
        {
            "root": SingleTaskGP(
                train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
                train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            ),
            "child": SingleTaskGP(
                train_X=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
                train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            ),
        },
        num_network_samples=16,
        sampling_seed=5,
    )
    monkeypatch.setattr(campaign, "_fit_node_models", lambda: None)

    recommendation = campaign.recommend_configuration(
        sampling_seed=47,
        n_mc_samples=16,
        num_restarts=2,
        raw_samples=8,
    )

    assert list(recommendation.columns) == ["experiment_id", "x", "z"]
    assert recommendation.loc[0, "experiment_id"] == "exp-2"
    assert 0.0 <= recommendation.loc[0, "x"] <= 1.0
    assert 0.0 <= recommendation.loc[0, "z"] <= 1.0
    reserved = campaign.data._contexts["exp-2"].configuration
    assert reserved == {
        "x": recommendation.loc[0, "x"],
        "z": recommendation.loc[0, "z"],
    }
