"""Functional tests for the product-free factorized input layout and DAG opt-in.

The tests use small local fixtures. The only parameter product in this module is the
small ``SearchSpace.from_product`` oracle used for parity checks; the product-free
paths are additionally instrumented so that no product builder can be invoked.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from baybe.exceptions import IncompatibilityError
from baybe.multistep import MultiStepDAG, MultiStepNode
from baybe.multistep._layout import _FactorizedInputLayout
from baybe.multistep._layout_fit import _fit_layout_surrogate
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.core import SearchSpace
from baybe.searchspace.discrete import SubspaceDiscrete
from baybe.targets.numerical import NumericalTarget


def _make_mixed_parameters() -> tuple:
    """Parameters covering continuous, numerical, OHE, and INT inputs."""
    return (
        NumericalContinuousParameter("continuous", bounds=(0.0, 2.0)),
        NumericalDiscreteParameter("discrete", values=(0.0, 0.25, 1.0)),
        CategoricalParameter(
            "categorical_ohe",
            values=("a", "b", "c"),
            encoding=CategoricalEncoding.OHE,
        ),
        CategoricalParameter(
            "categorical_int",
            values=("x", "y"),
            encoding=CategoricalEncoding.INT,
        ),
    )


def _make_mixed_rows() -> pd.DataFrame:
    """Representative experimental rows for the mixed parameter set."""
    return pd.DataFrame(
        {
            "continuous": [0.5, 1.5, 2.0, 0.0, 1.0],
            "discrete": [0.25, 1.0, 0.0, 0.25, 1.0],
            "categorical_ohe": ["b", "c", "a", "b", "c"],
            "categorical_int": ["y", "x", "y", "x", "y"],
        }
    )


def _target_values() -> list[float]:
    """Deterministic target values for the mixed row frame."""
    return [0.0, 0.3, -0.2, 0.9, 0.5]


@pytest.fixture(name="mixed_parameters")
def fixture_mixed_parameters() -> tuple:
    """Parameters covering continuous, uneven numerical, OHE, and INT inputs."""
    return _make_mixed_parameters()


@pytest.fixture(name="mixed_rows")
def fixture_mixed_rows() -> pd.DataFrame:
    """Representative experimental rows for the mixed parameter fixture."""
    return _make_mixed_rows()


def _make_single_node_dag(
    parameters: tuple, *, product_free: bool = False, output: str = "y"
) -> MultiStepDAG:
    """Build a one-node DAG from the given parameters."""
    node = MultiStepNode(name="root", output=output, inputs=parameters)
    return MultiStepDAG(
        nodes=(node,),
        objective=SingleTargetObjective(NumericalTarget(output)),
        product_free=product_free,
    )


def _forbid_product_builders(monkeypatch) -> list[tuple]:
    """Instrument every parameter product builder to record and fail."""
    calls: list[tuple] = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("A parameter product builder was called.")

    monkeypatch.setattr(SearchSpace, "from_product", forbidden)
    monkeypatch.setattr(SubspaceDiscrete, "from_product", forbidden)
    monkeypatch.setattr(SubspaceContinuous, "from_product", forbidden)
    return calls


def _seed_deterministic_fit() -> None:
    """Reproduce the same random stream before every paired fit."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)


def _assert_tensors_close(actual, expected) -> None:
    """Compare tensors with the project-wide parity tolerance."""
    import torch

    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)


def _spy_folds(monkeypatch, model_module) -> dict[str, list[int]]:
    """Track the current train/test positions while the cross-fit iterates folds."""
    state: dict[str, list[int]] = {"train": [], "test": []}
    original = model_module._make_grouped_folds

    def fold_iter(*args, **kwargs):
        for train_positions, test_positions in original(*args, **kwargs):
            state["train"] = list(train_positions)
            state["test"] = list(test_positions)
            yield train_positions, test_positions

    monkeypatch.setattr(model_module, "_make_grouped_folds", fold_iter)
    return state


def test_layout_matches_small_real_searchspace(mixed_parameters, mixed_rows):
    """The layout reproduces the real small search space exactly."""
    old = SearchSpace.from_product(mixed_parameters)
    new = _FactorizedInputLayout.from_parameters(mixed_parameters)
    assert new.parameter_names == old.parameter_names
    assert new.comp_rep_columns == old.comp_rep_columns
    assert_frame_equal(new.scaling_bounds, old.scaling_bounds)
    assert_frame_equal(new.transform(mixed_rows), old.transform(mixed_rows))


@pytest.mark.parametrize(
    "encoding",
    [CategoricalEncoding.OHE, CategoricalEncoding.INT],
    ids=["ohe", "int"],
)
def test_layout_complete_columns_and_bounds_for_inactive_levels(encoding):
    """Inactive levels keep their comp-rep columns and scaling bounds."""
    parameter = CategoricalParameter(
        "c",
        values=("a", "b", "c"),
        active_values=("b",),
        encoding=encoding,
    )
    old = SearchSpace.from_product((parameter,))
    new = _FactorizedInputLayout.from_parameters((parameter,))

    assert new.comp_rep_columns == old.comp_rep_columns
    assert_frame_equal(new.scaling_bounds, old.scaling_bounds)
    assert tuple(old.discrete.exp_rep["c"]) == ("b",)
    assert new.active_levels("c") == ("b",)

    if encoding is CategoricalEncoding.OHE:
        assert new.comp_rep_columns == ("c_a", "c_b", "c_c")
        assert new.scaling_bounds.loc["max", "c_a"] == 1.0
        assert new.scaling_bounds.loc["min", "c_c"] == 0.0
    else:
        assert new.comp_rep_columns == ("c",)
        assert new.scaling_bounds.loc["min", "c"] == 0.0
        assert new.scaling_bounds.loc["max", "c"] == 2.0


def test_product_free_dag_allows_parent_only_non_root_node():
    """A non-root node may rely entirely on parent outputs."""
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    root = MultiStepNode(name="root", output="root_out", inputs=(x0,))
    child = MultiStepNode(name="child", output="child_out", parents=(root,))
    dag = MultiStepDAG(
        nodes=(root, child),
        objective=SingleTargetObjective(NumericalTarget("child_out")),
        product_free=True,
    )

    assert dag._factorized_layout.parameter_names == ("x0",)
    assert child._own_parameter_layout.comp_rep_columns == ()
    assert child._own_comp_rep_columns == ()


def test_product_free_dag_allows_inactive_levels_but_default_rejects():
    """The opt-in accepts inactive levels while the legacy validation stays intact."""
    parameter = CategoricalParameter("c", values=("a", "b", "c"), active_values=("b",))

    dag = _make_single_node_dag((parameter,), product_free=True)
    assert dag._factorized_layout.active_levels("c") == ("b",)
    assert dag._factorized_layout.comp_rep_columns == ("c_a", "c_b", "c_c")

    with pytest.raises(IncompatibilityError, match="restricted discrete candidate set"):
        _make_single_node_dag((parameter,), product_free=False)


def _make_serialization_dag(product_free: bool) -> MultiStepDAG:
    """Build a minimal DAG for serialization round trips."""
    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    return _make_single_node_dag((x0,), product_free=product_free)


def test_default_dag_and_campaign_omit_product_free_in_serialization():
    """The default DAG keeps the historical fieldless payload, also when nested."""
    from baybe.multistep.campaign import MultiStepCampaign

    dag = _make_serialization_dag(product_free=False)
    dag_specs = dag.to_dict()
    assert "product_free" not in dag_specs

    campaign_specs = MultiStepCampaign(dag=dag).to_dict()
    assert "product_free" not in campaign_specs["dag"]

    # Fieldless legacy snapshots still load with the legacy default.
    legacy_dag_specs = dict(dag_specs)
    legacy_dag_specs.pop("product_free", None)
    assert MultiStepDAG.from_dict(legacy_dag_specs).product_free is False

    legacy_campaign_specs = dict(campaign_specs)
    legacy_campaign_specs["dag"] = dict(campaign_specs["dag"])
    legacy_campaign_specs["dag"].pop("product_free", None)
    restored = MultiStepCampaign.from_dict(legacy_campaign_specs)
    assert restored.dag.product_free is False

    # Explicit legacy False snapshots continue to structure as before.
    explicit_false_dag = {**dag_specs, "product_free": False}
    assert MultiStepDAG.from_dict(explicit_false_dag).product_free is False
    explicit_false_campaign = {
        **campaign_specs,
        "dag": {**campaign_specs["dag"], "product_free": False},
    }
    assert (
        MultiStepCampaign.from_dict(explicit_false_campaign).dag.product_free is False
    )


def test_product_free_true_is_serialized_and_round_trips():
    """The explicit opt-in is written and survives DAG and campaign round trips."""
    from baybe.multistep.campaign import MultiStepCampaign

    dag = _make_serialization_dag(product_free=True)
    dag_specs = dag.to_dict()
    assert dag_specs["product_free"] is True
    assert MultiStepDAG.from_dict(dag_specs).product_free is True

    campaign_specs = MultiStepCampaign(dag=dag).to_dict()
    assert campaign_specs["dag"]["product_free"] is True
    assert MultiStepCampaign.from_dict(campaign_specs).dag.product_free is True


def _make_large_parameter_set() -> tuple[NumericalDiscreteParameter, ...]:
    """Create 20 ten-level discrete parameters whose product is ``10**20``."""
    levels = tuple(float(level) for level in range(10))
    return tuple(
        NumericalDiscreteParameter(f"parameter_{index:02d}", values=levels)
        for index in range(20)
    )


def test_product_free_dag_construction_never_builds_parameter_product(monkeypatch):
    """A 20x10 product-free DAG must not invoke any product builder."""
    calls = _forbid_product_builders(monkeypatch)

    parameters = _make_large_parameter_set()
    dag = _make_single_node_dag(parameters, product_free=True)

    assert calls == []
    assert dag.product_free is True
    assert dag._factorized_layout.parameter_names == tuple(
        parameter.name for parameter in parameters
    )
    assert dag._factorized_layout.comp_rep_columns == tuple(
        parameter.name for parameter in parameters
    )
    assert dag._factorized_layout.n_tasks == 1


_GP_PRESETS = [
    pytest.param(None, id="default"),
    "baybe",
    "botorch",
    "chen",
    "edbo",
    "edbo_smoothed",
    "hvarfner",
]


@pytest.mark.parametrize("preset", _GP_PRESETS)
def test_gp_layout_fit_matches_real_searchspace_preset(
    preset, mixed_parameters, mixed_rows
):
    """The default GP and every built-in preset fit identically on the layout."""
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
    from baybe.utils.dataframe import to_tensor

    frame = mixed_rows.assign(y=_target_values())
    objective = NumericalTarget("y").to_objective()
    old = SearchSpace.from_product(mixed_parameters)
    layout = _FactorizedInputLayout.from_parameters(mixed_parameters)

    def make_surrogate():
        if preset is None:
            return GaussianProcessSurrogate()
        return GaussianProcessSurrogate.from_preset(preset.upper())

    legacy, free = make_surrogate(), make_surrogate()
    _seed_deterministic_fit()
    legacy.fit(old, objective, frame)
    _seed_deterministic_fit()
    _fit_layout_surrogate(free, layout, objective, frame)

    assert_frame_equal(
        layout.transform(frame, allow_extra=True),
        old.transform(frame, allow_extra=True),
    )
    assert free._searchspace is layout

    expected = legacy._posterior_comp(
        to_tensor(old.transform(frame, allow_extra=True)).unsqueeze(-2)
    )
    actual = free._posterior_comp(
        to_tensor(layout.transform(frame, allow_extra=True)).unsqueeze(-2)
    )
    _assert_tensors_close(actual.mean, expected.mean)
    _assert_tensors_close(actual.variance, expected.variance)

    expected_state = legacy._model.state_dict()
    actual_state = free._model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        _assert_tensors_close(actual_state[key], value)


def test_classifier_layout_prior_and_fit_match_real_searchspace(
    mixed_parameters, mixed_rows
):
    """The feasibility classifier prior and posterior match the real space."""
    from baybe.multistep.model import _MultiStepGPClassifier
    from baybe.targets.binary import BinaryTarget
    from baybe.utils.dataframe import to_tensor

    frame = mixed_rows.assign(ok=[True, False, True, True, False])
    objective = BinaryTarget(
        "ok", success_value=True, failure_value=False
    ).to_objective()
    old = SearchSpace.from_product(mixed_parameters)
    layout = _FactorizedInputLayout.from_parameters(mixed_parameters)

    legacy_prior, free_prior = _MultiStepGPClassifier(), _MultiStepGPClassifier()
    legacy_prior.initialize_prior(old, objective)
    free_prior.initialize_prior(layout, objective)
    assert free_prior._searchspace is layout

    legacy, free = _MultiStepGPClassifier(), _MultiStepGPClassifier()
    _seed_deterministic_fit()
    legacy.fit(old, objective, frame)
    _seed_deterministic_fit()
    _fit_layout_surrogate(free, layout, objective, frame)
    assert free._searchspace is layout

    candidates = _make_mixed_rows()
    expected_prior = legacy_prior.posterior_probabilities(
        to_tensor(old.transform(candidates)).unsqueeze(-2)
    )
    actual_prior = free_prior.posterior_probabilities(
        to_tensor(layout.transform(candidates)).unsqueeze(-2)
    )
    _assert_tensors_close(actual_prior, expected_prior)

    expected = legacy.posterior_probabilities(
        to_tensor(old.transform(candidates)).unsqueeze(-2)
    )
    actual = free.posterior_probabilities(
        to_tensor(layout.transform(candidates)).unsqueeze(-2)
    )
    _assert_tensors_close(actual, expected)


def _make_condition_case(mode, *, product_free: bool):
    """Build the mixed success/failure dataset used by the legacy tests."""
    from baybe.multistep.data import MultiStepData, NodeMeasurement
    from baybe.multistep.node import FeasibilityCondition
    from baybe.recommenders.pure.experimental_feasibility import (
        InfeasibleMeasurementMode,
    )

    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    options = {}
    if mode is InfeasibleMeasurementMode.DUMMY:
        options["infeasible_dummy_value"] = -999.0
    node = MultiStepNode(
        name="n",
        output="y",
        inputs=(x0,),
        feasibility_conditions=(FeasibilityCondition(column="ok", operator="is_true"),),
        infeasible_measurement_mode=mode,
        **options,
    )
    dag = MultiStepDAG(
        nodes=(node,),
        objective=SingleTargetObjective(NumericalTarget("y")),
        product_free=product_free,
    )
    data = MultiStepData.empty().add(
        NodeMeasurement(
            node="n", inputs={"x0": 0.75}, output=4.0, condition_values={"ok": False}
        ),
        NodeMeasurement(
            node="n", inputs={"x0": 0.0}, output=1.0, condition_values={"ok": True}
        ),
        NodeMeasurement(node="n", inputs={"x0": 0.5}, output=3.0),
        NodeMeasurement(
            node="n", inputs={"x0": 0.125}, output=1.5, condition_values={"ok": True}
        ),
        NodeMeasurement(
            node="n", inputs={"x0": 0.25}, output=2.0, condition_values={"ok": False}
        ),
    )
    return dag, data


@pytest.mark.parametrize(
    "mode_name",
    ["OBSERVED", "EXCLUDE", "DUMMY", "PREDICTED_MEAN"],
)
def test_product_free_node_training_data_matches_legacy(mode_name, monkeypatch):
    """All failure modes keep their rows, order, and predicted values."""
    import torch

    from baybe.recommenders.pure.experimental_feasibility import (
        InfeasibleMeasurementMode,
    )

    mode = InfeasibleMeasurementMode[mode_name]
    legacy_dag, data = _make_condition_case(mode, product_free=False)
    free_dag, _ = _make_condition_case(mode, product_free=True)

    torch.manual_seed(3)
    expected = data.node_training_data(legacy_dag, "n")

    calls = _forbid_product_builders(monkeypatch)
    torch.manual_seed(3)
    actual = data.node_training_data(free_dag, "n")

    assert calls == []
    assert_frame_equal(actual, expected, atol=1e-8, rtol=1e-6)


def _make_product_free_model_case():
    """Build a single-node product-free DAG with successes and failures."""
    from baybe.multistep.data import MultiStepData, NodeMeasurement
    from baybe.multistep.node import FeasibilityCondition
    from baybe.recommenders.pure.experimental_feasibility import (
        InfeasibleMeasurementMode,
    )

    node = MultiStepNode(
        name="n",
        output="y",
        inputs=_make_mixed_parameters(),
        feasibility_conditions=(FeasibilityCondition(column="ok", operator="is_true"),),
        infeasible_measurement_mode=InfeasibleMeasurementMode.PREDICTED_MEAN,
    )
    dag = MultiStepDAG(
        nodes=(node,),
        objective=SingleTargetObjective(NumericalTarget("y")),
        product_free=True,
    )
    rows = _make_mixed_rows().assign(y=_target_values())
    data = MultiStepData.empty().add(
        *(
            NodeMeasurement(
                node="n",
                inputs=row.drop(labels=["y"]).to_dict(),
                output=row["y"],
                condition_values={"ok": index % 3 != 1},
            )
            for index, row in rows.iterrows()
        )
    )
    return dag, data


def test_product_free_model_fit_avoids_products_and_evaluates(monkeypatch):
    """A real product-free fit builds no product and yields usable predictions."""
    import torch

    from baybe.multistep.model import MultiStepModel
    from baybe.utils.dataframe import to_tensor

    dag, data = _make_product_free_model_case()
    node_frames = {
        node.name: data.node_training_data(dag, node.name) for node in dag.nodes
    }
    feasibility_frames = {
        node.name: data.node_feasibility_training_data(dag, node.name)
        for node in dag.nodes
        if node.feasibility_conditions
    }
    model = MultiStepModel(dag=dag, num_network_samples=4, sampling_seed=7)

    calls = _forbid_product_builders(monkeypatch)
    model.fit_node_data(
        node_frames, allow_partial=True, feasibility_frames=feasibility_frames
    )

    assert calls == []
    surrogate = model._node_surrogates["n"]["y"]
    assert isinstance(surrogate._searchspace, _FactorizedInputLayout)
    classifier = model._feasibility_models["n"]
    assert isinstance(classifier._searchspace, _FactorizedInputLayout)
    assert model._feasibility_feature_names["n"] == {}

    layout = _FactorizedInputLayout.from_parameters(_make_mixed_parameters())
    candidates = to_tensor(layout.transform(_make_mixed_rows())).unsqueeze(0)

    with torch.no_grad():
        posterior = model.get_node_model("n").posterior(candidates)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    probabilities = model.predict_feasibility("n", _make_mixed_rows())
    assert probabilities.between(0.0, 1.0).all()


def _make_oof_classifier_case(mode):
    """Build numeric-output feasibility data with real success and failure rows."""
    from baybe.multistep.data import MultiStepData, NodeMeasurement
    from baybe.multistep.node import FeasibilityCondition

    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))

    def make_dag(*, product_free: bool):
        node = MultiStepNode(
            name="n",
            output="y",
            inputs=(x0,),
            feasibility_conditions=(
                FeasibilityCondition(column="y", operator=">=", threshold=1.0),
            ),
            infeasible_measurement_mode=mode,
        )
        return MultiStepDAG(
            nodes=(node,),
            objective=SingleTargetObjective(NumericalTarget("y")),
            product_free=product_free,
        )

    data = MultiStepData.empty().add(
        *(
            NodeMeasurement(node="n", inputs={"x0": value}, output=output)
            for value, output in [
                (0.1, 2.0),
                (0.15, 0.5),
                (0.2, 2.5),
                (0.35, 1.5),
                (0.5, 3.0),
                (0.6, 0.8),
                (0.65, 2.2),
                (0.8, 4.0),
            ]
        )
    )
    return make_dag(product_free=False), make_dag(product_free=True), data


@pytest.mark.parametrize(
    "mode_name", ["OBSERVED", "PREDICTED_MEAN"], ids=["observed", "predicted-mean"]
)
def test_product_free_cross_fitted_classifier_matches_legacy(mode_name, monkeypatch):
    """The out-of-fold numeric features and classifier match the real space."""
    import math

    import torch

    import baybe.multistep.model as model_module
    from baybe.multistep.model import MultiStepModel
    from baybe.recommenders.pure.experimental_feasibility import (
        InfeasibleMeasurementMode,
    )

    mode = InfeasibleMeasurementMode[mode_name]
    legacy_dag, free_dag, data = _make_oof_classifier_case(mode)
    frame = data.node_feasibility_training_data(free_dag, "n")

    legacy_model = MultiStepModel(dag=legacy_dag)
    free_model = MultiStepModel(dag=free_dag)

    torch.manual_seed(5)
    expected, expected_names = legacy_model._fit_cross_fitted_classifier(
        legacy_dag.nodes[0], frame
    )

    auxiliary_fits: list[tuple] = []
    real_predicted_mean_rows = model_module._predicted_mean_rows
    fold_state = _spy_folds(monkeypatch, model_module)

    def record_predicted_mean(dag, node, successes, failures):
        rows = real_predicted_mean_rows(dag, node, successes, failures)
        auxiliary_fits.append(
            (
                successes,
                failures,
                rows,
                list(fold_state["train"]),
                list(fold_state["test"]),
            )
        )
        return rows

    calls = _forbid_product_builders(monkeypatch)
    monkeypatch.setattr(model_module, "_predicted_mean_rows", record_predicted_mean)
    torch.manual_seed(5)
    actual, actual_names = free_model._fit_cross_fitted_classifier(
        free_dag.nodes[0], frame
    )

    assert calls == []
    assert actual_names == expected_names
    assert free_model._feasibility_models == {}
    assert actual._searchspace is not None
    assert isinstance(actual._searchspace, _FactorizedInputLayout)

    if mode is InfeasibleMeasurementMode.PREDICTED_MEAN:
        assert auxiliary_fits
        for successes, failures, rows, train, test in auxiliary_fits:
            train_values = set(frame.iloc[train]["x0"])
            test_values = set(frame.iloc[test]["x0"])
            used_values = {inputs["x0"] for inputs, _ in successes}
            used_values |= {inputs["x0"] for inputs in failures}
            assert successes
            assert failures
            assert all(values["y"] >= 1.0 for _, values in successes)
            assert used_values <= train_values
            assert used_values.isdisjoint(test_values)
            assert len(rows) == len(failures)
            assert all(math.isfinite(row["y"]) for row in rows)
    else:
        assert auxiliary_fits == []

    expected_state = expected._model.state_dict()
    actual_state = actual._model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        _assert_tensors_close(actual_state[key], value)


def _make_two_output_oof_case():
    """Build a numerically conditioned two-output node with finite observations."""
    from baybe.multistep.data import MultiStepData, NodeMeasurement
    from baybe.multistep.node import FeasibilityCondition
    from baybe.objectives.pareto import ParetoObjective

    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    node = MultiStepNode(
        name="n",
        output=("a", "b"),
        inputs=(x0,),
        feasibility_conditions=(
            FeasibilityCondition(column="a", operator=">=", threshold=1.0),
            FeasibilityCondition(column="b", operator=">=", threshold=1.0),
        ),
    )
    dag = MultiStepDAG(
        nodes=(node,),
        objective=ParetoObjective(targets=(NumericalTarget("a"), NumericalTarget("b"))),
        product_free=True,
    )
    values = [
        (0.05, 2.0, 2.0),
        (0.15, 1.5, 1.8),
        (0.25, 0.5, 1.4),
        (0.35, 1.2, 0.6),
        (0.45, 2.5, 2.2),
        (0.55, 0.8, 1.9),
        (0.65, 1.8, 1.1),
        (0.75, 1.1, 1.3),
    ]
    data = MultiStepData.empty().add(
        *(
            NodeMeasurement(
                node="n",
                inputs={"x0": x0_value},
                output={"a": a_value, "b": b_value},
            )
            for x0_value, a_value, b_value in values
        )
    )
    return node, dag, data.node_feasibility_training_data(dag, "n")


def _make_partial_output_predicted_mean_case(product_free):
    """Build a PREDICTED_MEAN node whose successful row misses one output."""
    from baybe.multistep.data import MultiStepData, NodeMeasurement
    from baybe.multistep.node import FeasibilityCondition
    from baybe.objectives.pareto import ParetoObjective
    from baybe.recommenders.pure.experimental_feasibility import (
        InfeasibleMeasurementMode,
    )

    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    node = MultiStepNode(
        name="n",
        output=("a", "b"),
        inputs=(x0,),
        feasibility_conditions=(
            FeasibilityCondition(column="a", operator=">=", threshold=1.0),
        ),
        infeasible_measurement_mode=InfeasibleMeasurementMode.PREDICTED_MEAN,
    )
    dag = MultiStepDAG(
        nodes=(node,),
        objective=ParetoObjective(targets=(NumericalTarget("a"), NumericalTarget("b"))),
        product_free=product_free,
    )
    data = MultiStepData.empty().add(
        NodeMeasurement(node="n", inputs={"x0": 0.1}, output={"a": 2.0, "b": 2.0}),
        NodeMeasurement(node="n", inputs={"x0": 0.2}, output={"a": 2.5}),
        NodeMeasurement(node="n", inputs={"x0": 0.3}, output={"a": 3.0, "b": 3.0}),
        NodeMeasurement(node="n", inputs={"x0": 0.4}, output={"a": 0.0, "b": 1.0}),
    )
    return dag, data


def _lower_tail_probability(node, output_name, posterior):
    """Compute P(Y >= lower) for an output conditioned only from below."""
    import torch

    thresholds = [
        condition.threshold
        for condition in node.feasibility_conditions
        if condition.column == output_name
        and condition.operator == ">="
        and condition.threshold is not None
    ]
    mean = posterior.mean[..., 0]
    standard_deviation = posterior.variance[..., 0].clamp_min(0.0).sqrt()
    normal = torch.distributions.Normal(mean, standard_deviation)
    lower = max(thresholds)
    return 1.0 - normal.cdf(torch.as_tensor(lower, dtype=mean.dtype))


def test_product_free_multi_output_oof_uses_each_local_output(monkeypatch):
    """Both numerically conditioned outputs produce correct finite OOF features."""
    import numpy as np
    import torch

    import baybe.multistep.model as model_module
    from baybe.multistep.model import MultiStepModel
    from baybe.utils.dataframe import to_tensor

    node, dag, raw = _make_two_output_oof_case()
    model = MultiStepModel(dag=dag)

    fold_state = _spy_folds(monkeypatch, model_module)

    fits: list[dict] = []
    original_surrogate = model_module._make_local_surrogate

    def surrogate_spy(
        node_object, fit_context, target, measurements, label, template=None
    ):
        surrogate = original_surrogate(
            node_object, fit_context, target, measurements, label, template
        )
        fits.append(
            {
                "label": label,
                "train": list(fold_state["train"]),
                "test": list(fold_state["test"]),
                "measurements": measurements.copy(),
                "surrogate": surrogate,
                "fit_context": fit_context,
            }
        )
        return surrogate

    records: list[dict] = []
    original_probability = model_module._interval_probability

    def probability_spy(node_object, output_name, posterior, **kwargs):
        result = original_probability(node_object, output_name, posterior, **kwargs)
        records.append(
            {
                "output": output_name,
                "train": list(fold_state["train"]),
                "test": list(fold_state["test"]),
                "posterior": posterior,
                "values": result.detach().clone(),
            }
        )
        return result

    calls = _forbid_product_builders(monkeypatch)
    monkeypatch.setattr(model_module, "_make_local_surrogate", surrogate_spy)
    monkeypatch.setattr(model_module, "_interval_probability", probability_spy)
    _seed_deterministic_fit()

    classifier, feature_names = model._fit_cross_fitted_classifier(node, raw)

    assert calls == []
    assert feature_names.keys() == {"a", "b"}
    assert isinstance(classifier._searchspace, _FactorizedInputLayout)
    assert classifier._model is not None
    assert records
    assert {record["output"] for record in records} == {"a", "b"}

    expected: dict[str, dict[int, float]] = {"a": {}, "b": {}}
    for record in records:
        posterior = record["posterior"]
        assert posterior.mean.shape[-1] == 1
        assert record["values"].numel() == len(record["test"])
        analytic = _lower_tail_probability(node, record["output"], posterior)
        torch.testing.assert_close(
            record["values"].reshape(-1),
            analytic.reshape(-1),
            atol=1e-8,
            rtol=1e-6,
        )

        # The posterior must stem from the local model fitted on this fold's
        # output-specific training data, not merely from any passed posterior.
        matches = [
            fit
            for fit in fits
            if fit["label"] == record["output"]
            and fit["train"] == record["train"]
            and fit["test"] == record["test"]
        ]
        assert len(matches) == 1
        fit = matches[0]
        train_frame = raw.iloc[fit["train"]]
        assert list(fit["measurements"].columns) == ["x0", fit["label"]]
        assert fit["measurements"]["x0"].tolist() == train_frame["x0"].tolist()
        assert (
            fit["measurements"][fit["label"]].tolist()
            == train_frame[fit["label"]].tolist()
        )
        test_inputs = raw.iloc[fit["test"]][["x0"]]
        refreshed = (
            fit["surrogate"]
            .to_botorch()
            .posterior(
                to_tensor(fit["fit_context"].transform(test_inputs)).unsqueeze(-2)
            )
        )
        _assert_tensors_close(refreshed.mean, posterior.mean)
        _assert_tensors_close(refreshed.variance, posterior.variance)

        for position, value in zip(
            record["test"], record["values"].reshape(-1).tolist()
        ):
            expected[record["output"]][position] = value

    measurements = classifier._measurements
    for output_name, feature_name in feature_names.items():
        values = measurements[feature_name].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()
        for position, value in expected[output_name].items():
            assert np.isclose(values[position], value, atol=1e-8, rtol=1e-6)

    # The records cover every row exactly once per output.
    assert set(expected["a"]) == set(range(len(raw)))
    assert set(expected["b"]) == set(range(len(raw)))


@pytest.mark.parametrize("product_free", [False, True], ids=["legacy", "product-free"])
def test_predicted_mean_oof_partial_success_respects_path_filter(
    product_free, monkeypatch
):
    """Only the product-free OOF path drops NaN partial successes."""
    import numpy as np

    from baybe.multistep.model import MultiStepModel

    dag, data = _make_partial_output_predicted_mean_case(product_free)
    frame = data.node_feasibility_training_data(dag, "n")
    model = MultiStepModel(dag=dag)
    calls = _forbid_product_builders(monkeypatch) if product_free else []

    if product_free:
        classifier, feature_names = model._fit_cross_fitted_classifier(
            dag.nodes[0], frame
        )
        assert calls == []
        assert set(feature_names) == {"a"}
        assert np.isfinite(classifier._measurements.to_numpy(dtype=float)).all()
    else:
        with pytest.raises(IncompatibilityError, match="PREDICTED_MEAN.*output 'b'"):
            model._fit_cross_fitted_classifier(dag.nodes[0], frame)


@pytest.mark.parametrize("n_observations", [0, 1])
def test_local_prior_layout_matches_real_searchspace(
    n_observations, mixed_parameters, mixed_rows
):
    """Zero/one-observation local priors match on both fit contexts."""
    from baybe.multistep.data import _make_local_searchspace
    from baybe.multistep.model import _make_local_surrogate
    from baybe.utils.dataframe import to_tensor

    node = MultiStepNode(name="n", output="y", inputs=mixed_parameters)
    dag = MultiStepDAG(
        nodes=(node,),
        objective=SingleTargetObjective(NumericalTarget("y")),
        product_free=True,
    )
    frame = mixed_rows.assign(y=_target_values()).iloc[:n_observations]
    target = NumericalTarget("y")
    legacy_context = _make_local_searchspace(dag.context, node, frame)
    layout = _FactorizedInputLayout.from_parameters(dag.context._parameters(node))

    _seed_deterministic_fit()
    legacy = _make_local_surrogate(node, legacy_context, target, frame, "y")
    _seed_deterministic_fit()
    free = _make_local_surrogate(node, layout, target, frame, "y")

    assert legacy._searchspace is legacy_context
    assert free._searchspace is layout

    expected_state = legacy._model.state_dict()
    actual_state = free._model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        _assert_tensors_close(actual_state[key], value)

    candidates = mixed_rows
    expected = legacy._posterior_comp(
        to_tensor(legacy_context.transform(candidates)).unsqueeze(-2)
    )
    actual = free._posterior_comp(to_tensor(layout.transform(candidates)).unsqueeze(-2))
    _assert_tensors_close(actual.mean, expected.mean)
    _assert_tensors_close(actual.variance, expected.variance)


def _make_public_fit_rows() -> pd.DataFrame:
    """Complete single-node DAG evaluation rows for the mixed parameter set."""
    return _make_mixed_rows().assign(y=_target_values())


def _make_public_fit_oracle(parameters):
    """Create a legacy oracle fitted on the same complete rows."""
    from baybe.multistep.model import MultiStepModel

    rows = _make_public_fit_rows()
    legacy_dag = _make_single_node_dag(parameters, product_free=False)
    legacy_space = SearchSpace.from_product(parameters)
    oracle = MultiStepModel(dag=legacy_dag, num_network_samples=4, sampling_seed=7)
    _seed_deterministic_fit()
    oracle.fit(None, legacy_dag.objective, rows)
    return oracle, legacy_space, rows


@pytest.mark.parametrize("searchspace_kind", ["none", "real"])
def test_product_free_public_fit_avoids_products_and_matches_oracle(
    searchspace_kind, monkeypatch
):
    """``fit(None, ...)`` and a pre-built real space fit without any product."""
    from baybe.multistep.model import MultiStepModel
    from baybe.utils.dataframe import to_tensor

    parameters = _make_mixed_parameters()
    oracle, legacy_space, rows = _make_public_fit_oracle(parameters)
    free_dag = _make_single_node_dag(parameters, product_free=True)
    layout = _FactorizedInputLayout.from_parameters(parameters)
    provided = SearchSpace.from_product(parameters)
    model = MultiStepModel(dag=free_dag, num_network_samples=4, sampling_seed=7)

    calls = _forbid_product_builders(monkeypatch)
    _seed_deterministic_fit()
    model.fit(
        None if searchspace_kind == "none" else provided, free_dag.objective, rows
    )

    assert calls == []
    surrogate = model._node_surrogates["root"]["y"]
    assert surrogate._searchspace == layout

    expected = oracle.get_node_model("root").posterior(
        to_tensor(legacy_space.transform(rows, allow_extra=True)).unsqueeze(0)
    )
    actual = model.get_node_model("root").posterior(
        to_tensor(layout.transform(rows, allow_extra=True)).unsqueeze(0)
    )
    _assert_tensors_close(actual.mean, expected.mean)
    _assert_tensors_close(actual.variance, expected.variance)


def test_product_free_actions_use_active_levels_and_real_parents():
    """New actions use active levels while historical parent levels stay valid."""
    from baybe.multistep import ContextMode, MultiStepContext
    from baybe.multistep.data import (
        MultiStepData,
        NodeMeasurement,
        _EvaluationRef,
        register_action,
    )

    parent_level = CategoricalParameter("p", values=("lo", "hi"), active_values=("hi",))
    own_level = CategoricalParameter("c", values=("a", "b"), active_values=("a",))
    parent = MultiStepNode(name="parent", output="parent_out", inputs=(parent_level,))
    child = MultiStepNode(
        name="child", output="child_out", inputs=(own_level,), parents=(parent,)
    )
    dag = MultiStepDAG(
        nodes=(parent, child),
        objective=SingleTargetObjective(NumericalTarget("child_out")),
        context=MultiStepContext(ContextMode.PARENT_INPUTS_OUTPUTS),
        product_free=True,
    )
    ref = _EvaluationRef(index=0, node="parent")
    historical_parent = NodeMeasurement(
        node="parent",
        inputs={"p": "lo"},
        output=1.5,
        experiment_id="exp-1",
    )
    data = MultiStepData.empty().add(historical_parent)
    data.validate(dag)

    contexts: dict = {}
    with pytest.raises(IncompatibilityError, match="not an active level"):
        register_action(
            contexts,
            0,
            experiment_id="exp-2",
            node="child",
            inputs={"c": "b", "p": "lo", "parent_out": 1.5},
            parent_refs={"parent": ref},
            data=data,
            dag=dag,
        )
    assert contexts == {}

    contexts, _ = register_action(
        contexts,
        0,
        experiment_id="exp-2",
        node="child",
        inputs={"c": "a", "p": "lo", "parent_out": 1.5},
        parent_refs={"parent": ref},
        data=data,
        dag=dag,
    )
    pending = contexts["exp-2"].pending["child"]
    assert dict(pending.inputs) == {"c": "a", "p": "lo", "parent_out": 1.5}
    assert dict(pending.parent_refs) == {"parent": ref}
    assert pending.parent_refs["parent"] is ref


def _make_direct_predicted_mean_refit_case():
    """Build two roots where the second uses PREDICTED_MEAN and a numeric condition."""
    from baybe.multistep.data import MultiStepData, NodeMeasurement
    from baybe.multistep.node import FeasibilityCondition
    from baybe.objectives.pareto import ParetoObjective
    from baybe.recommenders.pure.experimental_feasibility import (
        InfeasibleMeasurementMode,
    )

    x0 = NumericalContinuousParameter("x0", bounds=(0.0, 1.0))
    x1 = NumericalContinuousParameter("x1", bounds=(0.0, 1.0))
    first = MultiStepNode(name="first", output="y1", inputs=(x0,))
    second = MultiStepNode(
        name="second",
        output="y2",
        inputs=(x1,),
        feasibility_conditions=(
            FeasibilityCondition(column="y2", operator=">=", threshold=1.0),
        ),
        infeasible_measurement_mode=InfeasibleMeasurementMode.PREDICTED_MEAN,
    )
    dag = MultiStepDAG(
        nodes=(first, second),
        objective=ParetoObjective(
            targets=(NumericalTarget("y1"), NumericalTarget("y2"))
        ),
        product_free=True,
    )
    data = MultiStepData.empty().add(
        NodeMeasurement(node="first", inputs={"x0": 0.1}, output=1.0),
        NodeMeasurement(node="first", inputs={"x0": 0.5}, output=2.0),
        NodeMeasurement(node="first", inputs={"x0": 0.9}, output=3.0),
        NodeMeasurement(node="second", inputs={"x1": 0.1}, output=2.0),
        NodeMeasurement(node="second", inputs={"x1": 0.3}, output=3.0),
        NodeMeasurement(node="second", inputs={"x1": 0.5}, output=4.0),
        NodeMeasurement(node="second", inputs={"x1": 0.7}, output=5.0),
    )
    return dag, data


def test_product_free_campaign_predicted_mean_fit_avoids_products(monkeypatch):
    """A valid multi-node PREDICTED_MEAN campaign fit passes without products."""
    import baybe.multistep.data as data_module
    from baybe.multistep.campaign import MultiStepCampaign
    from baybe.multistep.data import NodeMeasurement

    dag, data = _make_direct_predicted_mean_refit_case()
    data = data.add(NodeMeasurement(node="second", inputs={"x1": 0.9}, output=0.0))
    campaign = MultiStepCampaign(dag=dag, data=data)
    product_calls = _forbid_product_builders(monkeypatch)

    auxiliary_calls: list[tuple] = []
    original = data_module._predicted_mean_rows

    def counting(*args, **kwargs):
        auxiliary_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(data_module, "_predicted_mean_rows", counting)

    campaign._fit_node_models()

    assert auxiliary_calls
    assert product_calls == []
    for args, _ in auxiliary_calls:
        successes, failures = args[2], args[3]
        assert successes
        assert failures
        assert all(values["y2"] >= 1.0 for _, values in successes)
    assert set(campaign.model._node_surrogates) == {"first", "second"}
    assert set(campaign.model._feasibility_models) == {"second"}
    for surrogates in campaign.model._node_surrogates.values():
        for surrogate in surrogates.values():
            assert isinstance(surrogate._searchspace, _FactorizedInputLayout)
