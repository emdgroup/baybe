"""Tests for factorized multi-step configuration layouts."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from baybe.exceptions import IncompatibilityError
from baybe.multistep import MultiStepDAG, MultiStepModel, MultiStepNode
from baybe.multistep._configuration_space import _ConfigurationSpace
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.targets.numerical import NumericalTarget


@pytest.fixture(name="mixed_dag")
def fixture_mixed_dag():
    """Create an unsorted mixed-parameter DAG with categorical encoding."""
    node = MultiStepNode(
        name="root",
        output="utility",
        inputs=(
            NumericalContinuousParameter("z_cont", (2.0, 5.0)),
            CategoricalParameter("category", ("b", "a")),
            NumericalDiscreteParameter("d_disc", (0.0, 2.0, 4.0)),
            NumericalContinuousParameter("a_cont", (-1.0, 1.0)),
        ),
    )
    return MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )


def test_configuration_space_matches_small_product(mixed_dag):
    """Keep the old product's transform, order, and bounds exactly."""
    layout = _ConfigurationSpace.from_dag(mixed_dag)
    product = mixed_dag.full_searchspace
    frame = pd.DataFrame(
        {
            "z_cont": [2.5, 4.0],
            "category": ["a", "b"],
            "d_disc": [0.0, 4.0],
            "a_cont": [0.0, -0.5],
        }
    )

    assert layout.parameter_names == tuple(
        parameter.name for node in mixed_dag.nodes for parameter in node.inputs
    )
    assert layout.comp_rep_columns == product.comp_rep_columns
    assert_frame_equal(layout.transform(frame), product.transform(frame))
    assert_frame_equal(layout.bounds, product.comp_rep_bounds)
    layout.validate(frame.iloc[0].to_dict())


def test_sampling_is_factorized_and_seeded(mixed_dag):
    """Sample each owned parameter reproducibly without product enumeration."""
    layout = _ConfigurationSpace.from_dag(mixed_dag)

    first = layout.sample(sampling_seed=31)
    second = layout.sample(sampling_seed=31)

    assert_frame_equal(first, second)
    assert len(first) == 1
    row = first.iloc[0]
    assert -1.0 <= row["a_cont"] <= 1.0
    assert 2.0 <= row["z_cont"] <= 5.0
    assert row["d_disc"] in (0.0, 2.0, 4.0)
    assert row["category"] in ("a", "b")


def test_parameterless_node_and_empty_layout_have_one_sample_row():
    """Keep parameterless nodes and zero-dimensional layouts well-defined."""
    root = MultiStepNode(
        name="root",
        output="intermediate",
        inputs=(NumericalContinuousParameter("x", (0.0, 1.0)),),
    )
    terminal = MultiStepNode(name="terminal", output="utility", parents=(root,))
    dag = MultiStepDAG(
        nodes=(root, terminal), objective=NumericalTarget("utility").to_objective()
    )
    layout = _ConfigurationSpace.from_dag(dag)
    empty_layout = _ConfigurationSpace.from_dag(SimpleNamespace(parameter_by_name={}))

    assert layout.parameter_names == ("x",)
    assert empty_layout.sample(sampling_seed=0).shape == (1, 0)
    assert empty_layout.transform(pd.DataFrame(index=[0])).shape == (1, 0)


def test_transform_rejects_duplicate_parameter_columns():
    """Reject ambiguous experimental input columns before encoding."""
    layout = _ConfigurationSpace(
        parameters=(NumericalContinuousParameter("x", (0.0, 1.0)),)
    )
    frame = pd.DataFrame([[0.2, 0.8]], columns=["x", "x"])

    with pytest.raises(IncompatibilityError, match="unique"):
        layout.transform(frame)


def test_network_posterior_avoids_global_product(monkeypatch):
    """Use factorized comp-rep columns for posterior validation and gradients."""
    import torch
    from botorch.models import SingleTaskGP

    parameter = NumericalContinuousParameter("x", (0.0, 1.0))
    node = MultiStepNode(name="root", output="utility", inputs=(parameter,))
    dag = MultiStepDAG(
        nodes=(node,), objective=NumericalTarget("utility").to_objective()
    )
    node_model = SingleTaskGP(
        train_X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        train_Y=torch.tensor([[0.0], [1.0]], dtype=torch.double),
    )
    model = MultiStepModel.from_node_models(dag, {"root": node_model})
    product_columns = dag.full_searchspace.comp_rep_columns
    model.__dict__["_full_comp_rep_index"] = {
        column: index for index, column in enumerate(product_columns)
    }
    reference_candidates = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    reference_candidates.requires_grad_(True)
    reference_posterior = model.posterior(reference_candidates)
    reference_gradient = torch.autograd.grad(
        reference_posterior.mean.sum(), reference_candidates
    )[0]
    del model.__dict__["_full_comp_rep_index"]

    monkeypatch.setattr(
        type(dag),
        "full_searchspace",
        property(lambda _self: pytest.fail("global product was accessed")),
    )
    candidates = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    candidates.requires_grad_(True)

    posterior = model.posterior(candidates)
    gradient = torch.autograd.grad(posterior.mean.sum(), candidates)[0]

    assert tuple(posterior.mean.shape) == (2, 1, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(posterior.mean, reference_posterior.mean)
    torch.testing.assert_close(posterior.variance, reference_posterior.variance)
    torch.testing.assert_close(gradient, reference_gradient)
