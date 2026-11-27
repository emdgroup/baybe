"""Focused tests for model-reference expected improvement."""

from __future__ import annotations

from types import SimpleNamespace

import hypothesis.strategies as st
import pytest
from hypothesis import given

from baybe.exceptions import IncompatibilityError
from baybe.multistep import MREI, ModelReferenceExpectedImprovement
from baybe.multistep.mrei.acquisition import (
    _compute_sample_expected_improvement,
    _transform_final_samples,
)
from tests.serialization.utils import assert_roundtrip_consistency


def _model_reference_ei_strategy() -> st.SearchStrategy[
    ModelReferenceExpectedImprovement
]:
    """Generate valid MREI configurations for serialization roundtrips."""
    return st.builds(
        ModelReferenceExpectedImprovement,
        reference=st.sampled_from(("max_mean", "mean", "thompson_max")),
        n_network_samples=st.integers(min_value=1, max_value=32),
        n_reference_points=st.integers(min_value=1, max_value=32),
        n_path_features=st.integers(min_value=1, max_value=32),
        sampling_seed=st.integers(min_value=0, max_value=2**32 - 1),
    )


@pytest.mark.parametrize("reference", ["max_mean", "mean", "thompson_max"])
def test_reference_accepts_only_three_modes(reference):
    assert ModelReferenceExpectedImprovement(reference=reference).reference == reference


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference": "median"},
        {"n_network_samples": 0},
        {"n_reference_points": True},
        {"n_path_features": 1.5},
        {"sampling_seed": -1},
        {"sampling_seed": False},
    ],
)
def test_configuration_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ModelReferenceExpectedImprovement(**kwargs)


def test_configuration_public_capabilities_and_alias():
    acquisition = MREI()
    assert acquisition.reference == "max_mean"
    assert acquisition.n_network_samples == 256
    assert acquisition.n_reference_points == 1024
    assert acquisition.n_path_features == 2048
    assert acquisition.sampling_seed == 0
    assert not acquisition.is_analytic
    assert not acquisition.supports_batching
    assert acquisition.supports_pending_experiments


@given(data=st.data())
def test_mrei_hypothesis_serialization_roundtrip(data):
    acquisition = data.draw(_model_reference_ei_strategy())

    assert_roundtrip_consistency(acquisition)


def test_ordinary_botorch_conversion_is_rejected():
    with pytest.raises(IncompatibilityError, match="MultiStepCampaign.recommend"):
        MREI().to_botorch(None, None, None, None)


def test_sample_ei_averages_positive_part_per_sample():
    import torch

    samples = torch.tensor([[[0.0]], [[2.0]], [[4.0]]], dtype=torch.double)
    assert _compute_sample_expected_improvement(samples, 2.0).item() == pytest.approx(
        2.0 / 3.0
    )


@pytest.mark.parametrize(
    ("reference", "expected"),
    [(2.0, 1.0), (3.0, 1.0 / 3.0), (4.0, 0.0)],
)
def test_sample_ei_deterministic_values_above_and_below_reference(reference, expected):
    import torch

    samples = torch.tensor([[[3.0]], [[2.0]], [[4.0]]], dtype=torch.double)
    result = _compute_sample_expected_improvement(samples, reference)
    assert result.tolist() == pytest.approx([expected])


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        pytest.param(2.0, 1.0, id="reference-below-draw"),
        pytest.param(3.0, 0.0, id="reference-equals-draw"),
        pytest.param(4.0, 0.0, id="reference-above-draw"),
    ],
)
def test_deterministic_sample_three_against_references_two_three_four(
    reference, expected
):
    import torch

    result = _compute_sample_expected_improvement(
        torch.tensor([[[3.0]]], dtype=torch.double), reference
    )

    assert result.item() == pytest.approx(expected)


def test_sample_ei_rejects_invalid_shape_and_nonfinite_samples():
    import torch

    with pytest.raises(IncompatibilityError, match="q=1"):
        _compute_sample_expected_improvement(torch.ones((2, 2, 2)), 0.0)
    with pytest.raises(IncompatibilityError, match="finite"):
        _compute_sample_expected_improvement(torch.tensor([[[float("nan")]]]), 0.0)
    with pytest.raises(IncompatibilityError, match="finite"):
        _compute_sample_expected_improvement(torch.tensor([[[float("inf")]]]), 0.0)


def test_sample_ei_matches_standard_normal_ei_oracle():
    from math import pi, sqrt

    import torch

    n_samples = 8192
    probabilities = (torch.arange(n_samples, dtype=torch.double) + 0.5) / n_samples
    samples = (
        torch.distributions.Normal(0.0, 1.0)
        .icdf(probabilities)
        .reshape(n_samples, 1, 1)
    )

    estimate = _compute_sample_expected_improvement(samples, 0.0).item()

    assert estimate == pytest.approx(1.0 / sqrt(2.0 * pi), abs=5e-3)


def test_sample_ei_reduces_nongaussian_samples_directly():
    import torch

    samples = torch.tensor([[[0.0]], [[0.0]], [[10.0]]], dtype=torch.double)

    assert _compute_sample_expected_improvement(samples, 1.0).item() == pytest.approx(
        3.0
    )


def test_sample_ei_is_one_for_draws_nine_and_twelve_above_reference_ten():
    import torch

    samples = torch.tensor([[[9.0]], [[12.0]]], dtype=torch.double)

    assert _compute_sample_expected_improvement(samples, 10.0).item() == pytest.approx(
        1.0
    )


def test_final_transform_orders_outputs_then_transforms_each_sample():
    import torch

    class SquareObjective:
        """Return the square of the first objective-ordered output."""

        @staticmethod
        def _full_transformation(samples, _X):
            return samples[..., 0].square()

    dag = SimpleNamespace(
        graph_output_names=("other", "selected"),
        objective_output_indices=(1,),
        objective=SquareObjective(),
    )
    raw = torch.tensor([[[[2.0, 3.0]]], [[[4.0, 5.0]]]], dtype=torch.double)
    result = _transform_final_samples(dag, raw)
    assert result.shape == (2, 1, 1)
    assert result.flatten().tolist() == pytest.approx([9.0, 25.0])


def test_final_transform_applies_minimization_orientation():
    import torch

    from baybe.objectives.single import SingleTargetObjective
    from baybe.targets.numerical import NumericalTarget

    target = NumericalTarget(name="loss", minimize=True)
    objective = SingleTargetObjective(target=target)
    dag = SimpleNamespace(
        graph_output_names=("loss",),
        objective_output_indices=(0,),
        objective=objective,
    )
    raw = torch.tensor([[[[2.0]]], [[[5.0]]]], dtype=torch.double)
    assert _transform_final_samples(dag, raw).flatten().tolist() == [-2.0, -5.0]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_final_transform_scalarizes_desirability_in_sample_dtype(dtype):
    import torch

    from baybe.objectives.desirability import DesirabilityObjective
    from baybe.targets.numerical import NumericalTarget

    torch_dtype = getattr(torch, dtype)
    objective = DesirabilityObjective(
        [NumericalTarget("a").clamp(min=0), NumericalTarget("b").clamp(min=0)],
        [1, 3],
        scalarizer="MEAN",
        require_normalization=False,
    )
    dag = SimpleNamespace(
        graph_output_names=("b", "a"),
        objective_output_indices=(1, 0),
        objective=objective,
    )
    samples = torch.tensor([[[[3.0, -1.0]]]], dtype=torch_dtype)

    result = _transform_final_samples(dag, samples)

    assert result.dtype == torch_dtype
    assert result.item() == pytest.approx(2.25)
