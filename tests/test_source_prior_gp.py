"""Tests for SourcePriorGaussianProcessSurrogate with DataFrame-based approach."""

import pandas as pd
import pytest
import torch

from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.core import TransferConfig
from baybe.surrogates.transfer_learning.source_prior import (
    SourcePriorGaussianProcessSurrogate,
)
from baybe.targets import NumericalTarget


def test_source_prior_gp_fit_and_posterior():
    """Test that SourcePriorGP can fit and produce posterior predictions."""
    # Create simple searchspace with TaskParameter
    task_param = TaskParameter(
        name="task", values=["source1", "source2", "target"], active_values=["target"]
    )
    cont_param = NumericalContinuousParameter(name="x", bounds=(0, 1))
    searchspace = SearchSpace.from_product([task_param, cont_param])

    # Create objective
    target = NumericalTarget(name="y", mode="MAX")
    objective = SingleTargetObjective(target=target)

    # Create training data with source and target tasks
    measurements = pd.DataFrame(
        {
            "task": ["source1", "source1", "source2", "source2", "target", "target"],
            "x": [0.1, 0.3, 0.2, 0.4, 0.5, 0.7],
            "y": [1.0, 2.0, 1.5, 2.5, 3.0, 4.0],
        }
    )

    # Create and fit surrogate
    surrogate = SourcePriorGaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    # Check that both source and target surrogates exist
    assert surrogate._source_surrogate is not None
    assert surrogate._target_surrogate is not None
    assert surrogate._reduced_searchspace is not None

    # Test posterior prediction
    candidates = pd.DataFrame({"task": ["target", "target"], "x": [0.6, 0.8]})

    posterior = surrogate.posterior(candidates)

    # Check posterior output dimensions
    assert posterior.mean.shape == (2, 1)  # 2 candidates, 1 target
    assert posterior.variance.shape == (2, 1)

    # Check values are reasonable (not NaN/inf)
    assert not torch.any(torch.isnan(posterior.mean))
    assert not torch.any(torch.isnan(posterior.variance))
    assert torch.all(posterior.variance > 0)  # Variance should be positive


def test_source_prior_gp_no_target_data():
    """Test behavior when no target data is provided."""
    task_param = TaskParameter(
        name="task", values=["source1", "target"], active_values=["target"]
    )
    cont_param = NumericalContinuousParameter(name="x", bounds=(0, 1))
    searchspace = SearchSpace.from_product([task_param, cont_param])

    target = NumericalTarget(name="y", mode="MAX")
    objective = SingleTargetObjective(target=target)

    # Only source data
    measurements = pd.DataFrame(
        {"task": ["source1", "source1"], "x": [0.1, 0.3], "y": [1.0, 2.0]}
    )

    surrogate = SourcePriorGaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    # Should still work (target GP = source GP)
    assert surrogate._source_surrogate is not None
    assert surrogate._target_surrogate is not None


def test_source_prior_gp_validation():
    """Test that proper validation occurs."""
    # SearchSpace without TaskParameter should fail
    cont_param = NumericalContinuousParameter(name="x", bounds=(0, 1))
    searchspace = SearchSpace.from_product([cont_param])

    target = NumericalTarget(name="y", mode="MAX")
    objective = SingleTargetObjective(target=target)

    measurements = pd.DataFrame({"x": [0.1, 0.3], "y": [1.0, 2.0]})

    surrogate = SourcePriorGaussianProcessSurrogate()

    with pytest.raises(
        ValueError, match="SourcePriorGaussianProcessSurrogate requires a TaskParameter"
    ):
        surrogate.fit(searchspace, objective, measurements)


def test_source_prior_gp_no_source_data():
    """Test that error is raised when no source data is available."""
    task_param = TaskParameter(
        name="task", values=["source", "target"], active_values=["target"]
    )
    cont_param = NumericalContinuousParameter(name="x", bounds=(0, 1))
    searchspace = SearchSpace.from_product([task_param, cont_param])

    target = NumericalTarget(name="y", mode="MAX")
    objective = SingleTargetObjective(target=target)

    # Only target data (no source data)
    measurements = pd.DataFrame(
        {"task": ["target", "target"], "x": [0.1, 0.3], "y": [1.0, 2.0]}
    )

    surrogate = SourcePriorGaussianProcessSurrogate()

    with pytest.raises(ValueError, match="No source data found"):
        surrogate.fit(searchspace, objective, measurements)


def test_source_prior_gp_to_botorch():
    """Test that to_botorch() method works correctly."""
    task_param = TaskParameter(
        name="task", values=["source", "target"], active_values=["target"]
    )
    cont_param = NumericalContinuousParameter(name="x", bounds=(0, 1))
    searchspace = SearchSpace.from_product([task_param, cont_param])

    target = NumericalTarget(name="y", mode="MAX")
    objective = SingleTargetObjective(target=target)

    measurements = pd.DataFrame(
        {
            "task": ["source", "source", "target", "target"],
            "x": [0.1, 0.2, 0.5, 0.7],
            "y": [1.0, 1.5, 3.0, 4.0],
        }
    )

    surrogate = SourcePriorGaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)

    # Test to_botorch method
    botorch_model = surrogate.to_botorch()
    assert botorch_model is not None
    assert hasattr(botorch_model, "posterior")
    assert hasattr(botorch_model, "num_outputs")
    assert botorch_model.num_outputs == 1


def test_source_prior_gp_transfer_modes():
    """Test different transfer learning modes."""
    task_param = TaskParameter(
        name="task", values=["source", "target"], active_values=["target"]
    )
    cont_param = NumericalContinuousParameter(name="x", bounds=(0, 1))
    searchspace = SearchSpace.from_product([task_param, cont_param])

    target = NumericalTarget(name="y", mode="MAX")
    objective = SingleTargetObjective(target=target)

    measurements = pd.DataFrame(
        {
            "task": ["source", "source", "target", "target"],
            "x": [0.1, 0.2, 0.5, 0.7],
            "y": [1.0, 1.5, 3.0, 4.0],
        }
    )

    # Test with prior mean transfer
    surrogate_mean = SourcePriorGaussianProcessSurrogate(
        mean_transfer=TransferConfig("freeze", "source"), covariance_transfer=None
    )
    surrogate_mean.fit(searchspace, objective, measurements)
    assert surrogate_mean._target_surrogate is not None

    # Test with prior kernel transfer
    surrogate_kernel = SourcePriorGaussianProcessSurrogate(
        mean_transfer=None, covariance_transfer=TransferConfig("freeze", "target")
    )
    surrogate_kernel.fit(searchspace, objective, measurements)
    assert surrogate_kernel._target_surrogate is not None

    with pytest.raises(
        ValueError,
    ):
        surrogate_independent = SourcePriorGaussianProcessSurrogate(
            mean_transfer=None, covariance_transfer=None
        )
        surrogate_independent.fit(searchspace, objective, measurements)
