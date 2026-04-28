"""Tests for RGPE surrogate."""

import pandas as pd
import pytest

from baybe import Campaign
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.categorical import TaskCorrelation
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates.rgpe import RGPESurrogate
from baybe.targets import NumericalTarget


@pytest.fixture
def tl_searchspace():
    """A simple TL search space with RANKED correlation."""
    params = [
        NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        TaskParameter(
            "task",
            values=["source_1", "source_2", "target"],
            active_values=["target"],
            task_correlation=TaskCorrelation.RANKED,
        ),
    ]
    return SearchSpace.from_product(parameters=params)


@pytest.fixture
def objective():
    """A simple single-target objective."""
    return NumericalTarget(name="y").to_objective()


@pytest.fixture
def measurements():
    """Training data with source and target tasks."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 5.0],
            "y": [
                1.0,
                4.0,
                9.0,
                16.0,
                25.0,  # source_1 (quadratic)
                2.0,
                5.0,
                10.0,
                17.0,
                26.0,  # source_2 (quadratic + 1)
                1.5,
                9.5,
                25.5,  # target (quadratic + noise)
            ],
            "task": ["source_1"] * 5 + ["source_2"] * 5 + ["target"] * 3,
        }
    )


class TestRGPESurrogateUnit:
    """Unit tests for RGPESurrogate."""

    def test_fit_and_posterior(self, tl_searchspace, objective, measurements):
        """RGPE surrogate can fit and produce predictions."""
        model = RGPESurrogate(num_posterior_samples=32)
        model.fit(tl_searchspace, objective, measurements)

        # Predict on target candidates
        candidates = pd.DataFrame({"x": [2.0, 4.0], "task": ["target", "target"]})
        posterior = model.posterior(candidates)
        assert posterior.mean.shape == (2, 1)

    def test_to_botorch_interface(self, tl_searchspace, objective, measurements):
        """to_botorch() returns a model that accepts full-dim input."""
        import torch

        model = RGPESurrogate(num_posterior_samples=32)
        model.fit(tl_searchspace, objective, measurements)
        botorch_model = model.to_botorch()

        # Full computational input (includes task column)
        # task column is the second column, target task index
        target_idx = tl_searchspace.target_task_idxs[0]
        test_x = torch.tensor(
            [[2.0, target_idx], [4.0, target_idx]], dtype=torch.double
        )
        posterior = botorch_model.posterior(test_x)
        assert posterior.mean.shape[-1] == 1

    def test_weights_sum_to_one(self, tl_searchspace, objective, measurements):
        """Rank weights should sum to 1."""
        model = RGPESurrogate(num_posterior_samples=64)
        model.fit(tl_searchspace, objective, measurements)
        weights = model._rgpe_model.weights
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_num_models_correct(self, tl_searchspace, objective, measurements):
        """Number of models = number of source tasks + 1 (target)."""
        model = RGPESurrogate(num_posterior_samples=32)
        model.fit(tl_searchspace, objective, measurements)
        # 2 source tasks + 1 target = 3 models
        assert len(model._rgpe_model.models) == 3


class TestRGPECampaignIntegration:
    """Integration tests: RGPE through Campaign."""

    def test_campaign_recommend(self, tl_searchspace, objective, measurements):
        """Campaign with RANKED task correlation produces recommendations."""
        campaign = Campaign(
            searchspace=tl_searchspace,
            objective=objective,
            recommender=BotorchRecommender(),
        )
        campaign.add_measurements(measurements)
        rec = campaign.recommend(1)

        # Recommendations should include task column with target value
        assert "task" in rec.columns
        assert all(rec["task"] == "target")
        # Should also include the feature parameter
        assert "x" in rec.columns

    def test_auto_dispatch(self, tl_searchspace, objective, measurements):
        """BayesianRecommender auto-swaps to RGPE for RANKED correlation."""
        recommender = BotorchRecommender()
        surrogate = recommender.get_surrogate(tl_searchspace, objective, measurements)
        assert isinstance(surrogate, RGPESurrogate)

    def test_multiple_recommendations(self, tl_searchspace, objective, measurements):
        """Can do multiple rounds of recommend-measure."""
        campaign = Campaign(
            searchspace=tl_searchspace,
            objective=objective,
            recommender=BotorchRecommender(),
        )
        campaign.add_measurements(measurements)
        rec1 = campaign.recommend(1)

        # Add the recommendation as a new measurement
        rec1_with_y = rec1.copy()
        rec1_with_y["y"] = 10.0
        campaign.add_measurements(rec1_with_y)
        rec2 = campaign.recommend(1)
        assert len(rec2) == 1


class TestRGPEEdgeCases:
    """Edge case tests."""

    def test_single_source_task(self):
        """Works with a single source task."""
        params = [
            NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0, 4.0, 5.0]),
            TaskParameter(
                "task",
                values=["source", "target"],
                active_values=["target"],
                task_correlation=TaskCorrelation.RANKED,
            ),
        ]
        ss = SearchSpace.from_product(parameters=params)
        obj = NumericalTarget(name="y").to_objective()
        data = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0],
                "y": [1.0, 4.0, 9.0, 16.0, 25.0, 5.0, 17.0],
                "task": ["source"] * 5 + ["target"] * 2,
            }
        )
        model = RGPESurrogate(num_posterior_samples=32)
        model.fit(ss, obj, data)
        candidates = pd.DataFrame({"x": [3.0], "task": ["target"]})
        posterior = model.posterior(candidates)
        assert posterior.mean.shape == (1, 1)

    def test_target_only_data(self, tl_searchspace, objective):
        """Works with only target task data (no source data)."""
        data = pd.DataFrame(
            {
                "x": [1.0, 3.0, 5.0],
                "y": [1.5, 9.5, 25.5],
                "task": ["target"] * 3,
            }
        )
        model = RGPESurrogate(num_posterior_samples=32)
        model.fit(tl_searchspace, objective, data)
        candidates = pd.DataFrame({"x": [2.0], "task": ["target"]})
        posterior = model.posterior(candidates)
        assert posterior.mean.shape == (1, 1)

    def test_ranked_validation_multiple_active(self):
        """RANKED with multiple active values raises ValueError."""
        with pytest.raises(ValueError, match="RANKED mode assumes a single"):
            TaskParameter(
                "task",
                values=["t1", "t2", "t3"],
                active_values=["t1", "t2"],
                task_correlation=TaskCorrelation.RANKED,
            )
