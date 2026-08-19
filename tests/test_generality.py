"""Tests for generality-oriented Bayesian optimization (CurryBO)."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest
import torch
from torch import nn

from baybe.aggregation import MeanAggregation, MinAggregation, SigmoidAggregation
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.parameters.categorical import GeneralityParameter
from baybe.searchspace import SearchSpace


class _NoOpTransform(nn.Module):
    """Identity transform matching BoTorch's (Y, Yvar) call signature."""

    def forward(self, Y, Yvar=None):
        return Y


# ─── Aggregation unit tests ───────────────────────────────────────────────────


class TestAggregation:
    """Unit tests for aggregation functions."""

    def test_mean(self):
        """MeanAggregation averages over last dim."""
        Y = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(MeanAggregation().forward(Y), torch.tensor([2.0]))

    def test_min(self):
        """MinAggregation takes minimum over last dim."""
        Y = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(MinAggregation().forward(Y), torch.tensor([1.0]))

    def test_sigmoid(self):
        """SigmoidAggregation gives fraction above threshold (smooth)."""
        Y = torch.tensor([[0.0, 1.0]])  # threshold=0.5 → one below, one above
        result = SigmoidAggregation(threshold=0.5, steepness=100.0).forward(Y)
        # With high steepness, should be close to 0.5 (1 of 2 above)
        assert 0.4 < result.item() < 0.6

    def test_single_context(self):
        """All aggregations handle r=1 (single context value)."""
        Y = torch.tensor([[7.0]])
        assert MeanAggregation().forward(Y).item() == pytest.approx(7.0)
        assert MinAggregation().forward(Y).item() == pytest.approx(7.0)
        sig = SigmoidAggregation(threshold=5.0, steepness=100.0)
        assert sig.forward(Y).item() > 0.9

    def test_batch_shape_preserved(self):
        """Aggregation preserves leading batch dimensions."""
        Y = torch.randn(4, 3, 5)  # (batch=4, q=3, r=5)
        result = MeanAggregation().forward(Y)
        assert result.shape == (4, 3)


# ─── SearchSpace._split_by_generality ─────────────────────────────────────────


@pytest.fixture(name="gen_searchspace")
def fixture_gen_searchspace():
    """A search space with one design param and one generality param."""
    x = NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0])
    w = GeneralityParameter(
        name="solvent",
        context=CategoricalParameter("solvent", values=["A", "B", "C"]),
    )
    return SearchSpace.from_product(parameters=[x, w])


class TestSplitByGenerality:
    """Tests for _split_by_generality."""

    def test_design_subspace_excludes_generality(self, gen_searchspace):
        """Design subspace contains only non-generality parameters."""
        design_ss, _, _, _ = gen_searchspace._split_by_generality()
        assert "x" in design_ss.parameter_names
        assert "solvent" not in design_ss.parameter_names

    def test_w_values_shape(self, gen_searchspace):
        """w_values has shape (n_contexts, d_w)."""
        _, _, _, w_values = gen_searchspace._split_by_generality()
        assert w_values.shape[0] == 3  # A, B, C

    def test_indices_cover_all_columns(self, gen_searchspace):
        """x_col_indices + w_col_indices = all column indices."""
        _, x_idx, w_idx, _ = gen_searchspace._split_by_generality()
        n_cols = len(gen_searchspace.comp_rep_columns)
        assert sorted(x_idx + w_idx) == list(range(n_cols))

    def test_constraint_across_design_and_context_raises(self):
        """Constraint spanning both design and context params raises."""
        from baybe.constraints.conditions import (
            SubSelectionCondition,
            ThresholdCondition,
        )
        from baybe.constraints.discrete import DiscreteExcludeConstraint
        from baybe.exceptions import IncompatibilityError

        x = NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0])
        w = GeneralityParameter(
            name="solvent",
            context=CategoricalParameter("solvent", values=["A", "B", "C"]),
        )
        ss = SearchSpace.from_product(
            parameters=[x, w],
            constraints=[
                DiscreteExcludeConstraint(
                    parameters=["x", "solvent"],
                    combiner="AND",
                    conditions=[
                        ThresholdCondition(threshold=2.0, operator=">"),
                        SubSelectionCondition(selection=["A"]),
                    ],
                )
            ],
        )
        with pytest.raises(IncompatibilityError, match="[Cc]onstraint"):
            ss._split_by_generality()


# ─── SearchSpace._fix_parameters ──────────────────────────────────────────────


class TestFixParameters:
    """Tests for _fix_parameters."""

    def test_returns_new_space_with_fixed_values(self, gen_searchspace):
        """Fixed values propagate to the new space's discrete subspace."""
        fixed = gen_searchspace._fix_parameters(x=2.0)
        assert fixed.discrete._fixed_values == {"x": 2.0}

    def test_unknown_param_raises(self, gen_searchspace):
        """Fixing an unknown parameter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            gen_searchspace._fix_parameters(bogus=1.0)

    def test_original_unchanged(self, gen_searchspace):
        """Original space is not mutated."""
        gen_searchspace._fix_parameters(x=2.0)
        assert gen_searchspace.discrete._fixed_values == {}


# ─── SearchSpace._comp_rep_to_exp_rep ─────────────────────────────────────────


class TestCompRepToExpRep:
    """Tests for _comp_rep_to_exp_rep."""

    def test_numerical_roundtrip(self):
        """Numerical discrete parameter round-trips through comp-rep."""
        x = NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0])
        ss = SearchSpace.from_product(parameters=[x])
        comp_cols = ss.comp_rep_columns
        result = ss._comp_rep_to_exp_rep({comp_cols[0]: 2.0})
        assert result["x"] == 2.0

    def test_onehot_categorical_roundtrip(self):
        """One-hot encoded categorical round-trips correctly."""
        cat = CategoricalParameter("color", values=["red", "blue", "green"])
        ss = SearchSpace.from_product(parameters=[cat])
        # Get comp-rep for "blue"
        comp_row = ss.discrete.comp_rep[ss.discrete.exp_rep["color"] == "blue"].iloc[0]
        comp_dict = comp_row.to_dict()
        result = ss._comp_rep_to_exp_rep(comp_dict)
        assert result["color"] == "blue"


# ─── _ReducedSearchSpace ──────────────────────────────────────────────────────


class TestReducedSearchSpace:
    """Tests for _ReducedSearchSpace attribute guard."""

    def test_allowed_attributes_work(self):
        """Allowed attributes are accessible."""
        x = NumericalDiscreteParameter("x", values=[1.0, 2.0])
        cat = CategoricalParameter("c", values=["A", "B"])
        ss = SearchSpace.from_product(parameters=[x, cat])
        reduced = ss._drop_parameters({"x"})
        # These should not raise
        _ = reduced.parameters
        _ = reduced.parameter_names
        _ = reduced.comp_rep_columns

    def test_disallowed_attribute_raises(self):
        """Accessing transform or other heavy methods raises AttributeError."""
        x = NumericalDiscreteParameter("x", values=[1.0, 2.0])
        cat = CategoricalParameter("c", values=["A", "B"])
        ss = SearchSpace.from_product(parameters=[x, cat])
        reduced = ss._drop_parameters({"x"})
        with pytest.raises(AttributeError, match="does not support"):
            reduced.transform(pd.DataFrame())


# ─── _GeneralityModel._expand ─────────────────────────────────────────────────


class TestGeneralityModelExpand:
    """Tests for _GeneralityModel._expand content correctness."""

    def test_expand_interleaving(self):
        """Columns are interleaved correctly per x_col_indices/w_col_indices."""
        from baybe.surrogates.generality import _GeneralityModel

        # x has 1 col at index 0, w has 2 cols at indices 1,2
        base_model = MagicMock()
        base_model.num_outputs = 1
        w_values = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        model = _GeneralityModel(
            base_model=base_model,
            w_values_comp=w_values,
            x_col_indices=[0],
            w_col_indices=[1, 2],
            aggregation=MeanAggregation(),
            target_transform=_NoOpTransform(),
        )

        X = torch.tensor([[7.0]])  # q=1, d_x=1
        result = model._expand(X)

        # Output shape: (q*r, d_full) = (3, 3)
        assert result.shape == (3, 3)
        # Column 0 (x) should be 7.0 for all rows
        assert torch.allclose(result[:, 0], torch.tensor([7.0, 7.0, 7.0]))
        # Columns 1,2 (w) should be the w_values rows
        assert torch.allclose(result[:, 1:], w_values)

    def test_expand_x_cols_not_contiguous(self):
        """Works when x columns are non-contiguous (e.g., indices 0 and 2)."""
        from baybe.surrogates.generality import _GeneralityModel

        base_model = MagicMock()
        base_model.num_outputs = 1
        w_values = torch.tensor([[0.5], [0.9]])  # r=2, d_w=1

        model = _GeneralityModel(
            base_model=base_model,
            w_values_comp=w_values,
            x_col_indices=[0, 2],  # x at positions 0 and 2
            w_col_indices=[1],  # w at position 1
            aggregation=MeanAggregation(),
            target_transform=_NoOpTransform(),
        )

        X = torch.tensor([[1.0, 3.0]])  # q=1, d_x=2
        result = model._expand(X)

        # Shape: (q*r, d_full) = (2, 3)
        assert result.shape == (2, 3)
        # Col 0: x[0] = 1.0
        assert torch.allclose(result[:, 0], torch.tensor([1.0, 1.0]))
        # Col 1: w values
        assert torch.allclose(result[:, 1], torch.tensor([0.5, 0.9]))
        # Col 2: x[1] = 3.0
        assert torch.allclose(result[:, 2], torch.tensor([3.0, 3.0]))

    def test_expand_batch_shape(self):
        """Handles batch dimensions correctly."""
        from baybe.surrogates.generality import _GeneralityModel

        base_model = MagicMock()
        base_model.num_outputs = 1
        w_values = torch.tensor([[0.1], [0.2]])

        model = _GeneralityModel(
            base_model=base_model,
            w_values_comp=w_values,
            x_col_indices=[0],
            w_col_indices=[1],
            aggregation=MeanAggregation(),
            target_transform=_NoOpTransform(),
        )

        X = torch.randn(5, 3, 1)  # batch=5, q=3, d_x=1
        result = model._expand(X)
        # (5, q*r, d_full) = (5, 6, 2)
        assert result.shape == (5, 6, 2)


# ─── _GeneralityPosterior._aggregate ──────────────────────────────────────────


class TestGeneralityPosteriorAggregate:
    """Tests for _GeneralityPosterior aggregation logic."""

    def _make_posterior(self, samples, q, r, m, aggregation=None):
        """Build a _GeneralityPosterior with a mock base that returns samples."""
        from baybe.surrogates.generality import _GeneralityPosterior

        if aggregation is None:
            aggregation = MeanAggregation()

        base_posterior = MagicMock()
        base_posterior.rsample = MagicMock(return_value=samples)
        base_posterior.device = torch.device("cpu")
        base_posterior.dtype = torch.float32

        return _GeneralityPosterior(
            base_posterior=base_posterior,
            q=q,
            r=r,
            m=m,
            aggregation=aggregation,
            target_transform=_NoOpTransform(),
        )

    def test_mean_aggregation_numerical(self):
        """Mean aggregation produces correct values with known input."""
        # q=1, r=3, m=1 → base returns shape (n_samples, q*r, m) = (1, 3, 1)
        # Values per context: 2, 4, 6 → mean = 4
        samples = torch.tensor([[[2.0], [4.0], [6.0]]])
        post = self._make_posterior(samples, q=1, r=3, m=1)
        result = post.rsample(torch.Size([1]))
        assert result.shape == (1, 1, 1)  # (n_samples, q, m)
        assert result.item() == pytest.approx(4.0)

    def test_min_aggregation_numerical(self):
        """Min aggregation picks worst context."""
        samples = torch.tensor([[[2.0], [4.0], [6.0]]])
        post = self._make_posterior(
            samples, q=1, r=3, m=1, aggregation=MinAggregation()
        )
        result = post.rsample(torch.Size([1]))
        assert result.item() == pytest.approx(2.0)

    def test_multi_q(self):
        """Aggregation works with multiple candidates (q>1)."""
        # q=2, r=2, m=1 → base shape (1, 4, 1)
        # Candidate 1: contexts [1, 3] → mean 2
        # Candidate 2: contexts [5, 7] → mean 6
        samples = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
        post = self._make_posterior(samples, q=2, r=2, m=1)
        result = post.rsample(torch.Size([1]))
        assert result.shape == (1, 2, 1)
        assert result[0, 0, 0].item() == pytest.approx(2.0)
        assert result[0, 1, 0].item() == pytest.approx(6.0)

    def test_mean_property(self):
        """The mean property aggregates correctly."""
        from baybe.surrogates.generality import _GeneralityPosterior

        base_posterior = MagicMock()
        type(base_posterior).mean = PropertyMock(
            return_value=torch.tensor([[[2.0], [4.0], [6.0]]])
        )
        base_posterior.device = torch.device("cpu")
        base_posterior.dtype = torch.float32

        post = _GeneralityPosterior(
            base_posterior=base_posterior,
            q=1,
            r=3,
            m=1,
            aggregation=MeanAggregation(),
            target_transform=_NoOpTransform(),
        )
        assert post.mean.item() == pytest.approx(4.0)


# ─── recommend_generality wiring ──────────────────────────────────────────────


class TestRecommendGeneralityWiring:
    """Tests for recommend_generality with mocked surrogate and optimizer."""

    @pytest.fixture(name="setup")
    def fixture_setup(self):
        """Set up a mocked recommender and search space for generality."""
        from baybe.objectives import SingleTargetObjective
        from baybe.targets import NumericalTarget

        x = NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0])
        w = GeneralityParameter(
            name="solvent",
            context=CategoricalParameter("solvent", values=["A", "B", "C"]),
        )
        ss = SearchSpace.from_product(parameters=[x, w])
        objective = SingleTargetObjective(target=NumericalTarget("yield"))

        measurements = pd.DataFrame(
            {"x": [1.0, 2.0], "solvent": ["A", "B"], "yield": [0.5, 0.8]}
        )

        return ss, objective, measurements

    def _make_recommender_mock(self, ss, objective):
        """Build a mocked recommender for recommend_generality."""
        from baybe.acquisition.acqfs import ExpectedImprovement

        design_ss, _, _, _ = ss._split_by_generality()

        mock_base_model = MagicMock()
        mock_base_model.num_outputs = 1
        mock_base_model.posterior = MagicMock(
            return_value=MagicMock(
                mean=torch.zeros(1, 1),
                variance=torch.ones(1, 1),
            )
        )

        recommender = MagicMock()
        recommender._objective = objective
        recommender._surrogate_model.to_botorch.return_value = mock_base_model
        recommender._get_acquisition_function = MagicMock(
            return_value=ExpectedImprovement()
        )
        return recommender, design_ss

    def _make_smart_optimizer(self, ss):
        """Return a fake optimizer that returns valid comp-rep for any subspace."""
        design_ss, _, _, _ = ss._split_by_generality()

        def fake_optimizer(batch_size, acqf, space):
            # Return the first row of comp-rep for whatever space is passed
            import numpy as np

            if not space.discrete.is_empty and len(space.discrete.comp_rep) > 0:
                row = np.array(space.discrete.comp_rep.iloc[0].values, dtype=float)
            else:
                n_cols = len(space.comp_rep_columns)
                row = np.zeros(n_cols)
            pts = torch.from_numpy(row).float().unsqueeze(0).expand(batch_size, -1)
            return pts, torch.zeros(batch_size)

        return fake_optimizer

    def test_optimizer_called_with_x_subspace(self, setup):
        """The optimizer is invoked on the x-only design subspace."""
        from baybe.recommenders.pure.bayesian.generality import recommend_generality

        ss, objective, measurements = setup
        recommender, _ = self._make_recommender_mock(ss, objective)
        recommender.optimizer = self._make_smart_optimizer(ss)

        result = recommend_generality(
            recommender, ss, batch_size=1, measurements=measurements
        )

        assert len(result) == 1
        assert "x" in result.columns
        assert "solvent" in result.columns

    def test_batch_produces_correct_count(self, setup):
        """recommend_generality returns exactly batch_size rows."""
        from baybe.recommenders.pure.bayesian.generality import recommend_generality

        ss, objective, measurements = setup
        recommender, _ = self._make_recommender_mock(ss, objective)
        recommender.optimizer = self._make_smart_optimizer(ss)

        result = recommend_generality(
            recommender, ss, batch_size=2, measurements=measurements
        )

        assert len(result) == 2


# ─── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    """Validation tests for generality in search spaces."""

    def test_multiple_generality_params_raises(self):
        """Cannot have more than one GeneralityParameter."""
        x = NumericalDiscreteParameter("x", values=[1.0, 2.0])
        w1 = GeneralityParameter(
            name="w1", context=CategoricalParameter("w1", values=["A", "B"])
        )
        w2 = GeneralityParameter(
            name="w2", context=CategoricalParameter("w2", values=["X", "Y"])
        )
        with pytest.raises(NotImplementedError, match="at most one"):
            SearchSpace.from_product(parameters=[x, w1, w2])

    def test_generality_with_task_param_raises(self):
        """Cannot combine GeneralityParameter with TaskParameter."""
        from baybe.parameters.categorical import TaskParameter

        x = NumericalDiscreteParameter("x", values=[1.0, 2.0])
        w = GeneralityParameter(
            name="solvent", context=CategoricalParameter("solvent", values=["A", "B"])
        )
        t = TaskParameter("task", values=["t1", "t2"])
        with pytest.raises(Exception, match="[Tt]ask|[Gg]enerality"):
            SearchSpace.from_product(parameters=[x, w, t])
