"""Tests for risk measure insights."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import param

from baybe import Campaign, Settings
from baybe._optional.info import INSIGHTS_INSTALLED
from baybe.constraints import ContinuousLinearConstraint
from baybe.exceptions import ModelNotTrainedError, NoMeasurementsError
from baybe.objectives import ParetoObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders import BotorchRecommender, RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import (
    CompositeSurrogate,
    GaussianProcessSurrogate,
    RandomForestSurrogate,
)
from baybe.targets import NumericalTarget

if not INSIGHTS_INSTALLED:
    pytest.skip(
        "Optional insights dependencies are not installed.", allow_module_level=True
    )

from baybe.insights.robustness import RobustnessInsight  # noqa: E402

_NAMES = ["x", "y", "z"]


def _make_data(n: int = 15) -> pd.DataFrame:
    """Make random measurements for the continuous test search space."""
    data = pd.DataFrame(np.random.default_rng(0).random((n, 3)), columns=_NAMES)
    data["t"] = np.sin(6 * data["x"]) + data["y"] - data["z"] ** 2
    data["t2"] = data["x"] - data["y"]
    return data


@pytest.fixture(name="searchspace")
def fixture_searchspace() -> SearchSpace:
    return SearchSpace.from_product(
        [NumericalContinuousParameter(n, (0, 1)) for n in _NAMES]
    )


@pytest.fixture(name="campaign")
def fixture_campaign(searchspace: SearchSpace) -> Campaign:
    campaign = Campaign(searchspace, NumericalTarget("t"))
    campaign.add_measurements(_make_data())
    return campaign


@pytest.fixture(name="posterior_spy")
def fixture_posterior_spy():
    """Spy on the inputs that are passed to the posterior computation."""
    original = GaussianProcessSurrogate._posterior_comp
    with mock.patch.object(
        GaussianProcessSurrogate, "_posterior_comp", autospec=True, side_effect=original
    ) as spy:
        yield spy


@pytest.fixture(name="make_insight")
def fixture_make_insight(campaign: Campaign):
    def make(**kwargs) -> RobustnessInsight:
        kwargs = {
            "input_noise": {"x": 0.1, "y": 0.1},
            "risk_measure": "Expectation",
        } | kwargs
        return RobustnessInsight.from_campaign(campaign, **kwargs)

    return make


def test_risk_measure_ordering(make_insight):
    """The risk measures are ordered according to their pessimism."""
    data = _make_data()
    with Settings(random_seed=0):
        values = {
            name: make_insight(
                risk_measure=name,
                alpha=alpha,
                n_perturbations=128,
                n_posterior_samples=256,
            )
            .evaluate(data)[name]
            .mean()
            for name, alpha in [
                ("WorstCase", None),
                ("CVaR", 0.8),
                ("VaR", 0.8),
                ("Expectation", None),
            ]
        }
    assert values["WorstCase"] < values["CVaR"] < values["VaR"]
    assert values["CVaR"] < values["Expectation"]


def test_expectation_without_noise_matches_posterior_mean(campaign, make_insight):
    """For negligible noise, the expectation matches the posterior mean."""
    data = _make_data()
    insight = make_insight(input_noise={"x": 1e-4}, n_perturbations=8)
    expected = campaign.get_surrogate().posterior(data).mean.detach().numpy().squeeze()
    with Settings(random_seed=0):
        result = insight.evaluate(data)["Expectation"]
    np.testing.assert_allclose(result, expected, atol=0.05)


def test_minimization_target(searchspace):
    """The risk measures are computed for the negated values of minimized targets."""
    data = _make_data()
    campaign = Campaign(searchspace, NumericalTarget("t", minimize=True))
    campaign.add_measurements(data)
    kwargs = {"n_perturbations": 8}
    insight = RobustnessInsight.from_campaign(
        campaign, input_noise={"x": 1e-4}, risk_measure="Expectation", **kwargs
    )
    mean = campaign.get_surrogate().posterior(data).mean.detach().numpy().squeeze()
    with Settings(random_seed=0):
        result = insight.evaluate(data)["Expectation"]
    np.testing.assert_allclose(result, -mean, atol=0.05)

    # The worst case is the largest raw value, i.e., smaller than the negated mean
    insight = RobustnessInsight.from_campaign(
        campaign, input_noise={"x": 0.2, "y": 0.2}, risk_measure="WorstCase", **kwargs
    )
    with Settings(random_seed=0):
        result = insight.evaluate(data)["WorstCase"]
    assert (result < -mean).all()


@pytest.mark.parametrize(
    ("risk_measure", "alpha"),
    [param("Expectation", None, id="expectation"), param("CVaR", 0.8, id="cvar")],
)
def test_reproducibility(make_insight, risk_measure, alpha):
    """The results are reproducible when the random seed is fixed."""
    data = _make_data()
    insight = make_insight(risk_measure=risk_measure, alpha=alpha)
    with Settings(random_seed=1337):
        first = insight.evaluate(data)
    with Settings(random_seed=1337):
        second = insight.evaluate(data)
    assert_frame_equal(first, second)


def test_perturbed_inputs_are_clipped(make_insight, posterior_spy):
    """The perturbed inputs passed to the surrogate stay within the bounds."""
    insight = make_insight(input_noise={"x": 5.0, "y": 5.0}, risk_measure="WorstCase")
    data = pd.DataFrame([[0.0, 1.0, 0.5], [0.5, 0.5, 0.5]], columns=_NAMES)
    insight.evaluate(data)

    perturbed = posterior_spy.call_args.args[1]
    assert perturbed.shape[:2] == (2, insight.n_perturbations)
    assert perturbed.min() >= 0
    assert perturbed.max() <= 1
    assert (perturbed[..., 2] == 0.5).all()  # noise-free parameter is untouched
    assert (perturbed[..., :2] == 0).any()  # clipping actually occurs


def test_diagonal_covariance_matches_standard_deviations(make_insight):
    """A diagonal covariance matrix is equivalent to standard deviations."""
    data = _make_data()
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09]), index=["x", "y"], columns=["x", "y"]
    )
    kwargs = {"risk_measure": "WorstCase"}
    with Settings(random_seed=1):
        by_std = make_insight(input_noise={"x": 0.2, "y": 0.3}, **kwargs).evaluate(data)
    with Settings(random_seed=1):
        by_cov = make_insight(input_noise=covariance, **kwargs).evaluate(data)
    assert_frame_equal(by_std, by_cov)


def test_covariance_is_reproduced_by_perturbations(make_insight, posterior_spy):
    """The perturbations drawn from a covariance matrix have this covariance."""
    covariance = pd.DataFrame(
        [[0.01, 0.006], [0.006, 0.01]], index=["x", "y"], columns=["x", "y"]
    )
    insight = make_insight(
        input_noise=covariance, risk_measure="WorstCase", n_perturbations=4096
    )
    data = pd.DataFrame([[0.5, 0.5, 0.5]], columns=_NAMES)
    insight.evaluate(data)

    perturbed = posterior_spy.call_args.args[1][0].numpy()
    np.testing.assert_allclose(np.cov(perturbed[:, :2].T), covariance, atol=2e-3)
    assert (perturbed[:, 2] == 0.5).all()


def test_correlated_noise_changes_result(searchspace):
    """Correlated noise is taken into account for the worst case."""
    data = _make_data()
    data["t"] = data["x"] - data["y"]  # noise along (1, 1) does not change the target
    campaign = Campaign(searchspace, NumericalTarget("t"))
    campaign.add_measurements(data)
    names = ["x", "y"]
    results = []
    for correlation in [1.0, -1.0]:
        covariance = pd.DataFrame(
            [[0.01, 0.01 * correlation], [0.01 * correlation, 0.01]],
            index=names,
            columns=names,
        )  # singular for perfectly correlated noise
        insight = RobustnessInsight.from_campaign(
            campaign, input_noise=covariance, risk_measure="WorstCase"
        )
        results.append(
            insight.evaluate(pd.DataFrame([[0.5, 0.5, 0.5]], columns=_NAMES))
        )
    # Fully correlated noise leaves x - y unchanged, anticorrelated noise does not
    assert results[0]["WorstCase"].iloc[0] > results[1]["WorstCase"].iloc[0]


@pytest.mark.parametrize(
    "row",
    [
        param([1.5, 0.5, 0.5], id="above_noisy"),
        param([0.5, -0.1, 0.5], id="below_noisy"),
        param([0.5, 0.5, 1.2], id="above_noise_free"),
    ],
)
def test_out_of_bounds_candidates_are_rejected(make_insight, row):
    """Candidates outside the search space bounds are rejected, not clipped."""
    with pytest.raises(ValueError, match="within the bounds"):
        make_insight().evaluate(pd.DataFrame([row], columns=_NAMES))


def test_evaluate_does_not_alter_candidates(make_insight):
    """The evaluation leaves the passed candidates untouched."""
    candidates = _make_data()
    expected = candidates.copy()
    make_insight().evaluate(candidates)
    assert_frame_equal(candidates, expected)


@pytest.mark.parametrize(
    "std",
    [
        param(1, id="int"),
        param(np.int64(1), id="np_int"),
        param(np.float32(1), id="np_float"),
    ],
)
def test_numeric_standard_deviations_are_converted(make_insight, std):
    """Real-valued standard deviations of any numeric type are accepted."""
    insight = make_insight(input_noise={"x": std})
    assert isinstance(insight.input_noise["x"], float)
    assert insight.input_noise["x"] == 1.0


def test_evaluate_output(make_insight):
    """The evaluation returns the configurations with an additional value column."""
    data = _make_data()
    result = make_insight().evaluate(data)
    assert list(result.columns) == [*_NAMES, "Expectation"]
    assert_frame_equal(result[_NAMES], data[_NAMES])


@pytest.mark.parametrize(
    ("risk_measure", "alpha"),
    [
        param("Expectation", None, id="expectation"),
        param("VaR", 0.8, id="var"),
        param("CVaR", 0.8, id="cvar"),
        param("WorstCase", None, id="worst_case"),
    ],
)
def test_risk_measure_is_botorch_callable(make_insight, risk_measure, alpha):
    """The risk measure is an instance of the corresponding callable BoTorch class."""
    from botorch.acquisition import risk_measures

    insight = make_insight(risk_measure=risk_measure, alpha=alpha, n_perturbations=4)
    measure = insight._make_risk_measure()
    assert isinstance(measure, getattr(risk_measures, risk_measure))
    assert measure.n_w == 4
    assert callable(measure)


def test_nonaffine_objective(searchspace):
    """Risk measures can be evaluated for objectives that are non-affine."""
    campaign = Campaign(searchspace, NumericalTarget.match_bell("t", 0.0, 1.0))
    campaign.add_measurements(_make_data())
    insight = RobustnessInsight.from_campaign(
        campaign, input_noise={"x": 0.1}, risk_measure="Expectation"
    )
    assert len(insight.evaluate(_make_data(3))) == 3


def _make_continuous_campaign() -> tuple[Campaign, pd.DataFrame]:
    searchspace = SearchSpace.from_product(
        [NumericalContinuousParameter(n, (0, 1)) for n in _NAMES],
        constraints=[
            ContinuousLinearConstraint(["x", "y"], "<=", coefficients=[1, 1], rhs=1)
        ],
    )
    data = _make_data().query("x + y <= 1")
    return Campaign(searchspace, NumericalTarget("t")), data


def _make_discrete_campaign() -> tuple[Campaign, pd.DataFrame]:
    searchspace = SearchSpace.from_product(
        [
            NumericalDiscreteParameter("x", [0.0, 0.25, 0.5, 0.75, 1.0]),
            NumericalDiscreteParameter("y", [0.0, 0.5, 1.0]),
            CategoricalParameter("c", ["a", "b"]),
        ]
    )
    return Campaign(searchspace, NumericalTarget("t")), _make_mixed_data()


def _make_hybrid_campaign() -> tuple[Campaign, pd.DataFrame]:
    searchspace = SearchSpace.from_product(
        [
            NumericalDiscreteParameter("x", [0.0, 0.25, 0.5, 0.75, 1.0]),
            NumericalContinuousParameter("y", (0, 1)),
            CategoricalParameter("c", ["a", "b"]),
        ]
    )
    return Campaign(searchspace, NumericalTarget("t")), _make_mixed_data()


def _make_mixed_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "x": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], 12),
            "y": rng.choice([0.0, 0.5, 1.0], 12),
            "c": ["a", "b"] * 6,
        }
    )
    data["t"] = np.sin(6 * data["x"]) + data["y"] + (data["c"] == "a")
    return data


@pytest.mark.parametrize(
    "make_campaign",
    [
        param(_make_continuous_campaign, id="continuous_constrained"),
        param(_make_discrete_campaign, id="discrete"),
        param(_make_hybrid_campaign, id="hybrid"),
    ],
)
@pytest.mark.parametrize(
    ("risk_measure", "alpha"),
    [param("Expectation", None), param("CVaR", 0.8), param("WorstCase", None)],
)
def test_search_space_types(make_campaign, risk_measure, alpha):
    """Risk measures can be evaluated for all search space types."""
    campaign, data = make_campaign()
    campaign.add_measurements(data)
    insight = RobustnessInsight.from_campaign(
        campaign,
        input_noise={"x": 0.1, "y": 0.1},
        risk_measure=risk_measure,
        alpha=alpha,
    )
    result = insight.evaluate(data)
    assert result[risk_measure].notna().all()
    assert_frame_equal(
        result[list(campaign.searchspace.parameter_names)],
        data[list(campaign.searchspace.parameter_names)],
    )


def test_only_numerical_parameters_are_perturbed(posterior_spy):
    """Non-numerical parameters are never perturbed, discrete ones may be."""
    campaign, data = _make_hybrid_campaign()
    campaign.add_measurements(data)
    insight = RobustnessInsight.from_campaign(
        campaign, input_noise={"x": 0.3, "y": 0.3}, risk_measure="WorstCase"
    )
    insight.evaluate(data.iloc[:1])

    columns = list(campaign.searchspace.comp_rep_columns)
    perturbed = posterior_spy.call_args.args[1][0]
    for name in ["x", "y"]:
        assert perturbed[:, columns.index(name)].std() > 0
    categorical = [c for c in columns if c.startswith("c")]
    assert categorical
    for name in categorical:
        values = perturbed[:, columns.index(name)]
        assert (values == values[0]).all()


def test_noise_on_non_numerical_parameter():
    """Input noise cannot be specified for non-numerical parameters."""
    campaign, data = _make_hybrid_campaign()
    campaign.add_measurements(data)
    with pytest.raises(ValueError, match="only be specified for numerical"):
        RobustnessInsight.from_campaign(
            campaign, input_noise={"x": 0.1, "c": 0.1}, risk_measure="Expectation"
        )


@pytest.mark.parametrize(
    ("risk_measure", "alpha"),
    [
        param("Expectation", None),
        param("VaR", 0.8),
        param("CVaR", 0.8),
        param("WorstCase", None),
    ],
)
def test_gradient_with_fixed_seed(make_insight, risk_measure, alpha):
    """The risk measure is a differentiable, deterministic function of the inputs."""
    import torch

    insight = make_insight(risk_measure=risk_measure, alpha=alpha)
    x = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.double, requires_grad=True)

    value = insight._evaluate_comp(x, seed=0)
    assert value.requires_grad
    value.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, :2].abs().sum() > 0  # noisy parameters influence the value

    # With a fixed seed, repeated calls give identical values
    assert torch.equal(
        insight._evaluate_comp(x.detach(), seed=0),
        insight._evaluate_comp(x.detach(), seed=0),
    )


def test_evaluate_does_not_track_gradients(make_insight):
    """The public evaluation computes the values without tracking gradients."""
    original = RobustnessInsight._evaluate_comp
    outputs = []

    def spy(self, *args, **kwargs):
        outputs.append(original(self, *args, **kwargs))
        return outputs[-1]

    insight = make_insight()
    with mock.patch.object(RobustnessInsight, "_evaluate_comp", spy):
        insight.evaluate(_make_data(3))
    assert not outputs[0].requires_grad


def test_seeds_of_perturbations_and_posterior_samples_differ(make_insight):
    """The perturbations and the posterior base samples use decorrelated seeds."""
    import importlib

    import torch

    sampling = importlib.import_module("botorch.utils.sampling")
    get_sampler_module = importlib.import_module("botorch.sampling.get_sampler")
    insight = make_insight(risk_measure="WorstCase")
    with (
        mock.patch.object(
            sampling,
            "draw_sobol_normal_samples",
            wraps=sampling.draw_sobol_normal_samples,
        ) as draw,
        mock.patch.object(
            get_sampler_module, "get_sampler", wraps=get_sampler_module.get_sampler
        ) as get,
    ):
        insight._evaluate_comp(torch.rand(2, 3, dtype=torch.double), seed=7)
    assert draw.call_args.kwargs["seed"] == 7
    assert get.call_args.kwargs["seed"] == 8


def test_from_recommender(searchspace):
    """The insight can be created from a recommender."""
    insight = RobustnessInsight.from_recommender(
        BotorchRecommender(),
        searchspace,
        NumericalTarget("t").to_objective(),
        _make_data(),
        input_noise={"x": 0.1},
        risk_measure="CVaR",
        alpha=0.5,
    )
    assert len(insight.evaluate(_make_data(3))) == 3


def test_from_recommender_without_surrogate(searchspace):
    """Recommenders without surrogate are rejected."""
    with pytest.raises(TypeError, match="does not provide a surrogate"):
        RobustnessInsight.from_recommender(
            RandomRecommender(),
            searchspace,
            NumericalTarget("t").to_objective(),
            _make_data(),
            input_noise={"x": 0.1},
            risk_measure="Expectation",
        )


def test_from_campaign_without_measurements(searchspace):
    """Campaigns without measurements are rejected."""
    with pytest.raises(NoMeasurementsError, match="does not contain any measurements"):
        RobustnessInsight.from_campaign(
            Campaign(searchspace, NumericalTarget("t")),
            input_noise={"x": 0.1},
            risk_measure="Expectation",
        )


def test_untrained_surrogate():
    """Untrained surrogates are rejected."""
    with pytest.raises(ModelNotTrainedError, match="must be trained"):
        RobustnessInsight(
            surrogate=GaussianProcessSurrogate(),
            input_noise={"x": 0.1},
            risk_measure="Expectation",
        )


def test_untrained_composite_surrogate():
    """Untrained composite surrogates are rejected."""
    with pytest.raises(ModelNotTrainedError, match="must be trained"):
        RobustnessInsight(
            surrogate=CompositeSurrogate.from_replication(GaussianProcessSurrogate()),
            input_noise={"x": 0.1},
            risk_measure="Expectation",
        )


def test_alpha_one_is_worst_case(make_insight):
    """A risk level of one corresponds to the worst case."""
    data = _make_data()
    with Settings(random_seed=0):
        cvar = make_insight(risk_measure="CVaR", alpha=1.0).evaluate(data)["CVaR"]
    with Settings(random_seed=0):
        worst = make_insight(risk_measure="WorstCase").evaluate(data)["WorstCase"]
    np.testing.assert_allclose(cvar, worst)


def _make_pareto_campaign() -> Campaign:
    searchspace = SearchSpace.from_product(
        [NumericalContinuousParameter(n, (0, 1)) for n in _NAMES]
    )
    objective = ParetoObjective([NumericalTarget("t"), NumericalTarget("t2")])
    campaign = Campaign(searchspace, objective)
    campaign.add_measurements(_make_data())
    return campaign


def _make_non_gp_campaign() -> Campaign:
    searchspace = SearchSpace.from_product(
        [NumericalContinuousParameter(n, (0, 1)) for n in _NAMES]
    )
    campaign = Campaign(
        searchspace,
        NumericalTarget("t"),
        BotorchRecommender(surrogate_model=RandomForestSurrogate()),
    )
    campaign.add_measurements(_make_data())
    return campaign


@pytest.mark.parametrize(
    ("make_campaign", "match"),
    [
        param(_make_non_gp_campaign, "only supports surrogates of type", id="non_gp"),
        param(_make_pareto_campaign, "single-output", id="multi_output"),
    ],
)
def test_unsupported_surrogates(make_campaign, match):
    """Unsupported surrogates raise a NotImplementedError."""
    with pytest.raises(NotImplementedError, match=match):
        RobustnessInsight.from_campaign(
            make_campaign(), input_noise={"x": 0.1}, risk_measure="Expectation"
        )


@pytest.mark.parametrize(
    "input_noise",
    [
        param(lambda x: x, id="callable"),
        param(object(), id="distribution"),
        param(np.zeros((8, 3)), id="perturbation_set"),
        param([0.1, 0.1, 0.1], id="sequence"),
    ],
)
def test_unsupported_input_noise(campaign, input_noise):
    """Input noise that is not a mapping of standard deviations is not supported."""
    with pytest.raises(NotImplementedError, match="Gaussian input noise"):
        RobustnessInsight.from_campaign(
            campaign, input_noise=input_noise, risk_measure="Expectation"
        )


def _invalid_covariances():
    """Create parameters for invalid covariance matrices."""
    names = ["x", "y"]

    def make(values, index=names, columns=names):
        return pd.DataFrame(values, index=index, columns=columns)

    return [
        param(
            {"input_noise": make([[1.0, 0.5], [0.4, 1.0]])},
            "symmetric",
            id="cov_asymmetric",
        ),
        param(
            {"input_noise": make([[1.0, 2.0], [2.0, 1.0]])},
            "positive semi-definite",
            id="cov_not_psd",
        ),
        param(
            {"input_noise": make([[1.0, np.nan], [np.nan, 1.0]])},
            "finite",
            id="cov_nan",
        ),
        param(
            {"input_noise": make([[1.0, 0.0], [0.0, 1.0]], columns=["x", "z"])},
            "same, unique",
            id="cov_mismatched_labels",
        ),
        param(
            {"input_noise": make([[1.0, 0.0]], index=["x"])},
            "same, unique",
            id="cov_not_square",
        ),
        param(
            {
                "input_noise": make(
                    [[1.0, 0.0], [0.0, 1.0]], ["x", "unknown"], ["x", "unknown"]
                )
            },
            "not part of",
            id="cov_unknown_parameter",
        ),
        param(
            {"input_noise": make([[0.0, 0.0], [0.0, 0.0]])},
            "At least one",
            id="cov_zero",
        ),
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        param({"input_noise": {}}, "At least one", id="empty_noise"),
        *_invalid_covariances(),
        param({"input_noise": {"unknown": 0.1}}, "not part of", id="unknown_parameter"),
        param({"input_noise": {"x": 0.0}}, "positive", id="zero_std"),
        param({"input_noise": {"x": float("inf")}}, "positive", id="infinite_std"),
        param({"input_noise": {"x": "a"}}, "positive", id="non_numeric_std"),
        param({"input_noise": {"x": True}}, "positive", id="bool_std"),
        param({"risk_measure": "CVaR"}, "requires a risk level", id="missing_alpha"),
        param(
            {"risk_measure": "VaR", "alpha": 0.0}, "requires a risk level", id="alpha_0"
        ),
        param({"alpha": 0.5}, "does not use a risk level", id="superfluous_alpha"),
        param({"risk_measure": "Unknown"}, "must be in", id="unknown_risk_measure"),
        param(
            {"risk_measure": "CVaR", "alpha": 1.5},
            "requires a risk level",
            id="alpha_>1",
        ),
        param(
            {"risk_measure": "CVaR", "alpha": -0.5},
            "requires a risk level",
            id="alpha_negative",
        ),
        param({"n_perturbations": 0}, "must be >", id="no_perturbations"),
        param({"n_posterior_samples": 0}, "must be >", id="no_posterior_samples"),
    ],
)
def test_invalid_arguments(make_insight, kwargs, match):
    """Invalid arguments are rejected."""
    with pytest.raises(ValueError, match=match):
        make_insight(**kwargs)
