"""Tests for fidelity parameters."""

import pandas as pd
import pytest
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
from pytest import param

from baybe.exceptions import IncompatibleSurrogateError
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.fit_criterion import FitCriterion
from baybe.surrogates.gaussian_process.components.generic import PlainGPComponentFactory
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.multi_fidelity import (
    GaussianProcessSurrogateSTMF,
)
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import create_fake_input

_num_fid_param = NumericalDiscreteFidelityParameter(
    "fidelity", values=[0.5, 1.0], costs=[1.0, 10.0]
)
_cat_fid_param = CategoricalFidelityParameter(
    "fidelity", values=["lo", "hi"], costs=[1.0, 10.0], zeta=[0.5, 0.0]
)
_design_param = NumericalDiscreteParameter("x", values=[1.0, 2.0, 3.0])

searchspace_num_fid = SearchSpace.from_product([_design_param, _num_fid_param])
searchspace_cat_fid = SearchSpace.from_product([_design_param, _cat_fid_param])

objective = NumericalTarget("t").to_objective()
measurements_num_fid = create_fake_input(
    searchspace_num_fid.parameters, objective.targets, n_rows=20
)
measurements_cat_fid = create_fake_input(
    searchspace_cat_fid.parameters, objective.targets, n_rows=20
)


def _dummy_likelihood_factory(*_args, **_kwargs) -> GPyTorchLikelihood:
    return GaussianLikelihood()


def test_categorical_fidelity_parameter_construction():
    """Equivalent zeta formats and value orderings produce equal objects."""
    p1 = CategoricalFidelityParameter("p", values=["h", "l"], costs=[1, 2], zeta=5)
    p2 = CategoricalFidelityParameter("p", values=["l", "h"], costs=[2, 1], zeta=[5, 0])
    assert p1 == p2


def test_numerical_discrete_fidelity_parameter_construction():
    """Fidelity values and costs are sorted according to numerical fidelity values."""
    p1 = NumericalDiscreteFidelityParameter("p", values=[0, 0.5, 1], costs=[1, 2, 3])
    p2 = NumericalDiscreteFidelityParameter("p", values=[0.5, 1, 0], costs=[2, 3, 1])
    assert p1 == p2


@pytest.mark.parametrize(
    ("parameter", "series", "expected"),
    [
        param(
            CategoricalFidelityParameter(
                "fidelity", values=["low", "high"], costs=[1, 2], zeta=[1, 0]
            ),
            pd.Series(["low", "high", "low"], name="fidelity"),
            [1.0, 0.0, 1.0],
            id="categorical",
        ),
        param(
            CategoricalFidelityParameter(
                "fidelity", values=["low", "high"], costs=[1, 2], zeta=5
            ),
            pd.Series(["low", "high", "low"], name="fidelity"),
            [1.0, 0.0, 1.0],
            id="categorical_scalar_zeta",
        ),
        param(
            NumericalDiscreteFidelityParameter(
                "fidelity", values=[0, 0.5, 1], costs=[1, 2, 3]
            ),
            pd.Series([0.5, 1.0, 0.0], name="fidelity"),
            [0.5, 1.0, 0.0],
            id="numerical_discrete",
        ),
    ],
)
def test_fidelity_parameter_transform(parameter, series, expected):
    """Transform must correctly map fidelity values to computational representation."""
    result = parameter.transform(series)
    assert list(result["fidelity"]) == expected


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        param(
            [
                CategoricalFidelityParameter(
                    "f1", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
                CategoricalFidelityParameter(
                    "f2", values=["a", "b"], costs=[1, 5], zeta=[0.3, 0.0]
                ),
            ],
            "at most one fidelity",
            id="two_categorical_fidelity",
        ),
        param(
            [
                CategoricalFidelityParameter(
                    "f1", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
                NumericalDiscreteFidelityParameter(
                    "f2", values=[0.5, 1.0], costs=[1, 10]
                ),
            ],
            "at most one fidelity",
            id="mixed_fidelity_types",
        ),
        param(
            [
                NumericalDiscreteFidelityParameter(
                    "f1", values=[0.5, 1.0], costs=[1, 10]
                ),
                NumericalDiscreteFidelityParameter(
                    "f2", values=[0.2, 1.0], costs=[1, 5]
                ),
            ],
            "at most one fidelity",
            id="two_numerical_fidelity",
        ),
        param(
            [
                TaskParameter("task", values=["a", "b"]),
                CategoricalFidelityParameter(
                    "f", values=["lo", "hi"], costs=[1, 10], zeta=[0.5, 0.0]
                ),
            ],
            "Combining task.*fidelity",
            id="task_plus_fidelity",
        ),
    ],
)
def test_invalid_fidelity_parameter_combinations(parameters, match):
    """Search spaces with invalid fidelity parameter combinations are rejected."""
    with pytest.raises(NotImplementedError, match=match):
        SearchSpace.from_product(parameters)


def test_standard_gp_rejects_numerical_fidelity():
    """GaussianProcessSurrogate raises when fitted on a numerical fidelity space."""
    surrogate = GaussianProcessSurrogate()
    with pytest.raises(IncompatibleSurrogateError, match="STMF"):
        surrogate.fit(searchspace_num_fid, objective, measurements_num_fid)


def test_stmf_rejects_categorical_fidelity():
    """STMF raises an error when fitted on a categorical fidelity search space."""
    surrogate = GaussianProcessSurrogateSTMF()
    measurements_cat_fid = create_fake_input(
        searchspace_cat_fid.parameters, objective.targets, n_rows=20
    )
    with pytest.raises(IncompatibleSurrogateError, match="GaussianProcessSurrogate"):
        surrogate.fit(searchspace_cat_fid, objective, measurements_cat_fid)


@pytest.mark.parametrize(
    ("component", "factory_attr", "expected_type"),
    [
        param(
            {"likelihood_or_factory": GaussianLikelihood()},
            "likelihood_factory",
            PlainGPComponentFactory,
            id="gpytorch_likelihood",
        ),
        param(
            {"likelihood_or_factory": _dummy_likelihood_factory},
            "likelihood_factory",
            type(_dummy_likelihood_factory),
            id="likelihood_factory_callable",
        ),
        param(
            {"fit_criterion_or_factory": FitCriterion.MARGINAL_LOG_LIKELIHOOD},
            "fit_criterion_factory",
            PlainGPComponentFactory,
            id="fit_criterion_enum",
        ),
    ],
)
def test_stmf_component_construction(component, factory_attr, expected_type):
    """STMF accepts GPyTorch objects and callables and wraps them correctly."""
    stmf = GaussianProcessSurrogateSTMF(**component)
    assert isinstance(getattr(stmf, factory_attr), expected_type)


def test_stmf_fit():
    """GaussianProcessSurrogateSTMF can be fitted on a numerical fidelity space."""
    surrogate = GaussianProcessSurrogateSTMF()
    surrogate.fit(searchspace_num_fid, objective, measurements_num_fid)
    stats = surrogate.posterior_stats(measurements_num_fid)
    assert set(stats.columns) == {"t_mean", "t_std"}
    assert len(stats) == len(measurements_num_fid)


@pytest.mark.parametrize(
    ("searchspace", "measurements", "expected_surrogate_type"),
    [
        param(
            searchspace_num_fid,
            measurements_num_fid,
            GaussianProcessSurrogateSTMF,
            id="numerical_fidelity_dispatches_to_stmf",
        ),
        param(
            searchspace_cat_fid,
            create_fake_input(
                searchspace_cat_fid.parameters,
                NumericalTarget("t").to_objective().targets,
                n_rows=20,
            ),
            GaussianProcessSurrogate,
            id="categorical_fidelity_dispatches_to_standard_gp",
        ),
    ],
)
def test_recommender_surrogate_dispatch(
    searchspace, measurements, expected_surrogate_type
):
    """BotorchRecommender auto-selects the correct surrogate for fidelity spaces."""
    recommender = BotorchRecommender()
    surrogate = recommender.get_surrogate(searchspace, objective, measurements)
    # The surrogate is wrapped in a CompositeSurrogate — check the inner template type
    assert isinstance(surrogate.surrogates.template, expected_surrogate_type)


def test_standard_gp_fit_categorical_fidelity():
    """GaussianProcessSurrogate can be fitted on a categorical fidelity space."""
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace_cat_fid, objective, measurements_cat_fid)
    stats = surrogate.posterior_stats(measurements_cat_fid)
    assert set(stats.columns) == {"t_mean", "t_std"}
    assert len(stats) == len(measurements_cat_fid)
