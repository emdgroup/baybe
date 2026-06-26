"""Tests for fidelity parameters."""

import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal
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
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset
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


def test_gp_fit_numerical_fidelity():
    """GaussianProcessSurrogate can be fitted on a numerical fidelity space."""
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace_num_fid, objective, measurements_num_fid)
    stats = surrogate.posterior_stats(measurements_num_fid)
    assert set(stats.columns) == {"t_mean", "t_std"}
    assert len(stats) == len(measurements_num_fid)


def test_gp_numerical_fidelity_kernel_structure():
    """Fitted GP on numerical fidelity contains a DownsamplingKernel component."""
    from botorch.models.kernels.downsampling import DownsamplingKernel
    from gpytorch.kernels import ProductKernel, ScaleKernel

    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace_num_fid, objective, measurements_num_fid)
    model = surrogate.to_botorch()
    kernel = model.covar_module

    assert isinstance(kernel, ScaleKernel)
    assert isinstance(kernel.base_kernel, ProductKernel)
    sub_kernels = list(kernel.base_kernel.kernels)
    assert any(isinstance(k, DownsamplingKernel) for k in sub_kernels)


def test_numerical_fidelity_matches_botorch_stmf():
    """BayBE GP on numerical fidelity matches BoTorch SingleTaskMultiFidelityGP."""
    import botorch
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from baybe.settings import active_settings
    from baybe.utils.dataframe import to_tensor

    train_x = to_tensor(
        searchspace_num_fid.transform(measurements_num_fid, allow_extra=True)
    )
    train_y = to_tensor(objective.transform(measurements_num_fid, allow_extra=True))
    fidelity_idx = searchspace_num_fid.fidelity_idx
    non_fidelity_idcs = [i for i in range(train_x.shape[-1]) if i != fidelity_idx]
    n_cols = len(searchspace_num_fid.comp_rep_columns)
    bounds = torch.from_numpy(searchspace_num_fid.scaling_bounds.to_numpy(copy=True))

    # BayBE path: default factories now match BoTorch's STMF behavior
    active_settings.random_seed = 1337
    gp = GaussianProcessSurrogate()
    gp.fit(searchspace_num_fid, objective, measurements_num_fid)
    posterior_baybe = gp.posterior_stats(measurements_num_fid)

    # Direct BoTorch path
    active_settings.random_seed = 1337
    model = botorch.models.SingleTaskMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=(fidelity_idx,),
        linear_truncated=False,
        input_transform=Normalize(d=n_cols, bounds=bounds, indices=non_fidelity_idcs),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    botorch.fit.fit_gpytorch_mll(mll)

    with torch.no_grad():
        posterior = model.posterior(train_x)
    mean = posterior.mean
    std = posterior.variance.sqrt()
    posterior_botorch = pd.DataFrame(
        {"t_mean": mean.numpy().ravel(), "t_std": std.numpy().ravel()}
    )

    assert_frame_equal(posterior_baybe, posterior_botorch)


@pytest.mark.parametrize(
    ("surrogate", "parameters"),
    [
        param(
            GaussianProcessSurrogate(),
            [TaskParameter("task", values=["a", "b"])],
            id="task_only",
        ),
        param(
            GaussianProcessSurrogate(),
            [_cat_fid_param],
            id="categorical_fidelity_only",
        ),
        param(
            GaussianProcessSurrogate(),
            [_num_fid_param],
            id="numerical_fidelity_only",
        ),
    ],
)
def test_surrogate_rejects_index_only_searchspace(surrogate, parameters):
    """GP surrogates raise for search spaces without regular model inputs."""
    searchspace = SearchSpace.from_product(parameters)
    measurements = create_fake_input(
        searchspace.parameters, objective.targets, n_rows=20
    )

    with pytest.raises(IncompatibleSurrogateError, match="non-task/non-fidelity"):
        surrogate.fit(searchspace, objective, measurements)


def test_recommender_surrogate_dispatch():
    """BotorchRecommender uses GaussianProcessSurrogate for all fidelity types."""
    recommender = BotorchRecommender()

    # Numerical fidelity
    surrogate = recommender.get_surrogate(
        searchspace_num_fid, objective, measurements_num_fid
    )
    assert isinstance(surrogate.surrogates.template, GaussianProcessSurrogate)

    # Categorical fidelity
    surrogate = recommender.get_surrogate(
        searchspace_cat_fid, objective, measurements_cat_fid
    )
    assert isinstance(surrogate.surrogates.template, GaussianProcessSurrogate)


def test_standard_gp_fit_categorical_fidelity():
    """GaussianProcessSurrogate can be fitted on a categorical fidelity space."""
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace_cat_fid, objective, measurements_cat_fid)
    stats = surrogate.posterior_stats(measurements_cat_fid)
    assert set(stats.columns) == {"t_mean", "t_std"}
    assert len(stats) == len(measurements_cat_fid)


@pytest.mark.parametrize(
    "preset",
    list(GaussianProcessPreset),
    ids=lambda preset: preset.value,
)
def test_gp_presets_fit_categorical_fidelity(preset):
    """All GP presets can be fitted on a categorical fidelity space."""
    surrogate = GaussianProcessSurrogate.from_preset(preset)
    surrogate.fit(searchspace_cat_fid, objective, measurements_cat_fid)
