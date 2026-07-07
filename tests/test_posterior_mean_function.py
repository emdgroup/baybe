"""Tests for ``GaussianProcessSurrogate.posterior_mean_function``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import torch

from baybe.exceptions import ModelNotTrainedError
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget

if TYPE_CHECKING:
    from baybe.objectives.base import Objective


_OBJECTIVE = NumericalTarget(name="y").to_objective()


def _searchspace(values: list[float]) -> SearchSpace:
    return SearchSpace.from_product([NumericalDiscreteParameter("x1", values=values)])


def _measurements(xs: list[float], ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x1": xs, "y": ys})


def _fitted_surrogate(
    searchspace: SearchSpace,
    objective: Objective,
    measurements: pd.DataFrame,
) -> GaussianProcessSurrogate:
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)
    return surrogate


def _predict_on_prior_mean(
    pretrained: GaussianProcessSurrogate, xs: list[float]
) -> pd.DataFrame:
    """Build measurements where the targets follow the pretrained posterior mean."""
    points = pd.DataFrame({"x1": xs})
    with torch.no_grad():
        targets = pretrained.posterior(points).mean
    return pd.DataFrame({"x1": xs, "y": targets.numpy().ravel()})


@pytest.fixture(name="pretrained")
def fixture_pretrained() -> GaussianProcessSurrogate:
    """A GP trained on a narrow search space with three points."""
    return _fitted_surrogate(
        _searchspace([0.0, 2.5, 5.0]),
        _OBJECTIVE,
        _measurements([0.0, 2.5, 5.0], [0.0, 5.0, 10.0]),
    )


@pytest.fixture(name="wider_searchspace")
def fixture_wider_searchspace() -> SearchSpace:
    return _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])


def test_matches_pretrained_at_held_out_point(
    pretrained: GaussianProcessSurrogate, wider_searchspace: SearchSpace
) -> None:
    """The transferred mean reproduces the pretrained posterior mean.

    When the new GP is trained on targets sampled from the pretrained posterior mean,
    the outer kernel sees zero residual and the prior mean alone explains the data,
    so the prediction at a held-out point must match the pretrained's prediction.
    """
    expected = pretrained.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    actual = new.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    assert abs(actual - expected) < 1e-4


def test_raises_if_not_fitted() -> None:
    """An untrained surrogate cannot produce a posterior mean function."""
    ss = _searchspace([0.0, 1.0])
    with pytest.raises(ModelNotTrainedError, match="must be fitted"):
        GaussianProcessSurrogate().posterior_mean_function(
            ss, _OBJECTIVE, _measurements([0.0, 1.0], [0.0, 1.0])
        )


def test_pretrained_hyperparameters_unaffected_by_new_fit(
    pretrained: GaussianProcessSurrogate,
    wider_searchspace: SearchSpace,
) -> None:
    """The pretrained surrogate's parameters are unaffected by fitting the new GP."""
    pretrained_lengthscale = pretrained._model.covar_module.lengthscale.detach().clone()
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    assert torch.equal(
        pretrained._model.covar_module.lengthscale,
        pretrained_lengthscale,
    )


def test_inner_parameters_constant_after_fit(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """The wrapped GP's hyperparameters are untouched by the outer MLL optimization."""
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 2.5, 5.0, 7.5, 10.0])
    module = pretrained.posterior_mean_function(new_ss, _OBJECTIVE, new_meas)
    before = module.gp.covar_module.lengthscale.detach().clone()

    new = GaussianProcessSurrogate(mean_or_factory=module)
    new.fit(new_ss, _OBJECTIVE, new_meas)

    after = module.gp.covar_module.lengthscale.detach()
    assert torch.equal(after, before)


def test_prebuilt_mean_module_can_be_reused(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """A pre-built mean module can be passed as the new GP's mean and fitted."""
    new_ss = _searchspace([0.0, 5.0])
    new_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    module = pretrained.posterior_mean_function(new_ss, _OBJECTIVE, new_meas)

    new = GaussianProcessSurrogate(mean_or_factory=module)
    new.fit(new_ss, _OBJECTIVE, new_meas)

    assert new._model is not None


def test_bound_method_factory_fits(
    pretrained: GaussianProcessSurrogate,
    wider_searchspace: SearchSpace,
) -> None:
    """Using the bound method as a factory builds the mean in the fit context."""
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    assert new._model is not None
