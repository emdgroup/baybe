"""Tests for ``GaussianProcessSurrogate.posterior_mean_function``."""

from __future__ import annotations

import math
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


def test_bound_method_factory(
    pretrained: GaussianProcessSurrogate, wider_searchspace: SearchSpace
) -> None:
    """Fitting a new GP via the bound-method factory transfers the mean correctly.

    A single new GP is fitted using the pretrained surrogate's bound method as the
    mean factory, and several properties are checked afterwards:

    * The new model is built in the fit context.
    * The new GP reproduces the pretrained posterior mean at a held-out point: the
      training targets lie exactly on the pretrained mean, so the outer kernel sees
      zero residual and the prior mean alone explains the data.
    * The pretrained surrogate's hyperparameters are left untouched by the outer
      optimization.
    """
    expected = pretrained.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()
    pretrained_lengthscale = pretrained._model.covar_module.lengthscale.detach().clone()

    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    assert new._model is not None

    actual = new.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()
    assert math.isclose(actual, expected, abs_tol=1e-4)

    assert torch.equal(
        pretrained._model.covar_module.lengthscale, pretrained_lengthscale
    )


def test_raises_if_not_fitted() -> None:
    """An untrained surrogate cannot produce a posterior mean function."""
    ss = _searchspace([0.0, 1.0])
    with pytest.raises(ModelNotTrainedError, match="must be fitted"):
        GaussianProcessSurrogate().posterior_mean_function(
            ss, _OBJECTIVE, _measurements([0.0, 1.0], [0.0, 1.0])
        )


def test_prebuilt_mean_module(pretrained: GaussianProcessSurrogate) -> None:
    """A pre-built mean module can be reused and freezes the wrapped GP.

    The mean module is built once and passed to a new GP. After fitting, we check
    that the new model was built and that the wrapped GP's hyperparameters are
    untouched by the outer MLL optimization.
    """
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 2.5, 5.0, 7.5, 10.0])
    module = pretrained.posterior_mean_function(new_ss, _OBJECTIVE, new_meas)
    before = module.gp.covar_module.lengthscale.detach().clone()

    new = GaussianProcessSurrogate(mean_or_factory=module)
    new.fit(new_ss, _OBJECTIVE, new_meas)

    assert new._model is not None
    after = module.gp.covar_module.lengthscale.detach()
    assert torch.equal(after, before)
