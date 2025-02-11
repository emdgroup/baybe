"""Tests for target minimization."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.testing import assert_close

from baybe.acquisition.acqfs import qKnowledgeGradient
from baybe.acquisition.base import AcquisitionFunction
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import get_subclasses
from baybe.utils.random import set_random_seed


def get_acqf_values(acqf_cls, surrogate, searchspace, objective, df):
    # TODO: Should be replace once a proper public interface is available
    acqf = acqf_cls().to_botorch(surrogate, searchspace, objective, df)
    return acqf(torch.tensor(searchspace.transform(df).values).unsqueeze(-2))


def compute_posterior_and_acqf(acqf_cls, df, searchspace, objective):
    surrogate_max = GaussianProcessSurrogate()
    surrogate_max.fit(searchspace, objective, df)
    with torch.no_grad():
        posterior = surrogate_max.posterior(df)
    acqf = get_acqf_values(acqf_cls, surrogate_max, searchspace, objective, df)
    return posterior, acqf


@pytest.mark.parametrize(
    "acqf_cls",
    [
        a
        for a in get_subclasses(AcquisitionFunction)
        if not issubclass(a, qKnowledgeGradient)  # TODO: not yet clear how to handle
    ],
)
def test_minimization(acqf_cls):
    """Maximizing targets is equivalent to minimizing target with inverted data."""
    values = np.linspace(10, 20)
    searchspace = NumericalDiscreteParameter("p", values).to_searchspace()

    # Maximization of plain targets
    set_random_seed(0)
    df_max = pd.DataFrame({"p": values, "t": values})
    obj_max = NumericalTarget("t", "MAX").to_objective()
    p_min, acqf_max = compute_posterior_and_acqf(acqf_cls, df_max, searchspace, obj_max)

    # Minimization of inverted targets
    set_random_seed(0)
    df_min = pd.DataFrame({"p": values, "t": -values})
    obj_min = NumericalTarget("t", "MIN").to_objective()
    p_max, acqf_min = compute_posterior_and_acqf(acqf_cls, df_min, searchspace, obj_min)

    # Both must yield identical posterior (modulo the sign) and acquisition values
    assert torch.equal(p_min.mean, -p_max.mean)
    assert torch.equal(p_min.mvn.covariance_matrix, p_max.mvn.covariance_matrix)
    # TODO: https://github.com/pytorch/botorch/issues/2681
    assert_close(acqf_max, acqf_min, rtol=0.0001, atol=0.1)
