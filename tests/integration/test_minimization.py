"""Tests for target minimization."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.testing import assert_close

from baybe.acquisition.acqfs import qKnowledgeGradient
from baybe.acquisition.base import AcquisitionFunction
from baybe.objectives.pareto import ParetoObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.basic import get_subclasses
from baybe.utils.random import set_random_seed


def compute_posterior_and_acqf(
    acqf_cls: type[AcquisitionFunction], df, searchspace, objective
):
    surrogate = GaussianProcessSurrogate()
    if acqf_cls.supports_multi_output:
        surrogate = surrogate.replicate()
    surrogate.fit(searchspace, objective, df)
    posterior = surrogate.posterior(df)
    acqf = acqf_cls().evaluate(
        df[list(searchspace.parameter_names)], surrogate, searchspace, objective, df
    )
    return posterior, torch.tensor(acqf)


@pytest.mark.parametrize(
    "acqf_cls",
    [
        a
        for a in get_subclasses(AcquisitionFunction)
        if not issubclass(a, qKnowledgeGradient)  # TODO: not yet clear how to handle
    ],
)
def test_minimization(acqf_cls: type[AcquisitionFunction]):
    """Maximizing targets is equivalent to minimizing target with inverted data."""
    values = np.linspace(10, 20)
    searchspace = NumericalDiscreteParameter("p", values).to_searchspace()

    # Maximization of plain targets
    set_random_seed(0)
    if acqf_cls.supports_multi_output:
        df_max = pd.DataFrame({"p": values, "t1": values, "t2": values})
        obj_max = ParetoObjective([NumericalTarget("t1"), NumericalTarget("t2")])
    else:
        df_max = pd.DataFrame({"p": values, "t": values})
        obj_max = NumericalTarget("t").to_objective()
    p_min, acqf_max = compute_posterior_and_acqf(acqf_cls, df_max, searchspace, obj_max)

    # Minimization of inverted targets
    set_random_seed(0)
    if acqf_cls.supports_multi_output:
        df_min = pd.DataFrame({"p": values, "t1": -values, "t2": -values})
        obj_min = ParetoObjective(
            [NumericalTarget("t1", minimize=True), NumericalTarget("t2", minimize=True)]
        )
    else:
        df_min = pd.DataFrame({"p": values, "t": -values})
        obj_min = NumericalTarget("t", minimize=True).to_objective()
    p_max, acqf_min = compute_posterior_and_acqf(acqf_cls, df_min, searchspace, obj_min)

    # Both must yield identical posterior (modulo the sign) and acquisition values
    assert torch.equal(p_min.mean, -p_max.mean)
    if acqf_cls.supports_multi_output:
        for pos_min, pos_max in zip(p_min.posteriors, p_max.posteriors):
            assert torch.equal(
                pos_min.mvn.covariance_matrix, pos_max.mvn.covariance_matrix
            )
    else:
        assert torch.equal(p_min.mvn.covariance_matrix, p_max.mvn.covariance_matrix)
    # TODO: https://github.com/pytorch/botorch/issues/2681
    assert_close(acqf_max, acqf_min, rtol=0.0001, atol=0.1)
