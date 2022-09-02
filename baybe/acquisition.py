# pylint: disable=unused-argument
"""
Adapter functionality to make BoTorch's acquisition functions work with other models.
"""
from inspect import signature
from typing import Type

import gpytorch.distributions
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP

from baybe.surrogate import SurrogateModel


def debotorchize(acqf: Type[AcquisitionFunction]):
    """
    Wraps a given BoTorch acquisition function such that becomes generally usable in
    combination with other non-BoTorch surrogate models.

    This is required since BoTorch's acquisition functions expect a
    `botorch.model.Model` to work with, hindering their general use with arbitrary
    probabilistic models. The wrapper class returned by this function resolves this
    issue by operating as an adapter that internally creates a dummy BoTorch model
    that is passed to the selected BoTorch acquisition function, carrying the posterior
    information provided from any other probabilistic model implementing BayBE's
    `SurrogateModel` interface.

    Example:
    --------
    from botorch.acquisition import ExpectedImprovement
    from baybe.surrogate import SurrogateModel
    acqf = debotorchize(ExpectedImprovement)(surrogate, best_f)
    acqf_scores = acqf(candidates)
    """

    class Wrapper:
        """Adapter acquisition function that accepts BayBE surrogate models."""

        def __init__(self, surrogate: SurrogateModel, best_f):
            self.surrogate = surrogate
            self.best_f = best_f
            self.acqf = acqf

        def __call__(self, candidates):
            mean, var = self.surrogate.posterior(candidates)
            mean = torch.from_numpy(mean)
            var = torch.from_numpy(var)
            mvn = gpytorch.distributions.MultivariateNormal(mean, var)
            model = self.DummyModel(mvn)
            required_params = {
                p: v
                for p, v in {"model": model, "best_f": self.best_f}.items()
                if p in signature(acqf).parameters
            }
            return self.acqf(**required_params)(candidates)

        class DummyModel(GPyTorchModel, ExactGP):
            """
            Dummy model to pass the posterior information to BoTorch's acquisition
            function.
            """

            def __init__(self, mvn):
                GPyTorchModel.__init__(self)
                ExactGP.__init__(self, None, None, GaussianLikelihood())
                self.mvn = mvn

            @property
            def num_outputs(self) -> int:
                return 1

            def posterior(self, *args, **kwargs) -> GPyTorchPosterior:
                return GPyTorchPosterior(self.mvn)

    return Wrapper
