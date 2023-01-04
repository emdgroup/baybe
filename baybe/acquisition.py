"""
Adapter functionality to make BoTorch's acquisition functions work with other models.
"""
from inspect import signature
from typing import Any, Callable, List, Optional, Type

import gpytorch.distributions
from botorch.acquisition import AcquisitionFunction
from botorch.models.gpytorch import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import Tensor

from baybe.surrogate import SurrogateModel


def debotorchize(acqf_cls: Type[AcquisitionFunction]):
    """
    Wraps a given BoTorch acquisition function such that becomes generally usable in
    combination with other non-BoTorch surrogate models.

    This is required since BoTorch's acquisition functions expect a
    `botorch.model.Model` to work with, hindering their general use with arbitrary
    probabilistic models. The wrapper class returned by this function resolves this
    issue by operating as an adapter that internally creates a helper BoTorch model,
    which serves as a translation layer and is passed to the selected BoTorch
    acquisition function, carrying the posterior information provided from any other
    probabilistic model implementing BayBE's `SurrogateModel` interface.

    Parameters
    ----------
    acqf_cls : Type[AcquisitionFunction]
        An arbitrary BoTorch acquisition function class.

    Returns
    -------
    A wrapped version of the class that accepts non-BoTorch surrogate models.

    Example
    -------
    from botorch.acquisition import ExpectedImprovement
    from baybe.surrogate import BayesianLinearModel
    surrogate = BayesianLinearModel(*args, **kwargs)
    surrogate.fit(train_x, train_y)
    best_f = train_y.max()
    acqf = debotorchize(ExpectedImprovement)(surrogate, best_f)
    acqf_scores = acqf(candidates)
    """

    class Wrapper:
        """Adapter acquisition function that accepts BayBE surrogate models."""

        def __init__(self, surrogate: SurrogateModel, best_f):
            self.model = AdapterModel(surrogate)
            self.best_f = best_f

            required_params = {
                p: v
                for p, v in {"model": self.model, "best_f": self.best_f}.items()
                if p in signature(acqf_cls).parameters
            }
            self.acqf = acqf_cls(**required_params)

        def __call__(self, candidates):
            return self.acqf(candidates)

        def __getattr__(self, item):
            return getattr(self.acqf, item)

    return Wrapper


class AdapterModel(Model):
    """
    A BoTorch model that internally uses a BayBE surrogate model for posterior
    computation. Can be used, for example, as an adapter layer for making a BayBE
    surrogate model usable in conjunction with BoTorch acquisition functions.
    """

    def __init__(self, surrogate: SurrogateModel):
        super().__init__()
        self._surrogate = surrogate

    @property
    def num_outputs(self) -> int:
        """See base class."""
        # TODO: So far, the usage is limited to single-output models.
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        """See base class."""
        mean, var = self._surrogate.posterior(X)
        mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        return GPyTorchPosterior(mvn)
