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
from torch import cat, squeeze, Tensor

from baybe.surrogate import Surrogate


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
    probabilistic model implementing BayBE's `Surrogate` interface.

    Parameters
    ----------
    acqf_cls : Type[AcquisitionFunction]
        An arbitrary BoTorch acquisition function class.

    Returns
    -------
    A wrapped version of the class that accepts non-BoTorch surrogate models.

    Example
    -------
    # TODO: Turn into doctest
    from botorch.acquisition import ExpectedImprovement
    from baybe.surrogate import BayesianLinearSurrogate
    surrogate = BayesianLinearSurrogate(*args, **kwargs)
    surrogate.fit(train_x, train_y)
    best_f = train_y.max()
    acqf = debotorchize(ExpectedImprovement)(surrogate, best_f)
    acqf_scores = acqf(candidates)
    """

    class Wrapper:
        """Adapter acquisition function that accepts BayBE surrogate models."""

        def __init__(self, surrogate: Surrogate, best_f):
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

    def __init__(self, surrogate: Surrogate):
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


class PartialAcquisitionFunction:
    """
    Acquisition function for evaluating points in a searchspace where of the subspaces
    is fixed.

    It can either pin the discrete or the continuous part. The pinned part is assumed
    to be a tensor of dimension dx1 where d is the computational dimension of the
    searchspace that is to be pinned. The acquisition function is assumed to be defined
    for the full hybrid space.

    Parameters
    ----------
    acqf: AcquisitionFunction
        The acquisition function for the hybrid space
    pinned_part: Tensor
        The values that will be attached whenever evaaluating the acquisition function
    pin_discrete: bool
        A flag for denoting whether the pinned_part corresponds to the discrete subspace
    """

    def __init__(
        self, acqf: AcquisitionFunction, pinned_part: Tensor, pin_discrete: bool
    ):
        self.acqf = acqf
        self.pinned_part = pinned_part
        self.pin_discrete = pin_discrete

    def _lift(self, partial_part: Tensor):
        """
        Lift the partial_part to the original hybrid space.

        Depending on whether the discrete or the variable part of the searchspace is
        pinned, this function identifies whether the partial_part is the continuous
        or discrete part and then constructs the full tensor accordingly.

        Parameters
        ----------
        partial_part: Tensor
            The part of the tensor that is to be evaluated in the partial space
        """
        # Might be necessary to insert a dummy dimension
        if partial_part.ndim == 2:
            partial_part = partial_part.unsqueeze(-2)
        # Repeat the pinned part such that it matches the dimension of the partial_part
        pinned_part = self.pinned_part.repeat(
            (partial_part.shape[0], partial_part.shape[1], 1)
        )
        # Check which part is discrete and which is continuous
        if self.pin_discrete:
            disc_part = pinned_part
            cont_part = partial_part
        else:
            disc_part = partial_part
            cont_part = pinned_part
        # Concat the parts and return the concatenated point
        full_point = cat((disc_part, cont_part), -1)
        return full_point

    def __call__(self, variable_part):
        # Lift the point to the hybrid space, then evaluate the acquisition function
        # in the hybrid space.
        full_point = self._lift(variable_part)
        return self.acqf(full_point)

    def __getattr__(self, item):
        return getattr(self.acqf, item)

    def set_X_pending(self, X_pending: Optional[Tensor]):  # pylint: disable=C0103
        """
        Informs the acquisition function about pending design points.

        Enhances the original 'set_X_pending' function from the full acquisition
        function as we need to store the full point, i.e., the point in the hybrid space
        for the PartialAcquisitionFunction to work properly.

        Parameters (according to original documentation)
        ----------
        X_pending: Tensor, optional
            n x d Tensor with n d-dim design points that have been submitted for
            evaluation but have not yet been evaluated.
        """
        if X_pending is not None:  # Lift point to hybrid space and add additional dim
            X_pending = self._lift(X_pending)
            X_pending = squeeze(X_pending, -2)
        # Now use the original set_X_pending function
        self.acqf.set_X_pending(X_pending)
