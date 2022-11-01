# pylint: disable=too-few-public-methods
"""
Surrogate models, such as Gaussian processes, random forests, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from ngboost import NGBRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from sklearn.pipeline import Pipeline

from torch import Tensor

from .scaler import DefaultScaler
from .utils import isabstract, to_tensor


def _check_x(x: Tensor):
    """Helper function to validate the input x"""
    if len(x) == 0:
        raise ValueError("The input dataset must be non-empty")


def _check_y(y: Tensor):
    """Helper function to validate the input y"""
    if y.shape[1] != 1:
        raise NotImplementedError("The model currently supports only one target.")


def _hallucinate(x: Tensor, y: Tensor):
    """Helper function to create an extra data point for certain models"""
    # Previous approach: copy data point - theoretical variance of this is 0
    # return
    # (x.repeat((2,)+(1,)*(len(x.shape)-1)), y.repeat((2,)+(1,)*(len(x.shape)-1)))

    # Current approach: add a "noisy" zero data point
    amplitude = 1e-3
    fake_x = amplitude * torch.randn(x.shape)
    fake_y = amplitude * torch.randn(y.shape)
    return (torch.cat((x, fake_x)), torch.cat((y, fake_y)))


def _smooth_var(covar: Tensor):
    """
    Helper function to smooth variance to avoid nearing zero (numerical instability)
    """
    # Add fixed var of amplitude
    amplitude = 1e-6
    return covar + amplitude


def scale_model(model: Type[SurrogateModel]):
    """A wrapper for models to be scaled"""

    class ScaledModel(model):
        """A scaled model"""

        def __init__(self, *args):
            """Init with underlying surrogate and scaler"""
            self.model = model
            self.scaler = None
            self.searchspace = args[0]  # searchspace as an argument

        def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """Scaled posterior"""
            # Scale input
            candidates = self.scaler.transform(candidates)
            # Call posterior
            mean, covar = self.model.posterior(self.model, candidates)
            # Unscale output
            mean, covar = self.scaler.untransform(mean, covar)
            # Smooth variance
            covar = _smooth_var(covar)
            return mean, covar

        def fit(self, train_x: Tensor, train_y: Tensor) -> None:
            """Fit scaler and model"""
            # Initialize scaler
            self.scaler = DefaultScaler(self.searchspace)
            # Scale inputs
            train_x, train_y = self.scaler.fit_transform(train_x, train_y)
            # Call model fit
            self.model.fit(self.model, train_x, train_y)

        def __getattribute__(self, attr):
            """Getter for all other attributes"""
            # Attributes for Scaled Model
            try:
                val = super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                return val

            # Additional attributes for underlying scaled model, if needed
            return self.model.__getattribute__(attr)

    return ScaledModel


def batch_untransform(
    posterior: Callable[[SurrogateModel, Tensor], Tuple[Tensor, Tensor]]
) -> Callable[[SurrogateModel, Tensor], Tuple[Tensor, Tensor]]:
    """A wrapper for posterior functions incompatible with t, q batchings"""

    @wraps(posterior)
    def decorated(model: SurrogateModel, candidates: Tensor) -> [Tensor, Tensor]:
        """Helper function to remove t, q batching"""

        # Check if batching is needed
        if len(candidates.shape) > 2:
            # Keep track of dimension
            t_shape = candidates.shape[:-2]
            q_shape = candidates.shape[-2]

            # Remove all batching
            untransformed = candidates.flatten(end_dim=-2)

            # Call function
            mean, covar = posterior(model, untransformed)
            var = covar.diag()

            # Transform back
            mean = mean.reshape(t_shape + (q_shape,))
            var = var.reshape(t_shape + (q_shape,)).flatten(end_dim=-2)

            covar = torch.cat(tuple(torch.diag(v).unsqueeze(0) for v in var.unbind(-2)))
            covar = covar.reshape(t_shape + (q_shape, q_shape))

        else:
            mean, covar = posterior(model, candidates)

        return mean, covar

    return decorated


class SurrogateModel(ABC):
    """Abstract base class for all surrogate models."""

    # TODO: to support other models than GPs, an interface to botorch's acquisition
    #  functions must be created (e.g. via a dedicated 'predict' method)

    type: str
    SUBCLASSES: Dict[str, Type[SurrogateModel]] = {}

    @abstractmethod
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Evaluates the surrogate model at the given candidate points.

        Parameters
        ----------
        candidates : torch.Tensor
            The candidate points, represented as a tensor of shape (*t, q, d), where
            't' denotes the "t-batch" shape, 'q' denotes the "q-batch" shape, and
            'd' is the input dimension. For more details about batch shapes, see:
            https://botorch.org/docs/batching

        Returns
        -------
        Tuple[Tensor, Tensor]
            The posterior means and posterior covariance matrices of the t-batched
            candidate points.
        """

    @abstractmethod
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Trains the surrogate model on the provided data."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls


class GaussianProcessModel(SurrogateModel):
    """A Gaussian process surrogate model."""

    type = "GP"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[SingleTaskGP] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        posterior = self.model.posterior(candidates)
        return posterior.mvn.mean, posterior.mvn.covariance_matrix

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""

        # validate input
        if len(train_x) == 0:
            raise ValueError("The training data set must be non-empty.")
        if train_y.shape[1] != 1:
            raise NotImplementedError("The model currently supports only one target.")

        # get the input bounds from the search space
        searchspace = to_tensor(self.searchspace)
        bounds = torch.vstack(
            [torch.min(searchspace, dim=0)[0], torch.max(searchspace, dim=0)[0]]
        )
        # TODO: use target value bounds when explicitly provided

        # define the input and outcome transforms
        # TODO [Scaling]: scaling should be handled by searchspace object
        input_transform = Normalize(train_x.shape[1], bounds=bounds)
        outcome_transform = Standardize(train_y.shape[1])

        # ---------- GP prior selection ---------- #
        # TODO: temporary prior choices adapted from edbo, replace later on

        mordred = any("MORDRED" in col for col in self.searchspace.columns) or any(
            "RDKIT" in col for col in self.searchspace.columns
        )
        if mordred and train_x.shape[-1] < 50:
            mordred = False

        # low D priors
        if train_x.shape[-1] < 5:
            lengthscale_prior = [GammaPrior(1.2, 1.1), 0.2]
            outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
            noise_prior = [GammaPrior(1.05, 0.5), 0.1]

        # DFT optimized priors
        elif mordred and train_x.shape[-1] < 100:
            lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
            outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]

        # Mordred optimized priors
        elif mordred:
            lengthscale_prior = [GammaPrior(2.0, 0.1), 10.0]
            outputscale_prior = [GammaPrior(2.0, 0.1), 10.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]

        # OHE optimized priors
        else:
            lengthscale_prior = [GammaPrior(3.0, 1.0), 2.0]
            outputscale_prior = [GammaPrior(5.0, 0.2), 20.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]

        # ---------- End: GP prior selection ---------- #

        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]

        # create GP mean
        mean_module = ConstantMean(batch_shape=batch_shape)

        # create GP covariance
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
                batch_shape=batch_shape,
                lengthscale_prior=lengthscale_prior[0],
            ),
            batch_shape=batch_shape,
            outputscale_prior=outputscale_prior[0],
        )
        covar_module.outputscale = torch.tensor([outputscale_prior[1]])
        covar_module.base_kernel.lengthscale = torch.tensor([lengthscale_prior[1]])

        # create GP likelihood
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior[0], batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self.model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={"disp": False})


class TrivialModel(SurrogateModel):
    """A trivial surrogate model"""

    type = "TM"

    def __init__(self, searchspace: pd.DataFrame):
        self.model = None
        self.searchspace = searchspace

    @batch_untransform
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        # Predicts the mean of training data
        mean = self.model * torch.ones([len(candidates)])
        # Covariance is the identity matrix
        covar = torch.eye(len(candidates))
        return mean, covar

    def fit(self, train_x: Tensor, train_y: Tensor):
        """See base class."""
        # Keep track of training data
        self.model = float(torch.mean(train_y.ravel()))


@scale_model
class RandomForestModel(SurrogateModel):
    """A random forest surrogate model"""

    type = "RF"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[RandomForestRegressor] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    @batch_untransform
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""

        # Get predictions
        mean = Tensor(self.model.predict(candidates))

        # Get ensemble predictions
        var = Tensor(
            np.var(
                [
                    self.model.estimators_[tree].predict(candidates)
                    for tree in range(self.model.n_estimators)
                ],
                axis=0,
            )
        )

        return mean, torch.diag(var)

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        # Validate Input
        _check_x(train_x)
        _check_y(train_y)

        # Slightly modify input if necessary
        if len(train_x) == 1:
            train_x, train_y = _hallucinate(train_x, train_y)

        # Create Model
        self.model = RandomForestRegressor()
        # Train model
        self.model.fit(train_x, train_y.ravel())


@scale_model
class NGBoostModel(SurrogateModel):
    """A natural-gradient-boosting surrogate model"""

    type = "NG"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[NGBRegressor] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    @batch_untransform
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        # Get Predictions
        dists = self.model.pred_dist(candidates)

        # Split into mean and variance
        mean = Tensor([d.mean() for d in dists])
        var = Tensor([d.std() ** 2 for d in dists])

        return mean, torch.diag(var)

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        # Validate Input
        _check_x(train_x)
        _check_y(train_y)

        # Slightly modify input if necessary
        if len(train_x) == 1:
            train_x, train_y = _hallucinate(train_x, train_y)

        # Create and Train model
        self.model = NGBRegressor(n_estimators=25, verbose=False).fit(
            train_x, train_y.ravel()
        )


@scale_model
class BayesianLinearModel(SurrogateModel):
    """A Bayesian linear regression surrogate model"""

    type = "BL"

    def __init__(self, searchspace: pd.DataFrame, degree=1):
        self.model: Optional[Pipeline] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

        # TODO: Add in degree option for the BL model
        self.degree = degree

    @batch_untransform
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        # Get predictions
        dists = self.model.predict(np.array(candidates), return_std=True)

        # Split into mean and variance
        mean = Tensor(dists[0])
        var = Tensor([d**2 for d in dists[1]])

        return mean, torch.diag(var)

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        # Validate Input
        _check_x(train_x)
        _check_y(train_y)

        # Slightly modify input if necessary
        if len(train_x) == 1:
            train_x, train_y = _hallucinate(train_x, train_y)

        # Create Model
        self.model = ARDRegression()
        # self.model = make_pipeline(
        #     PolynomialFeatures(degree=self.degree),
        #     ARDRegression()
        # )

        # Train model
        self.model.fit(train_x, train_y.ravel())
