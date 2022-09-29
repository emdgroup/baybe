# pylint: disable=too-few-public-methods
"""
Surrogate models, such as Gaussian processes, random forests, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from ngboost import NGBRegressor

from torch import Tensor

from .utils import isabstract, to_tensor

def check_x(X: Tensor):
    """Helper function to validate the input x"""
    if len(X) == 0:
        raise ValueError("The input dataset must be non-empty")
        
def check_y(Y: Tensor):
    """Helper function to validate the input y"""
    if Y.shape[1] != 1:
        raise NotImplementedError("The model currently supports only one target.")


def hallucinate(X: Tensor, Y: Tensor):
    """Helper function to create an extra data point for certain models"""
    # Previous approach: copy data point - theoretical variance of this is 0
    # return (X.repeat((2,)+(1,)*(len(X.shape)-1)), Y.repeat((2,)+(1,)*(len(X.shape)-1)))
    
    # Current approach: add a zero data point
    return (torch.cat((X,torch.zeros(X.shape))), torch.cat((Y,torch.zeros(Y.shape))))

def add_noise(X: Tensor, Y: Tensor):
    """Helper function to add noise to avoid variance nearing zero (numerical instability)"""
    # Add noise of amplitude
    AMP = 1e-3
    return (X + AMP*torch.randn(X.shape), Y + AMP*torch.randn(Y.shape))
    
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


class RandomForestModel(SurrogateModel):
    """A random forest surrogate model"""

    type = "RF"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[RandomForestRegressor] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace
    
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""

        # TODO: Input/Output Transforms
        # Not Needed - Ensemble Methods

        # TODO: Decide between the below methods (1,2,3)
        # Method 1 (Old), can only handle q = 1
        # candidates = candidates.squeeze(1)
        # mean2 = Tensor(self.model.predict(candidates)).unsqueeze(1)

        # epred = [self.model.estimators_[tree].predict(candidates) 
        #             for tree in range(self.model.n_estimators)]
        # Calculate variance
        # var = torch.diag(Tensor(np.var(epred, axis=0))).unsqueeze(2)
        # var2 = Tensor(np.var(epred, axis=0)).unsqueeze(1).unsqueeze(1)

        # method 2, can only handle t spanning one dimension
        # Predict mean (assuming size *t, q, d)
        # means = [
        #     Tensor(self.model.predict(t)).unsqueeze(-1) for t in candidates.unbind(dim=-2)
        # ]

        # mean = torch.cat(tuple(means),dim=-1)

        # epreds = [
        #     Tensor(np.var([self.model.estimators_[tree].predict(t) 
        #             for tree in range(self.model.n_estimators)], axis=0))
            
        #     for t in candidates.unbind(dim=-2)
        # ]

        # vars = torch.cat(tuple([ep.unsqueeze(-1) for ep in epreds]), dim=-1)

        # var = torch.cat(tuple([torch.diag(v).unsqueeze(0) for v in vars.unbind(-2)]))

        # Method 3
        # TODO: Understand how multi-dimensional t-batch works

        # Flatten t-batch
        flattened = candidates.flatten(end_dim=-3)

        # Get means for each q-batch
        means = [
            Tensor(self.model.predict(t)).unsqueeze(-1) for t in flattened.unbind(dim=-2)
        ]

        # Combine the means and reshape t-batch
        mean = torch.cat(tuple(means),dim=-1).reshape(candidates.shape[:-1])

        # Printouts
        # print(mean)
        # print(mean.size())
        
        # Get Ensemble predictions (assuming size *t, q, d)

        # Get q-batch dimension
        q = candidates.shape[-2]

        # Get variance for each q-batch
        epreds = [
            Tensor(np.var([self.model.estimators_[tree].predict(t) 
                    for tree in range(self.model.n_estimators)], axis=0))
            
            for t in flattened.unbind(dim=-2)
        ]

        # Combine variances
        vars = torch.cat(tuple([ep.unsqueeze(-1) for ep in epreds]), dim=-1)

        # Construct diagonal covariance matrices
        covar = torch.cat(tuple([torch.diag(v).unsqueeze(0) for v in vars.unbind(-2)]))
        
        # Reshape t-batch
        covar = covar.reshape(candidates.shape[:-2]+(q,q))
        
        # Printouts
        # print(covar)
        # print(covar.size())
        return mean,covar
    
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        # Validate Input
        check_x(train_x)
        check_y(train_y)

        # TODO: Input/Output Transforms
        # Not needed - Ensemble Method

        # Slightly modify input if necessary
        if len(train_x) == 1:
            train_x, train_y = hallucinate(train_x, train_y)
        if train_x.var() < 1e-6:
            train_x, train_y = add_noise(train_x, train_y)

        # Create Model
        self.model = RandomForestRegressor()
        # Train model
        self.model.fit(train_x, train_y.ravel())


class NGBoostModel(SurrogateModel):
    """A natural-gradient-boosting surrogate model"""

    type = "NG"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[NGBRegressor] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""

        # TODO: Input/Output Transforms
        # Not helpful - Ensemble method

        # Method 1 (Old), can only handle q=1
        # Predict
        # candidates = candidates.squeeze(1)
        # pred = self.model.pred_dist(candidates)
        
        # Get mean and variance
        # mean = Tensor(pred.mean()).unsqueeze(1)
        # var = torch.diag(Tensor(pred.std()**2))
        # var = Tensor(pred.std()**2).unsqueeze(1).unsqueeze(1)

        # Method 3
        # TODO: Understand how multi-dimensional t-batch works

        # Get q-batch dimension 
        q = candidates.shape[-2]

        # Flatten t-batch
        flattened = candidates.flatten(end_dim=-3)

        # Get distribution for each q-batch
        dists = [
            self.model.pred_dist(t) for t in flattened.unbind(dim=-2)
        ]

        # Extract means and vars
        means = [Tensor(d.mean()).unsqueeze(-1) for d in dists]
        vars = [Tensor(d.std()**2).unsqueeze(-1) for d in dists]

        # Combine means and reshape t-batch
        mean = torch.cat(tuple(means),dim=-1).reshape(candidates.shape[:-1])

        # Combine variances
        vars = torch.cat(tuple(vars), dim=-1)

        # Construct diagonal covariance matrices
        covar = torch.cat(tuple([torch.diag(v).unsqueeze(0) for v in vars.unbind(-2)]))

        # Reshape t-batch
        covar = covar.reshape(candidates.shape[:-2]+(q,q))

        return mean, covar
    
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        # Validate Input
        check_x(train_x)
        check_y(train_y)

        # Slightly modify input if necessary
        if len(train_x) == 1:
            train_x, train_y = hallucinate(train_x, train_y)
        if train_x.var() < 1e-6:
            train_x, train_y = add_noise(train_x, train_y)

        # TODO: Input/Output Transforms
        # Not needed - Ensemble method

        # Create and Train model
        self.model = NGBRegressor(n_estimators=100,verbose=False).fit(train_x, train_y.ravel())


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
    
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""

        # TODO: Input/Output Transforms
        # Added with SK-Learn Standard Scaler

        # Method 1 (Old), can only handle q = 1
        # Convert tensors to numpy array
        # candidates = np.array(candidates.squeeze(1))

        # Predict
        # pred, std = self.model.predict(candidates, return_std=True)

        # mean = Tensor(pred).unsqueeze(1)
        # var = Tensor(std**2).unsqueeze(1).unsqueeze(1)

        # Method 3
        # TODO: Understand how multi-dimensional t-batch works

        # Get q-batch dimension
        q = candidates.shape[-2]

        # Flatten t-batch
        flattened = candidates.flatten(end_dim=-3)

        # Get distribution for each q-batch
        dists = [
            self.model.predict(np.array(t), return_std=True) 
                for t in flattened.unbind(dim=-2)
        ]

        # Extract means and vars
        means = [Tensor(d[0]).unsqueeze(-1) for d in dists]
        vars = [Tensor(d[1]**2).unsqueeze(-1) for d in dists]

        # Combine means and reshape t-batch
        mean = torch.cat(tuple(means),dim=-1).reshape(candidates.shape[:-1])

        # Combine variances
        vars = torch.cat(tuple(vars), dim=-1)

        # Construct diagonal covariance matrices
        covar = torch.cat(tuple([torch.diag(v).unsqueeze(0) for v in vars.unbind(-2)]))

        # Reshape t-batch
        covar = covar.reshape(candidates.shape[:-2]+(q,q))

        return mean, covar
    
    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        # Validate Input
        check_x(train_x)
        check_y(train_y)

        # Slightly modify input if necessary
        if len(train_x) == 1:
            train_x, train_y = hallucinate(train_x, train_y)
        if train_x.var() < 1e-6:
            train_x, train_y = add_noise(train_x, train_y)

        # TODO: Input/Output Transforms
        # searchspace = self.searchspace.to_numpy()
        # bounds = (np.min(searchspace), np.max(searchspace))

        # Create Model
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=self.degree),
            ARDRegression()
        )

        # Train model
        self.model.fit(train_x, train_y.ravel())
