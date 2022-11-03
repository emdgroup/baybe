# pylint: disable=too-few-public-methods
"""
Surrogate models, such as Gaussian processes, random forests, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, Optional, Tuple, Type

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
from torch import Tensor

from .scaler import DefaultScaler
from .utils import isabstract, to_tensor


MIN_TARGET_STD = 1e-6


def _check_x(x: Tensor) -> None:
    """Helper function to validate the model input."""
    if len(x) == 0:
        raise ValueError("The model input must be non-empty.")


def _check_y(y: Tensor) -> None:
    """Helper function to validate the model targets."""
    if y.shape[1] != 1:
        raise NotImplementedError("The model currently supports only one target.")


def catch_constant_targets(model_cls: Type[SurrogateModel]):
    """
    Wraps a given `SurrogateModel` class that cannot handle constant training target
    values such that these cases are handled by a separate model type.
    """

    class SplitModel(SurrogateModel):
        """
        A surrogate model that applies a separate strategy for cases where the training
        targets are all constant and no variance can be estimated.
        """

        # Overwrite the registered subclass with the wrapped version
        type = model_cls.type

        def __init__(self, *args, **kwargs):
            """Stores an instance of the underlying model class."""
            self.model = model_cls(*args, **kwargs)

        def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """Calls the posterior function of the internal model instance."""
            return self.model.posterior(candidates)

        def fit(self, train_x: Tensor, train_y: Tensor) -> None:
            """Selects a model based on the variance of the targets and fits it."""
            # Validate the training data
            # TODO: move the validation to the surrogate model class
            _check_x(train_x)
            _check_y(train_y)

            # https://github.com/pytorch/pytorch/issues/29372
            # Needs 'unbiased=False' (otherwise, the result will be NaN for scalars)
            if torch.std(train_y.ravel(), unbiased=False) < MIN_TARGET_STD:
                self.model = MeanPredictionModel(self.model.searchspace)

            # Fit the selected model with the training data
            self.model.fit(train_x, train_y)

        def __getattribute__(self, attr):
            """
            Accesses the attributes of the class instance if available, otherwise uses
            the attributes of the internal model instance.
            """
            # Try to retrieve the attribute in the class
            try:
                val = super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                return val

            # If the attribute has not been overwritten, use that of the internal model
            return self.model.__getattribute__(attr)

    return SplitModel


def scale_model(model_cls: Type[SurrogateModel]):
    """
    Wraps a given `SurrogateModel` class such that it operates with scaled
    representations of the training and test data.
    """

    class ScaledModel(model_cls):
        """Overrides the methods of the given model class such the use scaled data."""

        def __init__(self, *args, **kwargs):
            """Stores an instance of the underlying model class and a scaler object."""
            self.model = model_cls(*args, **kwargs)
            self.scaler = None

        def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Calls the posterior function of the internal model instance on
            a scaled version of the test data and rescales the output accordingly.
            """
            candidates = self.scaler.transform(candidates)
            mean, covar = self.model.posterior(candidates)
            return self.scaler.untransform(mean, covar)

        def fit(self, train_x: Tensor, train_y: Tensor) -> None:
            """Fits the scaler and the model using the scaled training data."""
            self.scaler = DefaultScaler(self.model.searchspace)
            train_x, train_y = self.scaler.fit_transform(train_x, train_y)
            self.model.fit(train_x, train_y)

        def __getattribute__(self, attr):
            """
            Accesses the attributes of the class instance if available, otherwise uses
            the attributes of the internal model instance.
            """
            # Try to retrieve the attribute in the class
            try:
                val = super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                return val

            # If the attribute has not been overwritten, use that of the internal model
            return self.model.__getattribute__(attr)

    return ScaledModel


def batchify(
    posterior: Callable[[SurrogateModel, Tensor], Tuple[Tensor, Tensor]]
) -> Callable[[SurrogateModel, Tensor], Tuple[Tensor, Tensor]]:
    """
    Wraps `SurrogateModel` posterior functions that are incompatible with t- and
    q-batching such that they become able to process batched inputs.
    """

    @wraps(posterior)
    def sequential_posterior(
        model: SurrogateModel, candidates: Tensor
    ) -> [Tensor, Tensor]:
        """A posterior function replacement that processes batches sequentially."""

        # If no batch dimensions are given, call the model directly
        if candidates.ndim == 2:
            return posterior(model, candidates)

        # Keep track of batch dimensions
        t_shape = candidates.shape[:-2]
        q_shape = candidates.shape[-2]

        # Flatten all t-batch dimensions into a single one
        flattened = candidates.flatten(end_dim=-3)

        # Call the model on each (flattened) t-batch
        out = (posterior(model, batch) for batch in flattened)

        # Collect the results and restore the batch dimensions
        mean, covar = zip(*out)
        mean = torch.reshape(torch.stack(mean), t_shape + (q_shape,))
        covar = torch.reshape(torch.stack(covar), t_shape + (q_shape, q_shape))

        return mean, covar

    return sequential_posterior


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


class MeanPredictionModel(SurrogateModel):
    """
    A trivial surrogate model that uses the average value of the training targets as
    posterior mean and a (data-independent) identity covariance matrix as posterior
    covariance.
    """

    type = "MP"

    def __init__(self, searchspace: pd.DataFrame):  # pylint: disable=unused-argument
        self.target_value = None

    @batchify
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        # TODO: use target value bounds for covariance scaling when explicitly provided
        mean = self.target_value * torch.ones([len(candidates)])
        covar = torch.eye(len(candidates))
        return mean, covar

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        self.target_value = train_y.mean().item()


@catch_constant_targets
@scale_model
class RandomForestModel(SurrogateModel):
    """A random forest surrogate model."""

    type = "RF"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[RandomForestRegressor] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    @batchify
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""

        # Evaluate all trees
        predictions = torch.as_tensor(
            [
                self.model.estimators_[tree].predict(candidates)
                for tree in range(self.model.n_estimators)
            ]
        )

        # Compute posterior mean and variance
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)

        return mean, torch.diag(var)

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        self.model = RandomForestRegressor()
        self.model.fit(train_x, train_y.ravel())


@catch_constant_targets
@scale_model
class NGBoostModel(SurrogateModel):
    """A natural-gradient-boosting surrogate model."""

    type = "NG"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[NGBRegressor] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    @batchify
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        # Get predictions
        dists = self.model.pred_dist(candidates)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists.mean())
        var = torch.from_numpy(dists.var)

        return mean, torch.diag(var)

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        self.model = NGBRegressor(n_estimators=25, verbose=False).fit(
            train_x, train_y.ravel()
        )


@catch_constant_targets
@scale_model
class BayesianLinearModel(SurrogateModel):
    """A Bayesian linear regression surrogate model."""

    type = "BL"

    def __init__(self, searchspace: pd.DataFrame):
        self.model: Optional[ARDRegression] = None
        # TODO: the surrogate model should work entirely on Tensors (parameter name
        #  agnostic) -> the scaling information should not be provided in form of a
        #  DataFrame
        self.searchspace = searchspace

    @batchify
    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""
        # Get predictions
        dists = self.model.predict(candidates.numpy(), return_std=True)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists[0])
        var = torch.from_numpy(dists[1]).pow(2)

        return mean, torch.diag(var)

    def fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        self.model = ARDRegression()
        self.model.fit(train_x, train_y.ravel())
