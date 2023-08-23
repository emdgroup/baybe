"""Surrogate models, such as Gaussian processes, random forests, etc."""
# TODO: ForwardRefs via __future__ annotations are currently disabled due to this issue:
#  https://github.com/python-attrs/cattrs/issues/354

import gc
import sys
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type

import cattrs
import numpy as np
import onnxruntime as ort
import torch
from attrs import define, field
from botorch.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import IndexKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from ngboost import NGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from torch import Tensor

from baybe.scaler import DefaultScaler
from baybe.searchspace import SearchSpace
from baybe.utils import get_subclasses, SerialMixin, unstructure_base

# Use float64 (which is recommended at least for BoTorch models)
_DTYPE = torch.float64

# Define constants
_MIN_TARGET_STD = 1e-6
_MIN_VARIANCE = 1e-6
_WRAPPER_MODELS = ("SplitModel", "ScaledModel", "CustomArchitectureSurrogate")


def _prepare_inputs(x: Tensor) -> Tensor:
    """Validate and prepare the model input.

    Args:
        x: The "raw" model input.

    Returns:
        The prepared input.

    Raises:
        ValueError: If the model input is empty.
    """
    if len(x) == 0:
        raise ValueError("The model input must be non-empty.")
    return x.to(_DTYPE)


def _prepare_targets(y: Tensor) -> Tensor:
    """Validate and prepare the model targets.

    Args:
        y: The "raw" model targets.

    Returns:
        The prepared targets.

    Raises:
        NotImplementedError: If there is more than one target.
    """
    if y.shape[1] != 1:
        raise NotImplementedError(
            "The model currently supports only one target or multiple targets in "
            "DESIRABILITY mode."
        )
    return y.to(_DTYPE)


def _get_model_params_validator(model_init: Callable) -> Callable:
    """Construct a validator based on the model class."""

    def validate_model_params(obj, _, model_params: dict) -> None:
        # Get model class name
        model = obj.__class__.__name__

        # GP does not support additional model params
        if "GaussianProcess" in model and model_params:
            raise ValueError(f"{model} does not support model params.")

        # Invalid params
        invalid_params = ", ".join(
            [
                key
                for key in model_params.keys()
                if key not in model_init.__code__.co_varnames
            ]
        )

        if invalid_params:
            raise ValueError(f"Invalid model params for {model}: {invalid_params}.")

    return validate_model_params


def _validate_custom_pretrained_params(obj, _, model_params: dict) -> None:
    """Validates custom pretrain model params."""
    try:
        onnx_str = model_params["onnx"]
        _ = model_params["onnx_input_name"]
    except KeyError as exc:
        raise ValueError(
            f"Incomplete model params for {obj.__class__.__name__}"
        ) from exc

    try:
        ort.InferenceSession(onnx_str.encode("ISO-8859-1"))
    except Exception as exc:
        raise ValueError(f"Invalid onnx str for {obj.__class__.__name__}") from exc


def catch_constant_targets(model_cls: Type["Surrogate"]):
    """Wrap a ```Surrogate``` class that cannot handle constant training target values.

    In the wrapped class, these cases are handled by a separate model type.

    Args:
        model_cls: A ```Surrogate``` class that should be wrapped.

    Returns:
        A wrapped version of the class.
    """

    class SplitModel(*model_cls.__bases__):
        """The class that is used for wrapping.

        It applies a separate strategy for cases where the training
        targets are all constant and no variance can be estimated.

        It stores an instance of the underlying model class.
        """

        # The posterior mode is chosen to match that of the wrapped model class
        joint_posterior: ClassVar[bool] = model_cls.joint_posterior

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = model_cls(*args, **kwargs)
            self.__class__.__name__ = self.model.__class__.__name__
            self.model_params = self.model.model_params

        def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """Calls the posterior function of the internal model instance."""
            mean, var = self.model._posterior(  # pylint: disable=protected-access
                candidates
            )

            # If a joint posterior is expected but the model has been overriden by one
            # that does not provide covariance information, construct a diagonal
            # covariance matrix
            if self.joint_posterior and not self.model.joint_posterior:
                # Convert to tensor containing covariance matrices
                var = torch.diag_embed(var)

            return mean, var

        def _fit(
            self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
        ) -> None:
            """Select a model based on the variance of the targets and fits it."""
            # https://github.com/pytorch/pytorch/issues/29372
            # Needs 'unbiased=False' (otherwise, the result will be NaN for scalars)
            if torch.std(train_y.ravel(), unbiased=False) < _MIN_TARGET_STD:
                self.model = MeanPredictionSurrogate()

            # Fit the selected model with the training data
            self.model.fit(searchspace, train_x, train_y)

        def __getattribute__(self, attr):
            """Accesses the attributes of the class instance if available.

            If the attributes are not available, it uses the attributes of the internal
            model instance.
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

    # Wrapping a class using a decorator does not transfer the doc, resulting in the
    # autodocumentation not showing the correct docstring. We thus need to manually
    # assign the docstring of the class.
    SplitModel.__doc__ = model_cls.__doc__
    return SplitModel


def scale_model(model_cls: Type["Surrogate"]):
    """Wrap a ```Surrogate``` class such that it operates with scaled representations.

    Args:
        model_cls: A ```Surrogate``` model class that should be wrapped.

    Returns:
        A wrapped version of the class.
    """

    class ScaledModel(*model_cls.__bases__):
        """Overrides the methods of the given model class such the use scaled data.

        It stores an instance of the underlying model class and a scalar object.
        """

        # The posterior mode is chosen to match that of the wrapped model class
        joint_posterior: ClassVar[bool] = model_cls.joint_posterior

        def __init__(self, *args, **kwargs):
            self.model = model_cls(*args, **kwargs)
            self.__class__.__name__ = self.model.__class__.__name__
            self.model_params = self.model.model_params
            self.scaler = None

        def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
            """Call the posterior function of the internal model instance.

            This call is made on a scaled version of the test data and rescales the
            output accordingly.
            """
            candidates = self.scaler.transform(candidates)
            mean, covar = self.model._posterior(  # pylint: disable=protected-access
                candidates
            )
            return self.scaler.untransform(mean, covar)

        def _fit(
            self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
        ) -> None:
            """Fits the scaler and the model using the scaled training data."""
            self.scaler = DefaultScaler(searchspace.discrete.comp_rep)
            train_x, train_y = self.scaler.fit_transform(train_x, train_y)
            self.model.fit(searchspace, train_x, train_y)

        def __getattribute__(self, attr):
            """Accesses the attributes of the class instance if available.

            If the attributes are not available, it uses the attributes of the internal
            model instance.
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

    # Wrapping a class using a decorator does not transfer the doc, resulting in the
    # autodocumentation not showing the correct docstring. We thus need to manually
    # assign the docstring of the class.
    ScaledModel.__doc__ = model_cls.__doc__
    return ScaledModel


def register_custom_architecture(
    joint_posterior_attr: bool = False,
    constant_target_catching: bool = True,
    batchify_posterior: bool = True,
):
    """
    Wraps a given Custom Class with fit and posterior functions
    to enable BayBE to interface with custom architectures.
    """

    def construct_custom_architecture(model_cls):
        """Constructs a surrogate class wrapped around the custom class."""

        class CustomArchitectureSurrogate(Surrogate):
            """Wraps around a custom architecture class."""

            joint_posterior: ClassVar[bool] = joint_posterior_attr
            model_params: Dict[str, Any] = field(factory=dict)

            def __init__(self, *args, **kwargs):
                """Stores an instance of the underlying model class."""
                self.model = model_cls(*args, **kwargs)

            def _fit(
                self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
            ) -> None:
                """See base class."""
                return self.model._fit(  # pylint: disable=protected-access
                    searchspace, train_x, train_y
                )

            def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
                """See base class."""
                return self.model._posterior(  # pylint: disable=protected-access
                    candidates
                )

            def __get_attribute__(self, attr):
                """
                Accesses the attributes of the class instance if available,
                otherwise uses the attributes of the internal model instance.
                """
                # Try to retrieve the attribute in the class
                try:
                    val = super().__getattribute__(attr)
                except AttributeError:
                    pass
                else:
                    return val

                # If the attribute is not overwritten, use that of the internal model
                return self.model.__getattribute__(attr)

        # Catch constant targets if needed
        cls = (
            catch_constant_targets(CustomArchitectureSurrogate)
            if constant_target_catching
            else CustomArchitectureSurrogate
        )

        # batchify posterior if needed
        if batchify_posterior:
            cls._posterior = batchify(  # pylint: disable=protected-access
                cls._posterior  # pylint: disable=protected-access
            )

        return cls

    return construct_custom_architecture


def batchify(
    posterior: Callable[["Surrogate", Tensor], Tuple[Tensor, Tensor]]
) -> Callable[["Surrogate", Tensor], Tuple[Tensor, Tensor]]:
    """Wrap ```Surrogate``` posterior functions to enable proper batching.

    More precisely, this wraps model that are incompatible with t- and q-batching such
    that they become able to process batched inputs.

    Args:
        posterior: The original ```posterior``` function.

    Returns:
        The wrapped posterior function.
    """

    @wraps(posterior)
    def sequential_posterior(
        model: "Surrogate", candidates: Tensor
    ) -> [Tensor, Tensor]:
        """A posterior function replacement that processes batches sequentially.

        Args:
            model: The ```Surrogate``` model.
            candidates: The candidates tensor.

        Returns:
            The mean and the covariance.
        """
        # If no batch dimensions are given, call the model directly
        if candidates.ndim == 2:
            return posterior(model, candidates)

        # Keep track of batch dimensions
        t_shape = candidates.shape[:-2]
        q_shape = candidates.shape[-2]

        # If the posterior function provides full covariance information, call it
        # t-batch by t-batch
        if model.joint_posterior:  # pylint: disable=no-else-return

            # Flatten all t-batch dimensions into a single one
            flattened = candidates.flatten(end_dim=-3)

            # Call the model on each (flattened) t-batch
            out = (posterior(model, batch) for batch in flattened)

            # Collect the results and restore the batch dimensions
            mean, covar = zip(*out)
            mean = torch.reshape(torch.stack(mean), t_shape + (q_shape,))
            covar = torch.reshape(torch.stack(covar), t_shape + (q_shape, q_shape))

            return mean, covar

        # Otherwise, flatten all t- and q-batches into a single q-batch dimension
        # and evaluate the posterior function in one go
        else:

            # Flatten *all* batches into the q-batch dimension
            flattened = candidates.flatten(end_dim=-2)

            # Call the model on the entire input
            mean, var = posterior(model, flattened)

            # Restore the batch dimensions
            mean = torch.reshape(mean, t_shape + (q_shape,))
            var = torch.reshape(var, t_shape + (q_shape,))

            return mean, var

    return sequential_posterior


@define
class Surrogate(ABC, SerialMixin):
    """Abstract base class for all surrogate models.

    Args:
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool]
    """Class variable encoding whether or not a joint posterior is calculated."""
    supports_transfer_learning: ClassVar[bool]

    # Object variables
    # TODO: In a next refactoring, the user friendliness could be improved by directly
    #   exposing the individual model parameters via the constructor, instead of
    #   expecting them in the form of an unstructured dictionary. This would also
    #   remove the need for the current `_get_model_params_validator` logic.
    model_params: Dict[str, Any] = field(factory=dict)

    def posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate the surrogate model at the given candidate points.

        Args:
            candidates: The candidate points, represented as a tensor of shape
                ```(*t, q, d)```, where ```t``` denotes the "t-batch" shape, ```q```
                denotes the "q-batch" shape, and ```d``` is the input dimension. For
                more details about batch shapes, see: https://botorch.org/docs/batching

        Returns:
            The posterior means and posterior covariance matrices of the t-batched
            candidate points.
        """
        # Prepare the input
        candidates = _prepare_inputs(candidates)

        # Evaluate the posterior distribution
        mean, covar = self._posterior(candidates)

        # Apply covariance transformation for marginal posterior models
        if not self.joint_posterior:
            # Convert to tensor containing covariance matrices
            covar = torch.diag_embed(covar)

        # Add small diagonal variances for numerical stability
        covar.add_(torch.eye(covar.shape[-1]) * _MIN_VARIANCE)

        return mean, covar

    @abstractmethod
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the actual posterior evaluation logic.

        In contrast to its public counterpart
        :func:`baybe.surrogate.Surrogate.posterior`, no data
        validation/transformation is carried out but only the raw posterior computation
        is conducted.

        Note that the public ```posterior``` method *always* returns a full covariance
        matrix. By contrast, this method may return either a covariance matrix or a
        tensor of marginal variances, depending on the models ```joint_posterior```
        flag. The optional conversion to a covariance matrix is handled by the public
        method.

        See :func:`baybe.surrogate.Surrogate.posterior` for details on the
        parameters.

        Args:
            candidates: The candidates.

        Returns:
            See :func:`baybe.surrogate.Surrogate.posterior`.
        """

    def fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """Train the surrogate model on the provided data.

        Args:
            searchspace: The search space in which experiments are conducted.
            train_x: The training data points.
            train_y: The training data labels.

        Raises:
            ValueError: If the search space contains task parameters but the selected
                surrogate model type does not support transfer learning.
            NotImplementedError: When using a continuous search space and a non-GP
                model.
        """
        # Check if transfer learning capabilities are needed
        if (searchspace.n_tasks > 1) and (not self.supports_transfer_learning):
            raise ValueError(
                f"The search space contains task parameters but the selected "
                f"surrogate model type ({self.__class__.__name__}) does not "
                f"support transfer learning."
            )
        # TODO: Adjust scale_model decorator to support other model types as well.
        if (not searchspace.continuous.is_empty) and (
            "GaussianProcess" not in self.__class__.__name__
        ):
            raise NotImplementedError(
                "Continuous search spaces are currently only supported by GPs."
            )

        # Validate and prepare the training data
        train_x = _prepare_inputs(train_x)
        train_y = _prepare_targets(train_y)

        return self._fit(searchspace, train_x, train_y)

    @abstractmethod
    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """Perform the actual fitting logic.

        In contrast to its public counterpart :func:`baybe.surrogate.Surrogate.fit`,
        no data validation/transformation is carried out but only the raw fitting
        operation is conducted.

        See :func:`baybe.surrogate.Surrogate.fit` for details on the parameters.
        """


@define
class GaussianProcessSurrogate(Surrogate):
    """A Gaussian process surrogate model.

    Args:
        _model: The actual model.
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = True
    supports_transfer_learning: ClassVar[bool] = True

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=_get_model_params_validator(SingleTaskGP.__init__),
    )
    _model: Optional[SingleTaskGP] = field(init=False, default=None)

    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.
        posterior = self._model.posterior(candidates)
        return posterior.mvn.mean, posterior.mvn.covariance_matrix

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.

        # identify the indexes of the task and numeric dimensions
        # TODO: generalize to multiple task parameters
        task_idx = searchspace.task_idx
        n_task_params = 1 if task_idx else 0
        numeric_idxs = [i for i in range(train_x.shape[1]) if i != task_idx]

        # get the input bounds from the search space in BoTorch Format
        bounds = searchspace.param_bounds_comp
        # TODO: use target value bounds when explicitly provided

        # define the input and outcome transforms
        # TODO [Scaling]: scaling should be handled by search space object
        input_transform = Normalize(
            train_x.shape[1], bounds=bounds, indices=numeric_idxs
        )
        outcome_transform = Standardize(train_y.shape[1])

        # ---------- GP prior selection ---------- #
        # TODO: temporary prior choices adapted from edbo, replace later on

        mordred = searchspace.contains_mordred or searchspace.contains_rdkit
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

        # define the covariance module for the numeric dimensions
        base_covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1] - n_task_params,
                active_dims=numeric_idxs,
                batch_shape=batch_shape,
                lengthscale_prior=lengthscale_prior[0],
            ),
            batch_shape=batch_shape,
            outputscale_prior=outputscale_prior[0],
        )
        base_covar_module.outputscale = torch.tensor([outputscale_prior[1]])
        base_covar_module.base_kernel.lengthscale = torch.tensor([lengthscale_prior[1]])

        # create GP covariance
        if task_idx is None:
            covar_module = base_covar_module
        else:
            task_covar_module = IndexKernel(
                num_tasks=searchspace.n_tasks,
                active_dims=task_idx,
                rank=searchspace.n_tasks,  # TODO: make controllable
            )
            covar_module = base_covar_module * task_covar_module

        # create GP likelihood
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior[0], batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self._model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        # IMPROVE: The step_limit=100 stems from the former (deprecated)
        #  `fit_gpytorch_torch` function, for which this was the default. Probably,
        #   one should use a smarter logic here.
        fit_gpytorch_mll_torch(mll, step_limit=100)


@define
class MeanPredictionSurrogate(Surrogate):
    """A trivial surrogate model.

    It provides the average value of the training targets
    as posterior mean and a (data-independent) constant posterior variance.

    Args:
        target_value: The value of the posterior mean.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False

    # Object variables
    target_value: Optional[float] = field(init=False, default=None)

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.
        # TODO: use target value bounds for covariance scaling when explicitly provided
        mean = self.target_value * torch.ones([len(candidates)])
        var = torch.ones(len(candidates))
        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self.target_value = train_y.mean().item()


@catch_constant_targets
@scale_model
@define
class RandomForestSurrogate(Surrogate):
    """A random forest surrogate model.

    Args:
        _model: The actual model.
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=_get_model_params_validator(RandomForestRegressor.__init__),
    )
    _model: Optional[RandomForestRegressor] = field(init=False, default=None)

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.

        # Evaluate all trees
        # NOTE: explicit conversion to ndarray is needed due to a pytorch issue:
        # https://github.com/pytorch/pytorch/pull/51731
        # https://github.com/pytorch/pytorch/issues/13918
        predictions = torch.from_numpy(
            np.asarray(
                [
                    self._model.estimators_[tree].predict(candidates)
                    for tree in range(self._model.n_estimators)
                ]
            )
        )

        # Compute posterior mean and variance
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = RandomForestRegressor(**(self.model_params))
        self._model.fit(train_x, train_y.ravel())


@catch_constant_targets
@scale_model
@define
class NGBoostSurrogate(Surrogate):
    """A natural-gradient-boosting surrogate model.

    Args:
        _model: The actual model.
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False
    _default_model_params: ClassVar[dict] = {"n_estimators": 25, "verbose": False}
    """Class variable encoding the default model parameters."""

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=_get_model_params_validator(NGBRegressor.__init__),
    )
    _model: Optional[NGBRegressor] = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.model_params = {**self._default_model_params, **self.model_params}

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class. pylint:disable=missing-function-docstring
        # Get predictions
        dists = self._model.pred_dist(candidates)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists.mean())
        var = torch.from_numpy(dists.var)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class. pylint:disable=missing-function-docstring
        self._model = NGBRegressor(**(self.model_params)).fit(train_x, train_y.ravel())


@catch_constant_targets
@scale_model
@define
class BayesianLinearSurrogate(Surrogate):
    """A Bayesian linear regression surrogate model.

    Args:
        _model: The actual model.
        model_params: Optional model parameters.
    """

    # Class variables
    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=_get_model_params_validator(ARDRegression.__init__),
    )
    _model: Optional[ARDRegression] = field(init=False, default=None)

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        # See base class.
        # Get predictions
        dists = self._model.predict(candidates.numpy(), return_std=True)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists[0])
        var = torch.from_numpy(dists[1]).pow(2)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        # See base class.
        self._model = ARDRegression(**(self.model_params))
        self._model.fit(train_x, train_y.ravel())


@define
class CustomPretrainedSurrogate(Surrogate):
    """A wrapper class for custom pretrained surrogate models"""

    # Class variables
    joint_posterior: ClassVar[bool] = False

    # Object variables
    model_params: Dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=_validate_custom_pretrained_params,
    )

    _model: Optional[ort.InferenceSession] = field(init=False, default=None)

    @batchify
    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See base class."""

        model_inputs = {
            self.model_params["onnx_input_name"]: candidates.numpy().astype(np.float32)
        }

        results = self._model.run(None, model_inputs)

        return torch.from_numpy(results[0]), torch.from_numpy(results[1]).pow(2)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """See base class."""
        self._model = ort.InferenceSession(
            self.model_params["onnx"].encode("ISO-8859-1")
        )


def block_serialize_custom_architecture(raw_unstructure_hook):
    """Raises error if attempt to serialize a custom architecture surrogate."""

    def wrapper(obj):
        if obj.__class__.__name__ == "CustomArchitectureSurrogate":
            raise RuntimeError(
                "Custom Architecture Surrogate Serialization is not supported"
            )

        return raw_unstructure_hook(obj)

    return wrapper


def _remove_model(raw_unstructure_hook):
    """Removes the model in a surrogate for serialization."""
    # TODO: No longer required once the following feature is released:
    #   https://github.com/python-attrs/cattrs/issues/40

    def wrapper(obj):
        dict_ = raw_unstructure_hook(obj)
        dict_.pop("_model", None)
        dict_.pop("target_value", None)
        return dict_

    return wrapper


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Temporary workaround >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _structure_surrogate(val, _):
    """Structures a surrogate model."""
    # TODO [15436]
    # https://***REMOVED***/_boards/board/t/SDK%20Devs/Features/?workitem=15436

    # NOTE:
    # Due to above issue,
    # it is difficult to find the wrapped class in the subclass structure.
    # The renaming only happens in the init method of wrapper classes
    # (classes that haven't been initialized won't have the overwritten name)
    # Since any method revolving `cls()` will not work as expected,
    # we rely temporarily on `getattr` to allow the wrappers to be called on demand.

    _type = val["type"]

    cls = getattr(sys.modules[__name__], _type, None)
    # cls = getattr(baybe.surrogate, ...) if used in another module

    if cls is None:
        raise ValueError(f"Unknown subclass {_type}.")

    return cattrs.structure_attrs_fromdict(val, cls)


def get_available_surrogates() -> List[Type[Surrogate]]:
    """Lists all available surrogate models."""
    # List available names
    available_names = {
        cl.__name__
        for cl in get_subclasses(Surrogate)
        if cl.__name__ not in _WRAPPER_MODELS
    }

    # Convert them to classes
    available_classes = [
        getattr(sys.modules[__name__], mdl_name, None) for mdl_name in available_names
    ]

    # TODO: The initialization of the classes is currently necessary for the renaming
    #  to take place (see [15436] and NOTE in `structure_surrogate`).
    [  # pylint: disable=expression-not-assigned
        cl() for cl in available_classes if cl is not None
    ]

    return [cl for cl in available_classes if cl is not None]


# Register (un-)structure hooks
cattrs.register_unstructure_hook(
    Surrogate, _remove_model(block_serialize_custom_architecture(unstructure_base))
)
cattrs.register_structure_hook(Surrogate, _structure_surrogate)

# Related to [15436]
gc.collect()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Temporary workaround <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
