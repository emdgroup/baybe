# pylint: disable=not-callable, no-member  # TODO: due to validators --> find fix
"""
Strategies for Design of Experiments (DOE).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from numpy import unique
from numpy.typing import ArrayLike
from pydantic import BaseModel, Extra, validator
from scipy.stats import multivariate_normal

# model imports
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

from .acquisition import debotorchize
from .recommender import Recommender
from .surrogate import SurrogateModel
from .utils import check_if_in, isabstract, to_tensor
from .utils.sampling_algorithms import farthest_point_sampling


class InitialStrategy(ABC):
    """
    Abstract base class for all initial design strategies. They are used for selecting
    initial sets of candidate experiments, i.e. without considering experimental data.
    """

    type: str
    SUBCLASSES: Dict[str, Type[InitialStrategy]] = {}

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Selects a first subset of points from the given candidates.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected.
        """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls


class RandomInitialStrategy(InitialStrategy):
    """An initial strategy that selects candidates uniformly at random."""

    type = "RANDOM"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """See base class."""
        return pd.Index(
            np.random.choice(candidates.index, batch_quantity, replace=False)
        )


class BasicClusteringInitialStrategy(InitialStrategy, ABC):
    """Intermediate class for cluster-based initial selection of candidates. Suitable
    for sklearn-like models that have a fit and predict method. Specific model
    parameters and cluster sub-selection techniques can be declared in the derived
    classes."""

    # Properties that need to be defined by derived classes
    model_class = None
    model_params = None
    model_cluster_num_parameter_name = None

    # Properties that are only relevant to this class
    model = None
    candidates_scaled = None

    def _make_selection(self) -> List:
        """
        Internal method selects one candidate from each cluster at random. Derived
        classes are free to override this and implement different methods where
        suitable.

        Returns
        -------
        selection : numpy array
           Positional indices of the selected candidates
        """

        assigned_clusters = self.model.predict(self.candidates_scaled)
        selection = [
            np.random.choice(np.argwhere(cluster == assigned_clusters).flatten())
            for cluster in unique(assigned_clusters)
        ]
        return selection

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Initial candidate recommendation based on clustering.

        Parameters
        ----------
        candidates : pandas data frame
            A data frame describing the candidates in computational representation.
        batch_quantity : int
            The number of candidates that should be selected.

        Returns
        -------
        pandas index
            Index (actual, not positional) of selected entries in candidates.
        """

        # Scale candidates. A contiguous array is needed for some methods.
        scaler = StandardScaler()
        self.candidates_scaled = np.ascontiguousarray(scaler.fit_transform(candidates))

        # Set set model parameters and perform fit.
        self.model_params.update(
            {self.model_cluster_num_parameter_name: batch_quantity}
        )
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.candidates_scaled)

        # Perform algorithm-specific selection based on assigned clusters
        selection = self._make_selection()

        # Convert positional indices into absolute indices and return result
        return candidates.index[selection]


class PAMInitialStrategy(BasicClusteringInitialStrategy):
    """Partitioning around Medoids (PAM) initial clustering strategy"""

    type = "PAM"
    model_class = KMedoids
    model_params = {"max_iter": 1000, "metric": "euclidean"}
    model_cluster_num_parameter_name = "n_clusters"

    def _make_selection(self) -> ArrayLike:
        """see base class"""

        # In PAM cluster centers (medoids) correspond to actual data points
        selection = self.model.medoid_indices_.tolist()
        return selection


class KMeansInitialStrategy(BasicClusteringInitialStrategy):
    """K-means initial clustering strategy"""

    type = "KMEANS"
    model_class = KMeans
    model_params = {"n_init": 50, "max_iter": 1000}
    model_cluster_num_parameter_name = "n_clusters"

    def _make_selection(self) -> List:
        """see base class"""

        # In KMeans we pick the points closest to each cluster center
        distances = pairwise_distances(
            self.candidates_scaled, self.model.cluster_centers_
        )
        selection = np.unique(np.argmin(distances, axis=0)).tolist()

        return selection


class GaussianMixtureInitialStrategy(BasicClusteringInitialStrategy):
    """Gaussian mixture model (GMM) initial clustering strategy"""

    type = "GMM"
    model_class = GaussianMixture
    model_params = {}
    model_cluster_num_parameter_name = "n_components"

    def _make_selection(self) -> List:
        """see base class"""

        # In GMM the most central point is the one with the highest density
        selection = []
        for k in range(self.model.n_components):
            density = multivariate_normal(
                cov=self.model.covariances_[k], mean=self.model.means_[k]
            ).logpdf(self.candidates_scaled)
            selection.append(np.argmax(density))

        return selection


class FPSInitialStrategy(InitialStrategy):
    """An initial strategy that selects the candidates via Farthest Point Sampling."""

    type = "FPS"

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """See base class."""
        ilocs = farthest_point_sampling(candidates.values, batch_quantity)
        return candidates.index[ilocs]


class Strategy(BaseModel, extra=Extra.forbid, arbitrary_types_allowed=True):
    """Abstract base class for all DOE strategies."""

    # TODO: consider adding validators for the individual component classes of the
    #  strategy or introducing config classes for them (-> disable arbitrary types)

    # object variables
    searchspace: pd.DataFrame
    surrogate_model_cls: Union[str, Type[SurrogateModel]] = "GP"
    acquisition_function_cls: Union[
        Literal["PM", "PI", "EI", "UCB"], Type[AcquisitionFunction]
    ] = "EI"
    initial_strategy: Union[str, InitialStrategy] = "RANDOM"
    recommender_cls: Union[str, Type[Recommender]] = "UNRESTRICTED_RANKING"

    # TODO: The following member declarations become obsolete in pydantic 2.0 when
    #  __post_init_post_parse__ is available:
    #   - https://github.com/samuelcolvin/pydantic/issues/691
    #   - https://github.com/samuelcolvin/pydantic/issues/1729
    surrogate_model: Optional[SurrogateModel] = None
    best_f: Optional[float] = None
    use_initial_strategy: bool = True

    # TODO: introduce a reusable validator once they all perform the same operation

    @validator("surrogate_model_cls", always=True)
    def validate_surrogate_model(cls, model):
        """Validates if the given surrogate model type exists."""
        if isinstance(model, str):
            check_if_in(model, list(SurrogateModel.SUBCLASSES.keys()))
            return SurrogateModel.SUBCLASSES[model]
        return model

    @validator("acquisition_function_cls", always=True)
    def validate_acquisition_function(cls, fun):
        """Validates if the given acquisition function type exists."""
        if isinstance(fun, str):
            mapping = {
                "PM": PosteriorMean,
                "PI": ProbabilityOfImprovement,
                "EI": ExpectedImprovement,
                "UCB": partial(UpperConfidenceBound, beta=1.0),
            }
            fun = debotorchize(mapping[fun])
        return fun

    @validator("initial_strategy", always=True)
    def validate_initial_strategy(cls, init_strategy):
        """Validates if the given initial strategy type exists."""
        if isinstance(init_strategy, str):
            check_if_in(init_strategy, list(InitialStrategy.SUBCLASSES.keys()))
            return InitialStrategy.SUBCLASSES[init_strategy]()
        return init_strategy

    @validator("recommender_cls", always=True)
    def validate_recommender(cls, recommender):
        """Validates if the given recommender model type exists."""
        if isinstance(recommender, str):
            check_if_in(recommender, list(Recommender.SUBCLASSES.keys()))
            return Recommender.SUBCLASSES[recommender]
        return recommender

    def fit(self, train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
        """
        Uses the given data to train a fresh surrogate model instance for the DOE
        strategy. If available, previous training data will be overwritten.

        Parameters
        ----------
        train_x : pd.DataFrame
            The features of the conducted experiments.
        train_y : pd.DataFrame
            The corresponding response values.
        """
        # validate input
        if not train_x.index.equals(train_y.index):
            raise ValueError("Training inputs and targets must have the same index.")

        # if no data is provided, apply the initial selection strategy
        self.use_initial_strategy = len(train_x) == 0

        # if data is provided (and the strategy is not random), train the surrogate
        if (not self.use_initial_strategy) and (self.recommender_cls.type != "RANDOM"):
            self.surrogate_model = self.surrogate_model_cls(self.searchspace)
            self.surrogate_model.fit(*to_tensor(train_x, train_y))
            self.best_f = train_y.max()

    def recommend(self, candidates: pd.DataFrame, batch_quantity: int = 1) -> pd.Index:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        candidates : pd.DataFrame
            The features of all candidate experiments that could be conducted next.
        batch_quantity : int (default = 1)
            The number of experiments to be conducted in parallel.

        Returns
        -------
        The DataFrame indices of the specific experiments selected.
        """
        # if no training data exists, apply the strategy for initial recommendations
        if self.use_initial_strategy:
            return self.initial_strategy.recommend(candidates, batch_quantity)

        # construct the acquisition function
        acqf = (
            self.acquisition_function_cls(self.surrogate_model, self.best_f)
            if self.recommender_cls.type != "RANDOM"
            else None
        )

        # select the next experiments using the given recommender approach
        recommender = self.recommender_cls(acqf)
        idxs = recommender.recommend(candidates, batch_quantity)

        return idxs
