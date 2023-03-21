# pylint: disable=too-few-public-methods
"""
Recommender classes for optimizing acquisition functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, TypeVar

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

from .searchspace import SearchSpace
from .utils import (
    IncompatibleSearchSpaceError,
    isabstract,
    NoMCAcquisitionFunctionError,
    NotEnoughPointsLeftError,
    to_tensor,
)
from .utils.sampling_algorithms import farthest_point_sampling

SklearnModel = TypeVar("SklearnModel")


class Recommender(ABC):
    """
    Abstract base class for all recommenders.

    The job of a recommender is to select (i.e. "recommend") a subset of candidate
    experiments based on an underlying (batch) acquisition criterion.
    """

    type: str
    SUBCLASSES: Dict[str, Type[Recommender]] = {}

    # it is generally assumed that a recommender is model-based and needs data. If this
    # is not the case the derived class should override is_model_free
    is_model_free: bool = False
    compatible_discrete: bool = False
    compatible_continuous: bool = False

    def __init__(
        self,
        searchspace: SearchSpace,
        acquisition_function: Optional[AcquisitionFunction],
    ):
        self.searchspace = searchspace
        self.acquisition_function = acquisition_function
        self.check_searchspace_compatibility(self.searchspace)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Registers new subclasses dynamically."""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            cls.SUBCLASSES[cls.type] = cls

    def recommend(
        self,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """
        Recommends the next experiments to be conducted.

        Parameters
        ----------
        batch_quantity : int
            The number of experiments to be conducted in parallel.
        allow_repeated_recommendations : bool
            Whether points whose discrete parts were already recommended can be
            recommended again.
        allow_recommending_already_measured : bool
            Whether points whose discrete parts were already measured can be
            recommended again.

        Returns
        -------
        The DataFrame with the specific experiments recommended.
        """
        # TODO[11179]: Potentially move call to get_candidates from _recommend to here
        # TODO[11179]: Potentially move metadata update from _recommend to here

        # Validate the search space
        self.check_searchspace_compatibility(self.searchspace)

        return self._recommend(
            batch_quantity,
            allow_repeated_recommendations,
            allow_recommending_already_measured,
        )

    @abstractmethod
    def _recommend(
        self,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """
        Implements the actual recommendation logic. In contrast to its public
        counterpart, no validation or post-processing is carried out but only the raw
        recommendation computation is conducted.
        """

    def check_searchspace_compatibility(self, searchspace: SearchSpace) -> None:
        """
        Performs a compatibility check between the recommender type and the provided
        search space.

        Parameters
        ----------
        searchspace : SearchSpace
            The search space to check for compatibility.

        Raises
        ------
        IncompatibleSearchSpaceError
            In case the recommender is not compatible with the specified search space.
        """
        if (not self.compatible_discrete) and (not searchspace.discrete.empty):
            raise IncompatibleSearchSpaceError(
                f"The search space that was passed contained a discrete part, but the "
                f"chosen recommender of type {self.type} does not support discrete "
                f"search spaces."
            )

        if (not self.compatible_continuous) and (not searchspace.continuous.empty):
            raise IncompatibleSearchSpaceError(
                f"The search space that was passed contained a continuous part, but the"
                f" chosen recommender of type {self.type} does not support continuous "
                f"search spaces."
            )


class AbstractDiscreteRecommender(Recommender, ABC):
    """
    Abstract class for discrete recommenders
    """

    type = "ABSTRACT_DISCRETE_RECOMMENDER"
    compatible_discrete: bool = True

    @abstractmethod
    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """
        Returns indices of recommended candidates from a discrete search space

        Parameters
        ----------
        candidates_comp : pd.DataFrame
            valid candidates from the discrete search space
        batch_quantity : int
            number of requested recommendations

        Returns
        -------
        pd.Index
        """

    def _recommend(
        self,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""

        # Get discrete candidates. The metadata flags are ignored if a continuous space
        # is not empty
        _, candidates_comp = self.searchspace.discrete.get_candidates(
            allow_repeated_recommendations=allow_repeated_recommendations
            if self.searchspace.continuous.empty
            else True,
            allow_recommending_already_measured=allow_recommending_already_measured
            if self.searchspace.continuous.empty
            else True,
        )

        # Check if enough candidates are left
        if len(candidates_comp) < batch_quantity:
            raise NotEnoughPointsLeftError(
                f"Using the current settings, there are fewer than {batch_quantity} "
                "possible data points left to recommend. This can be "
                "either because all data points have been measured at some point "
                "(while 'allow_repeated_recommendations' or "
                "'allow_recommending_already_measured' being False) "
                "or because all data points are marked as 'dont_recommend'."
            )

        # Get recommendations
        idxs = self._recommend_discrete(candidates_comp, batch_quantity)
        rec = self.searchspace.discrete.exp_rep.loc[idxs, :]

        # Update Metadata
        self.searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        # Return recommendations
        return rec


class SequentialGreedyRecommender(AbstractDiscreteRecommender):
    """
    Recommends the next set of experiments by means of sequential greedy optimization,
    i.e. using a growing set of candidates points, where each new candidate is
    selected by optimizing the joint acquisition score of the current candidate set
    while having fixed all previous candidates.

    Note: This approach only works with Monte Carlo acquisition functions, i.e. of
        type `botorch.acquisition.monte_carlo.MCAcquisitionFunction`.
    """

    # TODO: generalize the approach to also support continuous spaces

    type = "SEQUENTIAL_GREEDY"

    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """See base class."""
        # determine the next set of points to be tested
        candidates_tensor = to_tensor(candidates_comp)
        try:
            points, _ = optimize_acqf_discrete(
                self.acquisition_function, batch_quantity, candidates_tensor
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        # retrieve the index of the points from the input dataframe
        # IMPROVE: The merging procedure is conceptually similar to what
        #   `SearchSpace._match_measurement_with_searchspace_indices` does, though using
        #   a simpler matching logic. When refactoring the SearchSpace class to
        #   handle continuous parameters, a corresponding utility could be extracted.
        idxs = pd.Index(
            pd.merge(
                candidates_comp.reset_index(),
                pd.DataFrame(points, columns=candidates_comp.columns),
                on=list(candidates_comp),
            )["index"]
        )
        assert len(points) == len(idxs)

        return idxs


class MarginalRankingRecommender(AbstractDiscreteRecommender):
    """
    Recommends the top experiments from the ranking obtained by evaluating the
    acquisition function on the marginal posterior predictive distribution of each
    candidate, i.e. by computing the score for each candidate individually in a
    non-batch fashion.
    """

    type = "UNRESTRICTED_RANKING"

    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """See base class."""
        # prepare the candidates in t-batches (= parallel marginal evaluation)
        candidates_tensor = to_tensor(candidates_comp).unsqueeze(1)

        # evaluate the acquisition function for each t-batch and construct the ranking
        acqf_values = self.acquisition_function(candidates_tensor)
        ilocs = torch.argsort(acqf_values, descending=True)

        # return top ranked candidates
        idxs = candidates_comp.index[ilocs[:batch_quantity].numpy()]

        return idxs


class SKLearnClusteringRecommender(AbstractDiscreteRecommender, ABC):
    """
    Intermediate class for cluster-based selection of discrete candidates.

    Suitable for sklearn-like models that have a `fit` and `predict` method. Specific
    model parameters and cluster sub-selection techniques can be declared in the
    derived classes.
    """

    type = "ABSTRACT_SKLEARN_CLUSTERING"
    is_model_free: bool = True

    # Properties that need to be defined by derived classes
    model_class: Type[SklearnModel]
    model_cluster_num_parameter_name: str

    def __init__(self, **kwargs):
        super().__init__(
            searchspace=kwargs.pop("searchspace"),
            acquisition_function=kwargs.pop("acquisition_function"),
        )
        self.model_params = kwargs
        self._use_custom_selector = False

        # Members that will be initialized during the recommendation process
        self.model: Optional[SklearnModel] = None
        self.candidates_scaled: Optional[pd.DataFrame] = None

        # Fit scaler on entire searchspace
        # TODO [Scaling]: scaling should be handled by searchspace object
        self.scaler = StandardScaler()
        self.scaler.fit(self.searchspace.discrete.comp_rep)

    def _make_selection_default(self) -> List[int]:
        """
        Basic model-agnostic method that selects one candidate from each cluster
        uniformly at random.

        Returns
        -------
        selection : List[int]
           Positional indices of the selected candidates.
        """
        assigned_clusters = self.model.predict(self.candidates_scaled)
        selection = [
            np.random.choice(np.argwhere(cluster == assigned_clusters).flatten())
            for cluster in np.unique(assigned_clusters)
        ]
        return selection

    def _make_selection_custom(self) -> List[int]:
        """
        A model-specific method to select candidates from the computed clustering.
        May be implemented by the derived class.
        """
        raise NotImplementedError("This line in the code should be unreachable. Sry.")

    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """See base class."""
        self.candidates_scaled = np.ascontiguousarray(
            self.scaler.transform(candidates_comp)
        )

        # Set model parameters and perform fit
        self.model_params.update(
            {self.model_cluster_num_parameter_name: batch_quantity}
        )
        self.model = self.model_class(**self.model_params)
        self.model.fit(self.candidates_scaled)

        # Perform selection based on assigned clusters
        if self._use_custom_selector:
            selection = self._make_selection_custom()
        else:
            selection = self._make_selection_default()

        # Convert positional indices into DataFrame indices and return result
        return candidates_comp.index[selection]


if KMedoids:

    class PAMClusteringRecommender(SKLearnClusteringRecommender):
        """Partitioning Around Medoids (PAM) initial clustering strategy."""

        type = "CLUSTERING_PAM"
        model_class = KMedoids
        model_cluster_num_parameter_name = "n_clusters"

        def __init__(
            self, use_custom_selector: bool = True, max_iter: int = 100, **kwargs
        ):
            super().__init__(max_iter=max_iter, init="k-medoids++", **kwargs)
            self._use_custom_selector = use_custom_selector

        def _make_selection_custom(self) -> List[int]:
            """
            In PAM, cluster centers (medoids) correspond to actual data points,
            which means they can be directly used for the selection.
            """
            selection = self.model.medoid_indices_.tolist()
            return selection


class KMeansClusteringRecommender(SKLearnClusteringRecommender):
    """K-means initial clustering strategy."""

    type = "CLUSTERING_KMEANS"
    model_class = KMeans
    model_cluster_num_parameter_name = "n_clusters"

    def __init__(
        self,
        use_custom_selector: bool = True,
        n_init: int = 50,
        max_iter: int = 1000,
        **kwargs,
    ):
        super().__init__(n_init=n_init, max_iter=max_iter, **kwargs)
        self._use_custom_selector = use_custom_selector

    def _make_selection_custom(self) -> List[int]:
        """
        For K-means, a reasonable choice is to pick the points closest to each
        cluster center.
        """
        distances = pairwise_distances(
            self.candidates_scaled, self.model.cluster_centers_
        )
        # Set the distances of points that were not assigned by the model to that
        # cluster to infinity. This assures that one unique point per cluster is
        # assigned.
        predicted_clusters = self.model.predict(self.candidates_scaled)
        for k_cluster in range(self.model.cluster_centers_.shape[0]):
            idxs = predicted_clusters != k_cluster
            distances[idxs, k_cluster] = np.inf
        selection = np.argmin(distances, axis=0).tolist()
        return selection


class GaussianMixtureClusteringRecommender(SKLearnClusteringRecommender):
    """Gaussian mixture model (GMM) initial clustering strategy."""

    type = "CLUSTERING_GMM"
    model_class = GaussianMixture
    model_cluster_num_parameter_name = "n_components"

    def __init__(self, use_custom_selector: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._use_custom_selector = use_custom_selector

    def _make_selection_custom(self) -> List[int]:
        """
        In a GMM, a reasonable choice is to pick the point with the highest
        probability densities for each cluster.
        """
        predicted_clusters = self.model.predict(self.candidates_scaled)
        selection = []
        for k_cluster in range(self.model.n_components):
            density = multivariate_normal(
                cov=self.model.covariances_[k_cluster],
                mean=self.model.means_[k_cluster],
            ).logpdf(self.candidates_scaled)

            # For selecting a point from this cluster we only consider points that were
            # assigned to the current cluster by the model, hence set the density of
            # others to 0
            density[predicted_clusters != k_cluster] = 0.0

            selection.append(np.argmax(density).item())
        return selection


class FPSRecommender(AbstractDiscreteRecommender):
    """An initial strategy that selects the candidates via Farthest Point Sampling."""

    type = "FPS"
    is_model_free: bool = True

    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """See base class."""
        ilocs = farthest_point_sampling(candidates_comp.values, batch_quantity)
        return candidates_comp.index[ilocs]


class AbstractContinuousRecommender(Recommender, ABC):
    """
    Abstract class for continuous recommenders
    """

    type = "ABSTRACT_CONTINUOUS_RECOMMENDER"
    compatible_continuous: bool = True

    @abstractmethod
    def _recommend_continuous(self, batch_quantity: int) -> pd.DataFrame:
        """
        Recommends candidates from a continuous space.

        Parameters
        ----------
        batch_quantity : int
            number of requested recommendations

        Returns
        -------
        pd.DataFrame
        """

    def _recommend(
        self,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""
        return self._recommend_continuous(batch_quantity)


# TODO finalize class and type name below
class PurelyContinuousRecommender(AbstractContinuousRecommender):
    """
    Recommends the next set of experiments by means of sequential greedy optimization
    in a purely continuous space.
    """

    type = "CONTI"

    def _recommend_continuous(self, batch_quantity: int) -> pd.DataFrame:
        """See base class."""

        try:
            points, _ = optimize_acqf(
                acq_function=self.acquisition_function,
                bounds=self.searchspace.param_bounds_comp,
                q=batch_quantity,
                num_restarts=5,  # TODO make choice for num_restarts
                raw_samples=10,  # TODO make choice for raw_samples
            )
        except AttributeError as ex:
            raise NoMCAcquisitionFunctionError(
                f"The '{self.__class__.__name__}' only works with Monte Carlo "
                f"acquisition functions."
            ) from ex

        rec = pd.DataFrame(points, columns=self.searchspace.continuous.param_names)

        return rec


class AbstractCompositeRecommender(Recommender, ABC):
    """
    Abstract composite recommender
    """

    type = "ABSTRACT_COMPOSITE"
    compatible_discrete: bool = True
    compatible_continuous: bool = True

    @abstractmethod
    def _recommend_continuous(self, batch_quantity: int) -> pd.DataFrame:
        """See AbstractContinuousRecommender._recommend_continuous"""

    @abstractmethod
    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """See AbstractDiscreteRecommender._recommend_discrete"""

    def _recommend(
        self,
        batch_quantity: int = 1,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = True,
    ) -> pd.DataFrame:
        """See base class."""

        # Discrete part if applicable
        rec_disc = pd.DataFrame()
        if not self.searchspace.discrete.empty:
            _, candidates_comp = self.searchspace.discrete.get_candidates(
                allow_repeated_recommendations,
                allow_recommending_already_measured,
            )

            # randomly select from discrete candidates
            idxs = self._recommend_discrete(candidates_comp, batch_quantity)
            rec_disc = self.searchspace.discrete.exp_rep.loc[idxs, :]
            self.searchspace.discrete.metadata.loc[idxs, "was_recommended"] = True

        # Continuous part if applicable
        rec_conti = pd.DataFrame()
        if not self.searchspace.continuous.empty:
            rec_conti = self._recommend_continuous(batch_quantity)

        # If both spaces are present assure matching indices. Since the discrete indices
        # have meaning we choose them
        if not (self.searchspace.discrete.empty or self.searchspace.continuous.empty):
            rec_conti.index = rec_disc.index

        # Merge sub-parts and reorder columns to match original order
        rec = pd.concat([rec_disc, rec_conti], axis=1).reindex(
            columns=[p.name for p in self.searchspace.parameters]
        )

        return rec


class RandomRecommender(AbstractCompositeRecommender):
    """
    Recommends experiments randomly.
    """

    type = "RANDOM"
    is_model_free: bool = True

    def _recommend_continuous(self, batch_quantity: int) -> pd.DataFrame:
        """See base class."""
        return self.searchspace.continuous.samples_random(n_points=batch_quantity)

    def _recommend_discrete(
        self, candidates_comp: pd.DataFrame, batch_quantity: int
    ) -> pd.Index:
        """See base class."""
        return candidates_comp.sample(n=batch_quantity).index
