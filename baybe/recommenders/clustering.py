"""Recommendation strategies based on clustering."""

from abc import ABC
from typing import ClassVar, List, Type, TypeVar

import numpy as np
import pandas as pd
from attrs import define, field
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

from baybe.recommenders.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SearchSpaceType

_ScikitLearnModel = TypeVar("_ScikitLearnModel")


@define
class SKLearnClusteringRecommender(NonPredictiveRecommender, ABC):
    """Intermediate class for cluster-based selection of discrete candidates.

    Suitable for ``sklearn``-like models that have a ``fit`` and ``predict``
    method. Specific model parameters and cluster sub-selection techniques can be
    declared in the derived classes.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # See base class.
    # TODO: "Type" should not appear in ClassVar. Both PyCharm and mypy complain, see
    #   also note in the mypy docs:
    #       https://peps.python.org/pep-0526/#class-and-instance-variable-annotations
    #   Figure out what is the right approach here. However, the issue might be
    #   ultimately related to an overly restrictive PEP:
    #       https://github.com/python/mypy/issues/5144
    # TODO: `use_custom_selector` can probably be replaced with a fallback mechanism
    #   that checks if a custom mechanism is implemented and uses default otherwise
    #   (similar to what is done in the recommenders)

    model_class: ClassVar[Type[_ScikitLearnModel]]
    """Class variable describing the model class."""

    model_cluster_num_parameter_name: ClassVar[str]
    """Class variable describing the name of the clustering parameter."""

    _use_custom_selector: ClassVar[bool] = False
    """Class variable flagging whether a custom selector is being used."""

    # Object variables
    model_params: dict = field(factory=dict)
    """The parameters for the used model. This is initialized with reasonable default
    values for the derived child classes."""

    def _make_selection_default(
        self,
        model: _ScikitLearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """Select one candidate from each cluster uniformly at random.

        This function is model-agnostic and can be used by any child class.

        Args:
            model: The used model.
            candidates_scaled: The already scaled candidates.

        Returns:
           A list with positional indices of the selected candidates.
        """
        assigned_clusters = model.predict(candidates_scaled)
        selection = [
            np.random.choice(np.argwhere(cluster == assigned_clusters).flatten())
            for cluster in np.unique(assigned_clusters)
        ]
        return selection

    def _make_selection_custom(
        self,
        model: _ScikitLearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """Select candidates from the computed clustering.

        This function is model-specific and may be implemented by the derived class.

        Args:
            model: The used model.
            candidates_scaled: The already scaled candidates.

        Returns:
           A list with positional indices of the selected candidates.

        Raises:
            NotImplementedError: If this function is not implemented. Should be
                unreachable.
        """
        raise NotImplementedError("This line in the code should be unreachable. Sry.")

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        # See base class.

        # Fit scaler on entire search space
        # TODO [Scaling]: scaling should be handled by search space object
        scaler = StandardScaler()
        scaler.fit(searchspace.discrete.comp_rep)

        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))

        # Set model parameters and perform fit
        model = self.model_class(
            **{self.model_cluster_num_parameter_name: batch_quantity},
            **self.model_params,
        )
        model.fit(candidates_scaled)

        # Perform selection based on assigned clusters
        if self._use_custom_selector:
            selection = self._make_selection_custom(model, candidates_scaled)
        else:
            selection = self._make_selection_default(model, candidates_scaled)

        # Convert positional indices into DataFrame indices and return result
        return candidates_comp.index[selection]


@define
class PAMClusteringRecommender(SKLearnClusteringRecommender):
    """Partitioning Around Medoids (PAM) initial clustering strategy."""

    model_class: ClassVar[Type[_ScikitLearnModel]] = KMedoids
    # See base class.

    model_cluster_num_parameter_name: ClassVar[str] = "n_clusters"
    # See base class.

    _use_custom_selector: ClassVar[bool] = True
    # See base class.

    # Object variables
    model_params: dict = field()
    # See base class.

    @model_params.default
    def _default_model_params(self) -> dict:
        """Create the default model parameters."""
        return {"max_iter": 100, "init": "k-medoids++"}

    def _make_selection_custom(
        self,
        model: _ScikitLearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """Select candidates from the computed clustering.

        In PAM, cluster centers (medoids) correspond to actual data points,
        which means they can be directly used for the selection.

        Args:
            model: The used model.
            candidates_scaled: The already scaled candidates. Unused.

        Returns:
           A list with positional indices of the selected candidates.
        """
        selection = model.medoid_indices_.tolist()
        return selection


@define
class KMeansClusteringRecommender(SKLearnClusteringRecommender):
    """K-means initial clustering strategy."""

    # Class variables
    model_class: ClassVar[Type[_ScikitLearnModel]] = KMeans
    # See base class.

    model_cluster_num_parameter_name: ClassVar[str] = "n_clusters"
    # See base class.

    _use_custom_selector: ClassVar[bool] = True
    # See base class.

    # Object variables
    model_params: dict = field()
    # See base class.

    @model_params.default
    def _default_model_params(self) -> dict:
        """Create the default model parameters."""
        return {"max_iter": 1000, "n_init": 50}

    def _make_selection_custom(
        self,
        model: _ScikitLearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """Select candidates from the computed clustering.

        For K-means, a reasonable choice is to pick the points closest to each
        cluster center.

        Args:
            model: The used model.
            candidates_scaled: The already scaled candidates.

        Returns:
           A list with positional indices of the selected candidates.
        """
        distances = pairwise_distances(candidates_scaled, model.cluster_centers_)
        # Set the distances of points that were not assigned by the model to that
        # cluster to infinity. This assures that one unique point per cluster is
        # assigned.
        predicted_clusters = model.predict(candidates_scaled)
        for k_cluster in range(model.cluster_centers_.shape[0]):
            idxs = predicted_clusters != k_cluster
            distances[idxs, k_cluster] = np.inf
        selection = np.argmin(distances, axis=0).tolist()
        return selection


@define
class GaussianMixtureClusteringRecommender(SKLearnClusteringRecommender):
    """Gaussian mixture model (GMM) initial clustering strategy."""

    # Class variables
    model_class: ClassVar[Type[_ScikitLearnModel]] = GaussianMixture
    # See base class.

    model_cluster_num_parameter_name: ClassVar[str] = "n_components"
    # See base class.

    def _make_selection_custom(
        self,
        model: _ScikitLearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """Select candidates from the computed clustering.

        In a GMM, a reasonable choice is to pick the point with the highest
        probability densities for each cluster.

        Args:
            model: The used model.
            candidates_scaled: The already scaled candidates.

        Returns:
           A list with positional indices of the selected candidates.
        """
        predicted_clusters = model.predict(candidates_scaled)
        selection = []
        for k_cluster in range(model.n_components):
            density = multivariate_normal(
                cov=model.covariances_[k_cluster],
                mean=model.means_[k_cluster],
            ).logpdf(candidates_scaled)

            # For selecting a point from this cluster we only consider points that were
            # assigned to the current cluster by the model, hence set the density of
            # others to 0
            density[predicted_clusters != k_cluster] = 0.0

            selection.append(np.argmax(density).item())
        return selection
