# pylint: disable=missing-class-docstring, missing-function-docstring
# TODO: add docstrings

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

from baybe.searchspace import SearchSpace, SearchSpaceType
from baybe.strategies.recommender import NonPredictiveRecommender

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None


SklearnModel = TypeVar("SklearnModel")


@define
class SKLearnClusteringRecommender(NonPredictiveRecommender, ABC):
    """
    Intermediate class for cluster-based selection of discrete candidates.

    Suitable for sklearn-like models that have a `fit` and `predict` method. Specific
    model parameters and cluster sub-selection techniques can be declared in the
    derived classes.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.DISCRETE
    # TODO: "Type" should not appear in ClassVar. Both PyCharm and mypy complain, see
    #   also note in the mypy docs:
    #       https://peps.python.org/pep-0526/#class-and-instance-variable-annotations
    #   Figure out what is the right approach here. However, the issue might be
    #   ultimately related to an overly restrictive PEP:
    #       https://github.com/python/mypy/issues/5144
    model_class: ClassVar[Type[SklearnModel]]
    model_cluster_num_parameter_name: ClassVar[str]

    # Object variables
    # TODO: `use_custom_selector` can probably be replaced with a fallback mechanism,
    #   similar to what is done in the recommenders
    model_params: dict = field(factory=dict)
    _use_custom_selector: bool = field(default=False)

    def _make_selection_default(
        self,
        model: SklearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """
        Basic model-agnostic method that selects one candidate from each cluster
        uniformly at random.

        Returns
        -------
        selection : List[int]
           Positional indices of the selected candidates.
        """
        assigned_clusters = model.predict(candidates_scaled)
        selection = [
            np.random.choice(np.argwhere(cluster == assigned_clusters).flatten())
            for cluster in np.unique(assigned_clusters)
        ]
        return selection

    def _make_selection_custom(
        self,
        model: SklearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """
        A model-specific method to select candidates from the computed clustering.
        May be implemented by the derived class.
        """
        raise NotImplementedError("This line in the code should be unreachable. Sry.")

    def _recommend_discrete(
        self,
        searchspace: SearchSpace,
        candidates_comp: pd.DataFrame,
        batch_quantity: int,
    ) -> pd.Index:
        """See base class."""
        # Fit scaler on entire searchspace
        # TODO [Scaling]: scaling should be handled by searchspace object
        scaler = StandardScaler()
        scaler.fit(searchspace.discrete.comp_rep)

        candidates_scaled = np.ascontiguousarray(scaler.transform(candidates_comp))

        # Set model parameters and perform fit
        model = self.model_class(
            **{self.model_cluster_num_parameter_name: batch_quantity},
            **self.model_params
        )
        model.fit(candidates_scaled)

        # Perform selection based on assigned clusters
        if self._use_custom_selector:
            selection = self._make_selection_custom(model, candidates_scaled)
        else:
            selection = self._make_selection_default(model, candidates_scaled)

        # Convert positional indices into DataFrame indices and return result
        return candidates_comp.index[selection]


if KMedoids:
    # TODO: Instead of hiding the class, raise an error when attempting to create the
    #   object. However, this requires to replace the current class-based handling of
    #   recommenders with an object-based logic, since otherwise the exception can be
    #   triggered arbitrarily late in the DOE process.

    @define
    class PAMClusteringRecommender(SKLearnClusteringRecommender):
        """Partitioning Around Medoids (PAM) initial clustering strategy."""

        # Class variables
        model_class: ClassVar[Type[SklearnModel]] = KMedoids
        model_cluster_num_parameter_name: ClassVar[str] = "n_clusters"

        # Object variables
        _use_custom_selector = field(default=True)
        model_params: dict = field()

        @model_params.default
        def default_model_params(self) -> dict:
            return {"max_iter": 100, "init": "k-medoids++"}

        def _make_selection_custom(
            self,
            model: SklearnModel,
            candidates_scaled: pd.DataFrame,
        ) -> List[int]:
            """
            In PAM, cluster centers (medoids) correspond to actual data points,
            which means they can be directly used for the selection.
            """
            selection = model.medoid_indices_.tolist()
            return selection


@define
class KMeansClusteringRecommender(SKLearnClusteringRecommender):
    """K-means initial clustering strategy."""

    # Class variables
    model_class: ClassVar[Type[SklearnModel]] = KMeans
    model_cluster_num_parameter_name: ClassVar[str] = "n_clusters"

    # Object variables
    _use_custom_selector = field(default=True)
    model_params: dict = field()

    @model_params.default
    def default_model_params(self) -> dict:
        return {"max_iter": 1000, "n_init": 50}

    def _make_selection_custom(
        self,
        model: SklearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """
        For K-means, a reasonable choice is to pick the points closest to each
        cluster center.
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
    model_class: ClassVar[Type[SklearnModel]] = GaussianMixture
    model_cluster_num_parameter_name: ClassVar[str] = "n_components"

    def _make_selection_custom(
        self,
        model: SklearnModel,
        candidates_scaled: pd.DataFrame,
    ) -> List[int]:
        """
        In a GMM, a reasonable choice is to pick the point with the highest
        probability densities for each cluster.
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
