"""Base classes for all meta recommenders."""

from abc import ABC, abstractmethod
from typing import Optional

import cattrs
import pandas as pd
from attrs import define, field

from baybe.exceptions import DeprecationError
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.deprecation import structure_recommender_protocol
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.serialization import SerialMixin, converter, unstructure_base


@define
class MetaRecommender(SerialMixin, RecommenderProtocol, ABC):
    """Abstract base class for all meta recommenders."""

    allow_repeated_recommendations: bool = field(default=None, kw_only=True)
    """Deprecated! The flag has become an attribute of
    :class:`baybe.recommenders.pure.base.PureRecommender`."""

    allow_recommending_already_measured: bool = field(default=None, kw_only=True)
    """Deprecated! The flag has become an attribute of
    :class:`baybe.recommenders.pure.base.PureRecommender`."""

    @allow_repeated_recommendations.validator
    def _validate_allow_repeated_recommendations(self, _, value):
        """Raise a ``DeprecationError`` if the flag is used."""
        if value is not None:
            raise DeprecationError(
                f"Passing 'allow_repeated_recommendations' to "
                f"'{self.__class__.__name__}' is deprecated. The flag has become an "
                f"attribute of the '{PureRecommender.__name__}' classes."
            )

    @allow_recommending_already_measured.validator
    def _validate_allow_recommending_already_measured(self, _, value):
        """Raise a ``DeprecationError`` if the flag is used."""
        if value is not None:
            raise DeprecationError(
                f"Passing 'allow_recommending_already_measured' to "
                f"{self.__class__.__name__} is deprecated. The flag has become an "
                f"attribute of {PureRecommender.__name__}."
            )

    @abstractmethod
    def select_recommender(
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> PureRecommender:
        """Select a pure recommender for the given experimentation context.

        Args:
            searchspace:
                See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.
            batch_size:
                See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.
            train_x: See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.
            train_y: See :func:`baybe.recommenders.meta.base.MetaRecommender.recommend`.

        Returns:
            The selected recommender.
        """

    def recommend(
        self,
        searchspace: SearchSpace,
        batch_size: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """See :func:`baybe.recommenders.base.RecommenderProtocol.recommend`."""
        recommender = self.select_recommender(searchspace, batch_size, train_x, train_y)
        return recommender.recommend(searchspace, batch_size, train_x, train_y)


# Register (un-)structure hooks
converter.register_unstructure_hook(
    MetaRecommender,
    lambda x: unstructure_base(
        x,
        # TODO: Remove once deprecation got expired:
        overrides=dict(
            allow_repeated_recommendations=cattrs.override(omit=True),
            allow_recommending_already_measured=cattrs.override(omit=True),
        ),
    ),
)
converter.register_structure_hook(MetaRecommender, structure_recommender_protocol)
