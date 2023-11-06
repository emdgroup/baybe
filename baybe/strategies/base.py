"""Base classes for all strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from attr import define, field

from baybe.searchspace import SearchSpace
from baybe.utils.serialization import (
    converter,
    get_base_structure_hook,
    SerialMixin,
    unstructure_base,
)


@define
class Strategy(SerialMixin, ABC):
    """Abstract base class for all BayBE strategies.

    Args:
        allow_repeated_recommendations: Allow to make recommendations that were
            already recommended earlier. This only has an influence in discrete
            search spaces.
        allow_recommending_already_measured: Allow to output recommendations that
            were measured previously. This only has an influence in discrete
            search spaces.
    """

    allow_repeated_recommendations: bool = field(default=False)
    allow_recommending_already_measured: bool = field(default=False)

    @abstractmethod
    def recommend(
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Recommend the next experiments to be conducted.

        Args:
            searchspace: The search space in which the experiments are conducted.
            batch_quantity: The number of experiments to be conducted in parallel.
            train_x: The features of the conducted experiments.
            train_y: The corresponding response values.

        Returns:
            The DataFrame with the specific experiments recommended.
        """


# Register (un-)structure hooks
converter.register_unstructure_hook(Strategy, unstructure_base)
converter.register_structure_hook(Strategy, get_base_structure_hook(Strategy))
