"""Strategies that switch recommenders depending on the experimentation progress."""

from typing import Iterable, Literal, Optional

import pandas as pd
from attrs import define, field
from attrs.validators import in_

from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender
from baybe.recommenders.base import Recommender
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy


@define(kw_only=True)
class TwoPhaseStrategy(Strategy):
    """A two-phased strategy that switches the recommender after a specified event.

    The recommender is switched when a new (batch) recommendation is requested **and**
    the criterion specified via ```mode``` is fulfilled:
    * "recommended_batches": The strategy has been queried ```switch_after``` times.
    * "recommended_experiments": The strategy has provided at least ```switch_after```
        experimental configurations.
    * "total_measurements": The total number of available experiments (including those
        gathered before the strategy was active) is at least ```switch_after```.

    Note:
        When ```batch_quantity=1``` is set throughout **all** queries, the strategy
        behaves identically for ```mode="recommended_batches"``` and
        ```mode="recommended_experiments"```.

    Note:
        Throughout each phase, the strategy reuses the **same** recommender object,
        that is, no new instances are created. Therefore, special attention is required
        when using the strategy with stateful recommenders.

    Args:
        initial_recommender: The initial recommender used by the strategy.
        recommender: The recommender used by the strategy after the switch.
        switch_after: The (minimum) number of "events" (depending on ```mode```) after
            which the recommender is switched.
        mode: The type of events to be counted to trigger the switch.
        _n_batches_recommended: Ignore (for internal use only).
        _n_experiments_recommended: Ignore (for internal use only).
    """

    # Exposed
    initial_recommender: Recommender = field(factory=RandomRecommender)
    recommender: Recommender = field(factory=SequentialGreedyRecommender)
    switch_after: int = field(default=1)
    mode: Literal[
        "recommended_batches", "recommended_experiments", "total_measurements"
    ] = field(
        default="total_measurements",
        validator=in_(
            ("recommended_batches", "recommended_experiments", "total_measurements")
        ),
    )

    # Private
    # TODO: These should **not** be exposed via the constructor but the workaround
    #   is currently needed for correct (de-)serialization. A proper approach would be
    #   to not set them via the constructor but through a custom hook in combination
    #   with `_cattrs_include_init_false=True`. However, the way
    #   `get_base_structure_hook` is currently designed prevents such a hook from
    #   taking action.
    _n_batches_recommended: int = field(default=0, alias="_n_batches_recommended")
    _n_experiments_recommended: int = field(
        default=0, alias="_n_experiments_recommended"
    )

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        n_done = {
            "recommended_batches": self._n_batches_recommended,
            "recommended_experiments": self._n_experiments_recommended,
            "total_measurements": len(train_x),
        }[self.mode]
        return (
            self.recommender
            if n_done >= self.switch_after
            else self.initial_recommender
        )

    def recommend(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        recommendation = super().recommend(
            searchspace, batch_quantity, train_x, train_y
        )
        self._n_batches_recommended += 1
        self._n_experiments_recommended += batch_quantity
        return recommendation


@define(kw_only=True)
class SequentialStrategy(Strategy):
    """A strategy that uses a pre-defined sequence of recommenders.

    A new recommender is taken from the sequence after each recommended batch until
    all recommenders are exhausted.

    Args:
        recommenders: An iterable providing the recommenders to be used.

    Raises:
        StopIteration: If more (batch) recommendations are requested than there are
            recommenders available.
    """

    recommenders: Iterable[Recommender] = field()

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        return next(self.recommenders)
