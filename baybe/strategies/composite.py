"""Strategies that switch recommenders depending on the experimentation progress."""

from typing import Iterable, Iterator, List, Literal, Optional

import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, in_, instance_of

from baybe.exceptions import NoRecommendersLeftError
from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender
from baybe.recommenders.base import Recommender
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy
from baybe.utils import block_deserialization_hook, block_serialization_hook
from baybe.utils.serialization import converter


@define(kw_only=True)
class TwoPhaseStrategy(Strategy):
    """A two-phased strategy that switches the recommender after a specified event.

    The recommender is switched when a new (batch) recommendation is requested **and**
    the criterion specified via ```mode``` is fulfilled:
    * "measurements": The total number of collected measurements (including those
        gathered before the strategy was active) is at least ```switch_after```.
    * "batches": The total number of collected batches (including those
        gathered before the strategy was active) is at least ```switch_after```.

    Note:
        Throughout each phase, the strategy reuses the **same** recommender object,
        that is, no new instances are created. Therefore, special attention is required
        when using the strategy with stateful recommenders.

    Args:
        initial_recommender: The initial recommender used by the strategy.
        recommender: The recommender used by the strategy after the switch.
        switch_after: The number of "events" (depending on ```mode```) after which
            the recommender is switched.
        mode: The type of events to be counted to trigger the switch.
    """

    initial_recommender: Recommender = field(factory=RandomRecommender)
    recommender: Recommender = field(factory=SequentialGreedyRecommender)
    switch_after: int = field(default=1)
    mode: Literal["measurements", "batches"] = field(
        default="measurements",
        validator=in_(("measurements", "batches")),
    )

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        n_batches_done: int,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        n_done = n_batches_done if self.mode == "batches" else len(train_x)
        return (
            self.recommender
            if n_done >= self.switch_after
            else self.initial_recommender
        )


@define(kw_only=True)
class SequentialStrategy(Strategy):
    """A strategy that uses a pre-defined sequence of recommenders.

    A new recommender is taken from the sequence after each recommended batch until
    all recommenders are exhausted.

    Note:
        The provided sequence of recommenders will be internally pre-collected into a
        list. If this is not acceptable, consider using
        :class:`baybe.strategies.composite.StreamingSequentialStrategy` instead.

    Args:
        recommenders: A finite-length sequence of recommenders to be used.
            (For infinite-length iterables, see
            :class:`baybe.strategies.composite.StreamingSequentialStrategy`)
        reuse_last: A flag indicating if the last recommender in the sequence shall be
            reused in case more queries are made than recommenders are available.
            Note: If ```True```, the strategy reuses the **same** recommender object,
            that is, no new instances are created. Therefore, special attention is
            required when using this option with stateful recommenders.

    Raises:
        NoRecommendersLeftError: If more (batch) recommendations are requested than
            there are recommenders available and ```reuse_last=False```.
    """

    # Exposed
    recommenders: List[Recommender] = field(
        converter=list, validator=deep_iterable(instance_of(Recommender))
    )
    reuse_last: bool = field(default=False)

    # Private
    # TODO: These should **not** be exposed via the constructor but the workaround
    #   is currently needed for correct (de-)serialization. A proper approach would be
    #   to not set them via the constructor but through a custom hook in combination
    #   with `_cattrs_include_init_false=True`. However, the way
    #   `get_base_structure_hook` is currently designed prevents such a hook from
    #   taking action.
    _step: int = field(default=0, alias="_step")

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        n_batches_done: int,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        # Get the index for retrieving the recommender
        idx = self._step
        if self.reuse_last:
            idx = min(idx, len(self.recommenders) - 1)

        try:
            recommender = self.recommenders[idx]
        except IndexError as ex:
            raise NoRecommendersLeftError(
                f"The strategy has been queried {self._step+1} time(s) but the "
                f"provided sequence of recommenders contains only "
                f"{self._step} element(s)."
            ) from ex
        self._step += 1
        return recommender


@define(kw_only=True)
class StreamingSequentialStrategy(Strategy):
    """A strategy that switches between recommenders from an iterable.

    Similar to :class:`baybe.strategies.composite.SequentialStrategy` but without
    explicit list conversion. Consequently, it supports arbitrary iterables, possibly
    of infinite length. The downside is that serialization is not supported.

    Args:
        recommenders: An iterable providing the recommenders to be used.

    Raises:
        StopIteration: If more (batch) recommendations are requested than there are
            recommenders available.
    """

    # Exposed
    recommenders: Iterable[Recommender] = field()

    # Private
    _iterator: Iterator = field(init=False)
    _step: int = field(init=False, default=0)

    @_iterator.default
    def default_iterator(self):
        """Initialize the recommender iterator."""
        return iter(self.recommenders)

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        n_batches_done: int,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        try:
            recommender = next(self._iterator)
        except StopIteration as ex:
            raise NoRecommendersLeftError(
                f"The strategy has been queried {self._step+1} time(s) but the "
                f"provided sequence of recommenders contains only "
                f"{self._step} element(s)."
            ) from ex
        self._step += 1
        return recommender


# The recommender iterable cannot be serialized
converter.register_unstructure_hook(
    StreamingSequentialStrategy, block_serialization_hook
)
converter.register_structure_hook(
    StreamingSequentialStrategy, block_deserialization_hook
)
