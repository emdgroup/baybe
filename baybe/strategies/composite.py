"""Strategies that switch recommenders depending on the experimentation progress."""

from typing import Iterable, Iterator, List, Literal, Optional

import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, in_, instance_of

from baybe.exceptions import NoRecommendersLeftError
from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender
from baybe.recommenders.base import NonPredictiveRecommender, Recommender
from baybe.searchspace import SearchSpace
from baybe.strategies.base import Strategy
from baybe.utils import block_deserialization_hook, block_serialization_hook
from baybe.utils.serialization import converter

# TODO: Make predictive recommenders handle empty training data
_unsupported_recommender_error = ValueError(
    f"For cases where no training is available, the selected recommender "
    f"must be a subclass of '{NonPredictiveRecommender.__name__}'."
)


@define
class TwoPhaseStrategy(Strategy):
    """A two-phased strategy that switches the recommender at a certain specified point.

    The recommender is switched when a new (batch) recommendation is requested and
    the training data set size (i.e., the total number of collected measurements
    including those gathered before the strategy was active) is equal to or greater
    than the number specified via the ``switch_after`` parameter.

    Note:
        Throughout each phase, the strategy reuses the **same** recommender object,
        that is, no new instances are created. Therefore, special attention is required
        when using the strategy with stateful recommenders.
    """

    initial_recommender: Recommender = field(factory=RandomRecommender)
    """The initial recommender used by the strategy."""

    recommender: Recommender = field(factory=SequentialGreedyRecommender)
    """The recommender used by the strategy after the switch."""

    switch_after: int = field(default=1)
    """The number of experiments after which the recommender is switched for the next
    requested batch."""

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        # FIXME: enable predictive recommenders for empty training data
        if (train_x is None or len(train_x) == 0) and not isinstance(
            self.initial_recommender, NonPredictiveRecommender
        ):
            raise _unsupported_recommender_error

        return (
            self.recommender
            if len(train_x) >= self.switch_after
            else self.initial_recommender
        )


@define
class SequentialStrategy(Strategy):
    """A strategy that uses a pre-defined sequence of recommenders.

    A new recommender is taken from the sequence whenever at least one new measurement
    is available, until all recommenders are exhausted. More precisely, a recommender
    change is triggered whenever the size of the training dataset increases; the
    actual content of the dataset is ignored.

    Note:
        The provided sequence of recommenders will be internally pre-collected into a
        list. If this is not acceptable, consider using
        :class:`baybe.strategies.composite.StreamingSequentialStrategy` instead.

    Raises:
        NoRecommendersLeftError: If more recommenders are requested than there are
            recommenders available and ``mode="raise"``.
    """

    # Exposed
    recommenders: List[Recommender] = field(
        converter=list, validator=deep_iterable(instance_of(Recommender))
    )
    """A finite-length sequence of recommenders to be used. For infinite-length
    iterables, see :class:`baybe.strategies.composite.StreamingSequentialStrategy`."""

    mode: Literal["raise", "reuse_last", "cyclic"] = field(
        default="raise",
        validator=in_(("raise", "reuse_last", "cyclic")),
    )
    """Defines what shall happen when the last recommender in the sequence has been
    consumed but additional recommender changes are triggered:

        * ``"raise"``: An error is raised.
        * ``"reuse_last"``: The last recommender in the sequence is used indefinitely.
        * ``"cycle"``: The selection restarts from the beginning of the sequence.
    """

    # Private
    # TODO: These should **not** be exposed via the constructor but the workaround
    #   is currently needed for correct (de-)serialization. A proper approach would be
    #   to not set them via the constructor but through a custom hook in combination
    #   with `_cattrs_include_init_false=True`. However, the way
    #   `get_base_structure_hook` is currently designed prevents such a hook from
    #   taking action.
    _step: int = field(default=-1, alias="_step")
    """Counts how often the recommender has already been switched."""

    _n_last_measurements: int = field(default=-1, alias="_n_last_measurements")
    """The number of measurements that were available at the last call."""

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        # If the training dataset size has increased, move to the next recommender
        if len(train_x) > self._n_last_measurements:
            self._step += 1
        # If the training dataset size has decreased, something went wrong
        elif len(train_x) < self._n_last_measurements:
            raise RuntimeError(
                f"The training dataset size decreased from {self._n_last_measurements} "
                f"to {len(train_x)} since the last function call, which indicates that "
                f"'{self.__class__.__name__}' was not used as intended."
            )

        # Get the right index for the "next" recommender
        idx = self._step
        if self.mode == "reuse_last":
            idx = min(idx, len(self.recommenders) - 1)
        elif self.mode == "cyclic":
            idx %= len(self.recommenders)

        # Get the recommender
        try:
            recommender = self.recommenders[idx]
        except IndexError as ex:
            raise NoRecommendersLeftError(
                f"A total of {self._step+1} recommender(s) was/were requested but the "
                f"provided sequence contains only {self._step} element(s)."
            ) from ex

        # Remember the training dataset size for the next call
        self._n_last_measurements = len(train_x)

        # FIXME: enable predictive recommenders for empty training data
        if (train_x is None or len(train_x) == 0) and not isinstance(
            recommender, NonPredictiveRecommender
        ):
            raise _unsupported_recommender_error

        return recommender


@define
class StreamingSequentialStrategy(Strategy):
    """A strategy that switches between recommenders from an iterable.

    Similar to :class:`baybe.strategies.composite.SequentialStrategy` but without
    explicit list conversion. Consequently, it supports arbitrary iterables, possibly
    of infinite length. The downside is that serialization is not supported.

    Raises:
        NoRecommendersLeftError: If more recommenders are requested than there are
            recommenders available.
    """

    # Exposed
    recommenders: Iterable[Recommender] = field()
    """An iterable providing the recommenders to be used."""

    # Private
    # TODO: See :class:`baybe.strategies.composite.SequentialStrategy`
    _step: int = field(init=False, default=-1)
    """Counts how often the recommender has already been switched."""

    _n_last_measurements: int = field(init=False, default=-1)
    """The number of measurements that were available at the last call."""

    _iterator: Iterator = field(init=False)
    """The iterator used to traverse the recommenders."""

    _last_recommender: Optional[Recommender] = field(init=False, default=None)
    """The recommender returned from the last call."""

    @_iterator.default
    def default_iterator(self):
        """Initialize the recommender iterator."""
        return iter(self.recommenders)

    def select_recommender(  # noqa: D102
        self,
        searchspace: SearchSpace,
        batch_quantity: int = 1,
        train_x: Optional[pd.DataFrame] = None,
        train_y: Optional[pd.DataFrame] = None,
    ) -> Recommender:
        # See base class.

        use_last = True

        # If the training dataset size has increased, move to the next recommender
        if len(train_x) > self._n_last_measurements:
            self._step += 1
            use_last = False
        # If the training dataset size has decreased, something went wrong
        elif len(train_x) < self._n_last_measurements:
            raise RuntimeError(
                f"The training dataset size decreased from {self._n_last_measurements} "
                f"to {len(train_x)} since the last function call, which indicates that "
                f"'{self.__class__.__name__}' was not used as intended."
            )

        # Get the recommender
        try:
            if not use_last:
                self._last_recommender = next(self._iterator)
        except StopIteration as ex:
            raise NoRecommendersLeftError(
                f"A total of {self._step+1} recommender(s) was/were requested but the "
                f"provided iterator provided only {self._step} element(s)."
            ) from ex

        # Remember the training dataset size for the next call
        self._n_last_measurements = len(train_x)

        # FIXME: enable predictive recommenders for empty training data
        if (train_x is None or len(train_x) == 0) and not isinstance(
            self._last_recommender, NonPredictiveRecommender
        ):
            raise _unsupported_recommender_error

        return self._last_recommender


# The recommender iterable cannot be serialized
converter.register_unstructure_hook(
    StreamingSequentialStrategy, block_serialization_hook
)
converter.register_structure_hook(
    StreamingSequentialStrategy, block_deserialization_hook
)
