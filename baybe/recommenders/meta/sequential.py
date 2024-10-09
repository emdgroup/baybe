"""Meta recommenders that switch recommenders based on the experimentation progress."""
# TODO After bayesian recommenders are enabled with no training data, a refactoring of
#  this file will resolve type errors
# mypy: disable-error-code="arg-type"

import gc
from collections.abc import Iterable, Iterator
from typing import Literal

import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, in_, instance_of

from baybe.exceptions import NoRecommendersLeftError
from baybe.objectives.base import Objective
from baybe.recommenders.meta.base import MetaRecommender
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.serialization import (
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
from baybe.utils.plotting import to_string


@define
class TwoPhaseMetaRecommender(MetaRecommender):
    """A two-phased meta recommender that switches at a certain specified point.

    The recommender is switched when a new (batch) recommendation is requested and
    the training data set size (i.e., the total number of collected measurements
    including those gathered before the meta recommender was active) is equal to or
    greater than the number specified via the ``switch_after`` parameter.

    Note:
        Throughout each phase, the meta recommender reuses the **same** recommender
        object, that is, no new instances are created. Therefore, special attention is
        required when using the meta recommender with stateful recommenders.
    """

    initial_recommender: PureRecommender = field(factory=RandomRecommender)
    """The initial recommender used by the meta recommender."""

    recommender: PureRecommender = field(factory=BotorchRecommender)
    """The recommender used by the meta recommender after the switch."""

    switch_after: int = field(default=1)
    """The number of experiments after which the recommender is switched for the next
    requested batch."""

    def select_recommender(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> PureRecommender:
        # See base class.

        return (
            self.recommender
            if (measurements is not None) and (len(measurements) >= self.switch_after)
            else self.initial_recommender
        )

    def __str__(self) -> str:
        fields = [
            to_string("Initial recommender", self.initial_recommender),
            to_string("Recommender", self.recommender),
            to_string("Switch after", self.switch_after, single_line=True),
        ]
        return to_string(self.__class__.__name__, *fields)


@define
class SequentialMetaRecommender(MetaRecommender):
    """A meta recommender that uses a pre-defined sequence of recommenders.

    A new recommender is taken from the sequence whenever at least one new measurement
    is available, until all recommenders are exhausted. More precisely, a recommender
    change is triggered whenever the size of the training dataset increases; the
    actual content of the dataset is ignored.

    Note:
        The provided sequence of recommenders will be internally pre-collected into a
        list. If this is not acceptable, consider using
        :class:`baybe.recommenders.meta.sequential.StreamingSequentialMetaRecommender`
        instead.

    Raises:
        RuntimeError: If the training dataset size decreased compared to the previous
            call.
        NoRecommendersLeftError: If more recommenders are requested than there are
            recommenders available and ``mode="raise"``.
    """

    # Exposed
    recommenders: list[PureRecommender] = field(
        converter=list, validator=deep_iterable(instance_of(PureRecommender))
    )
    """A finite-length sequence of recommenders to be used. For infinite-length
    iterables, see
    :class:`baybe.recommenders.meta.sequential.StreamingSequentialMetaRecommender`."""

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
        batch_size: int,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> PureRecommender:
        # See base class.

        n_data = len(measurements) if measurements is not None else 0

        # If the training dataset size has increased, move to the next recommender
        if n_data > self._n_last_measurements:
            self._step += 1

        # If the training dataset size has decreased, something went wrong
        elif n_data < self._n_last_measurements:
            raise RuntimeError(
                f"The training dataset size decreased from {self._n_last_measurements} "
                f"to {n_data} since the last function call, which indicates that "
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
        self._n_last_measurements = n_data

        return recommender

    def __str__(self) -> str:
        fields = [
            to_string("Recommenders", self.recommenders),
            to_string("Mode", self.mode, single_line=True),
        ]
        return to_string(self.__class__.__name__, *fields)


@define
class StreamingSequentialMetaRecommender(MetaRecommender):
    """A meta recommender that switches between recommenders from an iterable.

    Similar to :class:`baybe.recommenders.meta.sequential.SequentialMetaRecommender`
    but without explicit list conversion. Consequently, it supports arbitrary
    iterables, possibly of infinite length. The downside is that serialization is not
    supported.

    Raises:
        NoRecommendersLeftError: If more recommenders are requested than there are
            recommenders available.
    """

    # Exposed
    recommenders: Iterable[PureRecommender] = field()
    """An iterable providing the recommenders to be used."""

    # Private
    # TODO: See :class:`baybe.recommenders.meta.sequential.SequentialMetaRecommender`
    _step: int = field(init=False, default=-1)
    """Counts how often the recommender has already been switched."""

    _n_last_measurements: int = field(init=False, default=-1)
    """The number of measurements that were available at the last call."""

    _iterator: Iterator = field(init=False)
    """The iterator used to traverse the recommenders."""

    _last_recommender: PureRecommender | None = field(init=False, default=None)
    """The recommender returned from the last call."""

    @_iterator.default
    def default_iterator(self):
        """Initialize the recommender iterator."""
        return iter(self.recommenders)

    def select_recommender(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> PureRecommender:
        # See base class.

        use_last = True
        n_data = len(measurements) if measurements is not None else 0

        # If the training dataset size has increased, move to the next recommender
        if n_data > self._n_last_measurements:
            self._step += 1
            use_last = False

        # If the training dataset size has decreased, something went wrong
        elif n_data < self._n_last_measurements:
            raise RuntimeError(
                f"The training dataset size decreased from {self._n_last_measurements} "
                f"to {n_data} since the last function call, which indicates that "
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
        self._n_last_measurements = n_data

        return self._last_recommender  # type: ignore[return-value]

    def __str__(self) -> str:
        fields = [
            to_string("Recommenders", self.recommenders),
        ]
        return to_string(self.__class__.__name__, *fields)


# The recommender iterable cannot be serialized
converter.register_unstructure_hook(
    StreamingSequentialMetaRecommender, block_serialization_hook
)
converter.register_structure_hook(
    StreamingSequentialMetaRecommender, block_deserialization_hook
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
