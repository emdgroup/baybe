"""Meta recommenders that switch recommenders based on the experimentation progress."""
# TODO After bayesian recommenders are enabled with no training data, a refactoring of
#  this file will resolve type errors
# mypy: disable-error-code="arg-type"

import gc
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import Literal

import pandas as pd
from attrs import Factory, define, field, fields
from attrs.validators import deep_iterable, ge, in_, instance_of
from typing_extensions import override

from baybe.exceptions import NoRecommendersLeftError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.base import MetaRecommender
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

    initial_recommender: RecommenderProtocol = field(
        factory=RandomRecommender, validator=instance_of(RecommenderProtocol)
    )
    """The initial recommender used by the meta recommender."""

    recommender: RecommenderProtocol = field(
        factory=BotorchRecommender, validator=instance_of(RecommenderProtocol)
    )
    """The recommender used by the meta recommender after the switch."""

    switch_after: int = field(default=1, validator=[instance_of(int), ge(1)])
    """The number of experiments required for the recommender to switch."""

    remain_switched: bool = field(default=False, validator=instance_of(bool))
    """Determines if the recommender should remain switched even if the number of
    experiments falls below the threshold value in subsequent calls."""

    # TODO: These should **not** be exposed via the constructor but the workaround
    #   is currently needed for correct (de-)serialization. A proper approach would be
    #   to not set them via the constructor but through a custom hook in combination
    #   with `_cattrs_include_init_false=True`. However, the way
    #   `get_base_structure_hook` is currently designed prevents such a hook from
    #   taking action.
    _has_switched: bool = field(default=False, alias="_has_switched")
    """Indicates if the switch has already occurred."""

    @override
    @property
    def is_stateful(self) -> bool:
        return self.remain_switched

    @override
    def select_recommender(
        self,
        batch_size: int | None = None,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> RecommenderProtocol:
        n_data = len(measurements) if measurements is not None else 0
        if (n_data >= self.switch_after) or (
            self._has_switched and self.remain_switched
        ):
            self._has_switched = True
            return self.recommender

        return self.initial_recommender

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Initial recommender", self.initial_recommender),
            to_string("Recommender", self.recommender),
            to_string("Switch after", self.switch_after, single_line=True),
            to_string("Remain switched", self.remain_switched, single_line=True),
            to_string("Has switched", self._has_switched, single_line=True),
        ]
        return to_string(self.__class__.__name__, *fields)


@define
class BaseSequentialMetaRecommender(MetaRecommender):
    """Base class for sequential meta recommenders."""

    # TODO: These should **not** be exposed via the constructor but the workaround
    #   is currently needed for correct (de-)serialization. A proper approach would be
    #   to not set them via the constructor but through a custom hook in combination
    #   with `_cattrs_include_init_false=True`. However, the way
    #   `get_base_structure_hook` is currently designed prevents such a hook from
    #   taking action.
    _step: int = field(default=0, alias="_step", kw_only=True)
    """An index pointing to the current position in the recommender sequence."""

    _was_used: bool = field(default=False, alias="_was_used", kw_only=True)
    """Boolean flag indicating if the current recommender has been used."""

    _n_last_measurements: int = field(
        default=0, alias="_n_last_measurements", kw_only=True
    )
    """The number of measurements available at the last successful recommend call."""

    @override
    @property
    def is_stateful(self) -> bool:
        return True

    @abstractmethod
    def _get_recommender_at_current_step(self) -> RecommenderProtocol:
        """Get the recommender at the current sequence position."""

    @override
    def select_recommender(
        self,
        batch_size: int | None = None,
        searchspace: SearchSpace | None = None,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> RecommenderProtocol:
        # If the training dataset size has decreased, something went wrong
        if (
            n_data := len(measurements) if measurements is not None else 0
        ) < self._n_last_measurements:
            raise RuntimeError(
                f"The training dataset size decreased from {self._n_last_measurements} "
                f"to {n_data} since the last function call, which indicates that "
                f"'{self.__class__.__name__}' was not used as intended."
            )

        # Check conditions if the next recommender in sequence is required
        more_data = n_data > self._n_last_measurements
        if (not self._was_used) or (not more_data):
            return self._get_recommender_at_current_step()

        # Move on to the next recommender
        self._step += 1
        self._was_used = False
        return self._get_recommender_at_current_step()

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        recommendation = super().recommend(
            batch_size, searchspace, objective, measurements, pending_experiments
        )
        self._was_used = True

        # Remember the training dataset size for the next call
        self._n_last_measurements = len(measurements) if measurements is not None else 0

        return recommendation


@define
class SequentialMetaRecommender(BaseSequentialMetaRecommender):
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

    recommenders: list[RecommenderProtocol] = field(
        converter=list, validator=deep_iterable(instance_of(RecommenderProtocol))
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

    @override
    def _get_recommender_at_current_step(self) -> RecommenderProtocol:
        idx = self._step
        if self.mode == "reuse_last":
            idx = min(idx, len(self.recommenders) - 1)
        elif self.mode == "cyclic":
            idx %= len(self.recommenders)
        try:
            return self.recommenders[idx]
        except IndexError as ex:
            raise NoRecommendersLeftError(
                f"A total of {self._step+1} recommender(s) was/were requested but "
                f"the provided sequence contains only {self._step} element(s). "
                f"Add more recommenders or adjust the "
                f"'{fields(SequentialMetaRecommender).mode.name}' attribute."
            ) from ex

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Recommenders", self.recommenders),
            to_string("Mode", self.mode, single_line=True),
        ]
        return to_string(self.__class__.__name__, *fields)


@define
class StreamingSequentialMetaRecommender(BaseSequentialMetaRecommender):
    """A meta recommender that switches between recommenders from an iterable.

    Similar to :class:`baybe.recommenders.meta.sequential.SequentialMetaRecommender`
    but without explicit list conversion. This enables a number of advanced use cases:

    * It supports arbitrary iterables, allowing to configure recommender sequences
      of infinite length. This is useful when the total number of iterations unknown
      in advance.
    * It can be used to adaptively adjust the recommender sequence based on the
      latest context available outside the class, by modifying the iterable on the fly.

    The downside is that serialization is not supported.

    Raises:
        NoRecommendersLeftError: If more recommenders are requested than there are
            recommenders available.
    """

    recommenders: Iterable[RecommenderProtocol] = field()
    """An iterable providing the recommenders to be used."""

    _iterator: Iterator = field(
        init=False,
        default=Factory(lambda self: iter(self.recommenders), takes_self=True),
    )
    """The iterator used to traverse the recommenders."""

    _last_recommender: RecommenderProtocol | None = field(init=False, default=None)
    """The recommender returned from the last call."""

    _position_of_latest_recommender: int | None = field(init=False, default=None)
    """The position of the latest recommender fetched from the iterable."""

    @override
    def _get_recommender_at_current_step(self) -> RecommenderProtocol:
        if self._step != self._position_of_latest_recommender:
            try:
                self._last_recommender = next(self._iterator)
            except StopIteration as ex:
                raise NoRecommendersLeftError(
                    f"A total of {self._step+1} recommender(s) was/were requested but "
                    f"the provided iterator provided only {self._step} element(s). "
                ) from ex
            self._position_of_latest_recommender = self._step

        # By now, the first recommender has been fetched
        assert self._last_recommender is not None

        return self._last_recommender

    @override
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
