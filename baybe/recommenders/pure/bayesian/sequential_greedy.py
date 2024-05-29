"""Temporary namespace for backward compatibility."""

import warnings

from attrs import define

from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender


@define
class SequentialGreedyRecommender(BotorchRecommender):
    """A :class:`baybe.recommenders.pure.bayesian.botorch.BotorchRecommender` alias for backward compatibility."""  # noqa: D401, E501

    def __attrs_pre_init__(self):
        warnings.warn(
            f"The class `SequentialGreedyRecommender` has been deprecated and will be "
            f"removed in a future version. Please use `{BotorchRecommender.__name__}` "
            f"for a one-to-one replacement.",
            DeprecationWarning,
        )
