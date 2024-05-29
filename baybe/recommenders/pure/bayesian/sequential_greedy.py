"""Temporary namespace for backward compatibility."""

import warnings

from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender


def SequentialGreedyRecommender(*args, **kwargs):
    """A :class:`baybe.recommenders.pure.bayesian.botorch.BotorchRecommender` alias for backward compatibility."""  # noqa: D401, E501
    warnings.warn(
        f"The class `SequentialGreedyRecommender` has been deprecated and will be "
        f"removed in a future version. Please use `{BotorchRecommender.__name__}` for "
        f"a one-to-one replacement.",
        DeprecationWarning,
    )
    return BotorchRecommender(*args, **kwargs)
