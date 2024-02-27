"""Temporary functionality for backward compatibility."""

import warnings

from baybe.recommenders.meta import (
    SequentialMetaRecommender,
    StreamingSequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)


def Strategy(*args, **kwargs) -> TwoPhaseMetaRecommender:
    """A ``Strategy`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    warnings.warn(
        f"Using 'Strategy' directly is deprecated and will be removed in a future "
        f"version. Please use 'recommenders.{TwoPhaseMetaRecommender.__name__}' class "
        f"instead.",
        DeprecationWarning,
    )

    return TwoPhaseMetaRecommender(*args, **kwargs)


def TwoPhaseStrategy(*args, **kwargs) -> TwoPhaseMetaRecommender:
    """A ``TwoPhaseStrategy`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    warnings.warn(
        f"'TwoPhaseStrategy' is deprecated and will be removed in a future "
        f"version. Please use 'recommenders.{TwoPhaseMetaRecommender.__name__}' class "
        f"instead.",
        DeprecationWarning,
    )

    return TwoPhaseMetaRecommender(*args, **kwargs)


def SequentialStrategy(*args, **kwargs) -> SequentialMetaRecommender:
    """A ``SequentialStrategy`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    warnings.warn(
        f"'SequentialStrategy' is deprecated and will be removed in a future version. "
        f"Please use 'recommenders.{SequentialMetaRecommender.__name__}' class "
        f"instead.",
        DeprecationWarning,
    )

    return SequentialMetaRecommender(*args, **kwargs)


def StreamingSequentialStrategy(*args, **kwargs) -> StreamingSequentialMetaRecommender:
    """A ``StreamingSequentialStrategy`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    warnings.warn(
        f"'StreamingSequentialStrategy' is deprecated and will be removed in a future "
        f"version. Please use "
        f"'recommenders.{StreamingSequentialMetaRecommender.__name__}' class instead.",
        DeprecationWarning,
    )

    return StreamingSequentialMetaRecommender(*args, **kwargs)
