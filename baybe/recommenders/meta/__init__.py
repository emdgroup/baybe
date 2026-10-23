"""Meta recommenders.

Meta recommenders, in analogy to meta studies, consist of one or several pure
recommenders. According to their inner logic they choose which pure recommender to
query.
"""

from baybe.recommenders.meta.sequential import (
    SequentialMetaRecommender,
    StreamingSequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)

__all__ = [
    "TwoPhaseMetaRecommender",
    "SequentialMetaRecommender",
    "StreamingSequentialMetaRecommender",
]
