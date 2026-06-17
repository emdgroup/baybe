"""Shared definitions for the ``PriorGP`` (mean-transfer) transfer-learning models.

The ``PriorGP`` models wrap
:class:`~baybe.surrogates.mean_transfer.MeanTransferSurrogate` in the four
configurations relevant for the benchmarks. The naming scheme is
``PriorGP_{anchors}_{mean_kernel_init}``.
"""

from __future__ import annotations

from baybe.campaign import Campaign
from baybe.objectives.base import Objective
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates.mean_transfer import MeanTransferSurrogate

# The relevant ``PriorGP`` configurations: (suffix, anchors, mean_kernel_init).
PRIORGP_MODES: list[tuple[str, str, str]] = [
    ("PriorGP_pretrained_freeze", "pretrained", "freeze"),
    ("PriorGP_pretrained_warmstart", "pretrained", "warmstart"),
    ("PriorGP_combined_freeze", "combined", "freeze"),
    ("PriorGP_combined_discard", "combined", "discard"),
]


def make_mean_transfer_surrogate(
    anchors: str, mean_kernel_init: str
) -> MeanTransferSurrogate:
    """Create a configured mean-transfer surrogate.

    Args:
        anchors: The anchor selection of the surrogate.
        mean_kernel_init: The inner mean/kernel initialization of the surrogate.

    Returns:
        The configured surrogate.
    """
    return MeanTransferSurrogate(anchors=anchors, mean_kernel_init=mean_kernel_init)


def make_priorgp_recommender(
    anchors: str, mean_kernel_init: str
) -> TwoPhaseMetaRecommender:
    """Create a recommender using a mean-transfer surrogate.

    Args:
        anchors: The anchor selection of the surrogate.
        mean_kernel_init: The inner mean/kernel initialization of the surrogate.

    Returns:
        A two-phase meta recommender whose Bayesian phase uses the surrogate.
    """
    return TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(
            surrogate_model=make_mean_transfer_surrogate(anchors, mean_kernel_init)
        )
    )


def priorgp_scenarios(
    searchspace: SearchSpace,
    objective: Objective,
    *,
    prefix: str,
    n_source_tasks: int,
) -> dict[str, Campaign]:
    """Build the ``PriorGP`` scenario campaigns for a convergence benchmark.

    Args:
        searchspace: The transfer-learning search space (including the task parameter).
        objective: The benchmark objective.
        prefix: The scenario-label prefix (e.g. the source-data percentage).
        n_source_tasks: The number of source tasks of the benchmark.

    Returns:
        A mapping from scenario label ``f"{prefix}_{suffix}"`` to a campaign using the
        corresponding mean-transfer surrogate. The mapping is empty for benchmarks that
        do not have exactly one source task, since the mean-transfer surrogate is only
        applicable in the single-source setting.
    """
    if n_source_tasks != 1:
        return {}
    return {
        f"{prefix}_{suffix}": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=make_priorgp_recommender(anchors, mean_kernel_init),
        )
        for suffix, anchors, mean_kernel_init in PRIORGP_MODES
    }
