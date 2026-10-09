"""Meta recommenders based on Large Language Models (LLMs)."""

from __future__ import annotations

import gc
from typing import Any

from attrs import define, field
from attrs.validators import instance_of

from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.meta.sequential import (
    SequentialMetaRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.recommenders.pure.llm.llm import LLMRecommender


@define
class LLMTwoPhaseRecommender(TwoPhaseMetaRecommender):
    """A two-phase recommender that warm-starts with an :class:`LLMRecommender`.

    A convenience specialization of
    :class:`~baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender` whose
    :attr:`initial_recommender` must be an
    :class:`~baybe.recommenders.pure.llm.llm.LLMRecommender`. The language model
    proposes the first experiments, and the recommender switches to
    :attr:`recommender` once :attr:`switch_after` measurements have been collected.
    """

    initial_recommender: RecommenderProtocol = field(
        validator=instance_of(LLMRecommender), kw_only=True
    )
    """The language-model recommender used before the switch."""

    @classmethod
    def from_model(
        cls,
        model: str,
        experiment_description: str,
        *,
        litellm_args: dict[str, Any] | None = None,
        switch_after: int = 1,
        recommender: RecommenderProtocol | None = None,
    ) -> LLMTwoPhaseRecommender:
        """Create the recommender directly from a LiteLLM model identifier.

        Args:
            model: The LiteLLM model identifier for the initial LLM recommender.
            experiment_description: Textual description of the experiment for the LLM.
            litellm_args: Optional additional arguments passed to LiteLLM.
            switch_after: The number of collected experiments after which the
                recommender switches from the LLM to ``recommender``.
            recommender: The recommender used after the switch. Defaults to a
                :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`.

        Returns:
            The configured recommender.
        """
        from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

        return cls(
            initial_recommender=LLMRecommender(
                model=model,
                experiment_description=experiment_description,
                litellm_args=litellm_args or {},
            ),
            recommender=BotorchRecommender() if recommender is None else recommender,
            switch_after=switch_after,
        )


@define
class LLMAlternatingRecommender(SequentialMetaRecommender):
    """A recommender alternating between an :class:`LLMRecommender` and another one.

    A convenience specialization of
    :class:`~baybe.recommenders.meta.sequential.SequentialMetaRecommender` that cycles
    between a language-model recommender and another recommender, advancing whenever
    new measurements become available. The sequence must consist of exactly two
    recommenders, the first of which must be an
    :class:`~baybe.recommenders.pure.llm.llm.LLMRecommender`.
    """

    def __attrs_post_init__(self) -> None:
        """Validate the recommender sequence.

        Raises:
            ValueError: If the sequence does not consist of exactly two recommenders.
            ValueError: If the first recommender is not an
                :class:`~baybe.recommenders.pure.llm.llm.LLMRecommender`.
        """
        if len(self.recommenders) != 2:
            raise ValueError(
                f"'{self.__class__.__name__}' requires exactly two recommenders (an "
                f"'{LLMRecommender.__name__}' alternating with another recommender), "
                f"but {len(self.recommenders)} were provided."
            )
        if not isinstance(self.recommenders[0], LLMRecommender):
            raise ValueError(
                f"'{self.__class__.__name__}' requires the first recommender in the "
                f"sequence to be an '{LLMRecommender.__name__}'."
            )
        if not self.mode == "cyclic":
            raise ValueError(
                f"'{self.__class__.__name__}' requires the mode to be 'cyclic'."
            )

    @classmethod
    def from_model(
        cls,
        model: str,
        experiment_description: str,
        *,
        litellm_args: dict[str, Any] | None = None,
        recommender: RecommenderProtocol | None = None,
    ) -> LLMAlternatingRecommender:
        """Create the recommender directly from a LiteLLM model identifier.

        Args:
            model: The LiteLLM model identifier for the LLM recommender.
            experiment_description: Textual description of the experiment for the LLM.
            litellm_args: Optional additional arguments passed to LiteLLM.
            recommender: The recommender alternated with the LLM. Defaults to a
                :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`.

        Returns:
            The configured recommender, cycling between the LLM and ``recommender``.
        """
        from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

        return cls(
            recommenders=(
                LLMRecommender(
                    model=model,
                    experiment_description=experiment_description,
                    litellm_args=litellm_args or {},
                ),
                BotorchRecommender() if recommender is None else recommender,
            ),
            mode="cyclic",
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
