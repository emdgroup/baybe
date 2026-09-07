"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.exceptions import BatchSizeError, LLMResponseError
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.llm._parsing import parse_llm_response
from baybe.recommenders.pure.llm._prompts import make_prompt, make_recovery_prompt
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.serialization import SerialMixin
from baybe.utils.conversion import to_string
from baybe.utils.validation import preprocess_dataframe, validate_object_names

if TYPE_CHECKING:
    from baybe.recommenders.base import RecommenderProtocol
    from baybe.recommenders.meta.sequential import (
        SequentialMetaRecommender,
        TwoPhaseMetaRecommender,
    )

# Keys that are wired in by the recommender itself and must not be overridden.
_RESERVED_LITELLM_KEYS = frozenset({"model", "messages"})

# Credential keys that LiteLLM accepts inline but must be supplied via environment
# variables instead (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY). Blocking them at
# construction time prevents accidental exposure in logs, __str__, and serialization
# strings.
_CREDENTIAL_LITELLM_KEYS = frozenset({"api_key", "api_base", "api_version"})


@define(slots=False)
class LLMRecommender(PureRecommender, SerialMixin):
    """Recommender that uses a language model to suggest new experimental points."""

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    model: str = field(validator=(instance_of(str), min_len(1)))
    """The LiteLLM model identifier to use for recommendations."""

    experiment_description: str = field(validator=(instance_of(str), min_len(1)))
    """Textual description of the experiment.

    For best results, also set :attr:`baybe.objectives.base.Objective.metadata`
    on the objective passed to :meth:`recommend` to give the language model a
    description of what to optimize and per-target context such as units.
    """

    litellm_args: dict[str, Any] = field(factory=dict, converter=dict)
    """Additional arguments to pass to LiteLLM (e.g. ``temperature``, ``max_tokens``).

    API credentials must **not** be passed here — they would be stored in plain text
    and appear in logs and serialized campaign files. Configure them via the
    environment variables that LiteLLM reads automatically based on the model prefix
    (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
    """

    @litellm_args.validator
    def _validate_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate litellm_args does not contain reserved or credential keys."""
        conflicts = _RESERVED_LITELLM_KEYS & set(value.keys())
        if conflicts:
            raise ValueError(
                f"'{attribute.name}' must not contain keys that are set explicitly: "
                f"{conflicts}. Use the dedicated class attributes instead."
            )
        cred_conflicts = _CREDENTIAL_LITELLM_KEYS & set(value.keys())
        if cred_conflicts:
            raise ValueError(
                f"'{attribute.name}' must not contain credential keys "
                f"{cred_conflicts}. Supply credentials via environment variables "
                f"instead (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY)."
            )

    def _query_model(self, prompt: str) -> str:
        """Query the language model and return the raw response text.

        Args:
            prompt: The prompt to send to the language model.

        Returns:
            The raw text content of the model response.

        Raises:
            LLMResponseError: If the model call fails or returns no usable content.
        """
        from baybe._optional.llm import completion

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.litellm_args,
            )
        except Exception as e:
            raise LLMResponseError(
                f"The call to the language model failed ({type(e).__name__}): {e}. "
                f"Check your API credentials, network connection, and the model "
                f"identifier '{self.model}'."
            ) from e

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMResponseError(
                f"The language model returned an unexpected response structure: {e}."
            ) from e

        if content is None:
            raise LLMResponseError("The language model returned empty content (None).")

        return content

    def _validate_response(
        self, content: str, searchspace: SearchSpace, batch_size: int
    ) -> pd.DataFrame:
        """Parse and validate a raw response into a recommendation batch.

        Args:
            content: The raw text content of the model response.
            searchspace: The search space to validate recommendations against.
            batch_size: The number of recommendations to generate.

        Returns:
            A DataFrame with exactly ``batch_size`` recommendations.

        Raises:
            LLMResponseError: If the response cannot be parsed/validated, or if it
                contains fewer than ``batch_size`` recommendations.
        """
        output = parse_llm_response(content, searchspace)
        if len(output) < batch_size:
            raise BatchSizeError(
                f"The language model returned {len(output)} valid recommendation(s) "
                f"instead of the requested {batch_size}.",
                requested=batch_size,
                received=len(output),
            )
        # NOTE: Duplicate configurations within a batch are permitted.
        return output.head(batch_size)

    @override
    def recommend(
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate recommendations using the language model.

        Args:
            batch_size: The number of recommendations to generate.
            searchspace: The search space to generate recommendations for.
            objective: Optional objective to include in the prompt.
            measurements: Optional measurements to include in the prompt.
            pending_experiments: Optional pending experiments to include in the prompt.

        Returns:
            A DataFrame containing the recommendations as individual rows.

        Raises:
            LLMResponseError: If the call to the language model fails, or if its
                response cannot be turned into a valid recommendation batch even after
                a recovery attempt.
            ValueError: If ``batch_size`` is smaller than 1.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but "
                f"provided {batch_size=}."
            )

        if objective is not None:
            validate_object_names(searchspace.parameters + objective.targets)
            if objective.metadata.is_empty:
                warnings.warn(
                    "The objective has no metadata description. Without context on "
                    "what to optimize, the language model may produce suboptimal "
                    "suggestions. Set the 'description' field on the objective's "
                    "metadata to guide the LLM.",
                    UserWarning,
                    stacklevel=2,
                )

        if measurements is not None:
            measurements = preprocess_dataframe(
                measurements,
                searchspace,
                objective,
                numerical_measurements_must_be_within_tolerance=False,
            )

        if pending_experiments is not None:
            pending_experiments = preprocess_dataframe(
                pending_experiments,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        prompt = make_prompt(
            batch_size,
            searchspace,
            objective,
            measurements,
            pending_experiments,
            experiment_description=self.experiment_description,
        )
        content = self._query_model(prompt)
        try:
            return self._validate_response(content, searchspace, batch_size)
        except LLMResponseError as initial_error:
            # The recommendation had an issue. Make a single, error-specific recovery
            # attempt, informing the model what went wrong.
            recovery_prompt = make_recovery_prompt(
                searchspace, error=initial_error, original_response=content
            )
            recovery_content = self._query_model(recovery_prompt)
            try:
                return self._validate_response(
                    recovery_content, searchspace, batch_size
                )
            except LLMResponseError as recovery_error:
                raise LLMResponseError(
                    f"The language model failed to produce a valid recommendation, "
                    f"even after a recovery attempt. Initial problem: {initial_error} "
                    f"Remaining problem after recovery: {recovery_error}"
                ) from recovery_error

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Model", self.model, single_line=True),
            to_string("LiteLLM Args", self.litellm_args, single_line=True),
            to_string(
                "Experiment Description", self.experiment_description, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


def make_llm_two_phase_recommender(
    model: str,
    experiment_description: str,
    *,
    switch_after: int = 1,
    litellm_args: dict[str, Any] | None = None,
    recommender: RecommenderProtocol | None = None,
) -> TwoPhaseMetaRecommender:
    """Create a recommender that warm-starts with an LLM, then switches to Bayesian.

    The returned meta recommender uses an :class:`LLMRecommender` for the initial
    experiments and switches to ``recommender`` once ``switch_after`` measurements have
    been collected.

    Args:
        model: The LiteLLM model identifier for the initial LLM recommender.
        experiment_description: Textual description of the experiment for the LLM.
        switch_after: The number of collected experiments after which the recommender
            switches from the LLM to ``recommender``.
        litellm_args: Optional additional arguments passed to LiteLLM.
        recommender: The recommender used after the switch. Defaults to a
            :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`.

    Returns:
        A :class:`~baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender` using the
        LLM as its initial recommender.
    """
    from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
    from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

    return TwoPhaseMetaRecommender(
        initial_recommender=LLMRecommender(
            model=model,
            experiment_description=experiment_description,
            litellm_args=litellm_args or {},
        ),
        recommender=BotorchRecommender() if recommender is None else recommender,
        switch_after=switch_after,
    )


def make_llm_alternating_recommender(
    model: str,
    experiment_description: str,
    *,
    litellm_args: dict[str, Any] | None = None,
    recommender: RecommenderProtocol | None = None,
) -> SequentialMetaRecommender:
    """Create a recommender that alternates each round between an LLM and Bayesian.

    The returned meta recommender cycles indefinitely between an
    :class:`LLMRecommender` and ``recommender``, advancing to the next one whenever new
    measurements become available.

    Args:
        model: The LiteLLM model identifier for the LLM recommender.
        experiment_description: Textual description of the experiment for the LLM.
        litellm_args: Optional additional arguments passed to LiteLLM.
        recommender: The recommender alternated with the LLM. Defaults to a
            :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`.

    Returns:
        A :class:`~baybe.recommenders.meta.sequential.SequentialMetaRecommender` in
        cyclic mode, alternating between the LLM and ``recommender``.
    """
    from baybe.recommenders.meta.sequential import SequentialMetaRecommender
    from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

    recommenders: list[RecommenderProtocol] = [
        LLMRecommender(
            model=model,
            experiment_description=experiment_description,
            litellm_args=litellm_args or {},
        ),
        BotorchRecommender() if recommender is None else recommender,
    ]
    return SequentialMetaRecommender(recommenders=recommenders, mode="cyclic")


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
