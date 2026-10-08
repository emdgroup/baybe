"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import warnings
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import deep_mapping, instance_of, min_len
from typing_extensions import override

from baybe.exceptions import (
    LLMAuthenticationError,
    LLMBatchSizeError,
    LLMCallError,
    LLMResponseError,
)
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.llm._parsing import parse_llm_response
from baybe.recommenders.pure.llm._prompts import make_prompt, make_recovery_prompt
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.serialization import SerialMixin
from baybe.utils.conversion import to_string

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

    supports_discrete_subset_generating_constraints: ClassVar[bool] = True
    # See base class.

    model: str = field(validator=(instance_of(str), min_len(1)))
    """The LiteLLM model identifier to use for recommendations."""

    experiment_description: str = field(validator=(instance_of(str), min_len(1)))
    """Textual description of the experiment.

    For best results, also set :attr:`baybe.objectives.base.Objective.metadata`
    on the objective passed to :meth:`recommend` to give the language model a
    description of what to optimize and per-target context such as units.
    """

    litellm_args: dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=deep_mapping(
            key_validator=instance_of(str),
            # Values are intentionally unconstrained: LiteLLM accepts heterogeneous
            # argument types (str, int, float, bool, nested dicts, ...).
            value_validator=lambda *_: None,
            mapping_validator=instance_of(dict),
        ),
    )
    """Additional arguments to pass to LiteLLM (e.g. ``temperature``, ``max_tokens``).

    API credentials must **not** be passed here — they would be stored in plain text
    and appear in logs and serialized campaign files. Configure them via the
    environment variables that LiteLLM reads automatically based on the model prefix
    (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
    """

    # Stashed context necessary for prompt building
    _objective: Objective | None = field(default=None, init=False, eq=False)

    _measurements: pd.DataFrame | None = field(default=None, init=False, eq=False)

    _pending_experiments: pd.DataFrame | None = field(
        default=None, init=False, eq=False
    )

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
            LLMAuthenticationError: If authentication with the provider fails.
            LLMCallError: If the call fails for another reason (e.g. network, rate
                limiting, timeout, unknown model) before a response is produced.
            LLMResponseError: If a response is returned but contains no usable content.
        """
        from baybe._optional.llm import AuthenticationError, completion

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.litellm_args,
            )
        except AuthenticationError as e:
            raise LLMAuthenticationError(
                f"Authentication with the language model provider failed "
                f"({type(e).__name__}): {e}. Check the API credentials for model "
                f"'{self.model}' (e.g. the provider's API key environment variable)."
            ) from e
        except Exception as e:
            raise LLMCallError(
                f"The call to the language model failed ({type(e).__name__}): {e}. "
                f"Check your network connection and the model identifier "
                f"'{self.model}'."
            ) from e

        # NOTE: `completion()` can also return a stream/coroutine (via `litellm_args`),
        # whose types lack `.choices`; the `except` below handles those at runtime.
        try:
            choices = response.choices  # pyrefly: ignore[missing-attribute]
            content = choices[0].message.content
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
            raise LLMBatchSizeError(
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
            LLMAuthenticationError: If authentication with the provider fails.
            LLMCallError: If the call to the language model fails (e.g. network, rate
                limiting, timeout) before a response is produced.
            LLMResponseError: If the response cannot be turned into a valid
                recommendation batch even after a recovery attempt.
            ValueError: If ``batch_size`` is smaller than 1.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but "
                f"provided {batch_size=}."
            )

        if objective is not None and objective.metadata.is_empty:
            warnings.warn(
                "The objective has no metadata description. Without context on "
                "what to optimize, the language model may produce suboptimal "
                "suggestions. Set the 'description' field on the objective's "
                "metadata to guide the LLM.",
                UserWarning,
                stacklevel=2,
            )

        # Stash context for `_recommend_hybrid`, then delegate to the base `recommend`
        self._objective = objective
        self._measurements = measurements
        self._pending_experiments = pending_experiments

        return super().recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
        )

    @override
    def _recommend_hybrid(
        self,
        searchspace: SearchSpace,
        candidates_exp: pd.DataFrame,
        batch_size: int,
    ) -> pd.DataFrame:
        prompt = make_prompt(
            batch_size,
            searchspace,
            self._objective,
            self._measurements,
            self._pending_experiments,
            experiment_description=self.experiment_description,
        )
        content = self._query_model(prompt)
        try:
            return self._validate_response(content, searchspace, batch_size)
        except LLMResponseError as initial_error:
            # The recommendation had an issue. Make a single, error-specific recovery
            # attempt, informing the model what went wrong.
            recovery_prompt = make_recovery_prompt(
                searchspace,
                batch_size=batch_size,
                error=initial_error,
                original_response=content,
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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
