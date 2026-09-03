"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import warnings
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.exceptions import LLMResponseError
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.llm._parsing import parse_llm_response
from baybe.recommenders.pure.llm._prompts import make_prompt, make_recovery_prompt
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.serialization import SerialMixin
from baybe.utils.conversion import to_string
from baybe.utils.validation import preprocess_dataframe, validate_object_names

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

    recovery_model: str | None = field(default=None)
    """Optional model to use for recovery attempts.

    If ``None``, uses the same model as the main recommendations.
    """

    litellm_args: dict[str, Any] = field(factory=dict, converter=dict)
    """Additional arguments to pass to LiteLLM (e.g. ``temperature``, ``max_tokens``).

    API credentials must **not** be passed here — they would be stored in plain text
    and appear in logs and serialized campaign files. Configure them via the
    environment variables that LiteLLM reads automatically based on the model prefix
    (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
    """

    recovery_litellm_args: dict[str, Any] | None = field(default=None)
    """Optional arguments to pass to LiteLLM during recovery attempts.

    If ``None``, uses the same arguments as the main recommendations. The same
    credential restriction as for :attr:`litellm_args` applies.
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

    @recovery_litellm_args.validator
    def _validate_recovery_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate recovery_litellm_args has no reserved or credential keys."""
        if value is None:
            return
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

    def _attempt_recovery(
        self,
        error: Exception,
        original_response: str,
        searchspace: SearchSpace,
    ) -> pd.DataFrame:
        """Attempt to recover from a malformed LLM response by asking for correction.

        Args:
            error: The error that occurred during parsing.
            original_response: The original malformed response.
            searchspace: The search space to validate recommendations against.

        Returns:
            A DataFrame containing the corrected recommendations.

        Raises:
            LLMResponseError: If recovery fails.
        """
        from baybe._optional.llm import completion

        recovery_prompt = make_recovery_prompt(
            searchspace,
            error=error,
            original_response=original_response,
        )

        litellm_args = (
            self.recovery_litellm_args
            if self.recovery_litellm_args is not None
            else self.litellm_args
        )
        try:
            response = completion(
                model=self.recovery_model or self.model,
                messages=[{"role": "user", "content": recovery_prompt}],
                **litellm_args,
            )
        except Exception as e:
            raise LLMResponseError(
                f"Recovery LLM call failed ({type(e).__name__}): {e}. "
                f"Original error: {error}"
            ) from e

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMResponseError(
                f"Recovery response had unexpected structure: {e}. "
                f"Original error: {error}"
            ) from e

        if content is None:
            raise LLMResponseError(
                f"Recovery returned empty content (None). Original error: {error}"
            )

        try:
            return parse_llm_response(content, searchspace)
        except LLMResponseError as e:
            raise LLMResponseError(
                f"Recovery produced another malformed response: {e}. "
                f"Original error: {error}"
            ) from e

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
            LLMResponseError: If the call to the language model fails, if its
                response cannot be parsed and recovery fails, or if the number of
                eligible suggestions is less than the requested batch size.
            ValueError: If ``batch_size`` is smaller than 1.
        """
        from baybe._optional.llm import completion

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
            searchspace,
            batch_size=batch_size,
            experiment_description=self.experiment_description,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
        )
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
                f"LLM returned an unexpected response structure: {e}"
            ) from e

        if content is None:
            raise LLMResponseError("LLM returned empty content (None).")

        try:
            output = parse_llm_response(content, searchspace)
        except LLMResponseError as e:
            output = self._attempt_recovery(e, content, searchspace)

        if len(output) < batch_size:
            raise LLMResponseError(
                f"Only {len(output)} eligible suggestion(s) remained instead of the "
                f"requested {batch_size}. The language model may have returned too "
                f"few suggestions or proposed points excluded by the current "
                f"candidate filters."
            )

        # NOTE: Duplicate configurations within a batch are permitted
        return output.head(batch_size)

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
