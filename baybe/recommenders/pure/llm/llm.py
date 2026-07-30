"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import warnings
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.exceptions import LLMResponseError, LLMResponseWarning
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.llm._parsing import parse_llm_response
from baybe.recommenders.pure.llm._prompts import build_prompt, build_recovery_prompt
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.utils.conversion import to_string
from baybe.utils.validation import preprocess_dataframe

_RESERVED_LITELLM_KEYS = frozenset({"model", "messages"})


@define(slots=False)
class LLMRecommender(PureRecommender):
    """Recommender that uses a language model to suggest new experimental points.

    Unlike other pure recommenders, this recommender does not implement the
    ``_recommend_discrete``/``_recommend_continuous``/``_recommend_hybrid`` hooks.
    The language model returns a complete set of parameter configurations that already
    constitutes the recommendation, so there is no per-subspace selection step to
    delegate to those hooks. Instead, :meth:`recommend` is fully overridden and builds
    the result directly from the model response.
    """

    # Class variables
    compatibility: ClassVar[SearchSpaceType] = SearchSpaceType.HYBRID
    # See base class.

    model: str = field(validator=(instance_of(str), min_len(1)))
    """The LiteLLM model identifier to use for recommendations."""

    experiment_description: str = field(validator=(instance_of(str), min_len(1)))
    """Textual description of the experiment."""

    objective_description: str = field(validator=(instance_of(str), min_len(1)))
    """Textual description of the optimization objective."""

    format_instructions: str | None = field(default=None)
    """Optional custom instructions for formatting the LLM's response."""

    recovery_model: str | None = field(default=None)
    """Optional model to use for recovery attempts.

    If ``None``, uses the same model as the main recommendations.
    """

    litellm_args: dict[str, Any] = field(factory=dict, converter=dict)
    """Additional arguments to pass to LiteLLM."""

    recovery_litellm_args: dict[str, Any] | None = field(default=None)
    """Optional arguments to pass to LiteLLM during recovery attempts.

    If ``None``, uses the same arguments as the main recommendations.
    """

    @litellm_args.validator
    def _validate_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate litellm_args does not contain reserved keys."""
        conflicts = _RESERVED_LITELLM_KEYS & set(value.keys())
        if conflicts:
            raise ValueError(
                f"'litellm_args' must not contain keys that are set explicitly: "
                f"{conflicts}. Use the dedicated class attributes instead."
            )

    @recovery_litellm_args.validator
    def _validate_recovery_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate recovery_litellm_args does not contain reserved keys."""
        if value is None:
            return
        conflicts = _RESERVED_LITELLM_KEYS & set(value.keys())
        if conflicts:
            raise ValueError(
                f"'recovery_litellm_args' must not contain keys that are set "
                f"explicitly: {conflicts}. Use the dedicated class attributes instead."
            )

    def _construct_prompt(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        objective: Objective | None = None,
        measurements: pd.DataFrame | None = None,
        pending_experiments: pd.DataFrame | None = None,
    ) -> str:
        """Construct the prompt for the language model.

        Args:
            searchspace: The search space to generate recommendations for.
            batch_size: The number of recommendations to generate.
            objective: Optional objective to include in the prompt.
            measurements: Optional measurements to include in the prompt.
            pending_experiments: Optional pending experiments to include in the prompt.

        Returns:
            The constructed prompt.
        """
        return build_prompt(
            searchspace,
            recommender_name=self.__class__.__name__,
            batch_size=batch_size,
            experiment_description=self.experiment_description,
            objective_description=self.objective_description,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
            format_instructions=self.format_instructions,
        )

    def _parse_llm_response(
        self, response: str, searchspace: SearchSpace
    ) -> pd.DataFrame:
        """Parse the LLM response into a DataFrame of recommendations.

        Args:
            response: The response from the language model.
            searchspace: The search space to validate recommendations against.

        Returns:
            A DataFrame containing the parsed recommendations.
        """
        return parse_llm_response(response, searchspace)

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

        recovery_prompt = build_recovery_prompt(
            searchspace,
            recommender_name=self.__class__.__name__,
            error=error,
            original_response=original_response,
            format_instructions=self.format_instructions,
        )

        litellm_args = self.recovery_litellm_args or self.litellm_args
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
            return self._parse_llm_response(content, searchspace)
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
            LLMResponseError: If the call to the language model fails, or if its
                response cannot be parsed and recovery fails.
        """
        from baybe._optional.llm import completion

        if measurements is not None:
            measurements = preprocess_dataframe(
                measurements,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        if pending_experiments is not None:
            pending_experiments = preprocess_dataframe(
                pending_experiments,
                searchspace,
                numerical_measurements_must_be_within_tolerance=False,
            )

        prompt = self._construct_prompt(
            searchspace, batch_size, objective, measurements, pending_experiments
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
            output = self._parse_llm_response(content, searchspace)
        except LLMResponseError as e:
            output = self._attempt_recovery(e, content, searchspace)

        if len(output) < batch_size:
            warnings.warn(
                f"LLM returned {len(output)} suggestions instead of the "
                f"requested {batch_size}.",
                LLMResponseWarning,
                stacklevel=2,
            )

        return output.head(batch_size)

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Model", self.model, single_line=True),
            to_string("LiteLLM Args", self.litellm_args, single_line=True),
            to_string(
                "Experiment Description", self.experiment_description, single_line=True
            ),
            to_string(
                "Optimization Objective", self.objective_description, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
