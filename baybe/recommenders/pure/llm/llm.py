"""LLM-based recommender for experimental design."""

from __future__ import annotations

import gc
import warnings
from typing import Any, ClassVar

import pandas as pd
from attrs import define, field, fields
from attrs.validators import deep_mapping, instance_of, min_len
from typing_extensions import override

from baybe._optional.info import LLM_INSTALLED
from baybe.exceptions import (
    IncompatibilityError,
    LLMAuthenticationError,
    LLMCallError,
    LLMMalformedResponseError,
    LLMResponseError,
    NotEnoughPointsLeftError,
    OptionalImportError,
)
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.recommenders.pure.llm._parsing import parse_llm_response
from baybe.recommenders.pure.llm._prompts import make_prompt
from baybe.searchspace import SearchSpace
from baybe.searchspace.core import SearchSpaceType
from baybe.serialization import SerialMixin, converter
from baybe.utils.conversion import to_string
from baybe.utils.metadata import Metadata

# Keys that are controlled by the recommender itself and must not be overridden: the
# model and messages are set explicitly, and the response must not be streamed since
# it is parsed as a whole.
_RESERVED_LITELLM_KEYS = frozenset({"model", "messages", "stream"})


@define(slots=False)
class LLMRecommender(PureRecommender, SerialMixin):
    """Recommender that uses a language model to recommend new experimental points.

    When used in a :class:`~baybe.campaign.Campaign`, its recommendations are not marked
    as recommended, even if they match stored configurations. This temporary limitation
    remains until value-based recommendation tracking is available. The campaign's
    ``allow_recommending_already_recommended=False`` flag therefore cannot prevent
    repeats solely because the LLM previously proposed an experiment. Existing tracked
    recommendations, exclusions, and supplied pending experiments are still handled
    according to the campaign's settings.

    Measurements remain recorded and are included in the model's prompt. In discrete
    campaigns, set ``allow_recommending_already_measured=False`` to exclude tracked
    measured configurations. That flag defaults to ``True``, so adding results alone
    does not prevent repeats. Symmetry-equivalent configurations may remain untracked.
    """

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
        repr=False,
        validator=deep_mapping(
            key_validator=instance_of(str),
            # Values are intentionally unconstrained: LiteLLM accepts heterogeneous
            # argument types (str, int, float, bool, nested dicts, ...).
            value_validator=lambda *_: None,
        ),
    )
    """Additional arguments to pass to LiteLLM (e.g. ``temperature``, ``max_tokens``).

    Since these arguments may contain sensitive information, their values are neither
    shown in string representations nor serialized, i.e. they need to be provided
    again after deserialization. Nevertheless, prefer configuring API credentials via
    the environment variables that LiteLLM reads automatically based on the model
    prefix (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
    """

    # Stashed context necessary for prompt building
    _objective: Objective | None = field(default=None, init=False, eq=False)

    _measurements: pd.DataFrame | None = field(default=None, init=False, eq=False)

    _pending_experiments: pd.DataFrame | None = field(
        default=None, init=False, eq=False
    )

    @litellm_args.validator
    def _validate_litellm_args(self, attribute, value):  # noqa: DOC101, DOC103
        """Validate litellm_args does not contain reserved keys."""
        if conflicts := _RESERVED_LITELLM_KEYS & set(value.keys()):
            raise ValueError(
                f"'{attribute.name}' must not contain keys that are controlled by the "
                f"recommender itself: {sorted(conflicts)}."
            )

    @override
    def __attrs_post_init__(self) -> None:
        """Raise if the optional LLM dependencies are not installed."""
        super().__attrs_post_init__()
        if not LLM_INSTALLED:
            raise OptionalImportError(name="litellm", group="llm")

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
            LLMMalformedResponseError: If a response is returned but has an unexpected
                structure or no content.
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

        # NOTE: `completion()` is typed to also return a stream, which lacks `.choices`.
        #   Streaming is excluded via `_RESERVED_LITELLM_KEYS`, and the `except` below
        #   guards against any other unexpected response type at runtime.
        try:
            choices = response.choices  # pyrefly: ignore[missing-attribute]
            content = choices[0].message.content
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMMalformedResponseError(
                f"The language model returned an unexpected response structure: {e}."
            ) from e

        if content is None:
            raise LLMMalformedResponseError(
                "The language model returned empty content (None)."
            )

        return content

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
            A dataframe containing the recommendations as individual rows.

        Raises:
            IncompatibilityError: If the searchspace or objective contains a custom
                component that cannot be described to the language model.
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
                f"The objective has no metadata description. Without context on "
                f"what to optimize, the language model may produce suboptimal "
                f"recommendations. Set the "
                f"'{fields(Metadata).description.name}' field on the objective's "
                f"metadata to guide the LLM.",
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
        try:
            prompt = make_prompt(
                batch_size,
                searchspace,
                self._objective,
                self._measurements,
                self._pending_experiments,
                experiment_description=self.experiment_description,
            )
        except NotImplementedError as ex:
            raise IncompatibilityError(
                f"Could not construct a prompt for `{self.__class__.__name__}`"
                f"from the provided recommendations context."
            ) from ex
        # Stays `None` if the model returns no usable content
        content: str | None = None
        try:
            content = self._query_model(prompt)
            return parse_llm_response(content, searchspace, batch_size)
        except LLMResponseError as initial_error:
            # The recommendation had an issue. Make a recovery attempt.
            recovery_prompt = make_prompt(
                batch_size,
                searchspace,
                self._objective,
                self._measurements,
                self._pending_experiments,
                experiment_description=self.experiment_description,
                recovery=(initial_error, content),
            )
            try:
                recovery_content = self._query_model(recovery_prompt)
                return parse_llm_response(recovery_content, searchspace, batch_size)
            except LLMResponseError as recovery_error:
                raise LLMResponseError(
                    f"The language model failed to produce a valid recommendation, "
                    f"even after a recovery attempt. Initial problem: {initial_error} "
                    f"Remaining problem after recovery: {recovery_error}"
                ) from recovery_error

    @override
    def _recommend_with_discrete_parts(
        self,
        searchspace: SearchSpace,
        batch_size: int,
        pending_experiments: pd.DataFrame | None,
    ) -> pd.DataFrame:
        # The inherited discrete dispatch extracts candidate indices and reconstructs
        # stored rows. LLM responses instead contain generated values, so that path
        # would change the recommendations or fail for symmetry-equivalent variants.
        # Override it locally to preserve output without changing the parent class.
        candidates_exp, _ = searchspace.discrete.get_candidates()
        # This recommender supports subset-generating constraints. Only the base
        # dispatcher's candidate-count check must be repeated, before any model call.
        if (
            searchspace.type is SearchSpaceType.DISCRETE
            and len(candidates_exp) < batch_size
        ):
            raise NotEnoughPointsLeftError(
                f"Using the current settings, there are fewer than {batch_size} "
                f"possible data points left to recommend."
            )
        # TODO: Remove this override when discrete hooks return dataframes (#837).
        # Response parsing/recovery delegates validation to the search space; preserving
        # output and validating configurations do not establish trackable indices.
        return self._recommend_hybrid(searchspace, candidates_exp, batch_size)

    @override
    def __str__(self) -> str:
        fields = [
            to_string("Model", self.model, single_line=True),
            # Only the names are shown since the values may contain credentials
            to_string("LiteLLM Arg Names", list(self.litellm_args), single_line=True),
            to_string(
                "Experiment Description", self.experiment_description, single_line=True
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


_unstructure_with_litellm_args = converter.get_unstructure_hook(LLMRecommender)


def _unstructure_without_litellm_args(obj: LLMRecommender, /) -> dict[str, Any]:
    """Unstructure an LLM recommender without its LiteLLM arguments."""
    attribute = fields(LLMRecommender).litellm_args
    if obj.litellm_args:
        warnings.warn(
            f"The '{attribute.alias}' of '{obj.__class__.__name__}' "
            f"({sorted(obj.litellm_args)}) are not serialized since they may contain "
            f"sensitive information. Provide them again after deserialization.",
            UserWarning,
        )
    specs = _unstructure_with_litellm_args(obj)
    del specs[attribute.name]
    return specs


# The LiteLLM arguments may contain credentials and are hence never serialized
converter.register_unstructure_hook(LLMRecommender, _unstructure_without_litellm_args)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
