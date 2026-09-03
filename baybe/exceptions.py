"""Custom exceptions and warnings."""

import gc
from collections.abc import Collection
from typing import Any

import pandas as pd
from attr.validators import instance_of
from attrs import define, field
from typing_extensions import override

##### Warnings #####


class InputDataTypeWarning(UserWarning):
    """An input has unexpected data type."""


class LLMResponseWarning(UserWarning):
    """A language model response did not fully meet the request."""


class UnusedObjectWarning(UserWarning):
    """
    A method or function was called with undesired arguments which indicates an
    unintended user fault.
    """


@define
class SearchSpaceMatchWarning(UserWarning):
    """
    When trying to match data to entries in the search space, something unexpected
    happened.
    """

    message: str = field(validator=instance_of(str))
    data: pd.DataFrame = field(validator=instance_of(pd.DataFrame))

    def __attrs_pre_init(self):
        super().__init__(self.message)

    @override
    def __str__(self):
        return self.message


class MinimumCardinalityViolatedWarning(UserWarning):
    """Minimum cardinality constraints are violated."""


##### Exceptions #####


class IncompatibilityError(Exception):
    """Incompatible components are used together."""


class IncompatibleSearchSpaceError(IncompatibilityError):
    """
    A recommender is used with a search space that contains incompatible parts,
    e.g. a discrete recommender is used with a hybrid or continuous search space.
    """


class IncompatibleSurrogateError(IncompatibilityError):
    """An incompatible surrogate was selected."""


class IncompatibleAcquisitionFunctionError(IncompatibilityError):
    """An incompatible acquisition function was selected."""


class IncompatibleExplainerError(IncompatibilityError):
    """An explainer is incompatible with the data it is presented."""


class IncompatibleArgumentError(IncompatibilityError):
    """An incompatible argument was passed to a callable."""


class IncompatibleOverrideError(IncompatibilityError):
    """An override conflicts with another specification."""


class _UnsupportedSearchSpaceAttributeError(AttributeError):
    """Access to a blocked attribute on a reduced search space was attempted."""


class NonGaussianityError(Exception):
    """An operation assuming Gaussianity is attempted on a non-Gaussian distribution."""


class InfeasibilityError(Exception):
    """An optimization problem has no feasible solution."""


class LLMResponseError(Exception):
    """An error occurred while processing a language model response."""

    @property
    def recovery_instruction(self) -> str:
        """Guidance for the language model on how to correct its response."""
        return (
            "Your previous response could not be used. Please provide a new "
            "recommendation that follows the required format and respects the search "
            "space."
        )


class MalformedLLMResponseError(LLMResponseError):
    """A language model response could not be parsed into the expected structure."""

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            "Your previous response was not valid JSON in the required format. Return "
            "a JSON array of objects, each with an 'explanation' string and a "
            "'parameters' object mapping parameter names to values, and nothing else."
        )


class UnknownParameterError(LLMResponseError):
    """A language model response referenced parameters outside the search space."""

    def __init__(
        self,
        *args: Any,
        unknown_names: Collection[str],
        valid_names: Collection[str],
    ):
        super().__init__(*args)
        self.unknown_names = unknown_names
        self.valid_names = valid_names

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response used unknown parameter names "
            f"{sorted(self.unknown_names)}. Use only the following parameters: "
            f"{sorted(self.valid_names)}."
        )


class MissingParameterError(LLMResponseError):
    """A language model response omitted required search space parameters."""

    def __init__(self, *args: Any, parameters: Collection[str]):
        super().__init__(*args)
        self.parameters = parameters

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response did not specify values for the required "
            f"parameters {sorted(self.parameters)}. Every suggestion must provide a "
            f"value for all parameters."
        )


class NonNumericParameterError(LLMResponseError):
    """A language model response gave non-numeric values for a numerical parameter."""

    def __init__(self, *args: Any, detail: str):
        super().__init__(*args)
        self.detail = detail

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response provided non-numeric values for a numerical "
            f"parameter: {self.detail} Provide numeric values for all numerical "
            f"parameters."
        )


class InvalidParameterValueError(LLMResponseError):
    """A language model response contained invalid parameter values."""

    def __init__(self, *args: Any, detail: str):
        super().__init__(*args)
        self.detail = detail

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response contained invalid parameter values: {self.detail} "
            f"Provide values that lie within the allowed range (for numerical "
            f"parameters) or match an allowed choice exactly (for categorical "
            f"parameters)."
        )


class ConstraintViolationError(LLMResponseError):
    """A language model response violated a search space constraint."""

    def __init__(
        self,
        *args: Any,
        constraint_name: str,
        parameters: Collection[str],
    ):
        super().__init__(*args)
        self.constraint_name = constraint_name
        self.parameters = parameters

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response violated the '{self.constraint_name}' constraint "
            f"on parameters {list(self.parameters)}. Ensure all suggestions satisfy "
            f"this constraint."
        )


class IneligiblePointsError(LLMResponseError):
    """A language model response proposed points outside the eligible candidate set."""

    def __init__(self, *args: Any, n_ineligible: int):
        super().__init__(*args)
        self.n_ineligible = n_ineligible

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"{self.n_ineligible} of your proposed configurations are not eligible "
            f"candidates (they may already have been measured or recommended, or they "
            f"do not exist in the search space). Propose different points from the "
            f"available search space."
        )


class BatchSizeError(LLMResponseError):
    """A language model response contained the wrong number of recommendations."""

    def __init__(self, *args: Any, requested: int, received: int):
        super().__init__(*args)
        self.requested = requested
        self.received = received

    @property
    @override
    def recovery_instruction(self) -> str:
        return (
            f"Your previous response provided {self.received} valid recommendation(s), "
            f"but exactly {self.requested} are required. Provide {self.requested} "
            f"distinct, valid recommendations."
        )


class NotEnoughPointsLeftError(Exception):
    """
    More recommendations are requested than there are viable parameter configurations
    left in the search space.
    """


class NoMCAcquisitionFunctionError(Exception):
    """
    A Monte Carlo acquisition function is required but an analytical acquisition
    function has been selected by the user.
    """


class EmptySearchSpaceError(Exception):
    """The created search space contains no parameters."""


class NoMeasurementsError(Exception):
    """A context expected measurements but none were available."""


class IncompleteMeasurementsError(Exception):
    """A context expected complete measurements but none were available."""


class NothingToSimulateError(Exception):
    """There is nothing to simulate because there are no testable configurations."""


class NothingToComputeError(Exception):
    """There is nothing to compute because there are no inputs or existing data."""


class NoRecommendersLeftError(Exception):
    """A recommender is requested by a meta recommender but there are no recommenders
    left.
    """


class NumericalUnderflowError(Exception):
    """A computation would lead to numerical underflow."""


class OptionalImportError(ImportError):
    """An attempt was made to import an optional but uninstalled dependency."""

    def __init__(
        self,
        *args: Any,
        name: str | None = None,
        path: str | None = None,
        group: str | None = None,
    ):
        super().__init__(*args, name=name, path=path)

        # If no message has been explicitly set, create it from the context
        if self.msg is None and name is not None:
            group_str = f"`pip install 'baybe[{group}]'` or " if group else ""
            self.msg = (
                f"The requested functionality requires the optional "  # pyrefly: ignore[bad-assignment]
                f"'{self.name}' package, which is currently not installed. "
                f"Please install the dependency and try again. "
                f"You can do so manually (e.g. `pip install {self.name}`) "
                f"or using an appropriate optional dependency group "
                f"(e.g. {group_str}`pip install 'baybe[extras]'`)."
            )


class DeprecationError(Exception):
    """Signals the use of a deprecated mechanism to the user, interrupting execution."""


class UnidentifiedSubclassError(Exception):
    """A specified subclass cannot be found in the given class hierarchy."""


class ModelNotTrainedError(Exception):
    """A prediction/transformation is attempted before the model has been trained."""


class UnmatchedAttributeError(Exception):
    """An attribute cannot be matched against a certain callable signature."""


class InvalidTargetValueError(Exception):
    """A target value was entered that is not in the target space."""


class NotAllowedError(Exception):
    """An operation was attempted that is not allowed in the current context."""


class UnsupportedEarlyFilteringError(Exception):
    """A constraint does not support early filtering with the given parameters."""


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
