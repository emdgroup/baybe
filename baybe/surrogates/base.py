"""Base functionality for all BayBE surrogates."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, TypeAlias

import pandas as pd
from attrs import define, field
from joblib.hashing import hash
from typing_extensions import override

from baybe.exceptions import IncompatibleSurrogateError, ModelNotTrainedError
from baybe.objectives.base import Objective
from baybe.parameters.base import Parameter
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import handle_missing_values, to_tensor
from baybe.utils.scaling import ColumnTransformer

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import GPyTorchPosterior, Posterior
    from torch import Tensor

    from baybe.surrogates.composite import CompositeSurrogate

PosteriorStatistic: TypeAlias = float | Literal["mean", "std", "var", "mode"]
"""Type alias for requestable statistics (a float yields the corresponding quantile)."""


class _NoTransform(Enum):
    """Sentinel class."""

    IDENTITY_TRANSFORM = auto()


_IDENTITY_TRANSFORM = _NoTransform.IDENTITY_TRANSFORM
"""Sentinel to indicate the absence of a transform where `None` is ambiguous."""


class SurrogateProtocol(Protocol):
    """Type protocol specifying the interface surrogate models need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    # TODO: Final layout still to be optimized. For example, shall we require a
    #   `posterior` method?

    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit the surrogate to training data in a given modelling context.

        For details on the expected method arguments, see
        :meth:`baybe.recommenders.base.RecommenderProtocol`.
        """

    def to_botorch(self) -> Model:
        """Create the botorch-ready representation of the fitted model.

        The :class:`botorch.models.model.Model` created by this method needs to be
        configured such that it can be called with candidate points in **computational
        representation**, that is, input of the form as obtained via
        :meth:`baybe.searchspace.core.SearchSpace.transform`.
        """


@define
class Surrogate(ABC, SurrogateProtocol, SerialMixin):
    """Abstract base class for all surrogate models."""

    supports_transfer_learning: ClassVar[bool]
    """Class variable encoding whether or not the surrogate supports transfer
    learning."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding whether or not the surrogate is multi-output
    compatible."""

    _searchspace: SearchSpace | None = field(init=False, default=None, eq=False)
    """The search space on which the surrogate operates. Available after fitting."""

    _objective: Objective | None = field(init=False, default=None, eq=False)
    """The objective for which the surrogate was trained. Available after fitting."""

    _measurements_hash: str = field(init=False, default=None, eq=False)
    """The hash of the data the surrogate was trained on."""

    _input_scaler: ColumnTransformer | None = field(init=False, default=None, eq=False)
    """Scaler for transforming input values. Available after fitting.

    Scales a tensor containing parameter configurations in computational representation
    to make them digestible for the model-specific, scale-agnostic posterior logic."""

    # TODO: type should be
    #   `botorch.models.transforms.outcome.Standardize | _NoTransform` | None
    #   but is currently omitted due to:
    #   https://github.com/python-attrs/cattrs/issues/531
    _output_scaler = field(init=False, default=None, eq=False)
    """Scaler for transforming output values. Available after fitting.

    Scales a tensor containing target measurements in computational representation
    to make them digestible for the model-specific, scale-agnostic posterior logic."""

    @override
    def to_botorch(self) -> Model:
        from baybe.surrogates._adapter import AdapterModel

        return AdapterModel(self)

    def replicate(self) -> CompositeSurrogate:
        """Make the surrogate handle multiple targets via replication.

        If the surrogate only supports single targets, this method turns it into a
        multi-target surrogate by replicating the model architecture for each observed
        target. The resulting copies are trained independently, but share the same
        architecture.

        If the surrogate is itself already multi-target compatible, this operation
        effectively disables the model's inherent multi-target mechanism by treating
        it as a single-target surrogate and applying the same replication mechanism.
        """
        from baybe.surrogates.composite import CompositeSurrogate

        return CompositeSurrogate.from_replication(self)

    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        """Return the scaler factory to be used for the given parameter.

        This method is supposed to be overridden by subclasses to implement their
        custom parameter scaling logic. Otherwise, parameters will be normalized.
        """
        from botorch.models.transforms.input import Normalize

        return Normalize

    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        """Return the scaler factory to be used for target scaling.

        This method is supposed to be overridden by subclasses to implement their
        custom target scaling logic. Otherwise, targets will be standardized.
        """
        from botorch.models.transforms.outcome import Standardize

        return Standardize

    def _make_input_scaler(self, searchspace: SearchSpace) -> ColumnTransformer:
        """Make and fit the input scaler for transforming computational dataframes."""
        # Create a composite scaler from parameter-wise scaler objects
        mapping: dict[tuple[int, ...], InputTransform] = {}
        for p in searchspace.parameters:
            if (factory := self._make_parameter_scaler_factory(p)) is None:
                continue
            idxs = searchspace.get_comp_rep_parameter_indices(p.name)
            transformer = factory(len(idxs))
            mapping[idxs] = transformer
        scaler = ColumnTransformer(mapping)

        # Fit the scaler to the parameter bounds
        scaler.fit(to_tensor(searchspace.scaling_bounds))

        return scaler

    def _make_output_scaler(
        self, objective: Objective, measurements: pd.DataFrame
    ) -> OutcomeTransform | _NoTransform:
        """Make and fit the output scaler for transforming computational dataframes."""
        if (factory := self._make_target_scaler_factory()) is None:
            return _IDENTITY_TRANSFORM

        if objective.n_outputs != 1:
            # There is no execution path yet that could lead to this situation
            raise NotImplementedError(
                "Output scalers for multi-output models are not available."
            )
        scaler = factory(1)

        # TODO: Consider taking into account target boundaries when available
        scaler(to_tensor(objective._pre_transform(measurements, allow_extra=True)))
        scaler.eval()

        return scaler

    def posterior(self, candidates: pd.DataFrame) -> Posterior:
        """Compute the posterior for candidates in experimental representation.

        Takes a dataframe of parameter configurations in **experimental representation**
        and returns the corresponding posterior object. Therefore, the method serves as
        the user-facing entry point for accessing model predictions.

        Args:
            candidates: A dataframe containing parameter configurations in
                **experimental representation**.

        Raises:
            ModelNotTrainedError: When called before the model has been trained.

        Returns:
            A :class:`botorch.posteriors.Posterior` object representing the posterior
            distribution at the given candidate points, where the posterior is also
            described in **experimental representation**. That is, the posterior values
            lie in the same domain as the modelled targets/objective on which the
            surrogate was trained via :meth:`baybe.surrogates.base.Surrogate.fit`.
        """
        if self._searchspace is None:
            raise ModelNotTrainedError(
                "The surrogate must be trained before a posterior can be computed."
            )
        return self._posterior_comp(
            to_tensor(self._searchspace.transform(candidates, allow_extra=True))
        )

    def _posterior_comp(self, candidates_comp: Tensor, /) -> Posterior:
        """Compute the posterior for candidates in computational representation.

        Takes a tensor of parameter configurations in **computational representation**
        and returns the corresponding posterior object. Therefore, the method provides
        the entry point for queries coming from computational layers, for instance,
        BoTorch's `optimize_*` functions.

        Args:
            candidates_comp: A tensor containing parameter configurations in
                **computational representation**.

        Returns:
            The same :class:`botorch.posteriors.Posterior` object as returned via
            :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._input_scaler is not None

        p = self._posterior(self._input_scaler.transform(candidates_comp))
        if self._output_scaler is not _IDENTITY_TRANSFORM:
            p = self._output_scaler.untransform_posterior(p)
        return p

    @abstractmethod
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Perform the actual model-specific posterior evaluation logic.

        This method is supposed to be overridden by subclasses to implement their
        model-specific surrogate architecture. Internally, the method is called by the
        base class with a **scaled** tensor of candidates in **computational
        representation**, where the scaling is configurable by the subclass by
        overriding the default scaler factory methods of the base. The base class also
        takes care of transforming the returned posterior back to the original scale
        according to the defined scalers.

        This means:
        -----------
        Subclasses implementing this method do not have to bother about
        pre-/postprocessing of the in-/output. Instead, they only need to implement the
        mathematical operation of computing the posterior for the given input according
        to their model specifications and can implicitly assume that scaling is handled
        appropriately outside. In short: the returned posterior simply needs to be on
        the same scale as the given input.

        Args:
            candidates_comp_scaled: A tensor containing **scaled** parameter
                configurations in **computational representation**, as defined through
                the input scaler obtained via
                :meth:`baybe.surrogates.base.Surrogate._make_input_scaler`.

        Returns:
            A :class:`botorch.posteriors.Posterior` object representing the
            **scale-transformed** posterior distributions at the given candidate points,
            where the posterior is described on the scale dictated by the output scaler
            obtained via :meth:`baybe.surrogates.base.Surrogate._make_output_scaler`.
        """

    def posterior_stats(
        self,
        candidates: pd.DataFrame,
        stats: Sequence[PosteriorStatistic] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Return posterior statistics for each target.

        Args:
            candidates: The candidate points in experimental representation.
                For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
            stats: Sequence indicating which statistics to compute. Also accepts
                floats, for which the corresponding quantile point will be computed.

        Raises:
            ModelNotTrainedError: When called before the model has been trained.
            ValueError: If a requested quantile is outside the open interval (0,1).
            TypeError: If the posterior utilized by the surrogate does not support
                a requested statistic.

        Returns:
            A dataframe with posterior statistics for each target and candidate.
        """
        if self._objective is None:
            raise ModelNotTrainedError(
                "The surrogate must be trained before a posterior can be computed."
            )

        stat: PosteriorStatistic
        for stat in (x for x in stats if isinstance(x, float)):
            if not 0.0 < stat < 1.0:
                raise ValueError(
                    f"Posterior quantile statistics can only be computed for quantiles "
                    f"between 0 and 1 (non-inclusive). Provided value: '{stat}' as "
                    f"part of '{stats=}'."
                )
        posterior = self.posterior(candidates)

        import torch

        result = pd.DataFrame(index=candidates.index)
        with torch.no_grad():
            for stat in stats:
                try:
                    if isinstance(stat, float):  # Calculate quantile statistic
                        stat_name = f"Q_{stat}"
                        vals = posterior.quantile(torch.tensor(stat))
                    else:  # Calculate non-quantile statistic
                        stat_name = stat
                        vals = getattr(
                            posterior,
                            stat if stat not in ["std", "var"] else "variance",
                        )
                except (AttributeError, NotImplementedError) as e:
                    # We could arrive here because an invalid statistics string has
                    # been requested or because a quantile point has been requested,
                    # but the posterior type does not implement quantiles.
                    raise TypeError(
                        f"The utilized posterior of type "
                        f"'{posterior.__class__.__name__}' does not support the "
                        f"statistic associated with the requested input '{stat}'."
                    ) from e

                if stat == "std":
                    vals = torch.sqrt(vals)

                # Enforce a consistent shape
                # https://github.com/pytorch/botorch/issues/2958
                vals = vals.reshape((len(candidates), 1))

                result[
                    [
                        f"{name}_{stat_name}"
                        for name in self._objective._modeled_quantity_names
                    ]
                ] = vals.cpu().numpy()

        return result

    @override
    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Train the surrogate model on the provided data.

        Args:
            searchspace: The search space in which experiments are conducted.
            objective: The objective to be optimized.
            measurements: The training data in experimental representation.

        Raises:
            ValueError: If the search space contains task parameters but the selected
                surrogate model type does not support transfer learning.
            NotImplementedError: When using a continuous search space and a non-GP
                model.
        """
        # TODO: consider adding a validation step for `measurements`

        # Validate multi-target compatibility
        if objective.is_multi_output and not self.supports_multi_output:
            raise IncompatibleSurrogateError(
                f"You attempted to train a single-output surrogate in a "
                f"{len(objective.targets)}-target multi-output context. Either use "
                f"a proper multi-output surrogate or consider explicitly "
                f"replicating the current surrogate model using its "
                f"'.{self.replicate.__name__}' method."
            )

        # When the context is unchanged, no retraining is necessary
        if (
            searchspace == self._searchspace
            and objective == self._objective
            and hash(measurements) == self._measurements_hash
        ):
            return

        # Check if transfer learning capabilities are needed
        if (searchspace.n_tasks > 1) and (not self.supports_transfer_learning):
            raise ValueError(
                f"The search space contains task parameters but the selected "
                f"surrogate model type ({self.__class__.__name__}) does not "
                f"support transfer learning."
            )
        if (not searchspace.continuous.is_empty) and (
            "GaussianProcess" not in self.__class__.__name__
        ):
            raise NotImplementedError(
                "Continuous search spaces are currently only supported by GPs."
            )

        # Block partial measurements
        handle_missing_values(measurements, [t.name for t in objective.targets])

        # Remember the training context
        self._searchspace = searchspace
        self._objective = objective
        self._measurements_hash = hash(measurements)

        # Create context-specific transformations
        self._input_scaler = self._make_input_scaler(searchspace)
        self._output_scaler = self._make_output_scaler(objective, measurements)

        # Transform and fit
        # Note: The targets are only pre-transformed here. The remaining transformations
        #  are applied in form of BoTorch objectives. This has the consequence that:
        #  * The trained surrogate model can be called with pre-transformed target
        #    values, enabling predictions with input from the pre-transformed domain
        #   (this allows us to control precisely on which level the model is placed)
        #  * The main transformation is part of the computational backpropagation graph
        pre_transformed = objective._pre_transform(measurements, allow_extra=True)
        train_x_comp_rep, train_y_tensor = to_tensor(
            searchspace.transform(measurements, allow_extra=True), pre_transformed
        )
        train_x = self._input_scaler.transform(train_x_comp_rep)
        train_y = (
            train_y_tensor
            if self._output_scaler is _IDENTITY_TRANSFORM
            else self._output_scaler(train_y_tensor)[0]
        )

        self._fit(train_x, train_y)

    @abstractmethod
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Perform the actual fitting logic."""

    @override
    def __str__(self) -> str:
        fields = [
            to_string(
                "Supports Transfer Learning",
                self.supports_transfer_learning,
                single_line=True,
            ),
        ]
        return to_string(self.__class__.__name__, *fields)


@define
class IndependentGaussianSurrogate(Surrogate, ABC):
    """A surrogate base class providing independent Gaussian posteriors."""

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> GPyTorchPosterior:
        import torch
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        # Construct the Gaussian posterior from the estimated first and second moment
        mean, var = self._estimate_moments(candidates_comp_scaled)
        mvn = MultivariateNormal(mean, torch.diag_embed(var))
        return GPyTorchPosterior(mvn)

    @abstractmethod
    def _estimate_moments(
        self, candidates_comp_scaled: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        """Estimate first and second moments of the Gaussian posterior."""


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
