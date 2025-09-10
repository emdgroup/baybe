"""Available acquisition functions."""

from __future__ import annotations

import gc
import math
from abc import ABC
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from attr.converters import optional as optional_c
from attr.validators import optional as optional_v
from attrs import AttrsInstance, define, field, fields
from attrs.validators import gt, instance_of, le
from typing_extensions import override

from baybe.acquisition.base import AcquisitionFunction
from baybe.searchspace import SearchSpace
from baybe.utils.basic import classproperty, convert_to_float
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod, sample_numerical_df
from baybe.utils.validation import finite_float


########################################################################################
### Active Learning
@define(frozen=True)
class qNegIntegratedPosteriorVariance(AcquisitionFunction):
    """Monte Carlo based negative integrated posterior variance.

    This is typically used for active learning as it is a measure for global model
    uncertainty.
    """

    abbreviation: ClassVar[str] = "qNIPV"

    sampling_n_points: int | None = field(
        validator=optional_v([instance_of(int), gt(0)]),
        default=None,
    )
    """Number of data points sampled for integrating the posterior.

    Cannot be used if `sampling_fraction` is not `None`."""

    sampling_fraction: float | None = field(
        converter=optional_c(float),
        validator=optional_v([gt(0.0), le(1.0)]),
    )
    """Fraction of data sampled for integrating the posterior.

    Cannot be used if `sampling_n_points` is not `None`."""

    sampling_method: DiscreteSamplingMethod = field(
        converter=DiscreteSamplingMethod, default=DiscreteSamplingMethod.Random
    )
    """Sampling strategy used for integrating the posterior."""

    @sampling_fraction.default
    def _default_sampling_fraction(self) -> float | None:
        """If no sampling quantities are provided, use all points by default."""
        return 1.0 if self.sampling_n_points is None else None

    @sampling_fraction.validator
    def _validate_sampling_fraction(self, attr, value) -> None:
        """If both sampling quantities are specified, raise an error."""
        if None not in (self.sampling_fraction, self.sampling_n_points):
            raise ValueError(
                f"For '{self.__class__.__name__}', the attributes '{attr.name}' and "
                f"'{fields(self.__class__).sampling_n_points.name}' cannot "
                f"be specified at the same time."
            )

    @override
    @classproperty
    def _non_botorch_attrs(cls: type[AttrsInstance]) -> tuple[str, ...]:
        flds = fields(cls)
        return (
            flds.sampling_n_points.name,
            flds.sampling_method.name,
            flds.sampling_fraction.name,
        )

    def get_integration_points(self, searchspace: SearchSpace) -> pd.DataFrame:
        """Sample points from a search space for integration purposes.

        Sampling of the discrete part can be controlled via 'sampling_method', but
        sampling of the continuous part will always be random.

        Args:
            searchspace: The searchspace from which to sample integration points.

        Returns:
            The sampled data points.

        Raises:
            ValueError: If the search space is purely continuous and
                'sampling_n_points' was not provided.
        """
        # TODO: Move the core logic to `SearchSpace` and ``Subspace*`` classes

        sampled_parts: list[pd.DataFrame] = []
        n_candidates: int | None = None

        # Discrete part
        if not searchspace.discrete.is_empty:
            candidates_discrete = searchspace.discrete.comp_rep
            n_candidates = self.sampling_n_points or math.ceil(
                self.sampling_fraction * len(candidates_discrete)  # type: ignore[operator]
            )

            sampled_disc = sample_numerical_df(
                candidates_discrete, n_candidates, method=self.sampling_method
            )

            sampled_parts.append(sampled_disc)

        # Continuous part
        if not searchspace.continuous.is_empty:
            # If a discrete part has resulted in a particular choice for n_candidates,
            # take it. Otherwise, use the user specified number of points.
            if (n_candidates := n_candidates or self.sampling_n_points) is None:
                raise ValueError(
                    f"'{fields(self.__class__).sampling_n_points.name}' must be "
                    f"provided for '{self.__class__.__name__}' when sampling purely "
                    f"continuous search spaces."
                )
            sampled_conti = searchspace.continuous.sample_uniform(n_candidates)

            # Align indices if discrete part is present
            if len(sampled_parts) > 0:
                sampled_conti.index = sampled_parts[0].index
            sampled_parts.append(sampled_conti)

        # Combine different search space parts
        result = pd.concat(sampled_parts, axis=1)

        return result


########################################################################################
### Knowledge Gradient
@define(frozen=True)
class qKnowledgeGradient(AcquisitionFunction):
    """Monte Carlo based knowledge gradient.

    This acquisition function currently only supports purely continuous spaces.
    """

    abbreviation: ClassVar[str] = "qKG"

    num_fantasies: int = field(validator=[instance_of(int), gt(0)], default=128)
    """Number of fantasies to draw for approximating the knowledge gradient.

    More samples result in a better approximation, at the expense of both increased
    memory footprint and wall time."""


########################################################################################
### Posterior Statistics
@define(frozen=True)
class PosteriorMean(AcquisitionFunction):
    """Posterior mean."""

    abbreviation: ClassVar[str] = "PM"


@define(frozen=True)
class PosteriorStandardDeviation(AcquisitionFunction):
    """Posterior standard deviation."""

    abbreviation: ClassVar[str] = "PSTD"

    maximize: bool = field(default=True, validator=instance_of(bool))
    """If ``True``, points with maximum posterior standard deviation are selected.
    If ``False``, the acquisition function value is negated, yielding a selection
    with minimal posterior standard deviation."""


@define(frozen=True)
class qPosteriorStandardDeviation(AcquisitionFunction):
    """Monte Carlo based posterior standard deviation."""

    abbreviation: ClassVar[str] = "qPSTD"


########################################################################################
### Simple Regret
@define(frozen=True)
class qSimpleRegret(AcquisitionFunction):
    """Monte Carlo based simple regret."""

    abbreviation: ClassVar[str] = "qSR"


########################################################################################
### Expected Improvement
@define(frozen=True)
class ExpectedImprovement(AcquisitionFunction):
    """Analytical expected improvement."""

    abbreviation: ClassVar[str] = "EI"


@define(frozen=True)
class qExpectedImprovement(AcquisitionFunction):
    """Monte Carlo based expected improvement."""

    abbreviation: ClassVar[str] = "qEI"


@define(frozen=True)
class LogExpectedImprovement(AcquisitionFunction):
    """Logarithmic analytical expected improvement."""

    abbreviation: ClassVar[str] = "LogEI"


@define(frozen=True)
class qLogExpectedImprovement(AcquisitionFunction):
    """Logarithmic Monte Carlo based expected improvement."""

    abbreviation: ClassVar[str] = "qLogEI"


@define(frozen=True)
class qNoisyExpectedImprovement(AcquisitionFunction):
    """Monte Carlo based noisy expected improvement."""

    abbreviation: ClassVar[str] = "qNEI"

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


@define(frozen=True)
class qLogNoisyExpectedImprovement(AcquisitionFunction):
    """Logarithmic Monte Carlo based noisy expected improvement."""

    abbreviation: ClassVar[str] = "qLogNEI"

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


########################################################################################
### Probability of Improvement
@define(frozen=True)
class ProbabilityOfImprovement(AcquisitionFunction):
    """Analytical probability of improvement."""

    abbreviation: ClassVar[str] = "PI"


@define(frozen=True)
class qProbabilityOfImprovement(AcquisitionFunction):
    """Monte Carlo based probability of improvement."""

    abbreviation: ClassVar[str] = "qPI"


########################################################################################
### Upper Confidence Bound
@define(frozen=True)
class UpperConfidenceBound(AcquisitionFunction):
    """Analytical upper confidence bound."""

    abbreviation: ClassVar[str] = "UCB"

    beta: float = field(converter=float, validator=finite_float, default=0.2)
    """Trade-off parameter for mean and variance.

    * ``beta > 0``: Rewards uncertainty, takes more risk.
      Limit ``inf``: Pure exploration
    * ``beta < 0``: Punishes uncertainty, takes less risk.
      Limit ``-inf``: Pure exploitation
    * ``beta = 0``: Discards knowledge about uncertainty, i.e. neither rewards nor
      punishes it, is risk-neutral.
    """


@define(frozen=True)
class qUpperConfidenceBound(AcquisitionFunction):
    """Monte Carlo based upper confidence bound."""

    abbreviation: ClassVar[str] = "qUCB"

    beta: float = field(converter=float, validator=finite_float, default=0.2)
    """See :paramref:`UpperConfidenceBound.beta`."""


########################################################################################
### ThompsonSampling
@define(frozen=True)
class qThompsonSampling(qSimpleRegret):
    """Thomson sampling, implemented via simple regret. Inherently Monte Carlo based.

    This implementation exploits the fact that one-sample-based Thompson sampling
    (i.e. where the action probability is approximated using a single posterior sample)
    is equivalent to optimizing the Monte Carlo approximated posterior mean with
    sample size one. The latter can be achieved via `qSimpleRegret` and controlling
    its sample shape attribute.
    """

    abbreviation: ClassVar[str] = "qTS"

    n_mc_samples: int = field(default=1, init=False)
    """Number of Monte Carlo samples drawn from the posterior at each design point.

    Restring the the sample size to one allows us to emulate (one-sample based)
    Thompson sampling using the regular acquisition function machinery.
    """

    @override
    @classproperty
    def _non_botorch_attrs(cls: type[AttrsInstance]) -> tuple[str, ...]:
        flds = fields(cls)
        return (flds.n_mc_samples.name,)

    @override
    @classproperty
    def supports_batching(cls) -> bool:
        return False


########################################################################################
### Pareto Efficient Global Optimization (Chebyshev scalarization of targets)
@define(frozen=True)
class qLogNParEGO(AcquisitionFunction):
    """Pareto optimization via Chebyshev scalarization of the targets."""

    abbreviation: ClassVar[str] = "qLogNParEGO"
    supports_multi_output: ClassVar[bool] = True

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


########################################################################################
### Hypervolume Improvement
@define(frozen=True)
class _ExpectedHypervolumeImprovement(AcquisitionFunction, ABC):
    """Expected hypervolume improvement base class."""

    supports_multi_output: ClassVar[bool] = True

    reference_point: float | tuple[float, ...] | None = field(
        default=None, converter=optional_c(convert_to_float)
    )
    """The reference point for computing the hypervolume improvement.

    * When omitted, a default reference point is computed based on the provided data.
    * When specified as a float, the value is interpreted as a multiplicative factor
      determining the reference point location based on the difference between the best
      and worst target configuration in the provided data.
    * When specified as an iterable, the contained values are directly interpreted as
      the coordinates of the reference point.
    """

    @override
    @classproperty
    def _non_botorch_attrs(cls: type[AttrsInstance]) -> tuple[str, ...]:
        # While BoTorch's acquisition function also expects a `ref_point` argument,
        # the attribute defined here is more general and can hence not be directly
        # matched. Thus, we bypass the auto-matching mechanism and handle it manually.
        flds = fields(cls)
        return (flds.reference_point.name,)

    @staticmethod
    def compute_ref_point(
        array: npt.ArrayLike, maximize: npt.ArrayLike | None = None, factor: float = 0.1
    ) -> np.ndarray:
        """Compute a reference point for a given set of target configurations.

        The reference point is positioned relative to the worst point in the direction
        coming from the best point:

        * A factor of 0.0 results in the reference point being the worst point.
        * A factor > 0.0 moves the reference point further away from both worst and best
          points. (A factor of 1.0 exactly mirrors the best around the worst point.)
        * A factor < 0.0 moves the reference point closer to the best point.
          (A factor of -1.0 exactly places the reference point onto the best point.)

        Example:
            >>> from baybe.acquisition import qLogNEHVI

            >>> qLogNEHVI.compute_ref_point([[0, 10], [2, 20]], [True, True], 0.1)
            array([-0.2,  9. ])

            >>> qLogNEHVI.compute_ref_point([[0, 10], [2, 20]], [True, False], 0.2)
            array([-0.4, 22. ])

        Args:
            array: A 2-D array-like where each row represents a target configuration.
            maximize: An optional 1-D Boolean array indicating which targets are to be
                maximized.
            factor: A numeric value controlling the location of the reference point.

        Raises:
            ValueError: If the given target configuration array is not two-dimensional.
            ValueError: If the given Boolean array is not one-dimensional.

        Returns:
            The computed reference point.
        """
        if np.ndim(array) != 2:
            raise ValueError(
                "The specified data array must have exactly two dimensions."
            )
        if (maximize is not None) and (np.ndim(maximize) != 1):
            raise ValueError(
                "The specified Boolean array must have exactly one dimension."
            )

        # Convert arrays and set default optimization direction
        array = np.asarray(array)
        if maximize is None:
            maximize = [True for _ in range(array.shape[1])]
        maximize = np.where(maximize, 1.0, -1.0)

        # Compute bounds
        array = array * maximize[None, :]
        min = np.min(array, axis=0)
        max = np.max(array, axis=0)

        return (min - factor * (max - min)) * maximize


@define(frozen=True)
class qExpectedHypervolumeImprovement(_ExpectedHypervolumeImprovement):
    """Monte Carlo based expected hypervolume improvement."""

    abbreviation: ClassVar[str] = "qEHVI"

    alpha: float | None = field(default=None, converter=optional_c(float))
    """An optional threshold parameter controlling the shape of the partitioning."""

    @override
    @classproperty
    def _non_botorch_attrs(cls: type[AttrsInstance]) -> tuple[str, ...]:
        # BoTorch's acquisition functions do not directly expect an `alpha` argument
        # but a partitioning object derived from it, which we create on the fly.
        # Thus, we bypass the auto-matching mechanism and handle it manually.
        flds = fields(cls)
        return _ExpectedHypervolumeImprovement._non_botorch_attrs + (flds.alpha.name,)


@define(frozen=True)
class qLogExpectedHypervolumeImprovement(_ExpectedHypervolumeImprovement):
    """Logarithmic Monte Carlo based expected hypervolume improvement."""

    abbreviation: ClassVar[str] = "qLogEHVI"

    alpha: float | None = field(default=None, converter=optional_c(float))
    """An optional threshold parameter controlling the shape of the partitioning."""

    @override
    @classproperty
    def _non_botorch_attrs(cls: type[AttrsInstance]) -> tuple[str, ...]:
        # BoTorch's acquisition functions do not directly expect an `alpha` argument
        # but a partitioning object derived from it, which we create on the fly.
        # Thus, we bypass the auto-matching mechanism and handle it manually.
        flds = fields(cls)
        return _ExpectedHypervolumeImprovement._non_botorch_attrs + (flds.alpha.name,)


@define(frozen=True)
class qNoisyExpectedHypervolumeImprovement(_ExpectedHypervolumeImprovement):
    """Monte Carlo based noisy expected hypervolume improvement."""

    abbreviation: ClassVar[str] = "qNEHVI"

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


@define(frozen=True)
class qLogNoisyExpectedHypervolumeImprovement(_ExpectedHypervolumeImprovement):
    """Logarithmic Monte Carlo based noisy expected hypervolume improvement."""

    abbreviation: ClassVar[str] = "qLogNEHVI"

    prune_baseline: bool = field(default=True, validator=instance_of(bool))
    """Auto-prune candidates that are unlikely to be the best."""


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
