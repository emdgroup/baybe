"""Available acquisition functions."""

import gc
import math
from typing import ClassVar

import pandas as pd
from attr.converters import optional as optional_c
from attr.validators import optional as optional_v
from attrs import define, field, fields
from attrs.validators import ge, gt, instance_of, le

from baybe.acquisition.base import AcquisitionFunction
from baybe.searchspace import SearchSpace
from baybe.utils.basic import classproperty
from baybe.utils.sampling_algorithms import (
    DiscreteSamplingMethod,
    sample_numerical_df,
)


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

    @classproperty
    def _non_botorch_attrs(cls) -> tuple[str, ...]:
        flds = fields(qNegIntegratedPosteriorVariance)
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

    beta: float = field(converter=float, validator=ge(0.0), default=0.2)
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """


@define(frozen=True)
class qUpperConfidenceBound(AcquisitionFunction):
    """Monte Carlo based upper confidence bound."""

    abbreviation: ClassVar[str] = "qUCB"

    beta: float = field(converter=float, validator=ge(0.0), default=0.2)
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """


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

    @classproperty
    def _non_botorch_attrs(cls) -> tuple[str, ...]:
        flds = fields(qThompsonSampling)
        return (flds.n_mc_samples.name,)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
