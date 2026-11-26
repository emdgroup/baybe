"""Risk measure insights."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import gt, in_, instance_of
from pandas.api.types import is_numeric_dtype

from baybe.campaign import Campaign
from baybe.exceptions import ModelNotTrainedError, NoMeasurementsError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.searchspace import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.composite import CompositeSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from botorch.acquisition.risk_measures import RiskMeasureMCObjective
    from botorch.models.transforms.input import InputPerturbation
    from torch import Tensor

RiskMeasureName = Literal["Expectation", "VaR", "CVaR", "WorstCase"]
"""The names of the supported risk measures."""

_ALPHA_RISK_MEASURES = ("VaR", "CVaR")
"""The risk measures that are parameterized by a risk level."""

_NO_NOISE_MESSAGE = "At least one parameter must be subject to input noise."
"""The error message for missing input noise."""


def _convert_input_noise(value: object, /) -> object:
    """Copy the noise specification and convert standard deviations to floats.

    Args:
        value: The noise specification.

    Returns:
        A copy of the specification with real-valued (but not Boolean) standard
        deviations converted to floats. Unsupported input is returned unchanged so
        that it can be rejected by the validator.
    """
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if not isinstance(value, Mapping):
        return value
    return {
        k: float(v)
        if isinstance(v, int | np.integer | np.floating) and not isinstance(v, bool)
        else v
        for k, v in value.items()
    }


def _unwrap_surrogate(value: SurrogateProtocol, /) -> SurrogateProtocol:
    """Unwrap composite surrogates containing a single surrogate.

    Args:
        value: The surrogate to be unwrapped.

    Returns:
        The single surrogate contained in a composite surrogate or the unchanged input.

    Raises:
        ModelNotTrainedError: If the composite surrogate has not been trained.
        NotImplementedError: If the composite surrogate has multiple outputs.
    """
    if not isinstance(value, CompositeSurrogate):
        return value
    if value._modeled_quantity_names is None:
        raise ModelNotTrainedError(
            "The surrogate must be trained before risk measures can be computed."
        )
    if len(surrogates := value._surrogates_flat) != 1:
        raise NotImplementedError(
            "Risk measures are currently only supported for single-output surrogates."
        )
    return surrogates[0]


@define(frozen=True, kw_only=True, eq=False)
class RobustnessInsight:
    """Class for computing risk measures of a trained surrogate model.

    The risk measures quantify how well the surrogate model's predictions hold up
    when the inputs are subject to Gaussian noise. The input noise renders the objective
    value stochastic, and the risk measures offer a way to quantify the
    distribution of the resulting noisy objective values. Currently available risk
    measures include ``Expectation``, ``VaR``, ``CVaR``, and ``WorstCase`` ordered by
    their level of pessimism.

    Note that ``VaR`` and ``CVaR`` require the ``alpha`` parameter to be specified; VaR
    computes the (1-alpha)-quantile of the noise impacted objective distribution,
    while CVaR computes the expected value of the 1-alpha tail.

    This class evaluates the BoTorch risk measures, which assume that the objective is
    to be maximized. This is taken care of internally by the implementation, so users
    do not need to manually adjust for maximization.

    The current implementation is restricted to single-output Gaussian process
    surrogates and to zero-mean Gaussian input noise, specified via standard
    deviations or a covariance matrix. Continuous, discrete and hybrid search spaces
    are supported. Since only numerical parameters can be perturbed, input noise can
    only be specified for numerical parameters. All other parameters (e.g.,
    categorical ones) are kept fixed at their given values.

    Perturbed inputs are clipped to the box bounds of the computational
    representation of the search space. For discrete parameters, these are the minimum
    and maximum of the (constraint-filtered) search space entries, so they can be
    tighter than the parameter's declared range. Perturbed discrete values are not
    snapped back to the grid, and perturbed continuous values may violate constraints.

    Warning:
        All candidates passed to :meth:`evaluate` are processed in a single pass
        without chunking, so memory consumption grows linearly with their number (and
        with ``n_perturbations`` and ``n_posterior_samples``). As a rough guide,
        10,000 candidates with the default settings required about 2.5 GB. Split large
        sets of candidates manually.

    Note:
        The randomness involved in the computation is controlled via the BayBE random
        seed settings. The insight holds a reference to the given surrogate and does
        not copy it. If the surrogate is refitted later, e.g., by a campaign after
        adding new measurements, the insight reflects the refitted model.
    """

    alpha: float | None = field(default=None)
    """The risk level in (0, 1] for the risk measures ``VaR`` and ``CVaR``.

    The risk measure is evaluated on the worst ``1 - alpha`` fraction of outcomes, i.e.,
    ``alpha = 1`` corresponds to the worst case."""

    input_noise: Mapping[str, float] | pd.DataFrame = field(
        converter=_convert_input_noise
    )
    """The zero-mean Gaussian input noise.

    Either a mapping from parameter names to standard deviations, describing
    independent noise, or a covariance matrix as square dataframe whose index and
    columns are parameter names. Only numerical parameters can be subject to noise.
    Parameters that are not listed are noise-free."""

    n_perturbations: int = field(default=64, validator=(instance_of(int), gt(0)))
    """The number of input perturbations used to evaluate the risk measure.

    Quasi-random Sobol sequences are used, for which powers of two are recommended."""

    n_posterior_samples: int = field(default=128, validator=(instance_of(int), gt(0)))
    """The number of posterior samples used to evaluate the risk measure.

    Quasi-random Sobol sequences are used, for which powers of two are recommended."""

    risk_measure: RiskMeasureName = field(validator=in_(get_args(RiskMeasureName)))
    """The risk measure to be computed."""

    surrogate: SurrogateProtocol = field(converter=_unwrap_surrogate)
    """The trained surrogate model.

    Composite surrogates containing a single surrogate are unwrapped."""

    @staticmethod
    def _validate_covariance(covariance: pd.DataFrame, /) -> None:
        """Validate a noise covariance matrix.

        Args:
            covariance: The covariance matrix to be validated.

        Raises:
            ValueError: If the matrix is not a valid covariance matrix.
        """
        names = list(covariance.index)
        if not names:
            raise ValueError(_NO_NOISE_MESSAGE)
        if len(set(names)) != len(names) or set(names) != set(covariance.columns):
            raise ValueError(
                "The index and columns of the covariance matrix must contain the "
                "same, unique parameter names."
            )
        if not all(is_numeric_dtype(dtype) for dtype in covariance.dtypes):
            raise ValueError("The covariance matrix must be numeric.")
        matrix = covariance.loc[names, names].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("The covariance matrix must only contain finite values.")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("The covariance matrix must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() < -1e-8 * max(1.0, np.abs(eigenvalues).max()):
            raise ValueError("The covariance matrix must be positive semi-definite.")
        if eigenvalues.max() <= 0:
            raise ValueError(_NO_NOISE_MESSAGE)

    @input_noise.validator
    def _validate_input_noise(self, _, value: object) -> None:  # noqa: DOC101, DOC103
        """Validate the input noise specification.

        Raises:
            NotImplementedError: If the noise is not given as mapping or dataframe.
            ValueError: If the noise is empty or invalid.
        """
        if isinstance(value, pd.DataFrame):
            self._validate_covariance(value)
        elif isinstance(value, Mapping):
            if not value:
                raise ValueError(_NO_NOISE_MESSAGE)
            for name, std in value.items():
                if not isinstance(std, float) or not (math.isfinite(std) and std > 0):
                    raise ValueError(
                        f"The noise standard deviation of parameter '{name}' must be "
                        f"a positive, finite number. Given: {std!r}."
                    )
        else:
            raise NotImplementedError(
                f"Only Gaussian input noise specified as a mapping from parameter "
                f"names to standard deviations or as a covariance dataframe is "
                f"supported. Given: {type(value).__name__}."
            )

    @surrogate.validator
    def _validate_surrogate(self, _, value: SurrogateProtocol) -> None:  # noqa: DOC101, DOC103
        """Validate the surrogate.

        Raises:
            NotImplementedError: If the surrogate is not supported.
            ModelNotTrainedError: If the surrogate has not been trained.
        """
        if not isinstance(value, GaussianProcessSurrogate):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' currently only supports surrogates of "
                f"type '{GaussianProcessSurrogate.__name__}'. "
                f"Given: {type(value).__name__}."
            )
        if value._searchspace is None:
            raise ModelNotTrainedError(
                "The surrogate must be trained before risk measures can be computed."
            )

    def __attrs_post_init__(self) -> None:
        """Validate the settings that depend on multiple attributes.

        Raises:
            ValueError: If the noise refers to unknown or non-numerical parameters or
                the risk level does not match the risk measure.
        """
        params = {p.name: p for p in self._searchspace.parameters}
        noisy = set(self._noise_covariance.index)
        if unknown := noisy - set(params):
            raise ValueError(
                f"The input noise refers to parameters that are not part of the "
                f"search space: {sorted(unknown)}. Available parameters: "
                f"{self._parameter_names}."
            )
        if non_numerical := {n for n in noisy if not params[n].is_numerical}:
            raise ValueError(
                f"Input noise can only be specified for numerical parameters. "
                f"Invalid parameters: {sorted(non_numerical)}."
            )

        if self.risk_measure in _ALPHA_RISK_MEASURES:
            if self.alpha is None or not 0 < self.alpha <= 1:
                raise ValueError(
                    f"The risk measure '{self.risk_measure}' requires a risk level "
                    f"in the interval (0, 1]. Given: {self.alpha=}."
                )
        elif self.alpha is not None:
            raise ValueError(
                f"The risk measure '{self.risk_measure}' does not use a risk level "
                f"but {self.alpha=} was given."
            )

    @property
    def _gp(self) -> GaussianProcessSurrogate:
        """The surrogate as Gaussian process."""
        assert isinstance(self.surrogate, GaussianProcessSurrogate)
        return self.surrogate

    @property
    def _noise_covariance(self) -> pd.DataFrame:
        """The covariance matrix of the input noise."""
        if isinstance(self.input_noise, pd.DataFrame):
            names = list(self.input_noise.index)
            return self.input_noise.loc[names, names].astype(float)
        names = list(self.input_noise)
        return pd.DataFrame(
            np.diag([self.input_noise[n] ** 2 for n in names]),
            index=names,
            columns=names,
        )

    @property
    def _parameter_names(self) -> list[str]:
        """The names of the search space parameters."""
        return [p.name for p in self._searchspace.parameters]

    @property
    def _searchspace(self) -> SearchSpace:
        """The search space the surrogate was trained on."""
        assert self._gp._searchspace is not None
        return self._gp._searchspace

    def _evaluate_comp(
        self, candidates_comp: Tensor, /, *, seed: int | None = None
    ) -> Tensor:
        """Evaluate the risk measure for candidates in computational representation.

        The computation is differentiable with respect to the candidates. With a fixed
        seed, the perturbations and the posterior base samples are identical across
        calls, which makes the result a deterministic function of the candidates
        (sample average approximation), as required by gradient-based optimizers.

        Note:
            The method is internal groundwork for gradient-based searches of robust
            inputs. It is neither bounds-checked nor memory-bounded; see
            :meth:`evaluate` for the public entry point.

        Args:
            candidates_comp: A ``n x d``-dimensional tensor of candidates.
            seed: The seed for drawing the perturbations and the posterior base
                samples. If ``None``, they are drawn randomly in each call (but
                reproducibly under the BayBE random seed settings).

        Returns:
            An ``n``-dimensional tensor of risk measure values.
        """
        import torch
        from botorch.sampling.get_sampler import get_sampler

        perturb = self._make_input_perturbation(candidates_comp, seed=seed)

        # Each candidate is its own q-batch of perturbed points, so that the risk
        # measure is computed over the joint posterior of the perturbed inputs
        posterior = self._gp._posterior_comp(
            perturb.transform(candidates_comp.unsqueeze(-2))
        )

        # Use the same sampler selection mechanism as the acquisition functions. The
        # seed is offset to decorrelate the base samples from the perturbations,
        # which are drawn from a Sobol sequence with the original seed.
        sampler = get_sampler(
            posterior,  # type: ignore[arg-type]  # pyrefly: ignore[bad-argument-type]  # BoTorch annotates the dispatch only for TorchPosterior
            torch.Size([self.n_posterior_samples]),
            seed=None if seed is None else seed + 1,
        )
        return self._make_risk_measure()(sampler(posterior)).mean(dim=0).squeeze(-1)

    def _make_risk_measure(self) -> RiskMeasureMCObjective:
        """Make the callable BoTorch risk measure operating on posterior samples.

        The callable maps samples of shape ``n_samples x batch x n_perturbations x 1``
        to risk measure values of shape ``n_samples x batch x 1``. The objective is
        applied to the samples as preprocessing step.

        Returns:
            The BoTorch risk measure corresponding to :attr:`risk_measure`.
        """
        from botorch.acquisition.risk_measures import CVaR, Expectation, VaR, WorstCase

        assert self._gp._objective is not None
        n_w = self.n_perturbations
        preprocessing_function = self._gp._objective.to_botorch()

        if self.risk_measure in _ALPHA_RISK_MEASURES:
            assert self.alpha is not None  # guaranteed by validation
            alpha_measures = {"VaR": VaR, "CVaR": CVaR}
            return alpha_measures[self.risk_measure](
                alpha=self.alpha,
                n_w=n_w,
                preprocessing_function=preprocessing_function,
            )

        plain_measures = {"Expectation": Expectation, "WorstCase": WorstCase}
        return plain_measures[self.risk_measure](
            n_w=n_w, preprocessing_function=preprocessing_function
        )

    def _make_input_perturbation(
        self, candidates_comp: Tensor, /, *, seed: int | None
    ) -> InputPerturbation:
        """Make the input transform that perturbs candidates with the input noise.

        Args:
            candidates_comp: A ``n x d``-dimensional tensor of candidates in
                computational representation, which determines the dtype of the
                perturbations.
            seed: The seed for drawing the perturbations. If ``None``, they are drawn
                randomly.

        Returns:
            The transform expanding each candidate into ``n_perturbations`` perturbed
            candidates, which are clipped to the bounds of the search space.
        """
        import torch
        from botorch.models.transforms.input import InputPerturbation
        from botorch.utils.sampling import draw_sobol_normal_samples

        # Draw the perturbations once so that all candidates are treated equally.
        # Using an eigendecomposition for the factorization of the covariance matrix
        # also supports singular matrices, e.g., for perfectly correlated noise.
        # Parameters without noise (including non-numerical ones) are not perturbed.
        covariance = self._noise_covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.to_numpy(dtype=float))
        factor = torch.tensor(
            eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None)),
            dtype=candidates_comp.dtype,
        )
        columns = [
            self._searchspace.comp_rep_columns.index(name) for name in covariance.index
        ]
        perturbations = torch.zeros(
            self.n_perturbations, candidates_comp.shape[-1], dtype=candidates_comp.dtype
        )
        perturbations[:, columns] = (
            draw_sobol_normal_samples(
                d=len(columns),
                n=self.n_perturbations,
                dtype=candidates_comp.dtype,
                seed=seed,
            )
            @ factor.T
        )
        # Perturbed inputs are clipped to the bounds of the search space
        bounds = to_tensor(self._searchspace.comp_rep_bounds)
        return InputPerturbation(perturbations, bounds=bounds.to(candidates_comp))

    def evaluate(self, candidates: pd.DataFrame, /) -> pd.DataFrame:
        """Evaluate the risk measure for the given parameter configurations.

        Args:
            candidates: The candidate parameter configurations in experimental
                representation. They must lie within the bounds of the search space.

        Returns:
            The parameter configurations with an additional column named after the
            risk measure that contains the risk measure values.

        Raises:
            ValueError: If any candidate lies outside the bounds of the search space.
        """
        import torch

        candidates_comp = to_tensor(
            self._searchspace.transform(candidates, allow_extra=True)
        )
        bounds = to_tensor(self._searchspace.comp_rep_bounds).to(candidates_comp)
        outside = (candidates_comp < bounds[0]) | (candidates_comp > bounds[1])
        if outside.any():
            columns = [
                c
                for c, is_outside in zip(
                    self._searchspace.comp_rep_columns, outside.any(dim=0)
                )
                if is_outside
            ]
            raise ValueError(
                f"All candidates must lie within the bounds of the search space. "
                f"Violated for the computational representation columns: {columns}."
            )

        result = candidates[self._parameter_names].copy()
        with torch.no_grad():
            result[self.risk_measure] = self._evaluate_comp(candidates_comp).numpy()
        return result

    @classmethod
    def from_campaign(
        cls,
        campaign: Campaign,
        *,
        input_noise: Mapping[str, float] | pd.DataFrame,
        risk_measure: RiskMeasureName,
        alpha: float | None = None,
        n_perturbations: int = 64,
        n_posterior_samples: int = 128,
    ) -> RobustnessInsight:
        """Create a risk measure insight from a campaign.

        Args:
            campaign: A campaign using a surrogate-based recommender.
            input_noise: See :attr:`input_noise`.
            risk_measure: See :attr:`risk_measure`.
            alpha: See :attr:`alpha`.
            n_perturbations: See :attr:`n_perturbations`.
            n_posterior_samples: See :attr:`n_posterior_samples`.

        Returns:
            The risk measure insight.

        Raises:
            NoMeasurementsError: If the campaign does not contain any measurements.
        """
        if campaign.measurements.empty:
            raise NoMeasurementsError("The campaign does not contain any measurements.")

        return cls(
            surrogate=campaign.get_surrogate(),
            input_noise=input_noise,
            risk_measure=risk_measure,
            alpha=alpha,
            n_perturbations=n_perturbations,
            n_posterior_samples=n_posterior_samples,
        )

    @classmethod
    def from_recommender(
        cls,
        recommender: RecommenderProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        *,
        input_noise: Mapping[str, float] | pd.DataFrame,
        risk_measure: RiskMeasureName,
        alpha: float | None = None,
        n_perturbations: int = 64,
        n_posterior_samples: int = 128,
    ) -> RobustnessInsight:
        """Create a risk measure insight from a recommender.

        Args:
            recommender: A recommender using a surrogate model.
            searchspace: The searchspace for the recommender.
            objective: The objective for the recommender.
            measurements: The measurements for training the surrogate.
            input_noise: See :attr:`input_noise`.
            risk_measure: See :attr:`risk_measure`.
            alpha: See :attr:`alpha`.
            n_perturbations: See :attr:`n_perturbations`.
            n_posterior_samples: See :attr:`n_posterior_samples`.

        Returns:
            The risk measure insight.

        Raises:
            TypeError: If the recommender has no ``get_surrogate`` method.
        """
        if not hasattr(recommender, "get_surrogate"):
            raise TypeError(
                f"The provided recommender does not provide a surrogate model. "
                f"'{cls.__name__}' needs a surrogate model and thus only works with "
                f"model-based recommenders."
            )

        return cls(
            surrogate=recommender.get_surrogate(searchspace, objective, measurements),
            input_noise=input_noise,
            risk_measure=risk_measure,
            alpha=alpha,
            n_perturbations=n_perturbations,
            n_posterior_samples=n_posterior_samples,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
