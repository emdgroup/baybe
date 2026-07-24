"""Hypothesis strategies for surrogates."""

import hypothesis.strategies as st

from baybe.surrogates.gaussian_process.components.fit_criterion import FitCriterion
from baybe.surrogates.gaussian_process.components.likelihood import (
    LazyGaussianLikelihoodFactory,
)
from baybe.surrogates.gaussian_process.components.mean import LazyConstantMeanFactory
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEFitCriterionFactory,
    BayBELikelihoodFactory,
    BayBEMeanFactory,
)
from baybe.surrogates.gaussian_process.presets.chen import (
    ChenLikelihoodFactory,
    ChenMeanFactory,
)
from baybe.surrogates.gaussian_process.presets.edbo import (
    EDBOLikelihoodFactory,
    EDBOMeanFactory,
)
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOMeanFactory,
)
from tests.hypothesis_strategies.kernels import kernels

_MEAN_FACTORIES = [
    LazyConstantMeanFactory,
    BayBEMeanFactory,
    ChenMeanFactory,
    EDBOMeanFactory,
    SmoothedEDBOMeanFactory,
]

_LIKELIHOOD_FACTORIES = [
    LazyGaussianLikelihoodFactory,
    BayBELikelihoodFactory,
    ChenLikelihoodFactory,
    EDBOLikelihoodFactory,
]


def gaussian_process_surrogates():
    """A strategy generating GP surrogate instances with varying factory configurations.

    Each factory field is independently either ``None`` (defer to the BayBE default at
    fit time) or an explicit serializable value, covering all partial configurations.
    """
    return st.builds(
        GaussianProcessSurrogate,
        kernel_or_factory=st.one_of(st.none(), kernels()),
        mean_or_factory=st.one_of(
            st.none(),
            st.sampled_from([cls() for cls in _MEAN_FACTORIES]),
        ),
        likelihood_or_factory=st.one_of(
            st.none(),
            st.sampled_from([cls() for cls in _LIKELIHOOD_FACTORIES]),
        ),
        fit_criterion_or_factory=st.one_of(
            st.none(),
            st.sampled_from(FitCriterion),
            st.builds(BayBEFitCriterionFactory),
        ),
    )
