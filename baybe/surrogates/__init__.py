"""BayBE surrogates."""

from baybe.surrogates.bandits.base import MultiArmedBanditSurrogate
from baybe.surrogates.bandits.beta_bernoulli import (
    BetaBernoulliMultiArmedBanditSurrogate,
)
from baybe.surrogates.custom import CustomONNXSurrogate, register_custom_architecture
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.linear import BayesianLinearSurrogate
from baybe.surrogates.naive import MeanPredictionSurrogate
from baybe.surrogates.ngboost import NGBoostSurrogate
from baybe.surrogates.random_forest import RandomForestSurrogate

__all__ = [
    "register_custom_architecture",
    "BayesianLinearSurrogate",
    "BetaBernoulliMultiArmedBanditSurrogate",
    "CustomONNXSurrogate",
    "GaussianProcessSurrogate",
    "MeanPredictionSurrogate",
    "MultiArmedBanditSurrogate",
    "NGBoostSurrogate",
    "RandomForestSurrogate",
]
