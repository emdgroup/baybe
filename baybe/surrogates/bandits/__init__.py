"""Bandit surrogates."""

from baybe.surrogates.bandits.base import MultiArmedBanditSurrogate
from baybe.surrogates.bandits.beta_bernoulli import (
    BetaBernoulliMultiArmedBanditSurrogate,
)

__all__ = [
    "BetaBernoulliMultiArmedBanditSurrogate",
    "MultiArmedBanditSurrogate",
]
