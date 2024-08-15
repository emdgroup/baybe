from collections.abc import Iterable

from attrs import define
from scipy.stats import rv_continuous, rv_discrete


@define
class MultiArmedBanditModel:
    """Representation of a multi armed bandit."""

    real_distributions: list[rv_discrete | rv_continuous]
    """List of the reward distribution per arm."""

    def sample(self, arm_idxs: Iterable[int]):
        """Draw reward samples from the arms indexed in arm_idxs."""
        return [self.real_distributions[arm_idx].rvs() for arm_idx in arm_idxs]

    @property
    def means(self):
        """Return the real means of the reward distributions."""
        return [dist.stats(moments="m") for dist in self.real_distributions]
