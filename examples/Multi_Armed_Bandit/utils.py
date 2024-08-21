from collections.abc import Iterable

from scipy.integrate import quad
from scipy.stats import rv_continuous


def max_rv_distribution(
    dist_params: list[Iterable[float]], dist: rv_continuous
) -> list[float]:
    """Calculate the distribution of being the maximum RV in a set of independt RVs."""
    res = []
    for i, params in enumerate(dist_params):

        def integrand(x):
            product = dist.pdf(x, *params)
            for j, other_params in enumerate(dist_params):
                if j != i:
                    product *= dist.cdf(x, *other_params)
            return product

        probability, _ = quad(integrand, dist.a, dist.b)
        res.append(probability)
    return res
