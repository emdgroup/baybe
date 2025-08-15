"""Validation tests for Campaign."""

import pytest

from baybe import Campaign
from baybe.objectives import ParetoObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

_targets = (
    NumericalTarget("n1", "MAX"),
    NumericalTarget("n2", "MAX"),
    NumericalTarget("n3", "MAX"),
)
_parameters = (
    NumericalDiscreteParameter("n1", (1, 2, 3)),
    NumericalDiscreteParameter("bla", (4, 5, 6)),
    NumericalDiscreteParameter("n2", (7, 8, 9)),
)
_searchspace = SearchSpace.from_product(_parameters)
_objective = ParetoObjective(_targets)
_context = pytest.raises(
    ValueError, match="appear multiple times: {(?:'n1', 'n2'|'n2', 'n1')}"
)


def test_overlapping_target_parameter_names_campaign():
    """Overlapping names between parameters and targets are not allowed."""
    with _context:
        Campaign(searchspace=_searchspace, objective=_objective)


@pytest.mark.parametrize(
    "recommender",
    [
        # Note that a non-predictive recommender is not tested here because they do not
        # support objectives and are not passed an objective if they are part of
        # meta-recommenders.
        pytest.param(BayesianRecommender(), id="pure"),
        pytest.param(
            TwoPhaseMetaRecommender(initial_recommender=BayesianRecommender()),
            id="meta",
        ),
    ],
)
def test_overlapping_target_parameter_names_stateless(recommender):
    """Overlapping names between parameters and targets are not allowed."""
    with _context:
        recommender.recommend(
            2, searchspace=_searchspace, objective=_objective, measurements=None
        )
