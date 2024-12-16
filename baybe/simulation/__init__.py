"""Functionality to "simulate" Bayesian DOE given a lookup mechanism.

The term "simulation" can have two slightly different interpretations, depending on the
applied context:

*   It can refer to "backtesting" a particular DOE strategy on a fixed (finite)
    dataset. In this context, "simulation" means investigating what experimental
    trajectory we would have observed had we applied the recommender in a certain
    defined context and restricted the possible parameter configurations to those
    contained in the dataset.

*   It can refer to the simulation of an actual DOE loop (i.e. recommending experiments
    and retrieving the corresponding measurements) where the loop closure is realized
    in the form of a callable (black-box) function that can be queried during the
    optimization to provide target values.
"""

from baybe.simulation.core import simulate_experiment
from baybe.simulation.scenarios import simulate_scenarios
from baybe.simulation.transfer_learning import simulate_transfer_learning

__all__ = [
    "simulate_experiment",
    "simulate_scenarios",
    "simulate_transfer_learning",
]
