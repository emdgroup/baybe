## Registering Custom Hooks

# This example demonstrates the basic mechanics of the
# {func}`register_hooks <baybe.utils.basic.register_hooks>` utility,
# which lets you hook into any callable of your choice:
# * We define a hook that is compatible with the general
#   {meth}`RecommenderProtocol.recommend <baybe.recommenders.base.RecommenderProtocol.recommend>`
#   interface,
# * attach it to a recommender,
# * and watch it take action.


### Imports


from dataclasses import dataclass
from time import perf_counter

from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.utils.basic import register_hooks

### Defining the Hooks

# We start by defining a simple hook that lets us inspect the names of the parameters
# involved in the recommendation process.
# For this purpose, we match its signature to that of
# {meth}`RecommenderProtocol.recommend <baybe.recommenders.base.RecommenderProtocol.recommend>`:
#

# ```{admonition} Signature components
# :class: important
# Note that, if provided, annotations must match **exactly** those of the target signature.
# However, annotations are completely optional
# — only the names/order of the signature parameters and their defaults matter.
#
# For the sake of demonstration, we only provide the annotation for the
# relevant `searchspace` parameter of the hook in the example below.
# ```


def print_parameter_names_hook(
    self,
    batch_size,
    searchspace: SearchSpace,
    objective=None,
    measurements=None,
):
    """Print the names of the parameters spanning the search space."""
    print(f"Search space parameters: {[p.name for p in searchspace.parameters]}")


# Additionally, we set up a class that provides a combination of hooks for measuring
# the time needed to compute the recommendations:


@dataclass
class ElapsedTimePrinter:
    """Helper class for measuring the time between two calls."""

    last_call_time: float | None = None

    def start(
        instance,
        self,
        batch_size,
        searchspace: SearchSpace,
        objective=None,
        measurements=None,
    ):
        """Start the timer."""
        instance.last_call_time = perf_counter()

    def measure(
        instance,
        self,
        batch_size,
        searchspace: SearchSpace,
        objective=None,
        measurements=None,
    ):
        """Measure the elapsed time."""
        if instance.last_call_time is None:
            raise RuntimeError("Must call `start` first!")
        print(f"Elapsed time: {perf_counter() - instance.last_call_time}")


### Monkeypatching

# Next, we create our recommender and monkeypatch its `recommend` method:
timer = ElapsedTimePrinter()
recommender = RandomRecommender()
RandomRecommender.recommend = register_hooks(
    RandomRecommender.recommend,
    pre_hooks=[print_parameter_names_hook, timer.start],
    post_hooks=[timer.measure],
)

### Triggering the Hooks

# When we now apply the recommender in a specific context, we immediately see the
# effect of the hooks:

temperature = NumericalDiscreteParameter("Temperature", values=[90, 105, 120])
concentration = NumericalDiscreteParameter("Concentration", values=[0.057, 0.1, 0.153])
searchspace = SearchSpace.from_product([temperature, concentration])
recommendation = recommender.recommend(batch_size=3, searchspace=searchspace)
