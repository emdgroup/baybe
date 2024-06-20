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
from types import MethodType

from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.utils import register_hooks

### Defining the Hooks

# We start by defining a simple hook that lets us inspect the names of the parameters
# involved in the recommendation process.
# For this purpose, we match its signature to that of
# {meth}`RecommenderProtocol.recommend <baybe.recommenders.base.RecommenderProtocol.recommend>`:
#

# ```{admonition} Signature components
# :class: note
# Note that you are flexible in designing the signature of your hooks.
# For instance, function parameters and type annotations that you do not need in the
# hook body can simply be omitted.
# The exact rules to follow are described {func}`here <baybe.utils.basic.register_hooks>`.
# ```


def print_parameter_names_hook(self: RandomRecommender, searchspace: SearchSpace):
    """Print the names of the parameters spanning the search space."""
    print(f"Recommender type: {self.__class__.__name__}")
    print(f"Search space parameters: {[p.name for p in searchspace.parameters]}")


# Additionally, we set up a class that provides a combination of hooks for measuring
# the time needed to compute the recommendations:


@dataclass
class ElapsedTimePrinter:
    """Helper class for measuring the time between two calls."""

    last_call_time: float | None = None

    def start(printer_instance):
        """Start the timer."""
        printer_instance.last_call_time = perf_counter()

    def measure(printer_instance, self: RandomRecommender):
        """Measure the elapsed time."""
        if printer_instance.last_call_time is None:
            raise RuntimeError("Must call `start` first!")
        elapsed = perf_counter() - printer_instance.last_call_time
        print(f"Consumed time of {self.__class__.__name__}: {elapsed}")


# ```{admonition} Hook instance vs. target instance
# :class: important
# Notice the difference between the object belonging to the hook-providing class
# (named `printer_instance`) and the object whose method we intend to override
# (named `self`). This distinction is necessary because of
# {ref}`the particular way <BOUND_METHODS>` we attach the hook below, which binds `self`
# to the object carrying the target callable as a method.
# ```


### Monkeypatching

# Next, we create our recommender and monkeypatch its `recommend` method:

timer = ElapsedTimePrinter()
recommender = RandomRecommender()
recommender.recommend = MethodType(
    register_hooks(
        RandomRecommender.recommend,
        pre_hooks=[print_parameter_names_hook, timer.start],
        post_hooks=[timer.measure],
    ),
    recommender,
)

# (BOUND_METHODS)=
# ```{admonition} Bound methods
# :class: important
# Note that the explicit binding via `MethodType` above is required because we
# decorate the (unbound) `RandomRecommender.recommend` **function** with our hooks
# and attach it as an overridden **method** to the recommender instance.
#
# Alternatively, we could have ...
# * ... overridden the class callable itself via
#   `RandomRecommender.recommend = register_hooks(RandomRecommender.recommend, ...)`
#   which, however, would affect all instances of `RandomRecommender` or
# * ... used the bound method of the instance as reference via
#   `recommender.recommend = register_hooks(recommender.recommend, ...)` but then
#   the hooks would not have access to the recommender instance as it is not
#   explicitly exposed in the method's signature.
# ```

### Triggering the Hooks

# When we now apply the recommender in a specific context, we immediately see the
# effect of the hooks:

temperature = NumericalDiscreteParameter("Temperature", values=[90, 105, 120])
concentration = NumericalDiscreteParameter("Concentration", values=[0.057, 0.1, 0.153])
searchspace = SearchSpace.from_product([temperature, concentration])
recommendation = recommender.recommend(batch_size=3, searchspace=searchspace)
