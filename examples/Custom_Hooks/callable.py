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


from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.utils.basic import register_hooks

### Defining the Hook

# We start by defining a hook that lets us inspect the names of the parameters involved
# in the recommendation process.
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


### Monkeypatching

# Next, we create our recommender and monkeypatch its `recommend` method:

recommender = RandomRecommender()
RandomRecommender.recommend = register_hooks(
    RandomRecommender.recommend, pre_hooks=[print_parameter_names_hook]
)

### Triggering the Hook

# When we now apply the recommender in a specific context, we immediately see the
# effect of the hook:

temperature = NumericalDiscreteParameter("Temperature", values=[90, 105, 120])
concentration = NumericalDiscreteParameter("Concentration", values=[0.057, 0.1, 0.153])
searchspace = SearchSpace.from_product([temperature, concentration])
recommendation = recommender.recommend(batch_size=3, searchspace=searchspace)
