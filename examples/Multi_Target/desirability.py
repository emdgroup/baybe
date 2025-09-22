## Desirability Optimization

# This example demonstrates how to set up a multi-target optimization problem using the
# {class}`desirability approach <baybe.objectives.desirability.DesirabilityObjective>`.
# The focus lies on defining the target objects with the necessary transformations
# enabling the desirability computation, and the creation of the corresponding
# optimization objective.


### Imports

import pandas as pd

from baybe import Campaign
from baybe.objectives import DesirabilityObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.targets import NumericalTarget

### Defining the Search Space

# Because the search space is of secondary importance in this example, we keep it simple
# and consider only a single parameter, without loss of generality:

searchspace = NumericalContinuousParameter("parameter", (0, 1)).to_searchspace()


### Defining the Targets

# Next, we define our optimization targets. Because desirability computation relies on
# averaging target values, it is required that all targets are properly normalized.
# This can be achieved by applying appropriate target transformations, for which
# BayBE offers several built-in choices and also offers full customization for
# advanced use cases.

# ```{admonition} Target Normalization
# :class: note
# If you know what you are doing, you can also disable the normalization check
# via the
# {paramref}`~baybe.objectives.desirability.DesirabilityObjective.require_normalization`
# flag, with the consequence that the selected averaging method is executed on the
# target values no matter if they are normalized or not.
# ```

# For our example, we consider three simple targets reflecting different optimization
# goals. The first target takes values in the interval [0, 100] and is to be maximized.
# The {meth}`~baybe.targets.numerical.NumericalTarget.normalized_ramp` constructor helps
# us achieve this by applying an affine transformation whose output is clamped to the
# unit interval:

target_max = NumericalTarget.normalized_ramp("target_max", cutoffs=(0, 100))

# The second target takes values in the interval [-10, 0] and is to be minimized:

target_min = NumericalTarget.normalized_ramp(
    "target_min", cutoffs=(-10, 0), descending=True
)

# For the third target, we like to match a certain value. To do so, we apply a target
# transformation that penalizes the distance to this value using a
# {meth}`bell-shaped curve <baybe.targets.numerical.NumericalTarget.match_bell>`
# centered around it:

target_match = NumericalTarget.match_bell("target_match", match_value=50, sigma=5)

# ```{admonition} Customization
# :class: note
# Note that you can easily change the specifics of the applied transformations by
# resorting to other target constructors or specifying custom transformation
# logic. For more details, see our
# {ref}`target userguide <userguide/targets:NumericalTarget>`.
# ```


### Creating the Objective

# The targets are collected in a
# {class}`~baybe.objectives.desirability.DesirabilityObjective`, which takes care of the
# averaging process. The specifics of the averaging can be configured by specifying
# optional weights for the targets and the type of averaging to be used:

targets = [target_max, target_min, target_match]
objective = DesirabilityObjective(targets, weights=[20, 20, 60], scalarizer="MEAN")


### Getting Recommendations

# We can now use the objective, like any other, to
# {doc}`query recommendations </userguide/getting_recommendations>`, e.g. by setting
# up a {class}`~baybe.campaign.Campaign`:

campaign = Campaign(searchspace, objective)
recommendations = campaign.recommend(batch_size=3)
print(recommendations)


### Accessing Desirability Values

# Once the target measurements are available, ...

recommendations[target_max.name] = [65, 35, 87]
recommendations[target_min.name] = [-8, -3, -5]
recommendations[target_match.name] = [55, 25, 48]

# ... we can access the corresponding desirability values via the objective:

campaign.add_measurements(recommendations)
desirability = objective.transform(recommendations, allow_extra=True)
print(pd.concat([recommendations, desirability], axis=1))
