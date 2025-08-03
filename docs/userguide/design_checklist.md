# Checklist for designing BayBE experiments

- Are only some parameter values of interest/possible?

  > See how to exclude some 
[parameter values](https://emdgroup.github.io/baybe/stable/userguide/getting_recommendations.html#excluding-configurations) 
from being recommended, such as by defining
[bounds for continuous parameters](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#numericalcontinuousparameter).
or [active values for discrete parameters](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#discrete-parameters).

- Are only some parameter combinations of interest/possible?

  > See how to exclude some 
[parameter combinations](https://emdgroup.github.io/baybe/stable/userguide/constraints.html)
from being considered.

- Should be multiple target optimized simultaneously?

  > See how to use [multi-target objectives](https://emdgroup.github.io/baybe/stable/userguide/objectives.html).

- Is it possible to encode discrete parameters based on domain knowledge 
(e.g., ordered values, molecular fingerprints, model embeddings)?

  > See how to [encode](https://emdgroup.github.io/baybe/stable/userguide/parameters.html#discrete-parameters)
discrete parameters or provide custom encodings.

- Is additional data from historic or other partially-related experiments available?

  > Use [transfer learning](https://emdgroup.github.io/baybe/stable/userguide/transfer_learning.html).

- Is the aim to reduce the overall uncertainty across different regions of the search space 
rather than optimize a specific objective?

  > Use [active learning](https://emdgroup.github.io/baybe/stable/userguide/active_learning.html).

- Will the outcome measurements of different parameter setting become available at different times?

  > Use [asynchronous workflows](https://emdgroup.github.io/baybe/stable/userguide/async.html).


