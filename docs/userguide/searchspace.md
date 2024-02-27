# Search spaces

## General information

The term "search space" refers to the domain of possible values for the parameters that are being optimized during a campaign. A search space represents the space within which BayBE explores and searches for the optimal solution. It is implemented via the [`SearchSpace`](baybe.searchspace.core.SearchSpace) class.

A search space might be purely discrete, purely continuous, or hybrid. This is encoded by the ``type`` attribute (see [`SearchSpaceType`](baybe.searchspace.core.SearchSpaceType)). Furthermore, it is possible to restrict certain types of search spaces using [`Constraints`](./constraints.md).

A search space is constructed by providing either a discrete subspace, a continuous subspace, or both of them. 

## Understanding the different search spaces and subspaces

In general, subspaces are managed via the respective classes, that is, via [`SubspaceDiscrete`](baybe.searchspace.discrete.SubspaceDiscrete) and [`SubspaceContinuous`](baybe.searchspace.continuous.SubspaceContinuous). They are built from parameter definitions and optional constraints, and provide access to candidate sets and different parameter views.

Both kinds of subspaces share the following attributes:
* **A parameter list:** The list of the parameters of the subspace. Note that they need to be discrete resp. continuous, depending on the subspace (see [here](./parameters) for details.)
* **Optional constraint list(s):** A single resp. up to two lists can be provided for general constraints in discrete subspaces resp. linear equality and linear in-equality constraints in continuous subspaces. For more details on constraints, see [here](./constraints).

A discrete/continuous search space is a s searchspace that was constructed by only providing a discrete/continuous subspace upon initialization.

### Discrete subspaces

In addition to the ones noted above, a discrete subspace has the following attributes:
* **The experimental representation:** A ``DataFrame`` representing the experimental representation of the subspace.
* **The metadata:** A ``DataFrame`` keeping track of different metadata that is relevant for running a campaign.
* **An "empty" encoding flag:** A flag denoting whether an "empty" encoding should be used. This is useful, for instance, in combination with random recommenders that do not read the actual parameter values.
* **The computational representation:** The computational representation of the space. If not provided explicitly, it will be derived from the experimental representation.

Although it is possible to directly create a discrete subspace via the ``__init__`` function, it is intended to create themvia the [`from_dataframe`](baybe.searchspace.discrete.SubspaceDiscrete.from_dataframe) or [`from_product`](baybe.searchspace.discrete.SubspaceDiscrete.from_product) methods. These methods either require a ``DataFrame`` containing the experimental representation of the parameters and the optional explicit list of parameters (``from_dataframe``) or a list of parameters and optional constraints (``from_product``).

For details and examples on how to use a discrete search space, see the corresponding example [here](./../../examples/Searchspaces/discrete_space) and [here](./../../examples/Constraints_Discrete/Constraints_Discrete).

### Continuous subspaces

In contrast to discrete subspaces, continuous subspaces do *not* have dedicated experimental and computational representations since these are identical for continuous subspaces.

Although it is possible to directly create a continuous subspace via the ``__init__`` function, it is intended to create them via the [`from_dataframe`](baybe.searchspace.continuous.SubspaceContinuous.from_dataframe) or [`from_bounds`](baybe.searchspace.continuous.SubspaceContinuous.from_dataframe) methods. These methods both require a ``DataFrame`` containing the points that should be contained in the space resp. the bounds for the parameters.

For details and examples on how to use a continuous search space, see the corresponding examples [here](./../../examples/Searchspaces/continuous_space_botorch_function), [here](./../../examples/Searchspaces/continuous_space_custom_function)  and [here](./../../examples/Constraints_Continuous/Constraints_Continuous).

### Hybrid search spaces

If a search space has a discrete and a continuous subspace, it is a hybrid search space.
Note that constraints defined on the hybrid space itself, i.e., constraints that involve both subspaces, are not supported. However, constraints for the two subspaces can be used.

For details on how to use a hybrid searchspace, see the corresponding examples [here](./../../examples/Searchspaces/hybrid_space) and [here](./../../examples/Constraints_Continuous/hybrid_space).