# Objective

The [`Objective`](baybe.objective.Objective) instructs BayBE how to deal with multiple 
targets (if applicable).

To create an objective, it is necessary to provide at the following:
* An optimization ``mode``: This can be either ``SINGLE`` or ``DESIRABILITY``, 
  denoting either the optimization of a single target function or a combination of 
  different target functions.
* A list of ``targets``: The list of targets that are optimized. Note that the 
  ``SINGLE`` mode also requires a list containing the single target.
