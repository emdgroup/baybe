# Objective

## General Information

Objectives allows you to effectively manage the optimization objective and are implemented by the [`Objective`](baybe.objective.Objective) class.
By defining objectives, you can optimize a single target function or combine multiple target functions using different modes.
To create an objective, it is necessary to provide at least the following:
* **An optimization mode:** This can be either ``SINGLE`` or ``DESIRABILITY``, denoting either the optimization of a single target function or a combination of different target functions.
* **A list of targets:** The list of targets that are optimized. Note that the ``SINGLE`` mode also requires a list containing the single target. See [here](./targets) for details.

## Supported Optimization Modes

Currently, BayBE offers two optimization modes.

### Single Target Optimization: The ``SINGLE`` mode

In the ``SINGLE`` mode, objectives focus on maximizing or minimizing a single target. 
Nearly all of the examples use this objective mode.

### Combining multiple targets: The ``DESIRABILITY`` mode

The ``DESIRABILITY`` mode enables you to combine multiple targets into a unified objective using the ``combine_func``. This function effectively merges the list of targets into a single target, with mean and geometric mean beig the current available options. Additionally, you can assign weights to individual targets, enhancing their significance as needed.

For a practical example demonstrating this mode, see [here](./../../examples/Multi_Target/desirability).