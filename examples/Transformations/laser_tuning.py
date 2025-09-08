# # Optimizing a Laser's Wavelength

# In this example, we explore BayBE's capabilities for handling nonlinear
# optimization problems. Specifically, we demonstrate the two fundamental types of
# nonlinearities that can arise in optimization scenarios: known and unknown.
# We investigate a setting where:
# * we need to optimize a very specific **user-defined nonlinear** objective function of
#   a target quantity,
# * where the target quantity depends in a **nonlinear but unknown** way on a parameter
#   controlled by the user.

# ## Imports

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from polars import set_random_seed

from baybe.campaign import Campaign
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.simulation.scenarios import simulate_scenarios
from baybe.targets.numerical import NumericalTarget

# ## Settings

set_random_seed(1337)

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_MC_ITERATIONS = 2 if SMOKE_TEST else 20
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 20

# ## The Scenario

# Imagine you're in a busy optics lab, preparing to tune a high-performance laser for a
# critical experiment. Your goal is to align the laser's output wavelength to one of two
# specific channels in a [wavelength-division
# multiplexing](https://en.wikipedia.org/wiki/Wavelength-division_multiplexing) system.
# Both channels lead to the same detector, making them functionally equivalent for your
# measurements.

# The laser's wavelength is controlled by the voltage applied to its tuning element;
# however, this relationship is complex and nonlinear. It involves both linear drift
# and nonlinear cavity mode effects, but you lack a precise analytical model.
# Consequently, you turn to a data-driven optimization approach, with BayBE as your ally.

# ## The Laser Model

# To demonstrate this scenario through simulation, we need to define the laser's
# characteristics and system dynamics. In practice, this information would be unknown,
# but for benchmarking our optimization approach, we assume realistic values to build
# a black-box model.

# First, we define the physical constants that characterize our laser:

y0 = 1550.00  # unit: nm
α = 0.20  # unit: nm/V
A = 0.60  # unit: nm
P = 2.5  # unit: V

# We also capture the relationship between the applied voltage and the emitted
# wavelength in form of a lookup callable that can be used to close the simulation loop:


def lookup(df: pd.DataFrame) -> pd.DataFrame:
    """A lookup modelling the voltage-to-wavelength relationship of a tunable laser.

    This function captures both linear tuning and nonlinear cavity mode effects
    that are typical in real laser systems.
    """
    df["Wavelength"] = (
        y0 + α * df["Voltage"] + A * np.sin(2 * np.pi * df["Voltage"] / P)
    )
    return df


# ## The Optimization Problem

# Now, let us define our voltage parameter:

voltage = NumericalContinuousParameter("Voltage", [0, 10], metadata={"unit": "V"})

# We can visualize how it affects the laser's output wavelength by querying the lookup:

voltage_grid = np.linspace(*voltage.bounds.to_tuple(), 1000)
induced_λ = lookup(pd.DataFrame({"Voltage": voltage_grid}))["Wavelength"]

fig, axs = plt.subplots(2, 1, figsize=(8, 10))
axs[0].plot(voltage_grid, induced_λ, color="tab:blue")

# Our reference wavelengths for the two multiplexing channels and the acceptable
# tolerance around each are defined as follows:

λ1 = 1550.5  # unit: nm
λ2 = 1551.5  # unit: nm
σ = 0.1  # unit: nm

# Let us also add them to the plot:

axs[0].axhline(λ1, color="tab:red")
axs[0].axhline(λ2, color="tab:red")

# Our goal is to align the laser's output wavelength with either of the two reference
# wavelengths, treating both wavelenght as equally desirable. We can express this
# symmetric objective by creating "reward peaks" around each reference wavelength using
# Gaussian-shaped functions. The tolerance parameter $\sigma$ controls the width of
# these peaks, determining how precisely the laser must be tuned to achieve optimal
# performance:

# ```{math}
# \begin{equation*}
# \text{Objective}(\lambda) =
#   \exp\left(-\frac{(\lambda - \lambda_1)^2}{2\sigma^2}\right)
# + \exp\left(-\frac{(\lambda - \lambda_2)^2}{2\sigma^2}\right)
# \end{equation*}
# ```

# We can easily implement this mathematical objective in code using BayBE's
# transformation framework. To do so, we start by creating a target variable that
# lets us reference our observable quantity (the laser's output wavelength):

λ = NumericalTarget("Wavelength", metadata={"unit": "nm"})

# Using BayBE's ability to {ref}`compose transformations
# <userguide/transformations:Composite Transformations>`, the optimization objective can
# be imprinted onto the target using basic algebraic operations:

λ1_peak = ((λ - λ1).power(2) / (2 * σ**2)).negate().exp()
λ2_peak = ((λ - λ2).power(2) / (2 * σ**2)).negate().exp()
target = λ1_peak + λ2_peak

# ```{admonition} Convenience Constructor
# :class: tip
# Note that you can also use the
# {meth}`~baybe.targets.numerical.NumericalTarget.match_bell` convenience constructor to
# create the individual peaks instead, yielding {ref}`mathematically equivalent
# <userguide/transformations:Equality>` target objects.
# ```

# Let us overlay the relationship between observed wavelength and objective value in the
# existing plot:

λ_grid = pd.Series(np.linspace(induced_λ.min(), induced_λ.max(), 1000))
induced_target = target.transform(λ_grid)

ax2 = axs[0].twiny()
ax2.plot(induced_target, λ_grid, color="tab:orange")
ax2.set_xlabel("Objective Value", color="tab:orange")
ax2.tick_params(axis="x", labelcolor="tab:orange")
axs[0].set_xlabel(f"{voltage.name} ({voltage.metadata.unit})", color="tab:blue")
axs[0].set_ylabel(f"{target.name} ({target.metadata.unit})", color="tab:blue")
axs[0].tick_params(axis="x", labelcolor="tab:blue")
axs[0].tick_params(axis="y", labelcolor="tab:blue")

# ## Simulating the Optimization Loop

# Finally, we encapsulate both the search space and objective in a BayBE
# {class}`~baybe.campaign.Campaign`:

campaign = Campaign(
    searchspace=voltage.to_searchspace(),
    objective=target.to_objective(),
)

# In practice, we would iteratively query this campaign for voltage recommendations,
# measure the resulting wavelengths, and provide feedback for refined recommendations.
# Here, we simulate this closed-loop process using our black-box laser model:

results = simulate_scenarios(
    {"Laser": campaign},
    lookup,
    n_doe_iterations=N_DOE_ITERATIONS,
    n_mc_iterations=N_MC_ITERATIONS,
)

# ## Results

# Let us visualize the optimization process in the second subplot:

sns.lineplot(
    data=results.rename(
        columns={
            "Wavelength_CumBest": "Wavelength (nm)",
            "Num_Experiments": "# Experiments",
        }
    ),
    x="# Experiments",
    y="Wavelength (nm)",
    hue="Monte_Carlo_Run",
    marker="o",
    ax=axs[1],
    legend=False,
)
plt.tight_layout()
plt.show()


# The plot reveals trajectories converging to two distinct clusters,
# corresponding to the two reference wavelengths $\lambda_1$ and $\lambda_2$. This
# bimodal convergence occurs because both wavelengths – by construction – yield
# identical objective values according to the applied transformation.

# Each colored trajectory represents an independent Monte Carlo run of the optimization
# process, showing how different random initializations converge to one of the two
# equivalent optimal wavelengths without preferring one over the other. The early parts
# of each trajectory show larger variation in wavelength values (exploration phase),
# whereas the trajectories concentrate around the optimal wavelengths after several
# experiments (exploitation phase). Notice how trajectories can still escape their
# current basin of attraction in the hope of finding even better solutions in the
# other basin, which demonstrates the algorithm's ability to find the global
# optimum of the problem.
