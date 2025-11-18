# # Transfer Learning in Chemical Optimization

# In this example, we demonstrate BayBE's transfer learning capabilities using real
# experimental data from chemical synthesis. We explore how knowledge gained from
# optimization experiments under certain reaction conditions can be transferred to
# accelerate optimization under different conditions. Specifically, we investigate a
# setting where:
# * we have **historical experimental data** from chemical reactions conducted at
#   different temperatures
#   (taken from [Shields, B.J. et al.](https://doi.org/10.1038/s41586-021-03213-y)),
# * we want to **optimize reaction yield** at a new target temperature,
# * we can leverage the **relationships between reaction conditions** to accelerate
#   the optimization process.

# ## Imports and Settings

import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from baybe import Campaign
from baybe.parameters import (
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.random import set_random_seed

set_random_seed(1337)

SMOKE_TEST = "SMOKE_TEST" in os.environ
N_MC_ITERATIONS = 2 if SMOKE_TEST else 50
N_DOE_ITERATIONS = 2 if SMOKE_TEST else 25

# ## The Scenario

# Imagine you're a chemist in a pharmaceutical company, tasked with optimizing a
# direct arylation reaction to maximize product yield. This reaction involves
# combining different chemical components (solvents, bases, ligands) under varying
# temperature and concentration conditions. Each experiment is expensive and
# time-consuming, making efficient optimization crucial.

# However, you have an advantage: your laboratory has already conducted similar
# optimization campaigns at different temperatures in the past. The question arises:
# can you leverage this **historical data from related conditions** to accelerate
# optimization at your target temperature? This is where transfer learning becomes
# invaluable.

# ## The Chemical System

# Our experimental dataset comes from a real direct arylation study, containing
# measurements of reaction yields under systematically varied conditions. The key
# experimental variables are:

# * **Chemical components**: Different solvents, bases, and ligands
# * **Temperature**: Reaction temperature (90°C, 105°C, 120°C)
# * **Concentration**: Reagent concentration (0.057, 0.1, 0.153 mol/L)
# * **Target**: Reaction yield (percentage)


ENCODING = "RDKIT2DDESCRIPTORS"
BATCH_SIZE = 1

# We define the experimental conditions available in the dataset:

TEMPERATURES = [90, 105, 120]
CONCENTRATIONS = [0.057, 0.1, 0.153]
TARGET_TEMPERATURES = [90, 105, 120]
if not SMOKE_TEST:
    SAMPLE_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2]
else:
    SAMPLE_FRACTIONS = [0.01, 0.02]

# ## Loading the Experimental Dataset

try:
    lookup = pd.read_csv("benchmarks/data/direct_arylation/data.csv")
except FileNotFoundError:
    try:
        lookup = pd.read_csv("../../../benchmarks/data/direct_arylation/data.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Could not find the direct arylation dataset. Please ensure you're "
            "running from the correct directory and the data file exists."
        ) from e

# The dataset maps chemical substance names to their molecular representations (SMILES),
# enabling us to encode molecular structure information for the optimization algorithm:

substances = {
    "solvents": dict(zip(lookup["Solvent"], lookup["Solvent_SMILES"])),
    "bases": dict(zip(lookup["Base"], lookup["Base_SMILES"])),
    "ligands": dict(zip(lookup["Ligand"], lookup["Ligand_SMILES"])),
}

# ## The Optimization Problem

target = NumericalTarget(name="yield", minimize=False)
objective = target.to_objective()

# To enable transfer learning, we use a {class}`~baybe.parameters.categorical.TaskParameter`
# to represent temperature, which allows the algorithm to learn relationships between
# different temperature conditions and transfer knowledge between them.


def create_search_space(target_temperature):
    """Create a search space configured for transfer learning.

    The TaskParameter enables the algorithm to distinguish between different
    temperature conditions while learning relationships between them. We set
    ``active_values`` to specify which temperature we want to optimize for,
    while allowing the model to leverage data from all temperature conditions.
    """
    parameters = [
        SubstanceParameter(
            name="Solvent", data=substances["solvents"], encoding=ENCODING
        ),
        SubstanceParameter(name="Base", data=substances["bases"], encoding=ENCODING),
        SubstanceParameter(
            name="Ligand", data=substances["ligands"], encoding=ENCODING
        ),
        NumericalDiscreteParameter(
            name="Concentration", values=CONCENTRATIONS, tolerance=0.001
        ),
        TaskParameter(
            name="Temp_C",
            values=[str(t) for t in TEMPERATURES],
            active_values=[str(target_temperature)],
        ),
    ]
    return SearchSpace.from_product(parameters=parameters)


# We also need to prepare the lookup table with string temperatures for `TaskParameter`
# compatibility:

lookup_task = lookup.copy()
lookup_task["Temp_C"] = lookup_task["Temp_C"].astype(str)


# ## Simulating Transfer Learning

# To demonstrate the power of transfer learning, we simulate optimization campaigns
# for each temperature condition. For each target temperature, we compare two approaches:
#
# 1. **Baseline**: Optimization without any prior knowledge (cold start)
# 2. **Transfer Learning**: Optimization using varying amounts of data from other temperatures


all_results = {}

for target_temp in TARGET_TEMPERATURES:
    training_temps = [t for t in TEMPERATURES if t != target_temp]

    print(f"\nOptimizing for {target_temp}°C using data from {training_temps}°C...")

    searchspace = create_search_space(target_temp)

    training_temperatures = [str(t) for t in training_temps]
    lookup_training = lookup_task[
        lookup_task["Temp_C"].isin(training_temperatures)
    ].copy()

    results: list[pd.DataFrame] = []

    for fraction in SAMPLE_FRACTIONS:
        campaign = Campaign(searchspace=searchspace, objective=objective)

        # Create initial training datasets by sampling from other temperatures
        # Multiple random samples provide statistical robustness
        initial_data = [
            lookup_training.sample(frac=fraction, random_state=i)
            for i in range(N_MC_ITERATIONS)
        ]

        result = simulate_scenarios(
            {f"{int(100 * fraction)}": campaign},
            lookup_task,
            initial_data=initial_data,
            batch_size=BATCH_SIZE,
            n_doe_iterations=N_DOE_ITERATIONS,
        )
        results.append(result)

    # Baseline comparison: optimization without transfer learning
    result_baseline = simulate_scenarios(
        {"0": Campaign(searchspace=searchspace, objective=objective)},
        lookup_task,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
    )
    results = pd.concat([result_baseline, *results])

    all_results[target_temp] = results

# ## Results

# The results demonstrate the significant impact of transfer learning on optimization
# efficiency. Each subplot shows the optimization progress for a different target
# temperature, comparing baseline performance against transfer learning with various
# amounts of training data.

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

summary_data = []

for i, target_temp in enumerate(TARGET_TEMPERATURES):
    results = all_results[target_temp].copy()
    results.rename(columns={"Scenario": "% of data used"}, inplace=True)

    ax = axes[i]
    sns.lineplot(
        data=results,
        marker="o",
        markersize=6,
        x="Num_Experiments",
        y="yield_CumBest",
        hue="% of data used",
        ax=ax,
    )

    ax.set_xlabel("Number of Experiments")
    ax.set_ylabel("Best Yield Achieved (%)")
    ax.set_title(f"Target: {target_temp}°C")
    ax.grid(True, alpha=0.3)

    # Show legend only on the rightmost subplot
    if i < len(TARGET_TEMPERATURES) - 1:
        ax.legend().set_visible(False)
    else:
        ax.legend(
            title="Training data used", bbox_to_anchor=(1.05, 1), loc="upper left"
        )

    # Collect quantitative results for summary
    final_results = results.groupby("% of data used")["yield_CumBest"].max()
    baseline = final_results.loc["0"]
    best_transfer = final_results.drop("0").max()
    improvement = best_transfer - baseline

    training_temps = [t for t in TEMPERATURES if t != target_temp]
    summary_data.append(
        {
            "target_temp": target_temp,
            "training_temps": training_temps,
            "baseline": baseline,
            "best_transfer": best_transfer,
            "improvement": improvement,
        }
    )

plt.tight_layout()
if not SMOKE_TEST:
    plt.savefig("basic_transfer_learning.svg")


# The results reveal several key insights:
#
# 1. **Consistent Benefits**: Transfer learning provides substantial improvements
#    across all temperature conditions, demonstrating the robustness of the approach.
#
# 2. **Variable Effectiveness**: The magnitude of improvement varies by condition,
#    reflecting differences in how well knowledge transfers between specific
#    temperature pairs.
#
# 3. **Data Efficiency**: Even small amounts of training data (1-2%) can yield
#    significant acceleration, making this approach practical even with limited
#    historical data.
#
# 4. **Diminishing Returns**: Beyond a certain point, additional training data
#    provides marginal benefits, suggesting an optimal balance between data
#    utilization and computational efficiency.
