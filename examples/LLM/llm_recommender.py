## Warm-Starting a Campaign with a Language Model

# This example demonstrates the
# {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender`, which queries a large
# language model (LLM) to propose experiments. We use it to guide the *first* iterations
# of a chemical reaction optimization, hand over to a regular Bayesian recommender once
# enough data has been collected, and compare this strategy against a random baseline and
# BayBE's default recommender.

### How it works and why it helps early on

# On each call, the recommender turns the search space, the experiment and objective
# descriptions, and any collected measurements into a prompt, sends it to an LLM via
# [LiteLLM](https://docs.litellm.ai/), and parses the returned JSON into validated
# experiments (retrying once if the response is malformed).

# Bayesian optimization needs initial data before its surrogate becomes informative, so
# early experiments are usually random. An LLM can do better: needing no training data, it
# exploits literature and chemistry priors and understands the parameter semantics (names,
# units, allowed values) to propose sensible starting points. Once enough data exists, we
# switch to a {class}`~baybe.recommenders.pure.bayesian.botorch.BotorchRecommender` via a
# {class}`~baybe.recommenders.meta.sequential.TwoPhaseMetaRecommender`.


### Imports

import json
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import seaborn as sns

from baybe import Campaign, Settings
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.recommenders import (
    BotorchRecommender,
    LLMRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from examples.utils import create_example_plots

### Settings

# Let's define some general settings for our example:

SMOKE_TEST = "SMOKE_TEST" in os.environ

BATCH_SIZE = 1
N_WARMSTART_EXPERIMENTS = 1  # experiments handled by the LLM before switching
N_DOE_ITERATIONS = 1 if SMOKE_TEST else 3
# Kept low to bound the number of (potentially costly) LLM calls.
N_MC_ITERATIONS = 1 if SMOKE_TEST else 3

# LiteLLM reads the provider credentials from the environment. In CI (``SMOKE_TEST``) we
# fall back to a mocked LLM so the example runs offline; otherwise a real model is queried.


### Lookup Data

# We optimize a direct arylation reaction, the same chemical example used in other BayBE
# examples. Since all parameter combinations have been measured, we can use a lookup table
# to obtain the reaction yield for any recommendation. The data set was obtained from
# [Shields, B.J., Stevens et al. Nature 590, 89-96 (2021)](https://doi.org/10.1038/s41586-021-03213-y).

try:
    lookup = pd.read_csv("benchmarks/data/direct_arylation/data.csv")
except FileNotFoundError:
    lookup = pd.read_csv("../../../benchmarks/data/direct_arylation/data.csv")


### Defining the Optimization Problem

# The reaction is described by three chemical substances (solvent, base, and ligand) and
# two numerical process parameters (temperature and concentration). The parameter names
# must match the columns of the lookup table.

dict_solvent = {
    "DMAc": r"CC(N(C)C)=O",
    "Butyornitrile": r"CCCC#N",
    "Butyl Ester": r"CCCCOC(C)=O",
    "p-Xylene": r"CC1=CC=C(C)C=C1",
}

dict_base = {
    "Potassium acetate": r"O=C([O-])C.[K+]",
    "Potassium pivalate": r"O=C([O-])C(C)(C)C.[K+]",
    "Cesium acetate": r"O=C([O-])C.[Cs+]",
    "Cesium pivalate": r"O=C([O-])C(C)(C)C.[Cs+]",
}

dict_ligand = {
    "BrettPhos": r"CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)="
    "CC=C2OC",
    "Di-tert-butylphenylphosphine": r"CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
    "(t-Bu)PhCPhos": r"CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
}

# We define the substance parameters using the dictionaries above and add descriptive
# metadata to the numerical parameters. The metadata (descriptions and units) is picked
# up automatically by the ``LLMRecommender`` and rendered into the prompt, giving the
# language model additional context.

solvent = SubstanceParameter("Solvent", data=dict_solvent, encoding="MORDRED")
base = SubstanceParameter("Base", data=dict_base, encoding="MORDRED")
ligand = SubstanceParameter("Ligand", data=dict_ligand, encoding="MORDRED")

temperature = NumericalDiscreteParameter(
    "Temp_C",
    values=[90, 105, 120],
    tolerance=2,
    metadata={"description": "Reaction temperature", "unit": "°C"},
)
concentration = NumericalDiscreteParameter(
    "Concentration",
    values=[0.057, 0.1, 0.153],
    tolerance=0.005,
    metadata={"description": "Substrate concentration", "unit": "mol/L"},
)

searchspace = SearchSpace.from_product(
    [solvent, base, ligand, temperature, concentration]
)

# We maximize the reaction yield:

objective = SingleTargetObjective(target=NumericalTarget(name="yield"))


### Setting Up the Recommenders

# The ``LLMRecommender`` needs the model identifier plus textual descriptions of the
# experiment and objective, which convey the domain context to the language model.

llm_recommender = LLMRecommender(
    model="anthropic/claude-haiku-4-5",
    experiment_description=(
        "Optimization of a direct C-H arylation reaction. Each experiment selects a "
        "solvent, a base, and a ligand (given by their names) together with a reaction "
        "temperature and a substrate concentration."
    ),
    objective_description="Maximize the reaction yield in percent. Higher is better.",
)

# We compare three strategies:
# * **LLM + Bayesian**: the language model warm-starts the campaign and the meta
#   recommender switches to the Bayesian recommender after ``N_WARMSTART_EXPERIMENTS``,
# * **Default**: BayBE's default recommender (random initialization, then Bayesian), and
# * **Random**: a pure random baseline.

scenarios = {
    "LLM + Bayesian": Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=llm_recommender,
            recommender=BotorchRecommender(),
            switch_after=N_WARMSTART_EXPERIMENTS,
        ),
    ),
    "Default": Campaign(searchspace=searchspace, objective=objective),
    "Random": Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=RandomRecommender(),
    ),
}


### Mocked Fallback

# Without a credential, we patch the LiteLLM call to return random valid configurations,
# keeping the example runnable offline. With a real token, this is skipped.


def _make_mock_completion(space: SearchSpace):
    """Create a fake LiteLLM ``completion`` returning random valid suggestions."""
    candidates = space.discrete.exp_rep

    def _mock_completion(*args, **kwargs):
        rows = candidates.sample(min(len(candidates), 4 * BATCH_SIZE))
        suggestions = [
            {"explanation": "Mocked suggestion.", "parameters": row.to_dict()}
            for _, row in rows.iterrows()
        ]
        content = json.dumps(suggestions, default=str)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return _mock_completion


llm_context = (
    patch("baybe._optional.llm.completion", _make_mock_completion(searchspace))
    if SMOKE_TEST
    else nullcontext()
)

if SMOKE_TEST:
    Settings(parallelize_simulation_runs=False).activate()


### Running the Comparison

# We use `simulate_scenarios` to run a full optimization loop for each strategy, tracking
# the best yield found so far. The language model is queried through the (real or mocked)
# LiteLLM call for the duration of the simulation.


with llm_context:
    results = simulate_scenarios(
        scenarios,
        lookup,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
    )

print(results)


### Visualization

# We plot the best yield found so far against the number of conducted experiments. The
# language model gives the "LLM + Bayesian" strategy a head start in the early
# iterations, before the Bayesian recommender takes over.

results.rename(columns={"Scenario": "Strategy"}, inplace=True)
ax = sns.lineplot(
    data=results,
    marker="o",
    markersize=10,
    x="Num_Experiments",
    y="yield_CumBest",
    hue="Strategy",
)

# Note that the displayed images were generated from a real run with 30 MC and 30 DoE
# iterations to demonstrate the effect of the LLM warm-start.
create_example_plots(ax=ax, base_name="llm_recommender")
