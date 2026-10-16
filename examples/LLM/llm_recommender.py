# # Using Large Language Models for Recommendations

# Large Language Models (LLMs) are versatile tools that are part of many scientific and
# industrial applications. One of their biggest advantages is that they distilled vast
# amounts of knowledge from diverse sources, enabling them to make connections and
# draw conclusions from a completely different perspective compared to traditional
# Bayesian Optimization methods.

# In this example, we demonstrate the usage of the
# {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender` class for making
# recommendations based on large language models.

# ## Imports

import json
from unittest.mock import patch

import pandas as pd

from baybe import Campaign, active_settings
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
from baybe.recommenders import (
    LLMAlternatingRecommender,
    LLMRecommender,
    LLMTwoPhaseRecommender,
    RandomRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget

# ## Settings

N_DOE_ITERATIONS = 3
N_MC_ITERATIONS = 2
BATCH_SIZE = 2

# ## The Scenario

# LLMs shine where they can leverage the knowledge and information they have acquired
# during their training processes. They are thus in particular well-suited for tasks
# that benefit from knowledge that is hard to express in traditional rule-based systems
# or via explicit constraints. We therefore demonstrate their usage in a chemical
# reaction optimization scenario.

# We read the information about the conducted experiments from a CSV file.
# This data set was obtained from
# [Shields, B.J., Stevens et al. Nature 590, 89–96 (2021)](https://doi.org/10.1038/s41586-021-03213-y)
# and contains yield measurements for a direct C-H arylation reaction. The study
# systematically varies five reaction conditions: the **solvent**, the **base**, and
# the **phosphine ligand** (each identified by name and molecular structure), as well as
# the **reaction temperature** (in °C) and the **substrate concentration** (in mol/L).
# Together, these span 1728 unique experimental configurations, all of which have been
# measured and serve as the ground truth for the simulation below.
# Depending on your system and settings, you might need to slightly adjust the
# following paths.

try:
    lookup = pd.read_csv("benchmarks/data/direct_arylation/data.csv")
except FileNotFoundError as e:
    lookup = pd.read_csv("../../../benchmarks/data/direct_arylation/data.csv")
    print(e)

# ## Using Natural Language While Setting Up the Components

# BayBE's {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender` allows us to
# describe various aspects of the problem we want to solve in natural language. We can
# provide descriptions and metadata for the several BayBE objects and also supply
# an explicit experiment description that the LLM uses to make informed recommendations.

EXPERIMENT_DESCRIPTION = (
    "Optimization of a direct C-H arylation reaction. Each experiment selects a "
    "solvent, a base, and a ligand (given by their names) together with a reaction "
    "temperature and a substrate concentration."
)

# To leverage natural language for the parameters, we provide descriptive metadata,
# including a unit, a description, and additional context that the LLM can incorporate
# into its reasoning.

solvent = SubstanceParameter(
    "Solvent",
    data=dict(zip(lookup["Solvent"], lookup["Solvent_SMILES"])),
    encoding="MORDRED",
    metadata={
        "description": "Organic solvent used in the reaction",
        "role": "Controls polarity and solvation of reactants and intermediates",
    },
)
base = SubstanceParameter(
    "Base",
    data=dict(zip(lookup["Base"], lookup["Base_SMILES"])),
    encoding="MORDRED",
    metadata={
        "description": "Inorganic or organic base added to the reaction",
        "role": "Deprotonates the substrate and neutralizes acid generated during the cycle",
    },
)
ligand = SubstanceParameter(
    "Ligand",
    data=dict(zip(lookup["Ligand"], lookup["Ligand_SMILES"])),
    encoding="MORDRED",
    metadata={
        "description": "Phosphine ligand coordinating the palladium catalyst",
        "role": "Modulates catalyst reactivity and selectivity toward the arylation product",
    },
)
temperature = NumericalDiscreteParameter(
    "Temp_C",
    values=sorted(lookup["Temp_C"].unique()),
    tolerance=2,
    metadata={"description": "Reaction temperature", "unit": "°C"},
)
concentration = NumericalDiscreteParameter(
    "Concentration",
    values=sorted(lookup["Concentration"].unique()),
    tolerance=0.005,
    metadata={"description": "Substrate concentration", "unit": "mol/L"},
)

searchspace = SearchSpace.from_product(
    [solvent, base, ligand, temperature, concentration]
)

# In our scenario, we aim to maximize the yield of the reaction. We thus create an
# objective for the recommender, which we also enhance by using a description in natural
# language.

objective = SingleTargetObjective(
    target=NumericalTarget(
        name="yield",
        metadata={
            "description": "Reaction yield of a direct C-H arylation reaction",
            "unit": "%",
        },
    ),
    metadata={"description": "Maximize the reaction yield."},
)

# ## Creating the `LLMRecommender`

# We use [LiteLLM](https://docs.litellm.ai) as the model abstraction layer.
# LiteLLM provides a unified ``completion()`` interface that routes calls to
# Anthropic, OpenAI, Azure, Google, and dozens of other providers — all with the same
# API. BayBE passes the ``model`` string directly to LiteLLM, so any
# provider-prefixed identifier (e.g. ``"anthropic/claude-haiku-4-5"``,
# ``"openai/gpt-4o"``, ``"azure/my-deployment"``) works out of the box.

# Authentication is handled entirely via environment variables — BayBE deliberately
# does not accept credentials as constructor arguments to prevent them from appearing
# in logs or serialized campaign files. You thus need to set the variable that LiteLLM
# expects for your provider before when using the `LLMRecommender`. We refer to the
# [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for the full
# list of supported providers and their environment variables.

# We now create an instance of the {class}`~baybe.recommenders.pure.llm.llm.LLMRecommender`
# using the search space and objective defined above.

# **Note:** This example uses a mocked LLM that returns valid but arbitrary suggestions
# rather than real model responses. As a result, running this example will not produce
# meaningful optimization results — the LLM strategies will perform no better than
# random. Real performance results obtained with actual models are shown in the plots
# at the end of this page.

# In practice, replace ``"test/model"`` with a real LiteLLM model identifier such as
# ``"anthropic/claude-sonnet-4-6"`` and ensure the corresponding API key is set.

llm_recommender = LLMRecommender(
    model="test/model",
    experiment_description=EXPERIMENT_DESCRIPTION,
)


def _mock_query_model(self, prompt):
    """Return a JSON string with randomly sampled candidates."""
    batch = searchspace.discrete.exp_rep.sample(BATCH_SIZE)
    suggestions = [
        {
            "explanation": "Mock suggestion.",
            "parameters": {p.name: row[p.name] for p in searchspace.parameters},
        }
        for _, row in batch.iterrows()
    ]
    return json.dumps(suggestions)


# ## Making Best Use of the `LLMRecommender`

# Even though LLMs can provide valuable recommendations and leverage their domain
# knowledge, they are limited by what was seen during training. Since Bayesian
# Optimization is often applied to novel processes, LLMs will not always have detailed
# knowledge about specific reaction nuances. Therefore, combining LLM recommendations
# with classical Bayesian Optimization can lead to more effective exploration.

# BayBE offers two meta-recommenders that combine the LLM-based recommender with a
# classical Bayesian recommender:
#
# * The {class}`~baybe.recommenders.meta.llm.LLMTwoPhaseRecommender` uses the LLM to
#   propose the first few batches, then switches permanently to a Bayesian recommender
#   once a configurable number of measurements has been collected. This lets the LLM
#   guide early exploration using its prior knowledge, while Bayesian Optimization takes
#   over once enough data has been gathered for a reliable surrogate model.
# * The {class}`~baybe.recommenders.meta.llm.LLMAlternatingRecommender` cycles between
#   the LLM and a Bayesian recommender every batch throughout the entire campaign, so
#   that the LLM's prior knowledge and Bayesian exploitation continuously complement
#   each other.

# Both classes expose a ``from_model`` factory that constructs the recommender directly
# from a LiteLLM model identifier:

two_phase = LLMTwoPhaseRecommender.from_model(
    model="test/model",
    experiment_description=EXPERIMENT_DESCRIPTION,
    switch_after=5,
)
alternating = LLMAlternatingRecommender.from_model(
    model="test/model",
    experiment_description=EXPERIMENT_DESCRIPTION,
)

# ## Comparing Recommenders

# We compare the performance of the three LLM-based strategies against BayBE's default
# Bayesian recommender and a random baseline using a lookup-based simulation.

scenarios = {
    "LLM only": Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=llm_recommender,
    ),
    "Two-phase": Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=two_phase,
    ),
    "Alternating": Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=alternating,
    ),
    "Default": Campaign(searchspace=searchspace, objective=objective),
    "Random": Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=RandomRecommender(),
    ),
}

with patch.object(LLMRecommender, "_query_model", _mock_query_model):
    # We disable parallel simulation runs so that the mock LLM patch is visible across
    # all simulation workers (which run in the same process when parallelism is off).

    active_settings.parallelize_simulation_runs = False
    results = simulate_scenarios(
        scenarios,
        lookup,
        batch_size=BATCH_SIZE,
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
    )

# ## Results with Real Models

# The results above come from a mocked LLM and do not reflect real optimization
# performance. The plots below were generated by running the same comparison — LLM
# only, Two-phase, Alternating, Default, and Random — with actual language models on
# the full 1728-configuration search space over 20 DOE iterations and 35 Monte Carlo
# runs. Each plot shows all strategies for one model.

# ### Claude Sonnet 4.6

# ```{image} llm_recommender_sonnet_4_6_light.svg
# :align: center
# :class: only-light
# ```
# ```{image} llm_recommender_sonnet_4_6_dark.svg
# :align: center
# :class: only-dark
# ```

# ### Claude Sonnet 5

# ```{image} llm_recommender_sonnet_5_light.svg
# :align: center
# :class: only-light
# ```
# ```{image} llm_recommender_sonnet_5_dark.svg
# :align: center
# :class: only-dark
# ```


# ### Claude Opus 5.5

# ```{image} llm_recommender_opus_5_5_light.svg
# :align: center
# :class: only-light
# ```
# ```{image} llm_recommender_opus_5_5_dark.svg
# :align: center
# :class: only-dark
# ```
