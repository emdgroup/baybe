"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""

from baybe.core import add_fake_results, add_noise, BayBE, BayBEConfig
from baybe.utils import (
    smiles_to_fp_features,
    smiles_to_mordred_features,
    smiles_to_rdkit_features,
)

print(smiles_to_mordred_features(["C", "O", "c1ccccc1"]))
print("\n")
print(smiles_to_rdkit_features(["C", "O", "c1ccccc1"]))
print("\n")
print(smiles_to_fp_features(["C", "O", "c1ccccc1"]))
print("\n")

substances = {
    "Water": "O",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
    "Hexane": "CCCCCC",
    "Ethanol": "CCO",
}

config_dict = {
    "project_name": "Substance Parameter",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Categorical_1",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Num_disc_1",
            "type": "NUM_DISCRETE",
            "values": [11, 22, 33, 44],
            "tolerance": 0.3,
        },
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": substances,
            # "encoding": "RDKIT",
        },
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MAX",
            },
        ],
    },
    "strategy": {
        # "surrogate_model_cls": "GP",
        # "recommender_cls": "RANDOM"
    },
}

# Define some parameter values to define rows where the fake results should be good
good_reference_values = {"Num_disc_1": [1, 2, 3], "Categorical_1": ["OK"]}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)
print(baybe_obj)

N_ITERATIONS = 4
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=3)
    # print("\n### Recommended dataframe:\n", rec)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    if kIter % 2:
        add_noise(rec, baybe_obj, noise_level=0.1)
    print("### Recommended dataframe with fake results and eventual noise:\n", rec)

    baybe_obj.add_results(rec)
# print(
#     "\n\n### Internal measurement dataframe after data ingestion:\n",
#     baybe_obj.measurements_exp_rep,
# )
#
# print(
#     "\n\n### Internal measurement dataframe computational representation X:\n",
#     baybe_obj.measurements_comp_rep_x,
# )
#
# print(
#     "\n\n### Internal measurement dataframe computational representation Y:\n",
#     baybe_obj.measurements_comp_rep_y,
# )
#
# print(
#     "\n\n### Search Space Metadata\n",
#     baybe_obj.searchspace_metadata,
# )
