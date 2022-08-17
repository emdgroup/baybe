"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""
import pandas as pd

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter

custom_df = pd.DataFrame(
    {
        "Mol": ["mol1", "mol2", "mol3", "mol4", "mol5"],
        "D1": [1.1, 1.4, 1.7, 0.8, -0.2],
        "D2": [11, 23, 55, 23, 3],
        "D3": [-4, -13, 4, -2, 6],
        "D4": [0.1, 0.4, -1.3, -0.5, 2.1],
        "D5": [1, 2, 0, 0, 7],
    }
)
custom_df2 = pd.DataFrame(
    {
        "BuildingBlock": ["A", "B", "C"],
        "desc1": [1.1, 1.4, 1.7],
        "desc2": [55, 23, 3],
        "desc3": [4, 5, 6],
        "desc4": [-1.3, -0.5, 2.1],
    }
)


config_dict = {
    "project_name": "Custom Parameter",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": True,
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
            "values": [1, 2, 3, 4],
            "tolerance": 0.3,
        },
        {
            "name": "Custom_1",
            "type": "CUSTOM",
            "data": custom_df,
            # "identifier_col_idx": 0,
        },
        {
            "name": "Custom_2",
            "type": "CUSTOM",
            "data": custom_df2,
            # "identifier_col_idx": 0,
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

N_ITERATIONS = 5
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=3)
    # print("\n### Recommended dataframe:\n", rec)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)
    print("### Recommended dataframe with fake results and eventual noise:\n", rec)

    baybe_obj.add_results(rec)
    print(
        "\n\n### Internal measurement dataframe after data ingestion:\n",
        baybe_obj.measurements_exp_rep,
    )

    print(
        "\n\n### Internal measurement dataframe computational representation X:\n",
        baybe_obj.measurements_comp_rep_x,
    )

    print(
        "\n\n### Internal measurement dataframe computational representation Y:\n",
        baybe_obj.measurements_comp_rep_y,
    )

    print(
        "\n\n### Search Space Metadata\n",
        baybe_obj.searchspace_metadata,
    )
