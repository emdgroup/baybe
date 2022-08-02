"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""

from baybe.core import add_fake_results, add_noise, BayBE, BayBEConfig

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config_dict = {
    "project_name": "Connect Strategy",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": True,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Categorical_1",
            "type": "CAT",
            "values": [22, 33],
            "encoding": "OHE",
        },
        {
            "name": "Categorical_2",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Num_disc_1",
            "type": "NUM_DISCRETE",
            "values": [1, 2, 3, 4, 6, 8, 10],
            "tolerance": 0.3,
        },
        {
            "name": "Num_disc_2",
            "type": "NUM_DISCRETE",
            "values": [-1, -3, -6, -9],
            "tolerance": 0.3,
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
    },
}

# Define some parameter values to define rows where the fake results should be good
good_reference_values = {"Categorical_2": ["OK"], "Categorical_1": [22]}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)
print(baybe_obj)

N_ITERATIONS = 50
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=1)
    # print("\n### Recommended dataframe:\n", rec)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    if kIter % 2:
        add_noise(rec, baybe_obj, noise_level=0.1)
    print("\n\n### Recommended dataframe with fake results and eventual noise:\n", rec)

    baybe_obj.add_results(rec)
    # print(
    #     "\n\n### Internal measurement dataframe after data ingestion:\n",
    #     baybe_obj.measurements_exp_rep,
    # )

    # print(
    #     "\n\n### Internal measurement dataframe computational representation X:\n",
    #     baybe_obj.measurements_comp_rep_x,
    # )

    # print(
    #     "\n\n### Internal measurement dataframe computational representation Y:\n",
    #     baybe_obj.measurements_comp_rep_y,
    # )

    # Show metadata
    # print(
    #     "\n\n### Search Space Metadata\n",
    #     baybe_obj.searchspace_metadata,
    # )
