"""
Test for having multiple targets treated in a true multi-target fashion
"""

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

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
                "mode": "MIN",
                "bounds": (0, 100),
                # "bounds_transform_func": "BELL",
            },
        ],
    },
    "strategy": {},
}

# Define some parameter values to define rows where the fake results should be good
good_reference_values = {"Categorical_2": ["OK"], "Categorical_1": [22]}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)
print(baybe_obj)

N_ITERATIONS = 10
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=5)
    # print("\n### Recommended dataframe:\n", rec)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    if kIter % 2:
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)
    print("### Recommended dataframe with fake results and eventual noise:\n", rec)

    baybe_obj.add_results(rec)
    # print(
    #     "\n\n### Internal measurement dataframe after data ingestion:\n",
    #     baybe_obj.measurements_exp_rep,
    # )

    # print(
    #     "\n\n### Internal measurement dataframe computational representation X:\n",
    #     baybe_obj.measurements_comp_rep_x,
    # )

    print(
        "\n\n### Internal measurement dataframe computational representation Y:\n",
        baybe_obj.measurements_targets_comp,
    )

    # Show metadata
    # print(
    #     "\n\n### Search Space Metadata\n",
    #     baybe_obj.searchspace_metadata,
    # )
