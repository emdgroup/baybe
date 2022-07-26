"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""

from baybe.core import add_fake_results, add_noise, BayBE, BayBEConfig

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config_dict = {
    "project_name": "Input Output Debug",
    "allow_repeated_recommendations": True,
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
            "values": ["bad", "OK", "good"],
            "encoding": "Integer",
        },
        {
            "name": "Num_disc_1",
            "type": "NUM_DISCRETE",
            "values": [1, 2, 3],
            "tolerance": 0.3,
        },
        {
            "name": "Num_disc_2",
            "type": "NUM_DISCRETE",
            "values": [-1, -3, -6],
            "tolerance": 0.3,
        },
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "bounds": None,
                "mode": "MIN",
            },
        ],
    },
}

# Define some parameter values to define rows where the fake results should be good
good_reference_values = {"Categorical_2": ["OK"], "Categorical_1": [22]}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)
print(baybe_obj)

N_ITERATIONS = 4
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=3)
    print("\n\n### Recommended dataframe:\n", rec)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    if kIter % 2:
        print(kIter)
        add_noise(rec, baybe_obj, noise_level=0.1)
    print("\n\n### Recommended dataframe with fake results and eventual noise:\n", rec)

    # uncomment below to test error throw for disallowed value
    # baybe_obj.add_results(rec.replace(1, 11111))
    baybe_obj.add_results(rec.sample(n=1))
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

    # Show metadata
    print(
        "\n\n### Search Space Metadata\n",
        baybe_obj.searchspace_metadata,
    )
