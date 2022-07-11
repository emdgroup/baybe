"""Test for initial simple input, recommendation and adding fake results. Fake target
measurements are simulated for each round. Noise is added every second round.
From the three recommendations only one is actually added to test the matching and
metadata. Target objective is minimize to test computational transformation.
"""
import logging

from baybe.core import BayBE
from baybe.utils import add_fake_results, add_noise

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config = {
    "Project_Name": "Input Output Debug",
    "Allow_repeated_recommendations": True,
    "Allow_recommending_already_measured": True,
    "Num_measurements_must_be_within_tolerance": True,
    "Parameters": [
        {
            "Name": "Categorical_1",
            "Type": "CAT",
            "Values": [22, 33],
        },
        {
            "Name": "Categorical_2",
            "Type": "CAT",
            "Values": ["bad", "OK", "good"],
            "Encoding": "Integer",
        },
        {
            "Name": "Num_disc_1",
            "Type": "NUM_DISCRETE",
            "Values": [1, 2, 3],
            "Tolerance": 0.3,
        },
        {
            "Name": "Num_disc_2",
            "Type": "NUM_DISCRETE",
            "Values": [-1, -3, -6],
            "Tolerance": 0.3,
        },
    ],
    "Objective": {
        "Mode": "SINGLE",
        "Targets": [
            {"Name": "Target_1", "Bounds": None, "Mode": "Min"},
        ],
    },
}

# Define some parameter values to define rows where the fake results should be good
good_reference_values = [
    {"Parameter": "Categorical_2", "Value": "OK"},
    {"Parameter": "Categorical_1", "Value": 22},
]

# Create BayBE object, add fake results and print what happens to internal data
baybe_obj = BayBE(config=config)
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
