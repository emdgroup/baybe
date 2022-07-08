"""Test for initial simple input, recommendation and adding fake results.
Fake target measurements are simulated as well as noise added to the parameters.
"""
import logging

from baybe.core import BayBE
from baybe.utils import add_fake_results, add_noise

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config = {
    "Project_Name": "Input Output Debug Test",
    "Allow_repeated_recommendations": False,
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
            "Values": [1, 2, 8],
            "Tolerance": 0.3,
        },
    ],
    "Objective": {
        "Mode": "SINGLE",
        "Objectives": [{"Name": "Target_1", "Type": "NUM", "Bounds": None}],
    },
}

# Define some values where the fake results should be good
good_reference_values = [
    {"Parameter": "Categorical_2", "Value": "OK"},
    {"Parameter": "Categorical_1", "Value": 22},
]

# Create BayBE object, add fake results and print what happens to internal data
obj = BayBE(config=config)
print(obj)

N_ITERATIONS = 4
for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = obj.recommend(batch_quantity=3)
    print("\n\n### Recommended dataframe:\n", rec)

    add_fake_results(rec, obj, good_reference_values=good_reference_values)
    if kIter % 2:
        print(kIter)
        add_noise(rec, obj, noise_level=0.1)
    print("\n\n### Recommended dataframe with fake results and eventual noise:\n", rec)

    # uncomment below to test error throw for disallowed value
    # obj.add_results(rec.replace(1, 11111))
    obj.add_results(rec)
    print(
        "\n\n### Internal measurement dataframe after data ingestion:\n",
        obj.measurements_exp_rep,
    )

    print(
        "\n\n### Internal measurement dataframe computational representation X:\n",
        obj.measurements_comp_rep_x,
    )

    print(
        "\n\n### Internal measurement dataframe computational representation Y:\n",
        obj.measurements_comp_rep_y,
    )

# Show metadata
print("\n\n### Search Space Metadata after all iterations\n", obj.searchspace_metadata)
