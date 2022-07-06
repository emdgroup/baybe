"""Test for initial simple parameter parsing"""
import logging

from baybe.core import BayBE
from baybe.utils import add_fake_results

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config = {
    "Experiment_Name": "Initial Core Test",
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
good_values = [{"Parameter": "Categorical_2", "Value": "OK"}]

# Create BayBE object, add fake results and print what happens to internal data
obj = BayBE(config=config)
print(obj)

rec = obj.recommend()
print("\n\nRecommended in Iteration 1:\n", rec)

add_fake_results(rec, obj, good_values=good_values)
print("\n\nAfter Adding Fake Results Iteration 1:\n", rec)

rec = obj.recommend()
print("\n\nRecommended in Iteration 1:\n", rec)

add_fake_results(rec, obj, good_values=good_values)
print("\n\nAfter Adding Fake Results Iteration 2:\n", rec)

print("\n\nSearch Space Metadata after all iterations\n", obj.searchspace_metadata)
