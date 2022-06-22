"""Test for initial simple parameter parsing"""
import logging

from baybe.core import BayBE

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
            "Values": [1, 2],
        },
        {"Name": "Categorical_2", "Type": "CAT", "Values": ["on", "off"]},
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

# Create BayBE object and print a summary
obj = BayBE(config=config)

obj.print_summary()
