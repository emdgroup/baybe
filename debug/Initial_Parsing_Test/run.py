"""Test for initial simple parameter parsing"""

from baybe.core import BayBE


# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config = {
    "project_name": "Initial Core Test",
    "parameters": [
        {
            "name": "Categorical_1",
            "type": "CAT",
            "values": [22, 33],
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
            "values": [1, 2, 8],
            "tolerance": 0.3,
        },
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [{"name": "Target_1", "type": "NUM", "bounds": None}],
    },
}

# Create BayBE object and print a summary
obj = BayBE(config=config)

print(obj)
