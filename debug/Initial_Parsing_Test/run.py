"""Test for initial simple parameter parsing"""

from baybe.core import BayBE, BayBEConfig


# Simple example with one numerical target, two categorical and one numerical discrete
# parameter
config_dict = {
    "project_name": "Initial Core Test",
    "parameters": [
        {
            "name": "Categorical_1",
            "type": "CAT",
            "values": [22, 33],
            # "encoding": "OHE",
        },
        {
            "name": "Categorical_2",
            "type": "CAT",
            "values": ["bad", "OK", "good"],
            "encoding": "INT",
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
        "targets": [{"name": "Target_1", "type": "NUM", "bounds": None, "mode": "MIN"}],
    },
    "strategy": {
        # "surrogate_model_cls": "GP",
    },
}

# Create BayBE object and print a summary
config = BayBEConfig(**config_dict)
obj = BayBE(config=config)

print(obj)
