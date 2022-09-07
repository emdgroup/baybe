# BayBE -- A Bayesian Back End for Design of Experiments
This software package provides a general-purpose toolbox for **Design of Experiments 
(DOE)**.

It provides the necessary functionality to:
- manage experimental design spaces that can be defined through various types of experimental parameters.
- execute different kinds of DOE strategies, levering both classical and machine-learning-based models.
- handle measurements data and feeding it back into the experimental design.
- compare different DOE strategies using simulation on synthetic and real data.

### Getting Started
To get a batch of recommendations for the next set of experiments to be conducted,
define the underlying search space by listing the associated experimental parameters.
For example, a set of (discrete) numeric parameters, which can take only certain
specified values, may be defined as follows (see `baybe.parameters` for alternative 
options).
```
parameters = [
    {
        "name": "Temperature",
        "type": "NUM_DISCRETE",
        "values": [100, 110, 120, 130, 140, 150, 160],
        "tolerance": 3,
    },
    {
        "name": "Pressure",
        "type": "NUM_DISCRETE",
        "values": [6, 8, 10],
        "tolerance": 0.5,
    },
]
```

The corresponding optimization task is then specified through an `Objective`,
which may comprise one or multiple (potentially competing) `Targets` and defines how 
these should be balanced:
```
objective = {
    "mode": "SINGLE",
    "targets": [
        {
            "name": "Yield",
            "type": "NUM",
            "mode": "MAX",
        },
    ],
}
```

With this minimal setup (using default values for all other options), an initial
set of recommendations can be generated as follows:
```
from baybe.core import BayBE, BayBEConfig
config_dict = {
    "parameters": parameters,
    "objective": objective,
}
config = BayBEConfig(**config_dict)
baybe = BayBE(config)
baybe.recommend(batch_quantity=5)
```

At any point in time (also before querying the first recommendations), available 
measurements can be included into the design by passing a corresponding dataframe:
```
import pandas as pd
measurements = pd.DataFrame.from_records(
    [
        {"Temperature": 100, "Pressure": 6, "Yield": 0.8},
        {"Temperature": 160, "Pressure": 6, "Yield": 0.4},
    ]
)
baybe.add_results(measurements)
```
