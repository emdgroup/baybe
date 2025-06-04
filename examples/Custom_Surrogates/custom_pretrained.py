## Example for surrogate model with a custom pretrained model

# This example shows how to pre-train a model and use it as a surrogate.
# Please note that the model is not designed to be useful but to demonstrate the workflow.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports

import numpy as np
import torch
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.operator_converters.linear_regressor import convert_sklearn_bayesian_ridge
from sklearn.linear_model import BayesianRidge

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import CustomONNXSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements, to_tensor

### Experiment Setup

parameters = [
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    NumericalDiscreteParameter(
        name="Temperature[degree_C]",
        values=np.linspace(100, 200, 10),
    ),
]


### "Pre-training" stage

# Note that this example trains with several helpers built-in to BayBE.
# This can be done independently (and elsewhere).
# The only requirement that BayBE needs is that the model is in an onnx format.
# And The format should return both the mean and standard deviation.
# This example is based on a `BayesianRidge` regressor from `sklearn`.
# Its native conversion to onnx is supported via `skl2onnx`.
# Please also note that this example does not give a useful model.
# Its purpose is to show the workflow for using pre-trained surrogates in BayBE.

searchspace = SearchSpace.from_product(parameters=parameters, constraints=None)
train_x = to_tensor(searchspace.discrete.comp_rep)
train_y = torch.rand(train_x.size(dim=0))  # train with a random y vector

# Define model and fit

model = BayesianRidge()
model.fit(train_x, train_y)


### Convert model to onnx

# Need the option to return standard deviation

options = {type(model): {"return_std": True}}

# Specify what the input name is

ONNX_INPUT_NAME = "example_input_name"

# input dimensions and input type (should always be a float)

input_dim = train_x.size(dim=1)
initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, input_dim]))]

# Conversion

onnx_str = convert_sklearn(
    model,
    initial_types=initial_type,
    options=options,
    custom_conversion_functions={type(model): convert_sklearn_bayesian_ridge},
).SerializeToString()  # serialize to string to save in file


### Create a surrogate model with a pretrained model

surrogate_model = CustomONNXSurrogate(
    onnx_str=onnx_str,
    onnx_input_name=ONNX_INPUT_NAME,  # specify input name
)

### Create campaign

campaign = Campaign(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=SingleTargetObjective(target=NumericalTarget(name="Yield", mode="MAX")),
    recommender=TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(surrogate_model=surrogate_model),
        initial_recommender=FPSRecommender(),
    ),
)

### Iterate with recommendations and measurements

# Let's do a first round of recommendation
recommendation = campaign.recommend(batch_size=1)

print("Recommendation from campaign:")
print(recommendation)

# Add some fake results

add_fake_measurements(recommendation, campaign.targets)
campaign.add_measurements(recommendation)

### Model Outputs

# Do another round of recommendation
recommendation = campaign.recommend(batch_size=1)

# Print second round of recommendation

print("Recommendation from campaign:")
print(recommendation)


### Using configuration instead

# Note that this can be placed inside an overall baybe config
# Refer to [`create_from_config`](./../Serialization/create_from_config.md) for an example

CONFIG = {
    "type": "CustomONNXSurrogate",
    "onnx_str": onnx_str,
    "onnx_input_name": ONNX_INPUT_NAME,
}

### Model creation from dict (or json if string)
model_from_python = CustomONNXSurrogate(
    onnx_str=onnx_str, onnx_input_name=ONNX_INPUT_NAME
)
model_from_configs = CustomONNXSurrogate.from_dict(CONFIG)

# This configuration creates the same model

assert model_from_python == model_from_configs

# JSON configuration (expects onnx_str to be decoded with `ISO-8859-1`)

model_json = model_from_python.to_json()
assert model_from_python == CustomONNXSurrogate.from_json(model_json)
