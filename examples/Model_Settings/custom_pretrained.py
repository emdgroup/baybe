### Example for surrogate model with a custom pretrained model

"""
This example shows the creation of a BayBE object,
and the usage of pre-trained models as surrogates in BayBE.
"""

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`baybe_object`](./../Basics/baybe_object.md) for a basic example.

#### Necessary imports

import numpy as np
import torch

from baybe.core import BayBE
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.searchspace import SearchSpace
from baybe.strategies import FPSRecommender, SequentialGreedyRecommender, Strategy
from baybe.surrogate import CustomPretrainedSurrogate
from baybe.targets import NumericalTarget, Objective
from baybe.utils import add_fake_results, to_tensor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.operator_converters.linear_regressor import convert_sklearn_bayesian_ridge

from sklearn.linear_model import BayesianRidge


#### Experiment Setup

parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    NumericalDiscreteParameter(
        name="Temperature[degree_C]",
        values=np.linspace(100, 200, 10),
    ),
    SubstanceParameter(
        name="Solvent",
        data={
            "Solvent A": "COC",
            "Solvent B": "CCC",
            "Solvent C": "O",
            "Solvent D": "CS(=O)C",
        },
        encoding="MORDRED",
    ),
]


#### "Pre-training" stage

# Note that this example trains with several helpers built-in to BayBE
# but this can be done independently (and elsewhere)

# The only requirement that BayBE needs is that the model is
# in an onnx format which would return both the mean and standard deviation

# This example is based on a `BayesianRidge` regressor from `sklearn`
# where native conversion to onnx is supported via `skl2onnx`

# Please also note that this example does not give a useful model
# but to show what the workflow is for using pre-trained surrogates in BayBE

searchspace = SearchSpace.from_product(parameters=parameters, constraints=None)
train_x = to_tensor(searchspace.discrete.comp_rep)
train_y = torch.rand(train_x.size(dim=0))  # train with a random y vector

# Define model and fit
model = BayesianRidge()
model.fit(train_x, train_y)


#### Convert model to onnx

# Need the option to return standard devication
options = {type(model): {"return_std": True}}

# Specify what the input name is
ONNX_INPUT_NAME = "example_input_name"

# input dimensions and input type (shold always be a float)
input_dim = train_x.size(dim=1)
initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, input_dim]))]

# Conversion
onnx_str = convert_sklearn(
    model,
    initial_types=initial_type,
    options=options,
    custom_conversion_functions={type(model): convert_sklearn_bayesian_ridge},
).SerializeToString()  # serialize to string to save in file


#### Create a surrogate model with a pretrained model

# onnx string must decoded with ISO-8859-1 for serialization purposes
model_params = {
    "onnx": onnx_str.decode("ISO-8859-1"),
    "onnx_input_name": ONNX_INPUT_NAME,  # specify input name
}

surrogate_model = CustomPretrainedSurrogate(model_params=model_params)


#### Validation

invalid_model_params_no_name = {
    "onnx": onnx_str.decode("ISO-8859-1"),
}

invalid_model_params_onnx = {
    "onnx": onnx_str,
    "onnx_input_name": ONNX_INPUT_NAME,  # specify input name
}

for invalid_model_params in (invalid_model_params_no_name, invalid_model_params_onnx):
    try:
        _ = CustomPretrainedSurrogate(model_params=invalid_model_params)
    except ValueError as e:
        print(f"Error message: {e}")


#### Create BayBE object

baybe_obj = BayBE(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=Objective(
        mode="SINGLE", targets=[NumericalTarget(name="Yield", mode="MAX")]
    ),
    strategy=Strategy(
        recommender=SequentialGreedyRecommender(surrogate_model=surrogate_model),
        initial_recommender=FPSRecommender(),
    ),
)

#### Iterate with recommendations and measurements

# Let's do a first round of recommendation
recommendation = baybe_obj.recommend(batch_quantity=2)

print("Recommendation from baybe object:")
print(recommendation)

# Add some fake results
add_fake_results(recommendation, baybe_obj)
baybe_obj.add_measurements(recommendation)

#### Model Outputs

# Note that this model is only triggered when there is data.
print("Here you will see some model outputs as we set verbose to True")

# Do another round of recommendations
recommendation = baybe_obj.recommend(batch_quantity=2)

# Print second round of recommendations
print("Recommendation from baybe object:")
print(recommendation)


#### Using configuration instead

# Note that this can be placed inside an overall baybe config
# Refer to [`create_from_config`](./../Serialization/create_from_config.md) for an example

CONFIG = {"type": "CustomPretrainedSurrogate", "model_params": model_params}

#### Model creation from dict (or json if string)
model_from_python = CustomPretrainedSurrogate(model_params=model_params)
model_from_configs = CustomPretrainedSurrogate.from_dict(CONFIG)

# This configuration creates the same model
assert model_from_python == model_from_configs
