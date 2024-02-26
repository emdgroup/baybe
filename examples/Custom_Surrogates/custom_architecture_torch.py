## Example for surrogate model with a custom architecture using `pytorch`

# This example shows how to define a `pytorch` model architecture and use it as a surrogate.
# Please note that the model is not designed to be useful but to demonstrate the workflow.

# This example assumes some basic familiarity with using BayBE.
# We thus refer to [`campaign`](./../Basics/campaign.md) for a basic example.

### Necessary imports

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from baybe.campaign import Campaign
from baybe.objective import Objective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
)
from baybe.recommenders import (
    FPSRecommender,
    SequentialGreedyRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import register_custom_architecture
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import add_fake_results

### Architecture definition

# Note that the following is an example `PyTorch` Neural Network.
# Details of the setup is not the focus of BayBE but can be found in `Pytorch` guides.

# Model Configuration

INPUT_DIM = 10
OUTPUT_DIM = 1
DROPOUT = 0.5
NUM_NEURONS = [128, 32, 8]

# Model training hyperparameters

HYPERPARAMS = {
    "epochs": 10,
    "lr": 1e-3,
    "criterion": nn.MSELoss,
    "optimizer": torch.optim.Adam,
}

# MC Parameters

MC = 100


# Helper functions


def _create_linear_block(in_features: int, out_features: int) -> list:
    """Create a linear block with dropout and relu activation."""
    return [nn.Linear(in_features, out_features), nn.Dropout(p=DROPOUT), nn.ReLU()]


def _create_hidden_layers(num_neurons: List[int]) -> list:
    """Create all hidden layers comprised of linear blocks."""
    layers = []
    for in_features, out_features in zip(num_neurons, num_neurons[1:]):
        layers.extend(_create_linear_block(in_features, out_features))

    return layers


# Model Architecture


class NeuralNetDropout(nn.Module):
    """Pytorch implementation of Neural Network with Dropout."""

    def __init__(self):
        super().__init__()
        layers = [
            # Initial linear block with input
            *(_create_linear_block(INPUT_DIM, NUM_NEURONS[0])),
            # All hidden layers
            *(_create_hidden_layers(NUM_NEURONS)),
            # Last linear output
            nn.Linear(NUM_NEURONS[-1], OUTPUT_DIM),
        ]

        # Sequential with layers (Feed Forward)
        self.model = nn.Sequential(*layers)

    def forward(self, data: Tensor) -> Tensor:
        """Forward method for NN."""
        return self.model(data)


### Surrogate Definition with BayBE Registration

# The class must include `_fit` and `_posterior` functions with the correct signatures


# Registration


@register_custom_architecture(
    joint_posterior_attr=False, constant_target_catching=False, batchify_posterior=True
)
class NeuralNetDropoutSurrogate:
    """Surrogate that extracts posterior using monte carlo dropout simulations."""

    def __init__(self):
        self.model: Optional[nn.Module] = None

    def _posterior(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """See :class:`baybe.surrogates.Surrogate`."""
        self.model = self.model.train()  # keep dropout
        # Convert input from double to float
        candidates = candidates.float()
        # Run mc experiments through the NN with dropout
        predictions = torch.cat(
            [self.model(candidates).unsqueeze(dim=0) for _ in range(MC)]
        )

        # Compute posterior mean and variance
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """See :class:`baybe.surrogates.Surrogate`."""
        # Initialize Model
        self.model = NeuralNetDropout()

        # Training hyperparameters
        opt = HYPERPARAMS["optimizer"](self.model.parameters(), lr=HYPERPARAMS["lr"])
        criterion = HYPERPARAMS["criterion"]()

        # Convert input from double to float
        train_x = train_x.float()
        train_y = train_y.float()

        # Training loop
        for _ in range(HYPERPARAMS["epochs"]):
            opt.zero_grad()
            preds = self.model(train_x)
            loss = criterion(preds, train_y)
            loss.backward()
            opt.step()


### Experiment Setup

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


### Run DOE iterations with custom surrogate
# Create campaign

campaign = Campaign(
    searchspace=SearchSpace.from_product(parameters=parameters, constraints=None),
    objective=Objective(
        mode="SINGLE", targets=[NumericalTarget(name="Yield", mode="MAX")]
    ),
    recommender=TwoPhaseMetaRecommender(
        recommender=SequentialGreedyRecommender(
            surrogate_model=NeuralNetDropoutSurrogate()
        ),
        initial_recommender=FPSRecommender(),
    ),
)

# Let's do a first round of recommendation
recommendation = campaign.recommend(batch_size=2)

print("Recommendation from campaign:")
print(recommendation)

# Add some fake results

add_fake_results(recommendation, campaign)
campaign.add_measurements(recommendation)

# Do another round of recommendations
recommendation = campaign.recommend(batch_size=2)

# Print second round of recommendations

print("Recommendation from campaign:")
print(recommendation)

print()


### Serialization

# Serialization of custom models is not supported

try:
    campaign.to_json()
except RuntimeError as e:
    print(f"Serialization Error Message: {e}")
