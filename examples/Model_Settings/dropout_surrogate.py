### Custom Model example based on a Neural Network with Monte Carlo Dropout
# pylint: disable=unused-argument

"""
This example shows the creation of a Neural Network with Monte Carlo Dropout
using `pytorch` and the registration of the architecture for BayBE.
"""

# Please note that this is an example architecture
# The model is not designed to be useful but to demonstrate the workflow

#### Necessary imports

from typing import List, Optional, Tuple

import torch

from baybe.searchspace import SearchSpace
from baybe.surrogate import register_custom_architecture

from torch import nn, Tensor
from torch.autograd import Variable

#### Model Configuration

INPUT_DIM = 10
OUTPUT_DIM = 1
DROPOUT = 0.5
NUM_NEURONS = [128, 32, 8]

#### Model training hyperparameters

HYPERPARAMS = {
    "epochs": 10,
    "lr": 1e-3,
    "criterion": nn.MSELoss,
    "optimizer": torch.optim.Adam,
}

#### MC Parameters

MC = 100


#### Helper functions


def _create_linear_block(in_features: int, out_features: int) -> list:
    """Creates a linear block with dropout and relu activation"""
    return [nn.Linear(in_features, out_features), nn.Dropout(p=DROPOUT), nn.ReLU()]


def _create_hidden_layers(num_neurons: List[int]) -> list:
    """Creates all hidden layers comprised of linear blocks"""
    layers = []
    for in_features, out_features in zip(num_neurons, num_neurons[1:]):
        layers.extend(_create_linear_block(in_features, out_features))

    return layers


#### Model Architecture


class NeuralNetDropout(nn.Module):
    """Pytorch implementation of Neural Network with Dropout"""

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
        """Forward method for NN"""
        return self.model(data)


#### Surrogate Definition with BayBE Registration

# The class must include `_fit` and `_posterior` functions with the correct signatures

# Registration
@register_custom_architecture(
    joint_posterior_attr=False, constant_target_catching=False, batchify_posterior=True
)
class NeuralNetDropoutSurrogate:
    """
    Function definitions for a surrogate model that
    extracts posterior uncertainty using monte carlo
    dropout simulations.
    """

    def __init__(self):
        self.model: Optional[nn.Module] = None

    def _posterior(self, candidates: Tensor) -> Tuple[Tensor]:
        """See baybe.surrogate.Surrogate."""
        self.model = self.model.train()  # keep dropout
        # Convert input from double to float
        candidates = Variable(candidates.type(torch.FloatTensor))
        # Run mc experiments through the NN with dropout
        predictions = torch.cat(
            [self.model(candidates).unsqueeze(dim=0) for _ in range(MC)]
        )

        # Compute posterior mean and variance
        mean = predictions.mean(dim=0)
        var = predictions.var(dim=0)

        return mean, var

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        """See baybe.surrogate.Surrogate."""
        # Initialize Model
        self.model = NeuralNetDropout()

        # Training hyperparameters
        opt = HYPERPARAMS["optimizer"](self.model.parameters(), lr=HYPERPARAMS["lr"])
        criterion = HYPERPARAMS["criterion"]()

        # Convert input from double to float
        train_x = Variable(train_x.type(torch.FloatTensor))
        train_y = Variable(train_y.type(torch.FloatTensor))

        # Training loop
        for _ in range(HYPERPARAMS["epochs"]):
            opt.zero_grad()
            preds = self.model(train_x)
            loss = criterion(preds, train_y)
            loss.backward()
            opt.step()
