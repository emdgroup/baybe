"""Cost aware wrapping utilities for single-fidelity MC-based acquisiton functions."""

from typing import TYPE_CHECKING

from attrs import field
from attrs.validators import instance_of
from attrs.validators import optional as optional_v
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.models.deterministic import DeterministicModel
from typing_extensions import override

from baybe.parameters.fidelity import NumericalDiscreteFidelityParameter
from baybe.searchspace.core import SearchSpace, SearchSpaceCostType

if TYPE_CHECKING:
    from torch import Tensor


# Jordan MHS TODO: typing for fidelities_dict awkward since integer values in
# comp_df not explicitly typed. Seek help here.
def make_cost_tensor(searchspace: SearchSpace, /) -> tuple[Tensor, Tensor]:
    """Construct column indices, comp_df values and costs of costly parameters."""
    import torch

    if searchspace.cost_type == SearchSpaceCostType.MULTIFIDELITY:
        params = (
            p
            for p in searchspace.parameters
            if isinstance(p, NumericalDiscreteFidelityParameter)
        )

    elif searchspace.cost_type == SearchSpaceCostType.FIXEDDISCRETECOSTS:
        params = (p for p in searchspace.parameters if p.is_costly)

    values_dict = {i: tuple(p.comp_df.iloc[:, 0]) for i, p in enumerate(params)}

    costs_dict = {
        i: p.costs
        if getattr(p, "costs", None) is not None
        else tuple(0 for _ in p.values)
        for i, p in enumerate(params)
    }

    num_params = len(params)
    max_len = max(len(p.comp_df.iloc[:, 0]) for p in params)

    costly_indices = values_dict.keys()

    values = torch.full((num_params, max_len), float("nan"))
    costs = torch.zeros((num_params, max_len))

    for i in range(num_params):
        v = torch.tensor(values_dict[i])
        c = torch.tensor(costs_dict[i])

        values[i, : len(v)] = v
        costs[i, : len(c)] = c

    return costly_indices, values, costs


class discrete_cost_model(DeterministicModel):
    """A fixed cost model which matches discrete parameter values to their costs.

    A sum is taken over all costly parameters.
    """

    # Possible TODO: add validation that values and costs have the
    # same shape and are compatible with costly_indices.
    # Then move discrete_cost_model to its own file for custom
    # post-search space definition discrete cost models.
    costly_indices: Tensor = field(validator=instance_of(Tensor))
    """Computational representation of the parameters which are costly."""

    values: Tensor = field(validator=instance_of(Tensor))
    """Values for each costly parameter, padded with repeats to make rectangular."""

    costs: Tensor = field(validator=instance_of(Tensor))
    """Costs for each parameter, padded with repeats to make rectangular."""

    @override
    def foward(self, X: Tensor) -> Tensor:
        """Compute costs over a set of candidate values.

        Args:
            X: A `(b1 x ... bk) x q x d`-dim tensor of `d`-dim design.

        Returns:
            A `(b1 x ... bk) x q`-dim tensor of of evaluated costs.
        """
        X_sub = X.squeeze(-2)[..., self.costly_indices]

        matches = X_sub.unsqueeze(-1) == self.values

        cost_per_dim = (matches * self.costs).sum(dim=-1)

        return cost_per_dim.sum(dim=-1, keepdim=True)


class InverseCostWeightedAcquisitionObjective(MCAcquisitionObjective):
    """Wrapper for computing cost aware acquisiton values at the MC sample level."""

    base_objective: MCAcquisitionObjective | None = field(
        validator=optional_v(instance_of(MCAcquisitionObjective)), default=None
    )
    """Base objective to be wrapped with cost adjustment."""

    cost_model: DeterministicModel = field(
        validator=[instance_of(DeterministicModel)],
    )
    """Deterministic model of design choice."""

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        """Compute cost aware acquisition value within MC sampling."""
        cost = self.cost_model(samples)

        if self.base_objective is None:
            return cost
        else:
            base = self.base_objective(samples, X)
            return base / cost


def make_inverse_cost_utility(
    searchspace: SearchSpace, /
) -> InverseCostWeightedUtility:
    """Make an inverse cost wrapping utility from a searchspace, suitable for qMFKG."""
    costly_indices, values, costs = make_cost_tensor(searchspace)

    cost_model = discrete_cost_model(costly_indices, values, costs)

    return InverseCostWeightedUtility(cost_model=cost_model)


def wrap_cost_aware_objective(
    objective: MCAcquisitionObjective, searchspace: SearchSpace
):
    """Make an inverse cost-wrapped acquisition objective."""
    costly_indices, values, costs = make_cost_tensor(searchspace)

    cost_model = discrete_cost_model(costly_indices, values, costs)

    return InverseCostWeightedAcquisitionObjective(objective, cost_model)
