"""Utils for bayesian recommenders."""

from attrs import evolve

from baybe.parameters import CategoricalFidelityParameter
from baybe.searchspace import SearchSpace


def restricted_fidelity_searchspace(searchspace: SearchSpace, /) -> SearchSpace:
    """Evolve a multi-fidelity searchspace so the fidelity is fixed to the highest."""
    discrete_parameters_fixed_fidelities = tuple(
        evolve(
            p,
            values=(p.highest_fidelity,),
            costs=(p.highest_fidelity_cost,),
            zeta=(0.0,),
        )
        if isinstance(p, CategoricalFidelityParameter)
        else p
        for p in searchspace.discrete.parameters
    )

    discrete_subspace_fixed_fidelities = evolve(
        searchspace.discrete, parameters=discrete_parameters_fixed_fidelities
    )

    fixed_fidelity_searchspace = evolve(
        searchspace, discrete=discrete_subspace_fixed_fidelities
    )

    return fixed_fidelity_searchspace
