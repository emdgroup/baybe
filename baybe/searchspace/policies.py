"""Policies for transforming candidate sets.

A policy transforms a CandidatesProtocol into a new CandidatesProtocol.
The output is always finite (typically TableCandidates).

Policies are responsible for:
- Materializing infinite candidate spaces into finite frames.
- Filtering or resampling finite candidate sets.
- Generating entirely new candidates (e.g. from a generative model).
"""

from copy import deepcopy
from typing import Protocol, runtime_checkable

from attrs import define, field
from typing_extensions import override

from baybe.searchspace import CandidatesProtocol, TableCandidates


@runtime_checkable
class PolicyProtocol(Protocol):
    """Type Protocol specifying the interface policies have to implement policies."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    # TODO: tbd - Offer from different CandidatesTypes i.e. ProductCandidates or other

    def __call__(self, candidates: CandidatesProtocol) -> CandidatesProtocol:
        """Transform or materialize a candidate set.

        Args:
            candidates: The candidate generator to operate on. May be finite or
                infinite. If infinite, the policy is responsible for materializing the
                candidates.

        Returns:
            A CandidatesProtocol (typically TableCandidates) whose columns match
            those of the input parameters in their experimental representation.
        """


@define(frozen=True)
class PolicyChain(PolicyProtocol):
    """Composes multiple policies into a single policy, applied in ordered sequence."""

    policies: tuple[PolicyProtocol, ...] = field()
    """The ordered sequence of policies to apply."""

    @override
    def __call__(self, candidates: CandidatesProtocol) -> CandidatesProtocol:
        """Apply each policy in sequence, threading the output forward."""
        # TODO: Sequential Chaining of policies: Only the first one can be
        #  unmaterialized / infinite
        result = deepcopy(candidates)
        for policy in self.policies:
            result = policy(result)
        return result


@define
class RandomSamplingPolicy(PolicyProtocol):
    """Randomly samples a fixed number of candidates from the input candidate set."""

    n: int = field()
    """The number of candidates to sample."""

    seed: int | None = field(default=None)
    """Random seed for reproducibility."""

    @override
    def __call__(self, candidates: CandidatesProtocol) -> CandidatesProtocol:
        """Randomly sample n candidates from the input candidate set."""
        # TODO: Flag (?) Random subsampling from full candidates set or from  smaller
        #  grid (subsample parameters and then build cartesian product for subsampled)
        # TODO: Other strategy for other candidate types may be required
        # TODO: N has to be smaller than the number of candidates
        return TableCandidates(
            parameters=candidates.parameters,
            dataframe=candidates.to_lazy().collect().sample(n=self.n, seed=self.seed),
        )
