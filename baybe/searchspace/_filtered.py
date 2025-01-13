"""Search spaces with metadata."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import asdict, cmp_using, define, field
from attrs.validators import instance_of
from typing_extensions import Self, override

from baybe.searchspace import SubspaceDiscrete


@define
class FilteredSubspaceDiscrete(SubspaceDiscrete):
    """A filtered search space representing a reduced candidate set."""

    mask_keep: npt.NDArray[np.bool_] = field(
        validator=instance_of(np.ndarray),
        kw_only=True,
        eq=cmp_using(eq=np.array_equal),
    )
    """The filtering mask. ``True`` marks elements to be kept."""

    @mask_keep.validator
    def _validate_mask_keep(self, _, value) -> None:
        if not len(value) == len(self.exp_rep):
            raise ValueError("Filter mask must match the size of the space.")
        if not value.dtype == bool:
            raise ValueError("Filter mask must only contain Boolean values.")

    @classmethod
    def from_subspace(
        cls, subspace: SubspaceDiscrete, mask_keep: npt.NDArray[np.bool_]
    ) -> Self:
        """Filter an existing subspace."""
        return cls(
            **asdict(subspace, filter=lambda attr, _: attr.init, recurse=False),
            mask_keep=mask_keep,
        )

    @override
    def get_candidates(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask_todrop = self._excluded.copy()
        mask_todrop |= ~self.mask_keep
        return self.exp_rep.loc[~mask_todrop], self.comp_rep.loc[~mask_todrop]
