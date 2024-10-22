"""Search spaces with metadata."""

import pandas as pd
from attrs import asdict, define, field
from typing_extensions import Self, override

from baybe.searchspace import SubspaceDiscrete
from baybe.utils.boolean import eq_dataframe


@define
class AnnotatedSubspaceDiscrete(SubspaceDiscrete):
    """An annotated search space carrying additional metadata."""

    metadata: pd.DataFrame = field(kw_only=True, eq=eq_dataframe)
    """The metadata."""

    @classmethod
    def from_subspace(cls, subspace: SubspaceDiscrete, metadata: pd.DataFrame) -> Self:
        """Annotate an existing subspace with metadata."""
        return cls(
            **asdict(subspace, filter=lambda attr, _: attr.init, recurse=False),
            metadata=metadata,
        )

    @override
    def get_candidates(
        self,
        allow_repeated_recommendations: bool = False,
        allow_recommending_already_measured: bool = False,
        exclude: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from baybe.campaign import _DONT_RECOMMEND, _WAS_MEASURED, _WAS_RECOMMENDED

        # Exclude parts marked by metadata
        mask_todrop = self.metadata[_DONT_RECOMMEND].copy()
        if not allow_repeated_recommendations:
            mask_todrop |= self.metadata[_WAS_RECOMMENDED]
        if not allow_recommending_already_measured:
            mask_todrop |= self.metadata[_WAS_MEASURED]

        # Remove additional excludes
        if exclude is not None:
            mask_todrop |= pd.merge(self.exp_rep, exclude, indicator=True, how="left")[
                "_merge"
            ].eq("both")

        return self.exp_rep.loc[~mask_todrop], self.comp_rep.loc[~mask_todrop]
