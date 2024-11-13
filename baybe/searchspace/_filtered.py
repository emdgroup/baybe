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

    allow_repeated_recommendations: bool = field(kw_only=True)
    """Allow to make recommendations that were already recommended earlier.
    This only has an influence in discrete search spaces."""

    allow_recommending_already_measured: bool = field(kw_only=True)
    """Allow to make recommendations that were measured previously.
    This only has an influence in discrete search spaces."""

    @classmethod
    def from_subspace(
        cls,
        subspace: SubspaceDiscrete,
        *,
        metadata: pd.DataFrame,
        allow_repeated_recommendations: bool,
        allow_recommending_already_measured: bool,
    ) -> Self:
        """Annotate an existing subspace with metadata."""
        return cls(
            **asdict(subspace, filter=lambda attr, _: attr.init, recurse=False),
            metadata=metadata,
            allow_repeated_recommendations=allow_repeated_recommendations,
            allow_recommending_already_measured=allow_recommending_already_measured,
        )

    @override
    def get_candidates(
        self, exclude: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from baybe.campaign import _EXCLUDED, _MEASURED, _RECOMMENDED

        # Exclude parts marked by metadata
        mask_todrop = self.metadata[_EXCLUDED].copy()
        if not self.allow_repeated_recommendations:
            mask_todrop |= self.metadata[_RECOMMENDED]
        if not self.allow_recommending_already_measured:
            mask_todrop |= self.metadata[_MEASURED]

        # Remove additional excludes
        if exclude is not None:
            mask_todrop |= pd.merge(self.exp_rep, exclude, indicator=True, how="left")[
                "_merge"
            ].eq("both")

        return self.exp_rep.loc[~mask_todrop], self.comp_rep.loc[~mask_todrop]
