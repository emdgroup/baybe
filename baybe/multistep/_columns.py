"""Column names shared across the multi-step dataframe projections."""

from __future__ import annotations

NODE = "node"
"""Column identifying the node whose parameters a recommendation covers."""

STOCK_CAMPAIGN_COLUMNS = ("BatchNr", "FitNr")
"""Bookkeeping columns written by :class:`~baybe.campaign.Campaign`.

Parameter and output names must not shadow them so that terminal measurement
frames remain compatible with the stock campaign interface.
"""

RESERVED_COLUMNS = (
    NODE,
    *STOCK_CAMPAIGN_COLUMNS,
)
"""All columns that parameter and output names must not shadow."""
