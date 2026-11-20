"""Column names shared across the multi-step dataframe projections."""

from __future__ import annotations

NODE = "node"
"""Column identifying the node whose parameters a recommendation covers."""

EXPERIMENT_ID = "experiment_id"
"""Column identifying the experiment a recommendation or measurement belongs to."""

FEASIBILITY_LABEL = "__baybe_feasibility_success"
"""Internal binary label derived from a node's feasibility conditions."""

STOCK_CAMPAIGN_COLUMNS = ("BatchNr", "FitNr")
"""Bookkeeping columns written by :class:`~baybe.campaign.Campaign`.

Parameter and output names must not shadow them so that terminal measurement
frames remain compatible with the stock campaign interface.
"""

RESERVED_COLUMNS = (
    NODE,
    EXPERIMENT_ID,
    FEASIBILITY_LABEL,
    *STOCK_CAMPAIGN_COLUMNS,
)
"""All columns that parameter and output names must not shadow."""
