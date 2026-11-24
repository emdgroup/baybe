"""Column names shared across the multi-step dataframe projections."""

from __future__ import annotations

NODE = "node"
"""Column identifying the node whose parameters a recommendation covers."""

EXPERIMENT_ID = "experiment_id"
"""Column identifying the experiment a recommendation or measurement belongs to."""

FEASIBILITY_LABEL = "__baybe_feasibility_success"
"""Internal binary label derived from a node's feasibility conditions."""

MEASUREMENT_ID = "__baybe_measurement_id"
"""Stable identity of one raw node measurement used for cross-fitting."""

EXPERIMENT_GROUP = "__baybe_experiment_group"
"""Group key keeping measurements from one experiment in one cross-fit fold."""

FEASIBILITY_FEATURE_PREFIX = "__baybe_feasibility_probability_"
"""Prefix for collision-free output-posterior features of the node classifier."""

STOCK_CAMPAIGN_COLUMNS = ("BatchNr", "FitNr")
"""Bookkeeping columns written by :class:`~baybe.campaign.Campaign`.

Parameter and output names must not shadow them so that terminal measurement
frames remain compatible with the stock campaign interface.
"""

RESERVED_COLUMNS = (
    NODE,
    EXPERIMENT_ID,
    FEASIBILITY_LABEL,
    MEASUREMENT_ID,
    EXPERIMENT_GROUP,
    *STOCK_CAMPAIGN_COLUMNS,
)
"""All columns that parameter and output names must not shadow."""
