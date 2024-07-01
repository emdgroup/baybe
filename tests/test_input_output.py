"""Tests for basic input-output and iterative loop."""
import warnings

import numpy as np
import pandas as pd
import pytest

from baybe.constraints import DiscreteNoLabelDuplicatesConstraint
from baybe.exceptions import NoSearchspaceMatchWarning
from baybe.utils.augmentation import (
    df_apply_dependency_augmentation,
    df_apply_permutation_augmentation,
)
from baybe.utils.dataframe import add_fake_results


@pytest.mark.parametrize(
    "bad_val",
    [1337, np.nan, "asd"],
    ids=["not_within_tol", "nan", "string_instead_float"],
)
def test_bad_parameter_input_value(campaign, good_reference_values, bad_val, request):
    """Test attempting to read in an invalid parameter value."""
    rec = campaign.recommend(batch_size=3)
    add_fake_results(
        rec,
        campaign,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    with warnings.catch_warnings():
        # Ignore warning about incompatible data type assignment
        warnings.simplefilter("ignore", FutureWarning)
        rec.iloc[0, rec.columns.get_loc("Num_disc_1")] = bad_val

    with pytest.raises((ValueError, TypeError)):
        campaign.add_measurements(rec)


@pytest.mark.parametrize(
    "bad_val",
    [np.nan, "asd"],
    ids=["nan", "string_instead_float"],
)
def test_bad_target_input_value(campaign, good_reference_values, bad_val, request):
    """Test attempting to read in an invalid target value."""
    rec = campaign.recommend(batch_size=3)
    add_fake_results(
        rec,
        campaign,
        good_reference_values=good_reference_values,
    )

    # Add an invalid value
    with warnings.catch_warnings():
        # Ignore warning about incompatible data type assignment
        warnings.simplefilter("ignore", FutureWarning)
        rec.iloc[0, rec.columns.get_loc("Target_max")] = bad_val

    with pytest.raises((ValueError, TypeError)):
        campaign.add_measurements(rec)


# Reused parameter names for the mixture mock example
_mixture_columns = [
    "Solvent_1",
    "Solvent_2",
    "Solvent_3",
    "Fraction_1",
    "Fraction_2",
    "Fraction_3",
]


@pytest.mark.parametrize("n_grid_points", [5])
@pytest.mark.parametrize(
    "entry",
    [
        pd.DataFrame.from_records(
            [["THF", "Water", "DMF", 0.0, 25.0, 75.0]], columns=_mixture_columns
        ),
    ],
)
@pytest.mark.parametrize("parameter_names", [_mixture_columns])
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_7", "Constraint_11", "Constraint_12"]]
)
def test_permutation_invariant_input(campaign, entry):
    """Test whether permutation invariant measurements can be added."""
    add_fake_results(entry, campaign)

    # Create augmented combinations
    entries = df_apply_permutation_augmentation(
        entry,
        columns=["Solvent_1", "Solvent_2", "Solvent_3"],
        dependents=["Fraction_1", "Fraction_2", "Fraction_3"],
    )

    for _, row in entries.iterrows():
        # Reset searchspace metadata
        campaign.searchspace.discrete.metadata["was_measured"] = False

        # Assert that not NoSearchspaceMatchWarning is thrown
        with warnings.catch_warnings():
            print(row.to_frame().T)
            warnings.simplefilter("error", category=NoSearchspaceMatchWarning)
            campaign.add_measurements(pd.DataFrame([row]))

        # Assert exactly one searchspace entry has been marked
        num_nonzero = campaign.searchspace.discrete.metadata["was_measured"].sum()
        assert num_nonzero == 1, (
            "Measurement ingestion was successful, but did not correctly update the "
            f"searchspace metadata. Number of non-zero entries: {num_nonzero} "
            f"(expected 1)"
        )


@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
@pytest.mark.parametrize(
    "entry",
    [
        pd.DataFrame.from_records(
            [["THF", "Water", "DMF", 0.0, 25.0, 75.0]],
            columns=_mixture_columns,
        ),
        pd.DataFrame.from_records(
            [["THF", "Water", "DMF", 0.0, 0.0, 50.0]],
            columns=_mixture_columns,
        ),
    ],
    ids=["single_degen", "double_degen"],
)
@pytest.mark.parametrize("parameter_names", [_mixture_columns])
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_7", "Constraint_11", "Constraint_12"]]
)
def test_dependency_invariant_input(campaign, entry):
    """Test whether dependency invariant measurements can be added."""
    # Get an entry from the searchspace
    add_fake_results(entry, campaign)
    sol_vals = campaign.searchspace.get_parameters_by_name(["Solvent_1"])[0].values

    # Create augmented combinations
    entries = df_apply_dependency_augmentation(
        entry, causing=("Fraction_1", [0.0]), affected=[("Solvent_1", sol_vals)]
    )
    entries = df_apply_dependency_augmentation(
        entries, causing=("Fraction_2", [0.0]), affected=[("Solvent_2", sol_vals)]
    )
    entries = df_apply_dependency_augmentation(
        entries, causing=("Fraction_3", [0.0]), affected=[("Solvent_3", sol_vals)]
    )

    # Remove falsely created label duplicates
    entries.reset_index(drop=True, inplace=True)
    for c in campaign.searchspace.discrete.constraints:
        if isinstance(c, DiscreteNoLabelDuplicatesConstraint):
            entries.drop(index=c.get_invalid(entries), inplace=True)

    # Add nan entries for testing nan input in the invariant parameters
    entry_nan = entry.copy()
    entry_nan.loc[entry_nan["Fraction_1"] == 0.0, "Solvent_1"] = np.nan
    entry_nan.loc[entry_nan["Fraction_2"] == 0.0, "Solvent_2"] = np.nan
    entry_nan.loc[entry_nan["Fraction_3"] == 0.0, "Solvent_3"] = np.nan

    for _, row in pd.concat([entries, entry_nan]).iterrows():
        # Reset searchspace metadata
        campaign.searchspace.discrete.metadata["was_measured"] = False

        # Assert that not NoSearchspaceMatchWarning is thrown
        with warnings.catch_warnings():
            print(row.to_frame().T)
            warnings.simplefilter("error", category=NoSearchspaceMatchWarning)
            campaign.add_measurements(pd.DataFrame([row]))

        # Assert exactly one searchspace entry has been marked
        num_nonzero = campaign.searchspace.discrete.metadata["was_measured"].sum()
        assert num_nonzero == 1, (
            "Measurement ingestion was successful, but did not correctly update the "
            f"searchspace metadata. Number of non-zero entries: {num_nonzero} "
            f"(expected 1)"
        )
