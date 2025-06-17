"""Tests for the objective module."""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from cattrs import IterableValidationError
from chimera import Chimera
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes

from baybe.objectives.chimera import ChimeraObjective, ThresholdType
from baybe.objectives.desirability import DesirabilityObjective, scalarize
from baybe.objectives.enum import Scalarizer
from baybe.objectives.single import SingleTargetObjective
from baybe.targets import NumericalTarget

from .hypothesis_strategies.basic import finite_floats
from .hypothesis_strategies.objectives import chimera_objectives


class TestInvalidObjectiveCreation:
    """Invalid objective creation raises expected error."""

    @property
    def two_targets(self):
        """Return two example numerical targets for testing."""
        return [
            NumericalTarget(
                name="Target_1",
                mode="MAX",
                bounds=(0, 100),
            ),
            NumericalTarget(
                name="Target_2",
                mode="MIN",
                bounds=(0, 100),
            ),
        ]

    def test_empty_target_list(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(targets=[])

    def test_wrong_target_type(self):
        with pytest.raises(TypeError):
            SingleTargetObjective(target={"A": 1, "B": 2})

    def test_missing_bounds_for_desirability(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(
                targets=[
                    NumericalTarget(
                        name="Target_1",
                        mode="MAX",
                        bounds=(0, 100),
                    ),
                    NumericalTarget(
                        name="Target_2",
                        mode="MIN",
                    ),
                ],
            )

    def test_invalid_combination_function(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(
                targets=self.two_targets,
                scalarizer="FALSE_STUFF",
            )

    def test_wrong_number_of_weights(self):
        with pytest.raises(ValueError):
            DesirabilityObjective(
                targets=self.two_targets,
                weights=[1, 2, 3],
            )

    def test_non_numeric_weights(self):
        with pytest.raises(IterableValidationError):
            DesirabilityObjective(
                targets=self.two_targets,
                weights=[1, "ABC"],
            )

    def test_wrong_weights_type(self):
        with pytest.raises(IterableValidationError):
            DesirabilityObjective(
                targets=self.two_targets,
                weights="ABC",
            )


@pytest.mark.parametrize(
    ("values", "scalarizer", "weights", "expected"),
    [
        ([[1, 2]], Scalarizer.MEAN, [1, 1], [1.5]),
        ([[1, 2]], Scalarizer.MEAN, [1, 2], [5 / 3]),
        ([[1, 2]], Scalarizer.GEOM_MEAN, [1, 1], [np.sqrt(2)]),
        ([[1, 2]], Scalarizer.GEOM_MEAN, [1, 2], [np.power(4, 1 / 3)]),
    ],
)
def test_desirability_scalarization(values, scalarizer, weights, expected):
    """The desirability scalarization yields the expected result."""
    actual = scalarize(values, scalarizer, weights)
    assert np.array_equal(actual, expected), (expected, actual)


class TestInvalidChimeraObjectiveCreation:
    """Tests for invalid ChimeraObjective creation."""

    @property
    def two_targets(self):
        """Return two example numerical targets for testing."""
        return [
            NumericalTarget(name="Target_1", mode="MIN", bounds=(0, 100)),
            NumericalTarget(name="Target_2", mode="MIN", bounds=(0, 100)),
        ]

    def test_empty_target_list(self):
        with pytest.raises(ValueError):
            ChimeraObjective(targets=[])

    def test_wrong_target_type(self):
        valid_target = NumericalTarget(name="Target_1", mode="MIN", bounds=(0, 100))
        with pytest.raises(TypeError):
            ChimeraObjective(targets=[valid_target, {"name": "NotATarget"}])

    def test_duplicate_target_names(self):
        target1 = NumericalTarget(name="Target_1", mode="MIN", bounds=(0, 100))
        target2 = NumericalTarget(name="Target_1", mode="MIN", bounds=(0, 100))
        with pytest.raises(ValueError):
            ChimeraObjective(targets=[target1, target2])

    def test_mismatched_threshold_values(self):
        with pytest.raises(ValueError):
            ChimeraObjective(
                targets=self.two_targets,
                threshold_values=(0.1,),
            )

    def test_mismatched_threshold_types(self):
        with pytest.raises(ValueError):
            ChimeraObjective(
                targets=self.two_targets,
                threshold_types=(ThresholdType.FRACTION,),
            )

    def test_non_numeric_threshold_values(self):
        with pytest.raises(IterableValidationError):
            ChimeraObjective(
                targets=self.two_targets,
                threshold_values=[0.1, "ABC"],
            )

    def test_wrong_threshold_values_type(self):
        with pytest.raises(IterableValidationError):
            ChimeraObjective(
                targets=self.two_targets,
                threshold_values="ABC",
            )

    def test_wrong_threshold_types_type(self):
        with pytest.raises(ValueError):
            ChimeraObjective(
                targets=self.two_targets,
                threshold_types="ABC",
            )

    def test_invalid_threshold_type(self):
        with pytest.raises(ValueError):
            ChimeraObjective(
                targets=self.two_targets,
                threshold_types=["INVALID", "INVALID"],
            )


def compare_merits(
    target_vals: pd.DataFrame,
    ext_chimera: Chimera,
    int_chimera: ChimeraObjective,
    verbose: bool = False,
) -> bool:
    """Compare merit values computed by the external and internal Chimera objects.

    Args:
        target_vals: DataFrame of target values with columns matching target names.
        ext_chimera: External Chimera object with a scalarize method that accepts
                     a transposed array of shape (n_points, n_targets).
        int_chimera: Internal ChimeraObjective with a transform method that accepts
                     a pandas DataFrame.
        verbose: If True, print the computed merits.

    Returns:
        True if the merit values computed by both methods are close.
    """
    # 1) Run the internal transform once to populate all internal buffers
    _ = int_chimera.transform(target_vals)
    int_merits = int_chimera.transform(target_vals)["Merit"].values

    # 2) Grab the exact transformed DataFrame / thresholds BayBE uses
    ext_chimera.goals = ["max"] * len(int_chimera.targets)
    # after transform, all are maximization problems
    ext_chimera.tolerances = np.array(int_chimera._threshold_values_transformed)
    target_vals_transformed = int_chimera._targets_transformed.copy()

    # 3) Compute external merits
    ext_merits = ext_chimera.scalarize(target_vals_transformed.values)

    # 6) Compare final merits
    if verbose:
        stages = [
            ("adjusted", int_chimera._targets_adjusted, ext_chimera._objs),
            ("rescaled", int_chimera._targets_rescaled, ext_chimera._scaled_objs),
            ("shifted", int_chimera._targets_shifted, ext_chimera._shifted_objs),
        ]

        thr_stages = [
            (
                "thr_adj",
                int_chimera._threshold_values_adjusted,
                ext_chimera._thresholds,
            ),
            (
                "thr_res",
                int_chimera._threshold_values_rescaled,
                ext_chimera._scaled_thresholds,
            ),
            (
                "thr_shift",
                int_chimera._threshold_values_shifted,
                ext_chimera._shifted_thresholds,
            ),
        ]

        print("\n=== PER-STAGE OBJECTIVE COMPARISON ===")
        for name, a_buf, b_buf in stages:
            ok = np.allclose(a_buf, b_buf, atol=1e-8, equal_nan=True)
            print(f"{name:10s}: {'OK' if ok else 'DIFF'}")
            if not ok:
                print("  BayBE:", a_buf)
                print("  ext  :", b_buf)

        print("\n=== PER-STAGE THRESHOLD COMPARISON ===")
        for name, a_thr, b_thr in thr_stages:
            ok = np.allclose(a_thr, b_thr, atol=1e-8, equal_nan=True)
            print(f"{name:10s}: {'OK' if ok else 'DIFF'}")
            if not ok:
                print("  BayBE:", a_thr)
                print("  ext  :", b_thr)

        print("\nfinal merits:")
        print("  BayBE:", int_merits)
        print("  ext  :", ext_merits)

    # Normalize external merits to match internal implementation
    ext_range = np.amax(ext_merits) - np.amin(ext_merits)
    if ext_range > 0:
        ext_merits = (ext_merits - np.amin(ext_merits)) / ext_range
    else:
        ext_merits = np.zeros_like(ext_merits)  # Handle uniform values
    return np.allclose(ext_merits, int_merits, atol=1e-6)


@settings(
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,  # Remove time limits for complex cases
)
@given(chimera_obj=chimera_objectives(), data=st.data())
def test_chimera_merits(chimera_obj, data):
    """Validating chimerra merits value with external reference."""
    # Step 1: Build a list of columns, one per target.
    columns = [
        column(
            name=target.name,
            elements=finite_floats(
                min_value=target.bounds.lower,
                max_value=target.bounds.upper,
                exclude_min=True,
                exclude_max=True,
            ),
        )
        for target in chimera_obj.targets
    ]

    target_vals = data.draw(
        data_frames(columns=columns, index=range_indexes(min_size=2)).filter(
            lambda df: len(df.drop_duplicates()) > 1
        )
    )

    # Step 2: Initialize external Chimera using data from chimera_obj.
    threshold_vals = np.array(chimera_obj.threshold_values)
    ext_chimera = Chimera(
        tolerances=threshold_vals,
        absolutes=[
            tt.value.upper() == "ABSOLUTE" for tt in chimera_obj.threshold_types
        ],
        percentiles=[
            tt.value.upper() == "PERCENTILE" for tt in chimera_obj.threshold_types
        ],
        goals=["max"] * len(chimera_obj.targets),
        softness=chimera_obj.softness,
    )
    # Step 3: Compare merits
    assert compare_merits(target_vals, ext_chimera, chimera_obj)
