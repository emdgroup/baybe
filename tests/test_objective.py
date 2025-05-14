"""Tests for the objective module."""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from cattrs import IterableValidationError
from chimera import Chimera
from hypothesis import assume, given, settings
from hypothesis.extra.pandas import column, data_frames

from baybe.objectives.chimera import ChimeraObjective, ThresholdType
from baybe.objectives.desirability import DesirabilityObjective, scalarize
from baybe.objectives.enum import Scalarizer
from baybe.objectives.single import SingleTargetObjective
from baybe.targets import NumericalTarget

from .hypothesis_strategies.objectives import chimera_objectives


class TestInvalidObjectiveCreation:
    """Invalid objective creation raises expected error."""

    # Two example targets used in the tests
    two_targets = [
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

    two_targets = [
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
    verbose: bool = True,
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
    # Internal ChimeraObjective works directly with the DataFrame:
    int_merits = int_chimera.transform(target_vals).values.flatten()

    # External Chimera needs a transformed DataFrame and transformed threshold values
    # --- build transformed dataframe (identical to int_chimera._targets_transformed)
    target_vals_transformed = target_vals.copy()
    _threshold_values_transformed = np.array(int_chimera.threshold_values)
    ext_absolutes = []

    for target, ttype, tval in zip(
        int_chimera.targets, int_chimera.threshold_types, int_chimera.threshold_values
    ):
        # bounds-scale then invert the data column
        col_tr = 1.0 - target.transform(target_vals[target.name])
        target_vals_transformed[target.name] = col_tr

        if ttype is ThresholdType.ABSOLUTE:
            # transform + invert, but **do not** min-max scale here
            thr_tr = 1.0 - target.transform(pd.Series([tval])).values[0]
            _threshold_values_transformed[int_chimera.targets.index(target)] = thr_tr
            ext_absolutes.append(
                True
            )  # TODO: this is wrong, no append, always indexing
        else:
            ext_absolutes.append(False)

    # ----------- sanity prints --------------------------------------------------
    print(
        "transformed df are the same:",
        np.allclose(
            target_vals_transformed.values, int_chimera._targets_transformed.values
        ),
    )
    print(
        "transformed thresholds are the same:",
        np.allclose(
            _threshold_values_transformed, int_chimera._threshold_values_transformed
        ),
    )

    # ----------- feed external Chimera -----------------------------------------
    ext_chimera.absolutes = ext_absolutes
    ext_chimera.tolerances = _threshold_values_transformed
    ext_chimera.goals = ["min"] * len(ext_chimera.goals)

    print("Index Error?:", ext_absolutes)
    print("Index Error?:", target_vals_transformed.values)
    ext_merits = ext_chimera.scalarize(target_vals_transformed.values)

    are_close = np.allclose(ext_merits, int_merits, atol=1e-5)
    if verbose:
        print("Chimera merits:", ext_merits)
        print("BayBE Chimera merits:", int_merits)
    return are_close


@settings(max_examples=2)
@given(chimera_obj=chimera_objectives(), data=st.data())
def test_chimera_merits(chimera_obj, data):
    """Validating chimerra merits value with external reference."""
    # Initialize external Chimera using data from chimera_obj.
    goals = [t.mode.value.lower() for t in chimera_obj.targets]
    threshold_vals = np.array(chimera_obj.threshold_values)
    ext_chimera = Chimera(
        tolerances=threshold_vals,
        absolutes=[
            True
            for tt in chimera_obj.threshold_types
            # TODO: because of the transform, all become absolute
            # tt.value.upper() == "ABSOLUTE" for tt in chimera_obj.threshold_types
        ],
        goals=[g.lower() for g in goals],
        softness=chimera_obj.softness,
    )

    # Build a list of columns, one per target.
    columns = [
        column(
            name=t.name,
            elements=st.floats(
                min_value=t.bounds.lower
                if chimera_obj.threshold_types[i] == ThresholdType.ABSOLUTE
                else 0.0,
                max_value=t.bounds.upper
                if chimera_obj.threshold_types[i] == ThresholdType.ABSOLUTE
                else 1.0,
                allow_nan=False,
                allow_infinity=False,
                exclude_min=True,
                exclude_max=True,
            ),
        )
        for i, t in enumerate(chimera_obj.targets)
    ]

    target_vals = data.draw(data_frames(columns=columns))
    assume(target_vals.shape[0] >= 6)

    print("Target DataFrame:\n", target_vals)

    assert compare_merits(target_vals, ext_chimera, chimera_obj)
