# pylint: disable=missing-module-docstring, missing-function-docstring

import pytest
from baybe.searchspace import SearchSpace


# TODO: The serialization roundtrip does not yet yield an exact copy of the original
#   object for for all possible settings, due to the following reasons:
#   1)  Unions still need to be properly resolved (see also note in parameters.py).
#       That is: when serializing e.g. a field of typy Union[int, float], it must be
#       ensured that the deserialized type is correctly recovered, i.e. that a 1.0
#       is recovered as a float and not an int.
#   2)  The same is true e.g. for the types of the indexes in a pandas DataFrame.
#       For example, it was observed that the index type of an empty DataFrame changed
#       from "RangeIndex" to an Integer representation
#   3)  Finally, DataFrame equality should be checked only up to a certain numerical
#       precision, e.g. using pandas.testing.assert_frame_equal (similar to numpy's
#       allclose)


@pytest.mark.parametrize(
    "parameter_names",
    [
        ["Categorical_1", "Num_discrete_1"],
        pytest.param(
            ["Fraction_1"],
            marks=pytest.mark.xfail(
                reason="Serialization to be finalized (see TODO above)."
            ),
        ),
    ],
)
def test_searchspace_serialization(parameters):
    searchspace = SearchSpace.create(parameters)
    string = searchspace.to_json()
    searchspace2 = SearchSpace.from_json(string)
    assert searchspace == searchspace2
