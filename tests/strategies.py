"""Hypothesis strategies."""

import hypothesis.strategies as st
from hypothesis import assume

from baybe.exceptions import NumericalUnderflowError
from baybe.parameters.numerical import NumericalDiscreteParameter


@st.composite
def numerical_discrete_parameter(  # pylint: disable=inconsistent-return-statements
    draw: st.DrawFn,
):
    """Generates class:`baybe.parameters.numerical.NumericalDiscreteParameter`."""
    name = draw(st.text(min_size=1))
    values = draw(
        st.lists(
            st.one_of(
                st.integers(),
                st.floats(allow_infinity=False, allow_nan=False),
            ),
            min_size=2,
            unique=True,
        )
    )

    # Reject examples where the tolerance validator cannot be satisfied
    try:
        return NumericalDiscreteParameter(name=name, values=values)
    except NumericalUnderflowError:
        assume(False)


parameter = st.one_of([numerical_discrete_parameter()])
"""A strategy that creates parameters."""
